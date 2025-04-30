import os
from typing import Optional

import torch
from diffusers import SanaTransformer2DModel
from huggingface_hub import utils
from safetensors.torch import load_file
from torch import nn
from torch.nn import functional as F

from ..._C import QuantizedSanaModel
from ..._C import utils as cutils
from ...utils import get_precision
from .utils import NunchakuModelLoaderMixin

SVD_RANK = 32


class NunchakuSanaTransformerBlocks(nn.Module):
    def __init__(self, m: QuantizedSanaModel, dtype: torch.dtype, device: str | torch.device):
        super(NunchakuSanaTransformerBlocks, self).__init__()
        self.m = m
        self.dtype = dtype
        self.device = device

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        timestep: Optional[torch.LongTensor] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        skip_first_layer: Optional[bool] = False,
    ):

        batch_size = hidden_states.shape[0]
        img_tokens = hidden_states.shape[1]
        txt_tokens = encoder_hidden_states.shape[1]

        original_dtype = hidden_states.dtype
        original_device = hidden_states.device

        assert encoder_attention_mask is not None
        assert encoder_attention_mask.shape == (batch_size, 1, txt_tokens)

        mask = encoder_attention_mask.reshape(batch_size, txt_tokens)
        nunchaku_encoder_hidden_states = encoder_hidden_states[mask > -9000]

        cu_seqlens_txt = F.pad((mask > -9000).sum(dim=1).cumsum(dim=0), pad=(1, 0), value=0).to(torch.int32)
        cu_seqlens_img = torch.arange(
            0, (batch_size + 1) * img_tokens, img_tokens, dtype=torch.int32, device=self.device
        )

        if height is None and width is None:
            height = width = int(img_tokens**0.5)
        elif height is None:
            height = img_tokens // width
        elif width is None:
            width = img_tokens // height
        assert height * width == img_tokens

        return (
            self.m.forward(
                hidden_states.to(self.dtype).to(self.device),
                nunchaku_encoder_hidden_states.to(self.dtype).to(self.device),
                timestep.to(self.dtype).to(self.device),
                cu_seqlens_img.to(self.device),
                cu_seqlens_txt.to(self.device),
                height,
                width,
                batch_size % 3 == 0,  # pag is set when loading the model, FIXME: pag_scale == 0
                True,  # TODO: find a way to detect if we are doing CFG
                skip_first_layer,
            )
            .to(original_dtype)
            .to(original_device)
        )

    def forward_layer_at(
        self,
        idx: int,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        timestep: Optional[torch.LongTensor] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
    ):
        batch_size = hidden_states.shape[0]
        img_tokens = hidden_states.shape[1]
        txt_tokens = encoder_hidden_states.shape[1]

        original_dtype = hidden_states.dtype
        original_device = hidden_states.device

        assert encoder_attention_mask is not None
        assert encoder_attention_mask.shape == (batch_size, 1, txt_tokens)

        mask = encoder_attention_mask.reshape(batch_size, txt_tokens)
        nunchaku_encoder_hidden_states = encoder_hidden_states[mask > -9000]

        cu_seqlens_txt = F.pad((mask > -9000).sum(dim=1).cumsum(dim=0), pad=(1, 0), value=0).to(torch.int32)
        cu_seqlens_img = torch.arange(
            0, (batch_size + 1) * img_tokens, img_tokens, dtype=torch.int32, device=self.device
        )

        if height is None and width is None:
            height = width = int(img_tokens**0.5)
        elif height is None:
            height = img_tokens // width
        elif width is None:
            width = img_tokens // height
        assert height * width == img_tokens

        return (
            self.m.forward_layer(
                idx,
                hidden_states.to(self.dtype).to(self.device),
                nunchaku_encoder_hidden_states.to(self.dtype).to(self.device),
                timestep.to(self.dtype).to(self.device),
                cu_seqlens_img.to(self.device),
                cu_seqlens_txt.to(self.device),
                height,
                width,
                batch_size % 3 == 0,  # pag is set when loading the model, FIXME: pag_scale == 0
                True,  # TODO: find a way to detect if we are doing CFG
            )
            .to(original_dtype)
            .to(original_device)
        )

    def __del__(self):
        self.m.reset()


class NunchakuSanaTransformer2DModel(SanaTransformer2DModel, NunchakuModelLoaderMixin):
    @classmethod
    @utils.validate_hf_hub_args
    def from_pretrained(cls, pretrained_model_name_or_path: str | os.PathLike, **kwargs):
        device = kwargs.get("device", "cuda")
        pag_layers = kwargs.get("pag_layers", [])
        precision = get_precision(kwargs.get("precision", "auto"), device, pretrained_model_name_or_path)
        transformer, unquantized_part_path, transformer_block_path = cls._build_model(
            pretrained_model_name_or_path, **kwargs
        )
        m = load_quantized_module(
            transformer, transformer_block_path, device=device, pag_layers=pag_layers, use_fp4=precision == "fp4"
        )
        transformer.inject_quantized_module(m, device)
        transformer.to_empty(device=device)
        unquantized_state_dict = load_file(unquantized_part_path)
        transformer.load_state_dict(unquantized_state_dict, strict=False)
        return transformer

    def inject_quantized_module(self, m: QuantizedSanaModel, device: str | torch.device = "cuda"):
        self.transformer_blocks = torch.nn.ModuleList([NunchakuSanaTransformerBlocks(m, self.dtype, device)])
        return self


def load_quantized_module(
    net: SanaTransformer2DModel,
    path: str,
    device: str | torch.device = "cuda",
    pag_layers: int | list[int] | None = None,
    use_fp4: bool = False,
) -> QuantizedSanaModel:
    if pag_layers is None:
        pag_layers = []
    elif isinstance(pag_layers, int):
        pag_layers = [pag_layers]
    device = torch.device(device)
    assert device.type == "cuda"

    m = QuantizedSanaModel()
    cutils.disable_memory_auto_release()
    m.init(net.config, pag_layers, use_fp4, net.dtype == torch.bfloat16, 0 if device.index is None else device.index)
    m.load(path)
    return m


def inject_quantized_module(
    net: SanaTransformer2DModel, m: QuantizedSanaModel, device: torch.device
) -> SanaTransformer2DModel:
    net.transformer_blocks = torch.nn.ModuleList([NunchakuSanaTransformerBlocks(m, net.dtype, device)])
    return net
