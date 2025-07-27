"""
Implements the :class:`NunchakuSanaTransformer2DModel`,
a quantized Sana transformer for Diffusers with efficient inference support.
"""

import os
from pathlib import Path
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
    """
    Wrapper for quantized Sana transformer blocks.

    This module wraps a QuantizedSanaModel and provides forward methods compatible
    with the expected transformer block interface.

    Parameters
    ----------
    m : QuantizedSanaModel
        The quantized transformer model.
    dtype : torch.dtype
        The data type to use for computation.
    device : str or torch.device
        The device to run the model on.
    """

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
        """
        Forward pass through all quantized transformer blocks.

        Parameters
        ----------
        hidden_states : torch.Tensor
            Input hidden states of shape (batch_size, img_tokens, ...).
        attention_mask : torch.Tensor, optional
            Not used.
        encoder_hidden_states : torch.Tensor, optional
            Encoder hidden states of shape (batch_size, txt_tokens, ...).
        encoder_attention_mask : torch.Tensor, optional
            Encoder attention mask of shape (batch_size, 1, txt_tokens).
        timestep : torch.LongTensor, optional
            Timestep tensor.
        height : int, optional
            Image height.
        width : int, optional
            Image width.
        skip_first_layer : bool, optional
            Whether to skip the first layer.

        Returns
        -------
        torch.Tensor
            Output tensor after passing through the quantized transformer blocks.
        """
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
        """
        Forward pass through a specific quantized transformer layer.

        Parameters
        ----------
        idx : int
            Index of the layer to run.
        hidden_states : torch.Tensor
            Input hidden states.
        attention_mask : torch.Tensor, optional
            Not used.
        encoder_hidden_states : torch.Tensor, optional
            Encoder hidden states.
        encoder_attention_mask : torch.Tensor, optional
            Encoder attention mask.
        timestep : torch.LongTensor, optional
            Timestep tensor.
        height : int, optional
            Image height.
        width : int, optional
            Image width.

        Returns
        -------
        torch.Tensor
            Output tensor after passing through the specified quantized transformer layer.
        """
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
        """
        Destructor to reset the quantized model and free resources.
        """
        self.m.reset()


class NunchakuSanaTransformer2DModel(SanaTransformer2DModel, NunchakuModelLoaderMixin):
    """
    SanaTransformer2DModel with Nunchaku quantized backend support.

    This class extends the base SanaTransformer2DModel to support loading and
    injecting quantized transformer blocks using Nunchaku's custom backend.
    """

    @classmethod
    @utils.validate_hf_hub_args
    def from_pretrained(cls, pretrained_model_name_or_path: str | os.PathLike[str], **kwargs):
        """
        Load a pretrained NunchakuSanaTransformer2DModel from a local file or HuggingFace Hub.

        This method supports both quantized and unquantized checkpoints, and will
        automatically inject quantized transformer blocks if available.

        Parameters
        ----------
        pretrained_model_name_or_path : str or os.PathLike
            Path to the model checkpoint or HuggingFace Hub model name.
        **kwargs
            Additional keyword arguments for model loading.

        Returns
        -------
        NunchakuSanaTransformer2DModel or (NunchakuSanaTransformer2DModel, dict)
            The loaded model, and optionally metadata if ``return_metadata=True``.
        """
        device = kwargs.get("device", "cuda")
        if isinstance(device, str):
            device = torch.device(device)
        pag_layers = kwargs.get("pag_layers", [])
        precision = get_precision(kwargs.get("precision", "auto"), device, pretrained_model_name_or_path)
        metadata = None

        if isinstance(pretrained_model_name_or_path, str):
            pretrained_model_name_or_path = Path(pretrained_model_name_or_path)
        if pretrained_model_name_or_path.is_file() or pretrained_model_name_or_path.name.endswith(
            (".safetensors", ".sft")
        ):
            transformer, model_state_dict, metadata = cls._build_model(pretrained_model_name_or_path)
            quantized_part_sd = {}
            unquantized_part_sd = {}
            for k, v in model_state_dict.items():
                if k.startswith("transformer_blocks."):
                    quantized_part_sd[k] = v
                else:
                    unquantized_part_sd[k] = v
            m = load_quantized_module(
                transformer, quantized_part_sd, device=device, pag_layers=pag_layers, use_fp4=precision == "fp4"
            )
            transformer.inject_quantized_module(m, device)
            transformer.to_empty(device=device)
            transformer.load_state_dict(unquantized_part_sd, strict=False)
        else:
            transformer, unquantized_part_path, transformer_block_path = cls._build_model_legacy(
                pretrained_model_name_or_path, **kwargs
            )
            m = load_quantized_module(
                transformer, transformer_block_path, device=device, pag_layers=pag_layers, use_fp4=precision == "fp4"
            )
            transformer.inject_quantized_module(m, device)
            transformer.to_empty(device=device)
            unquantized_state_dict = load_file(unquantized_part_path)
            transformer.load_state_dict(unquantized_state_dict, strict=False)
        if kwargs.get("return_metadata", False):
            return transformer, metadata
        else:
            return transformer

    def inject_quantized_module(self, m: QuantizedSanaModel, device: str | torch.device = "cuda"):
        """
        Inject a quantized transformer module into this model.

        Parameters
        ----------
        m : QuantizedSanaModel
            The quantized transformer module to inject.
        device : str or torch.device, optional
            The device to place the module on (default: "cuda").

        Returns
        -------
        NunchakuSanaTransformer2DModel
            The model with the quantized module injected.
        """
        self.transformer_blocks = torch.nn.ModuleList([NunchakuSanaTransformerBlocks(m, self.dtype, device)])
        return self


def load_quantized_module(
    net: SanaTransformer2DModel,
    path_or_state_dict: str | os.PathLike[str] | dict[str, torch.Tensor],
    device: str | torch.device = "cuda",
    pag_layers: int | list[int] | None = None,
    use_fp4: bool = False,
) -> QuantizedSanaModel:
    """
    Load quantized weights into a QuantizedSanaModel.

    Parameters
    ----------
    net : SanaTransformer2DModel
        The base transformer model (for config and dtype).
    path_or_state_dict : str, os.PathLike, or dict
        Path to the quantized weights or a state dict.
    device : str or torch.device, optional
        Device to load the quantized model on (default: "cuda").
    pag_layers : int, list of int, or None, optional
        List of layers to use pag (default: None).
    use_fp4 : bool, optional
        Whether to use FP4 quantization (default: False).

    Returns
    -------
    QuantizedSanaModel
        The loaded quantized model.
    """
    if pag_layers is None:
        pag_layers = []
    elif isinstance(pag_layers, int):
        pag_layers = [pag_layers]
    device = torch.device(device)
    assert device.type == "cuda"

    m = QuantizedSanaModel()
    cutils.disable_memory_auto_release()
    m.init(net.config, pag_layers, use_fp4, net.dtype == torch.bfloat16, 0 if device.index is None else device.index)
    if isinstance(path_or_state_dict, dict):
        m.loadDict(path_or_state_dict, True)
    else:
        m.load(str(path_or_state_dict))
    return m


def inject_quantized_module(
    net: SanaTransformer2DModel, m: QuantizedSanaModel, device: torch.device
) -> SanaTransformer2DModel:
    """
    Inject a quantized transformer module into a SanaTransformer2DModel.

    Parameters
    ----------
    net : SanaTransformer2DModel
        The base transformer model.
    m : QuantizedSanaModel
        The quantized transformer module to inject.
    device : torch.device
        The device to place the module on.

    Returns
    -------
    SanaTransformer2DModel
        The model with the quantized module injected.
    """
    net.transformer_blocks = torch.nn.ModuleList([NunchakuSanaTransformerBlocks(m, net.dtype, device)])
    return net
