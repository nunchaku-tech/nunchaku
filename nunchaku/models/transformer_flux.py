import os

import diffusers
import torch
from diffusers import FluxTransformer2DModel
from diffusers.configuration_utils import register_to_config
from huggingface_hub import utils
from packaging.version import Version
from torch import nn

from .utils import NunchakuModelLoaderMixin, pad_tensor
from .._C import QuantizedFluxModel, utils as cutils
from ..utils import fetch_or_download

SVD_RANK = 32


class NunchakuFluxTransformerBlocks(nn.Module):
    def __init__(self, m: QuantizedFluxModel, device: str | torch.device):
        super(NunchakuFluxTransformerBlocks, self).__init__()
        self.m = m
        self.dtype = torch.bfloat16
        self.device = device

    def forward(
        self,
        hidden_states: torch.Tensor,
        temb: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        image_rotary_emb: torch.Tensor,
        joint_attention_kwargs=None,
        skip_first_layer=False,
    ):
        batch_size = hidden_states.shape[0]
        txt_tokens = encoder_hidden_states.shape[1]
        img_tokens = hidden_states.shape[1]

        original_dtype = hidden_states.dtype
        original_device = hidden_states.device

        hidden_states = hidden_states.to(self.dtype).to(self.device)
        encoder_hidden_states = encoder_hidden_states.to(self.dtype).to(self.device)
        temb = temb.to(self.dtype).to(self.device)
        image_rotary_emb = image_rotary_emb.to(self.device)

        assert image_rotary_emb.ndim == 6
        assert image_rotary_emb.shape[0] == 1
        assert image_rotary_emb.shape[1] == 1
        assert image_rotary_emb.shape[2] == batch_size * (txt_tokens + img_tokens)
        # [bs, tokens, head_dim / 2, 1, 2] (sincos)
        image_rotary_emb = image_rotary_emb.reshape([batch_size, txt_tokens + img_tokens, *image_rotary_emb.shape[3:]])
        rotary_emb_txt = image_rotary_emb[:, :txt_tokens, ...]  # .to(self.dtype)
        rotary_emb_img = image_rotary_emb[:, txt_tokens:, ...]  # .to(self.dtype)
        rotary_emb_single = image_rotary_emb  # .to(self.dtype)

        rotary_emb_txt = pad_tensor(rotary_emb_txt, 256, 1)
        rotary_emb_img = pad_tensor(rotary_emb_img, 256, 1)
        rotary_emb_single = pad_tensor(rotary_emb_single, 256, 1)

        hidden_states = self.m.forward(
            hidden_states, encoder_hidden_states, temb, rotary_emb_img, rotary_emb_txt, rotary_emb_single, skip_first_layer
        )

        hidden_states = hidden_states.to(original_dtype).to(original_device)

        encoder_hidden_states = hidden_states[:, :txt_tokens, ...]
        hidden_states = hidden_states[:, txt_tokens:, ...]

        return encoder_hidden_states, hidden_states

    def forward_layer_at(
        self,
        idx: int,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb: torch.Tensor,
        image_rotary_emb: torch.Tensor,
        joint_attention_kwargs=None,
    ):
        batch_size = hidden_states.shape[0]
        txt_tokens = encoder_hidden_states.shape[1]
        img_tokens = hidden_states.shape[1]

        original_dtype = hidden_states.dtype
        original_device = hidden_states.device

        hidden_states = hidden_states.to(self.dtype).to(self.device)
        encoder_hidden_states = encoder_hidden_states.to(self.dtype).to(self.device)
        temb = temb.to(self.dtype).to(self.device)
        image_rotary_emb = image_rotary_emb.to(self.device)

        assert image_rotary_emb.ndim == 6
        assert image_rotary_emb.shape[0] == 1
        assert image_rotary_emb.shape[1] == 1
        assert image_rotary_emb.shape[2] == batch_size * (txt_tokens + img_tokens)
        # [bs, tokens, head_dim / 2, 1, 2] (sincos)
        image_rotary_emb = image_rotary_emb.reshape([batch_size, txt_tokens + img_tokens, *image_rotary_emb.shape[3:]])
        rotary_emb_txt = image_rotary_emb[:, :txt_tokens, ...]  # .to(self.dtype)
        rotary_emb_img = image_rotary_emb[:, txt_tokens:, ...]  # .to(self.dtype)

        rotary_emb_txt = pad_tensor(rotary_emb_txt, 256, 1)
        rotary_emb_img = pad_tensor(rotary_emb_img, 256, 1)

        hidden_states, encoder_hidden_states = self.m.forward_layer(
                0, hidden_states, encoder_hidden_states, temb, rotary_emb_img, rotary_emb_txt)

        hidden_states = hidden_states.to(original_dtype).to(original_device)
        encoder_hidden_states = encoder_hidden_states.to(original_dtype).to(original_device)

        return encoder_hidden_states, hidden_states


## copied from diffusers 0.30.3
def rope(pos: torch.Tensor, dim: int, theta: int) -> torch.Tensor:
    assert dim % 2 == 0, "The dimension must be even."

    scale = torch.arange(0, dim, 2, dtype=torch.float64, device=pos.device) / dim
    omega = 1.0 / (theta**scale)

    batch_size, seq_length = pos.shape
    out = torch.einsum("...n,d->...nd", pos, omega)

    USE_SINCOS = True
    if USE_SINCOS:
        cos_out = torch.cos(out)
        sin_out = torch.sin(out)
        stacked_out = torch.stack([sin_out, cos_out], dim=-1)
        out = stacked_out.view(batch_size, -1, dim // 2, 1, 2)
    else:
        out = out.view(batch_size, -1, dim // 2, 1, 1)

    # stacked_out = torch.stack([cos_out, -sin_out, sin_out, cos_out], dim=-1)
    # out = stacked_out.view(batch_size, -1, dim // 2, 2, 2)
    return out.float()


class EmbedND(nn.Module):
    def __init__(self, dim: int, theta: int, axes_dim: list[int]):
        super(EmbedND, self).__init__()
        self.dim = dim
        self.theta = theta
        self.axes_dim = axes_dim

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        if Version(diffusers.__version__) >= Version("0.31.0"):
            ids = ids[None, ...]
        n_axes = ids.shape[-1]
        emb = torch.cat([rope(ids[..., i], self.axes_dim[i], self.theta) for i in range(n_axes)], dim=-3)
        return emb.unsqueeze(1)


def load_quantized_module(path: str, device: str | torch.device = "cuda", use_fp4: bool = False) -> QuantizedFluxModel:
    device = torch.device(device)
    assert device.type == "cuda"
    m = QuantizedFluxModel()
    cutils.disable_memory_auto_release()
    m.init(use_fp4, True, 0 if device.index is None else device.index)
    m.load(path)
    return m


class NunchakuFluxTransformer2dModel(FluxTransformer2DModel, NunchakuModelLoaderMixin):
    @register_to_config
    def __init__(
        self,
        patch_size: int = 1,
        in_channels: int = 64,
        out_channels: int | None = None,
        num_layers: int = 19,
        num_single_layers: int = 38,
        attention_head_dim: int = 128,
        num_attention_heads: int = 24,
        joint_attention_dim: int = 4096,
        pooled_projection_dim: int = 768,
        guidance_embeds: bool = False,
        axes_dims_rope: tuple[int] = (16, 56, 56),
    ):
        super(NunchakuFluxTransformer2dModel, self).__init__(
            patch_size=patch_size,
            in_channels=in_channels,
            out_channels=out_channels,
            num_layers=0,
            num_single_layers=0,
            attention_head_dim=attention_head_dim,
            num_attention_heads=num_attention_heads,
            joint_attention_dim=joint_attention_dim,
            pooled_projection_dim=pooled_projection_dim,
            guidance_embeds=guidance_embeds,
            axes_dims_rope=axes_dims_rope,
        )

    @classmethod
    @utils.validate_hf_hub_args
    def from_pretrained(cls, pretrained_model_name_or_path: str | os.PathLike, **kwargs):
        device = kwargs.get("device", "cuda")
        precision = kwargs.get("precision", "int4")
        assert precision in ["int4", "fp4"]
        transformer, transformer_block_path = cls._build_model(pretrained_model_name_or_path, **kwargs)
        m = load_quantized_module(transformer_block_path, device=device, use_fp4=precision == "fp4")
        transformer.inject_quantized_module(m, device)
        return transformer

    def update_lora_params(self, path: str):
        path = fetch_or_download(path)
        block = self.transformer_blocks[0]
        assert isinstance(block, NunchakuFluxTransformerBlocks)
        block.m.load(path, True)

    def set_lora_strength(self, strength: float = 1):
        block = self.transformer_blocks[0]
        assert isinstance(block, NunchakuFluxTransformerBlocks)
        block.m.setLoraScale(SVD_RANK, strength)

    def inject_quantized_module(self, m: QuantizedFluxModel, device: str | torch.device = "cuda"):
        print("Injecting quantized module")
        self.pos_embed = EmbedND(dim=self.inner_dim, theta=10000, axes_dim=[16, 56, 56])

        ### Compatible with the original forward method
        self.transformer_blocks = nn.ModuleList([NunchakuFluxTransformerBlocks(m, device)])
        self.single_transformer_blocks = nn.ModuleList([])

        return self
