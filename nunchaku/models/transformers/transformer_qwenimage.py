import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
from diffusers.models.attention_processor import Attention
from diffusers.models.transformers.transformer_qwenimage import QwenImageTransformer2DModel, QwenImageTransformerBlock
from huggingface_hub import utils

from ...utils import get_precision
from ..attention import NunchakuBaseAttention, NunchakuFeedForward
from ..linear import AWQW4A16Linear, SVDQW4A4Linear
from ..utils import fuse_linears
from .utils import NunchakuModelLoaderMixin


class NunchakuQwenAttention(NunchakuBaseAttention):
    def __init__(self, other: Attention, processor: str = "flashattn2", **kwargs):
        super(NunchakuQwenAttention, self).__init__(processor)
        self.inner_dim = other.inner_dim
        self.inner_kv_dim = other.inner_kv_dim
        self.query_dim = other.query_dim
        self.use_bias = other.use_bias
        self.is_cross_attention = other.is_cross_attention
        self.cross_attention_dim = other.cross_attention_dim
        self.upcast_attention = other.upcast_attention
        self.upcast_softmax = other.upcast_softmax
        self.rescale_output_factor = other.rescale_output_factor
        self.residual_connection = other.residual_connection
        self.dropout = other.dropout
        self.fused_projections = other.fused_projections
        self.out_dim = other.out_dim
        self.out_context_dim = other.out_context_dim
        self.context_pre_only = other.context_pre_only
        self.pre_only = other.pre_only
        self.is_causal = other.is_causal
        self.scale_qk = other.scale_qk
        self.scale = other.scale
        self.heads = other.heads
        self.sliceable_head_dim = other.sliceable_head_dim
        self.added_kv_proj_dim = other.added_kv_proj_dim
        self.only_cross_attention = other.only_cross_attention
        self.group_norm = other.group_norm
        self.spatial_norm = other.spatial_norm

        self.norm_cross = other.norm_cross

        self.norm_q = other.norm_q
        self.norm_k = other.norm_k
        self.norm_added_q = other.norm_added_q
        self.norm_added_k = other.norm_added_k

        # fuse the qkv
        with torch.device("meta"):
            to_qkv = fuse_linears([other.to_q, other.to_k, other.to_v])
        self.to_qkv = SVDQW4A4Linear.from_linear(to_qkv, **kwargs)
        self.to_out = other.to_out
        self.to_out[0] = SVDQW4A4Linear.from_linear(self.to_out[0], **kwargs)

        assert self.added_kv_proj_dim is not None
        # fuse the add_qkv
        with torch.device("meta"):
            add_qkv_proj = fuse_linears([other.add_q_proj, other.add_k_proj, other.add_v_proj])
        self.add_qkv_proj = SVDQW4A4Linear.from_linear(add_qkv_proj, **kwargs)
        self.to_add_out = SVDQW4A4Linear.from_linear(other.to_add_out, **kwargs)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        encoder_hidden_states_mask: torch.Tensor,
        attention_mask: torch.Tensor,
        image_rotary_emb: torch.Tensor,
        **kwargs,
    ):
        raise NotImplementedError("NunchakuQwenAttention is not implemented")


class NunchakuQwenImageTransformerBlock(QwenImageTransformerBlock):
    def __init__(self, other: QwenImageTransformerBlock, scale_shift: float = 0.0, **kwargs):
        super(QwenImageTransformerBlock, self).__init__()

        self.dim = other.dim
        self.img_mod = other.img_mod
        self.img_mod[1] = AWQW4A16Linear(self.img_mod[1])
        self.img_norm1 = self.img_norm1
        self.attn = NunchakuQwenAttention(other.attn)
        self.img_norm2 = self.img_norm2
        self.img_mlp = NunchakuFeedForward(self.img_mlp)

        # Text processing modules
        self.txt_mod = self.txt_mod
        self.txt_mod[1] = AWQW4A16Linear(self.txt_mod[1])
        self.txt_norm1 = self.txt_norm1
        # Text doesn't need separate attention - it's handled by img_attn joint computation
        self.txt_norm2 = self.txt_norm2
        self.txt_mlp = NunchakuFeedForward(self.txt_mlp)

        self.scale_shift = scale_shift

    def _modulate(self, x: torch.Tensor, mod_params: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply modulation to input tensor"""
        mod_params = mod_params.view(mod_params.shape[0], -1, 3).permute(2, 0, 1)
        shift, scale, gate = mod_params
        if self.scale_shift != 0:
            scale.add_(self.scale_shift)
        return x * scale.unsqueeze(1) + shift.unsqueeze(1), gate.unsqueeze(1)


class NunchakuQwenImageTransformer2DModel(QwenImageTransformer2DModel, NunchakuModelLoaderMixin):

    def _patch_model(self, **kwargs):
        for i, block in enumerate(self.transformer_blocks):
            self.transformer_blocks[i] = NunchakuQwenImageTransformerBlock(block, scale_shift=0, **kwargs)
        return self

    @classmethod
    @utils.validate_hf_hub_args
    def from_pretrained(cls, pretrained_model_name_or_path: str | os.PathLike[str], **kwargs):
        device = kwargs.get("device", "cpu")
        offload = kwargs.get("offload", False)

        if offload:
            raise NotImplementedError("Offload is not supported for FluxTransformer2DModelV2")

        torch_dtype = kwargs.get("torch_dtype", torch.bfloat16)

        if isinstance(pretrained_model_name_or_path, str):
            pretrained_model_name_or_path = Path(pretrained_model_name_or_path)

        assert pretrained_model_name_or_path.is_file() or pretrained_model_name_or_path.name.endswith(
            (".safetensors", ".sft")
        ), "Only safetensors are supported"
        transformer, model_state_dict, metadata = cls._build_model(pretrained_model_name_or_path, **kwargs)
        transformer = transformer.to(torch_dtype)

        precision = get_precision()
        if precision == "fp4":
            precision = "nvfp4"
        transformer._patch_model(precision=precision)

        transformer = transformer.to_empty(device=device)
        transformer.load_state_dict(model_state_dict)

        return transformer
