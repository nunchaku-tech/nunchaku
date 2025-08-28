import json
import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
from diffusers.models.attention_processor import Attention
from diffusers.models.transformers.transformer_qwenimage import (
    QwenEmbedRope,
    QwenImageTransformer2DModel,
    QwenImageTransformerBlock,
)
from huggingface_hub import utils

from ...utils import get_precision
from ..attention import NunchakuBaseAttention, NunchakuFeedForward
from ..attention_processors.qwenimage import NunchakuQwenImageNaiveFA2Processor
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
        hidden_states: torch.FloatTensor,  # Image stream
        encoder_hidden_states: torch.FloatTensor = None,  # Text stream
        encoder_hidden_states_mask: torch.FloatTensor = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        return self.processor(
            self,
            hidden_states,
            encoder_hidden_states,
            encoder_hidden_states_mask,
            attention_mask,
            image_rotary_emb,
            **kwargs,
        )

    def set_processor(self, processor: str):
        if processor == "flashattn2":
            self.processor = NunchakuQwenImageNaiveFA2Processor()
        else:
            raise ValueError(f"Processor {processor} is not supported")


class NunchakuQwenImageTransformerBlock(QwenImageTransformerBlock):
    def __init__(self, other: QwenImageTransformerBlock, scale_shift: float = 1.0, **kwargs):
        super(QwenImageTransformerBlock, self).__init__()

        self.dim = other.dim
        self.img_mod = other.img_mod
        self.img_mod[1] = AWQW4A16Linear.from_linear(other.img_mod[1], **kwargs)
        self.img_norm1 = other.img_norm1
        self.attn = NunchakuQwenAttention(other.attn, **kwargs)
        self.img_norm2 = other.img_norm2
        self.img_mlp = NunchakuFeedForward(other.img_mlp, **kwargs)

        # Text processing modules
        self.txt_mod = other.txt_mod
        self.txt_mod[1] = AWQW4A16Linear.from_linear(other.txt_mod[1], **kwargs)
        self.txt_norm1 = other.txt_norm1
        # Text doesn't need separate attention - it's handled by img_attn joint computation
        self.txt_norm2 = other.txt_norm2
        self.txt_mlp = NunchakuFeedForward(other.txt_mlp, **kwargs)

        self.scale_shift = scale_shift

    def _modulate(self, x: torch.Tensor, mod_params: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply modulation to input tensor"""
        shift, scale, gate = mod_params.chunk(3, dim=-1)
        if self.scale_shift != 0:
            scale.add_(self.scale_shift)
        return x * scale.unsqueeze(1) + shift.unsqueeze(1), gate.unsqueeze(1)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        encoder_hidden_states_mask: torch.Tensor,
        temb: torch.Tensor,
        image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Get modulation parameters for both streams
        img_mod_params = self.img_mod(temb)  # [B, 6*dim]
        txt_mod_params = self.txt_mod(temb)  # [B, 6*dim]

        # nunchaku's mod_params is [B, 6*dim] instead of [B, dim*6]
        img_mod_params = (
            img_mod_params.view(img_mod_params.shape[0], -1, 6).transpose(1, 2).reshape(img_mod_params.shape[0], -1)
        )
        txt_mod_params = (
            txt_mod_params.view(txt_mod_params.shape[0], -1, 6).transpose(1, 2).reshape(txt_mod_params.shape[0], -1)
        )

        img_mod1, img_mod2 = img_mod_params.chunk(2, dim=-1)  # Each [B, 3*dim]
        txt_mod1, txt_mod2 = txt_mod_params.chunk(2, dim=-1)  # Each [B, 3*dim]

        # Split modulation parameters for norm1 and norm2
        # img_mod1, img_mod2 = img_mod_params.chunk(2, dim=-1)  # Each [B, 3*dim]
        # txt_mod1, txt_mod2 = txt_mod_params.chunk(2, dim=-1)  # Each [B, 3*dim]

        # Process image stream - norm1 + modulation
        img_normed = self.img_norm1(hidden_states)
        img_modulated, img_gate1 = self._modulate(img_normed, img_mod1)

        # Process text stream - norm1 + modulation
        txt_normed = self.txt_norm1(encoder_hidden_states)
        txt_modulated, txt_gate1 = self._modulate(txt_normed, txt_mod1)

        # Use QwenAttnProcessor2_0 for joint attention computation
        # This directly implements the DoubleStreamLayerMegatron logic:
        # 1. Computes QKV for both streams
        # 2. Applies QK normalization and RoPE
        # 3. Concatenates and runs joint attention
        # 4. Splits results back to separate streams
        joint_attention_kwargs = joint_attention_kwargs or {}
        attn_output = self.attn(
            hidden_states=img_modulated,  # Image stream (will be processed as "sample")
            encoder_hidden_states=txt_modulated,  # Text stream (will be processed as "context")
            encoder_hidden_states_mask=encoder_hidden_states_mask,
            image_rotary_emb=image_rotary_emb,
            **joint_attention_kwargs,
        )

        # QwenAttnProcessor2_0 returns (img_output, txt_output) when encoder_hidden_states is provided
        img_attn_output, txt_attn_output = attn_output

        # Apply attention gates and add residual (like in Megatron)
        hidden_states = hidden_states + img_gate1 * img_attn_output
        encoder_hidden_states = encoder_hidden_states + txt_gate1 * txt_attn_output

        # Process image stream - norm2 + MLP
        img_normed2 = self.img_norm2(hidden_states)
        img_modulated2, img_gate2 = self._modulate(img_normed2, img_mod2)
        img_mlp_output = self.img_mlp(img_modulated2)
        hidden_states = hidden_states + img_gate2 * img_mlp_output

        # Process text stream - norm2 + MLP
        txt_normed2 = self.txt_norm2(encoder_hidden_states)
        txt_modulated2, txt_gate2 = self._modulate(txt_normed2, txt_mod2)
        txt_mlp_output = self.txt_mlp(txt_modulated2)
        encoder_hidden_states = encoder_hidden_states + txt_gate2 * txt_mlp_output

        # Clip to prevent overflow for fp16
        if encoder_hidden_states.dtype == torch.float16:
            encoder_hidden_states = encoder_hidden_states.clip(-65504, 65504)
        if hidden_states.dtype == torch.float16:
            hidden_states = hidden_states.clip(-65504, 65504)

        return encoder_hidden_states, hidden_states


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
        quantization_config = json.loads(metadata.get("quantization_config", "{}"))
        config = json.loads(metadata.get("config", "{}"))
        rank = quantization_config.get("rank", 32)
        transformer = transformer.to(torch_dtype)

        precision = get_precision()
        if precision == "fp4":
            precision = "nvfp4"
        transformer._patch_model(precision=precision, rank=rank)

        transformer = transformer.to_empty(device=device)
        # need to re-init the pos_embed as to_empty does not work on it
        transformer.pos_embed = QwenEmbedRope(
            theta=10000, axes_dim=list(config.get("axes_dims_rope", [16, 56, 56])), scale_rope=True
        )

        state_dict = transformer.state_dict()
        for k in state_dict.keys():
            if k not in model_state_dict:
                assert ".wtscale" in k or ".wcscales" in k
                model_state_dict[k] = torch.ones_like(state_dict[k])
            else:
                assert state_dict[k].dtype == model_state_dict[k].dtype
        transformer.load_state_dict(model_state_dict)

        return transformer
