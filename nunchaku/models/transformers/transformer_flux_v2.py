import json
import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
from diffusers.models.attention import FeedForward
from diffusers.models.embeddings import apply_rotary_emb
from diffusers.models.normalization import AdaLayerNormZero, AdaLayerNormZeroSingle
from diffusers.models.transformers.transformer_flux import (
    FluxAttention,
    FluxSingleTransformerBlock,
    FluxTransformer2DModel,
    FluxTransformerBlock,
)
from huggingface_hub import utils
from torch import nn
from torch.nn import functional as F

from ...utils import get_precision
from ..linear import AWQW4A16Linear, SVDQW4A4Linear
from ..utils import fuse_linears
from .utils import NunchakuModelLoaderMixin


def _patch_linear(module: nn.Module, linear_cls, **kwargs) -> nn.Module:
    for name, child in module.named_children():
        if isinstance(child, nn.Linear):
            setattr(module, name, linear_cls.from_linear(child, **kwargs))
        else:
            _patch_linear(child, linear_cls, **kwargs)
    return module


class NunchakuFeedForward(FeedForward):
    def __init__(self, ff: FeedForward, **kwargs):
        super(FeedForward, self).__init__()
        self.net = _patch_linear(ff.net, SVDQW4A4Linear, **kwargs)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        for module in self.net:
            hidden_states = module(hidden_states)
        return hidden_states


class NunchakuFluxAttention(nn.Module):
    def __init__(self, flux_attention: FluxAttention, processor: str = "flashattn2", **kwargs):
        super(NunchakuFluxAttention, self).__init__()

        self.head_dim = flux_attention.head_dim
        self.inner_dim = flux_attention.inner_dim
        self.query_dim = flux_attention.query_dim
        self.use_bias = flux_attention.use_bias
        self.dropout = flux_attention.dropout
        self.out_dim = flux_attention.out_dim
        self.context_pre_only = flux_attention.context_pre_only
        self.pre_only = flux_attention.pre_only
        self.heads = flux_attention.heads
        self.added_kv_proj_dim = flux_attention.added_kv_proj_dim
        self.added_proj_bias = flux_attention.added_proj_bias

        self.norm_q = flux_attention.norm_q
        self.norm_k = flux_attention.norm_k

        # fuse the qkv
        with torch.device("meta"):
            fused_qkv = fuse_linears([flux_attention.to_q, flux_attention.to_k, flux_attention.to_v])
        self.to_qkv = SVDQW4A4Linear.from_linear(fused_qkv, **kwargs)

        if not self.pre_only:
            self.to_out = flux_attention.to_out
            self.to_out[0] = SVDQW4A4Linear.from_linear(self.to_out[0], **kwargs)

        if self.added_kv_proj_dim is not None:
            self.norm_added_q = flux_attention.norm_added_q
            self.norm_added_k = flux_attention.norm_added_k

            # fuse the add_qkv
            with torch.device("meta"):
                fused_add_qkv = fuse_linears(
                    [flux_attention.add_q_proj, flux_attention.add_k_proj, flux_attention.add_v_proj]
                )
            self.add_qkv = SVDQW4A4Linear.from_linear(fused_add_qkv, **kwargs)
            self.to_add_out = SVDQW4A4Linear.from_linear(flux_attention.to_add_out, **kwargs)

        self.processor = processor

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        if attention_mask is not None:
            raise NotImplementedError("attention_mask is not supported")

        query, key, value = self.to_qkv(hidden_states).chunk(3, dim=-1)
        encoder_query, encoder_key, encoder_value = self.to_added_qkv(encoder_hidden_states).chunk(3, dim=-1)

        query = query.unflatten(-1, (self.heads, -1))
        key = key.unflatten(-1, (self.heads, -1))
        value = value.unflatten(-1, (self.heads, -1))

        query = self.norm_q(query)
        key = self.norm_k(key)

        if self.added_kv_proj_dim is not None:
            encoder_query = encoder_query.unflatten(-1, (self.heads, -1))
            encoder_key = encoder_key.unflatten(-1, (self.heads, -1))
            encoder_value = encoder_value.unflatten(-1, (self.heads, -1))

            encoder_query = self.norm_added_q(encoder_query)
            encoder_key = self.norm_added_k(encoder_key)

            query = torch.cat([encoder_query, query], dim=1)
            key = torch.cat([encoder_key, key], dim=1)
            value = torch.cat([encoder_value, value], dim=1)

        if image_rotary_emb is not None:
            query = apply_rotary_emb(query, image_rotary_emb, sequence_dim=1)
            key = apply_rotary_emb(key, image_rotary_emb, sequence_dim=1)

        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )

        hidden_states = hidden_states.flatten(2, 3)
        hidden_states = hidden_states.to(query.dtype)

        if encoder_hidden_states is not None:
            encoder_hidden_states, hidden_states = hidden_states.split_with_sizes(
                [encoder_hidden_states.shape[1], hidden_states.shape[1] - encoder_hidden_states.shape[1]], dim=1
            )
            hidden_states = self.to_out[0](hidden_states)
            hidden_states = self.to_out[1](hidden_states)
            encoder_hidden_states = self.to_add_out(encoder_hidden_states)

            return hidden_states, encoder_hidden_states
        else:
            return hidden_states


class NunchakuFluxTransformerBlock(FluxTransformerBlock):

    def __init__(self, block: FluxTransformerBlock, **kwargs):
        super(FluxTransformerBlock, self).__init__()

        self.norm1 = block.norm1
        self.norm1_context = block.norm1_context

        if isinstance(self.norm1, AdaLayerNormZero):
            self.norm1.linear = AWQW4A16Linear.from_linear(self.norm1.linear)
        if isinstance(self.norm1_context, AdaLayerNormZero):
            self.norm1_context.linear = AWQW4A16Linear.from_linear(self.norm1_context.linear)

        self.attn = NunchakuFluxAttention(block.attn, **kwargs)
        self.norm2 = block.norm2
        self.norm2_context = block.norm2_context
        self.ff = NunchakuFeedForward(block.ff, **kwargs)
        self.ff_context = NunchakuFeedForward(block.ff_context, **kwargs)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb: torch.Tensor,
        image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
    ):
        if joint_attention_kwargs is not None:
            raise NotImplementedError("joint_attention_kwargs is not supported")

        norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm1(hidden_states, emb=temb)

        norm_encoder_hidden_states, c_gate_msa, c_shift_mlp, c_scale_mlp, c_gate_mlp = self.norm1_context(
            encoder_hidden_states, emb=temb
        )

        joint_attention_kwargs = joint_attention_kwargs or {}

        # Attention.
        attention_outputs = self.attn(
            hidden_states=norm_hidden_states,
            encoder_hidden_states=norm_encoder_hidden_states,
            image_rotary_emb=image_rotary_emb,
            **joint_attention_kwargs,
        )

        if len(attention_outputs) == 2:
            attn_output, context_attn_output = attention_outputs
        elif len(attention_outputs) == 3:
            attn_output, context_attn_output, ip_attn_output = attention_outputs

        # Process attention outputs for the `hidden_states`.
        attn_output = gate_msa.unsqueeze(1) * attn_output
        hidden_states = hidden_states + attn_output

        norm_hidden_states = self.norm2(hidden_states)
        norm_hidden_states = norm_hidden_states * (1 + scale_mlp[:, None]) + shift_mlp[:, None]

        ff_output = self.ff(norm_hidden_states)
        ff_output = gate_mlp.unsqueeze(1) * ff_output

        hidden_states = hidden_states + ff_output
        if len(attention_outputs) == 3:
            hidden_states = hidden_states + ip_attn_output

        # Process attention outputs for the `encoder_hidden_states`.
        context_attn_output = c_gate_msa.unsqueeze(1) * context_attn_output
        encoder_hidden_states = encoder_hidden_states + context_attn_output

        norm_encoder_hidden_states = self.norm2_context(encoder_hidden_states)
        norm_encoder_hidden_states = norm_encoder_hidden_states * (1 + c_scale_mlp[:, None]) + c_shift_mlp[:, None]

        context_ff_output = self.ff_context(norm_encoder_hidden_states)
        encoder_hidden_states = encoder_hidden_states + c_gate_mlp.unsqueeze(1) * context_ff_output
        if encoder_hidden_states.dtype == torch.float16:
            encoder_hidden_states = encoder_hidden_states.clip(-65504, 65504)

        return encoder_hidden_states, hidden_states


class NunchakuFluxSingleTransformerBlock(FluxSingleTransformerBlock):
    def __init__(self, block: FluxSingleTransformerBlock, **kwargs):
        super(FluxSingleTransformerBlock, self).__init__()
        self.mlp_hidden_dim = block.mlp_hidden_dim
        self.norm = block.norm

        if isinstance(self.norm, AdaLayerNormZeroSingle):
            self.norm.linear = AWQW4A16Linear.from_linear(self.norm.linear, **kwargs)

        self.proj_mlp = SVDQW4A4Linear.from_linear(block.proj_mlp, **kwargs)
        self.act_mlp = block.act_mlp
        self.proj_out = SVDQW4A4Linear.from_linear(block.proj_out, **kwargs)

        self.attn = NunchakuFluxAttention(block.attn, **kwargs)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb: torch.Tensor,
        image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
    ) -> torch.Tensor:
        text_seq_len = encoder_hidden_states.shape[1]
        hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)

        residual = hidden_states
        norm_hidden_states, gate = self.norm(hidden_states, emb=temb)
        mlp_hidden_states = self.act_mlp(self.proj_mlp(norm_hidden_states))
        joint_attention_kwargs = joint_attention_kwargs or {}
        attn_output = self.attn(
            hidden_states=norm_hidden_states,
            image_rotary_emb=image_rotary_emb,
            **joint_attention_kwargs,
        )

        hidden_states = torch.cat([attn_output, mlp_hidden_states], dim=2)
        gate = gate.unsqueeze(1)
        hidden_states = gate * self.proj_out(hidden_states)
        hidden_states = residual + hidden_states
        if hidden_states.dtype == torch.float16:
            hidden_states = hidden_states.clip(-65504, 65504)

        encoder_hidden_states, hidden_states = hidden_states[:, :text_seq_len], hidden_states[:, text_seq_len:]
        return encoder_hidden_states, hidden_states


class NunchakuFluxTransformer2DModelV2(FluxTransformer2DModel, NunchakuModelLoaderMixin):

    def _patch_model(self, **kwargs):
        for i, block in enumerate(self.transformer_blocks):
            self.transformer_blocks[i] = NunchakuFluxTransformerBlock(block, **kwargs)
        for i, block in enumerate(self.single_transformer_blocks):
            self.single_transformer_blocks[i] = NunchakuFluxSingleTransformerBlock(block, **kwargs)
        return self

    @classmethod
    @utils.validate_hf_hub_args
    def from_pretrained(cls, pretrained_model_name_or_path: str | os.PathLike[str], **kwargs):
        device = kwargs.get("device", "cpu")
        offload = kwargs.get("offload", False)

        if offload:
            raise NotImplementedError("offload is not supported")

        torch_dtype = kwargs.get("torch_dtype", torch.bfloat16)
        metadata = None

        if isinstance(pretrained_model_name_or_path, str):
            pretrained_model_name_or_path = Path(pretrained_model_name_or_path)

        assert pretrained_model_name_or_path.is_file() or pretrained_model_name_or_path.name.endswith(
            (".safetensors", ".sft")
        ), "Only safetensors are supported"
        transformer, model_state_dict, metadata = cls._build_model(pretrained_model_name_or_path, **kwargs)
        quantization_config = json.loads(metadata["quantization_config"])

        precision = get_precision()
        if precision == "fp4":
            precision = "nvfp4"
        transformer._patch_model(precision=precision)
        return transformer
