import torch
from diffusers.models.normalization import AdaLayerNormZero, AdaLayerNormZeroSingle
from diffusers.models.transformers.transformer_flux import (
    FluxAttention,
    FluxSingleTransformerBlock,
    FluxTransformer2DModel,
    FluxTransformerBlock,
)
from torch import nn

from ..linear import AWQW4A16Linear, SVDQW4A4Linear
from ..utils import fuse_linears
from typing import Optional


class NunchakuFluxAttention(nn.Module):
    def __init__(self, flux_attention: FluxAttention, processor: str = "flashattn2"):
        super(FluxAttention, self).__init__()

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
        self.to_qkv = SVDQW4A4Linear.from_linear(fused_qkv)

        self.to_out = flux_attention.to_out

        self.norm_added_q = flux_attention.norm_added_q
        self.norm_added_k = flux_attention.norm_added_k

        # fuse the add_qkv
        with torch.device("meta"):
            fused_add_qkv = fuse_linears(
                [flux_attention.add_q_proj, flux_attention.add_k_proj, flux_attention.add_v_proj]
            )
        self.add_qkv = SVDQW4A4Linear.from_linear(fused_add_qkv)
        self.to_add_out = flux_attention.to_add_out

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
        if image_rotary_emb is not None:
            raise NotImplementedError("image_rotary_emb is not supported")
        if encoder_hidden_states is not None:
            raise NotImplementedError("encoder_hidden_states is not supported")


class NunchakuFluxTransformerBlock(FluxTransformerBlock):

    def __init__(self, block: FluxTransformerBlock):
        super(FluxTransformerBlock, self).__init__()

        self.norm1 = block.norm1
        self.norm1_context = block.norm1_context

        if isinstance(self.norm1, AdaLayerNormZero):
            self.norm1.linear = AWQW4A16Linear.from_linear(self.norm1.linear)
        if isinstance(self.norm1_context, AdaLayerNormZero):
            self.norm1_context.linear = AWQW4A16Linear.from_linear(self.norm1_context.linear)

        self.attn = block.attn
        self.norm2 = block.norm2
        self.norm2_context = block.norm2_context
        self.ff = block.ff
        self.ff_context = block.ff_context


class NunchakuFluxSingleTransformerBlock(FluxSingleTransformerBlock):
    def __init__(self, block: FluxSingleTransformerBlock):
        super(FluxSingleTransformerBlock, self).__init__()
        self.mlp_hidden_dim = block.mlp_hidden_dim
        self.norm = block.norm

        if isinstance(self.norm, AdaLayerNormZeroSingle):
            self.norm.linear = AWQW4A16Linear.from_linear(self.norm.linear)

        self.proj_mlp = block.proj_mlp
        self.act_mlp = block.act_mlp
        self.proj_out = block.proj_out
        self.attn = block.attn


class NunchakuFluxTransformer2DModelV2(FluxTransformer2DModel):

    def _patch_model(self):
        for i, block in enumerate(self.transformer_blocks):
            self.transformer_blocks[i] = NunchakuFluxTransformerBlock(block)
        for i, block in enumerate(self.transformer_blocks_single):
            self.transformer_blocks_single[i] = NunchakuFluxSingleTransformerBlock(block)
