import math
from typing import Optional, Tuple

import torch
from torch.nn import functional as F

from .._C.ops import attention_fp16
from ..ops.fused import fused_qkv_norm_rottary


class NunchakuFA2Processor:

    def __call__(
        self,
        attn,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        image_rotary_emb: Tuple[torch.Tensor, torch.Tensor] | torch.Tensor = None,
        **kwargs,
    ) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
        # Adapted from https://github.com/huggingface/diffusers/blob/50dea89dc6036e71a00bc3d57ac062a80206d9eb/src/diffusers/models/attention_processor.py#L2275
        if attention_mask is not None:
            raise NotImplementedError("attention_mask is not supported")

        batch_size, _, channels = hidden_states.shape
        assert channels == attn.heads * attn.head_dim
        qkv = fused_qkv_norm_rottary(
            hidden_states,
            attn.to_qkv,
            attn.norm_q,
            attn.norm_k,
            image_rotary_emb[0] if isinstance(image_rotary_emb, tuple) else image_rotary_emb,
        )

        if attn.added_kv_proj_dim is not None:
            assert encoder_hidden_states is not None
            assert isinstance(image_rotary_emb, tuple)
            qkv_context = fused_qkv_norm_rottary(
                encoder_hidden_states, attn.add_qkv_proj, attn.norm_added_q, attn.norm_added_k, image_rotary_emb[1]
            )
            qkv = torch.cat([qkv_context, qkv], dim=1)

        query, key, value = qkv.chunk(3, dim=-1)
        query = query.view(batch_size, -1, attn.heads, attn.head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, attn.head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, attn.head_dim).transpose(1, 2)
        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )
        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * attn.head_dim)
        hidden_states = hidden_states.to(query.dtype)

        if encoder_hidden_states is not None:
            encoder_hidden_states, hidden_states = (
                hidden_states[:, : encoder_hidden_states.shape[1]],
                hidden_states[:, encoder_hidden_states.shape[1] :],
            )
            # linear proj
            hidden_states = attn.to_out[0](hidden_states)
            # dropout
            hidden_states = attn.to_out[1](hidden_states)
            encoder_hidden_states = attn.to_add_out(encoder_hidden_states)
            return hidden_states, encoder_hidden_states
        else:
            # for single transformer block, we split the proj_out into two linear layers
            hidden_states = attn.to_out(hidden_states)
            return hidden_states


class NunchakuFP16AttnProcessor:

    def __init__(self, pad_size: int = 256):
        self.pad_size = pad_size

    def __call__(
        self,
        attn,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        image_rotary_emb: Tuple[torch.Tensor, torch.Tensor] | torch.Tensor = None,
        **kwargs,
    ) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
        pad_size = self.pad_size
        batch_size, _, channels = hidden_states.shape
        assert channels == attn.heads * attn.head_dim
        if encoder_hidden_states is None:
            # single transformer block
            assert attn.added_kv_proj_dim is None
            num_tokens = hidden_states.shape[1]
            num_tokens_pad = math.ceil(num_tokens / pad_size) * pad_size
            query = torch.empty(
                batch_size,
                attn.heads,
                num_tokens_pad,
                attn.head_dim,
                dtype=torch.float16,
                device=hidden_states.device,
            )
            key = torch.empty_like(query)
            value = torch.empty_like(query)

            assert torch.is_tensor(image_rotary_emb)
            fused_qkv_norm_rottary(
                hidden_states,
                attn.to_qkv,
                attn.norm_q,
                attn.norm_k,
                image_rotary_emb,
                output=(query, key, value),
                attn_tokens=num_tokens,
            )
        else:
            # joint transformer block
            assert attn.added_kv_proj_dim is not None
            num_txt_tokens = encoder_hidden_states.shape[1]
            num_img_tokens = hidden_states.shape[1]
            num_txt_tokens_pad = math.ceil(num_txt_tokens / pad_size) * pad_size
            num_img_tokens_pad = math.ceil(num_img_tokens / pad_size) * pad_size
            num_tokens_pad = num_txt_tokens_pad + num_img_tokens_pad
            query = torch.empty(
                batch_size,
                attn.heads,
                num_tokens_pad,
                attn.head_dim,
                dtype=torch.float16,
                device=hidden_states.device,
            )
            key = torch.empty_like(query)
            value = torch.empty_like(query)

            assert isinstance(image_rotary_emb, tuple)
            fused_qkv_norm_rottary(
                hidden_states,
                attn.to_qkv,
                attn.norm_q,
                attn.norm_k,
                image_rotary_emb[0],
                output=(
                    query[:, :, num_txt_tokens_pad:],
                    key[:, :, num_txt_tokens_pad:],
                    value[:, :, num_txt_tokens_pad:],
                ),
                attn_tokens=num_img_tokens,
            )
            fused_qkv_norm_rottary(
                encoder_hidden_states,
                attn.add_qkv_proj,
                attn.norm_added_q,
                attn.norm_added_k,
                image_rotary_emb[1],
                output=(
                    query[:, :, :num_txt_tokens_pad],
                    key[:, :, :num_txt_tokens_pad],
                    value[:, :, :num_txt_tokens_pad],
                ),
                attn_tokens=num_txt_tokens,
            )
        attention_output = torch.empty(
            batch_size,
            num_tokens_pad,
            attn.heads * attn.head_dim,
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )
        attention_fp16(query, key, value, attention_output, attn.head_dim ** (-0.5))
        hidden_states = attention_output

        if encoder_hidden_states is None:
            # for single transformer block, we split the proj_out into two linear layers
            hidden_states = hidden_states[:, :num_tokens]
            hidden_states = attn.to_out(hidden_states)
            return hidden_states
        else:
            encoder_hidden_states, hidden_states = (
                hidden_states[:, :num_txt_tokens],
                hidden_states[:, num_txt_tokens_pad : num_txt_tokens_pad + num_img_tokens],
            )
            # linear proj
            hidden_states = attn.to_out[0](hidden_states)
            # dropout
            hidden_states = attn.to_out[1](hidden_states)
            encoder_hidden_states = attn.to_add_out(encoder_hidden_states)
            return hidden_states, encoder_hidden_states
