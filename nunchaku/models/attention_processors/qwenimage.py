"""
Attention processors for :class:`~nunchaku.models.transformers.transformer_qwenimage.NunchakuQwenAttention`.
"""

from typing import Optional, Tuple

import torch
from diffusers.models.attention_dispatch import dispatch_attention_fn
from diffusers.models.transformers.transformer_qwenimage import apply_rotary_emb_qwen


class NunchakuQwenImageNaiveFA2Processor:
    """
    Naive attention processor for Qwen-Image joint text-image attention.
    """

    def __call__(
        self,
        attn,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: torch.FloatTensor = None,
        encoder_hidden_states_mask: torch.FloatTensor = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for joint text-image attention.

        Parameters
        ----------
        attn : :class:`~nunchaku.models.transformers.transformer_qwenimage.NunchakuQwenAttention`
            Attention module.
        hidden_states : torch.FloatTensor, shape (B, L, H*D)
            Image stream hidden states.
        encoder_hidden_states : torch.FloatTensor, shape (B, L_txt, H*D)
            Text stream hidden states.
        encoder_hidden_states_mask : torch.FloatTensor, optional
            Not used.
        attention_mask : Optional[torch.FloatTensor], shape (B, 1, L_total, L_total), optional
            Attention mask for joint attention.
        image_rotary_emb : Optional[Tuple[torch.Tensor, torch.Tensor]]
            Tuple of rotary embeddings for image and text streams.

        Returns
        -------
        img_attn_output : torch.Tensor, shape (B, L, H*D)
            Output for image stream after attention and projection.
        txt_attn_output : torch.Tensor, shape (B, L_txt, H*D)
            Output for text stream after attention and projection.

        Raises
        ------
        ValueError
            If ``encoder_hidden_states`` (text stream) is not provided.

        Notes
        -----
        - B: batch size
        - L: sequence length (image)
        - L_txt: sequence length (text)
        - H: number of attention heads
        - D: head dimension
        """
        if encoder_hidden_states is None:
            raise ValueError("NunchakuQwenImageFA2Processor requires encoder_hidden_states (text stream)")

        seq_txt = encoder_hidden_states.shape[1]

        # Compute QKV for image stream (sample projections)
        img_qkv = attn.to_qkv(hidden_states)
        img_query, img_key, img_value = img_qkv.chunk(3, dim=-1)

        # Compute QKV for text stream (context projections)
        txt_qkv = attn.add_qkv_proj(encoder_hidden_states)
        txt_query, txt_key, txt_value = txt_qkv.chunk(3, dim=-1)

        # Reshape for multi-head attention
        img_query = img_query.unflatten(-1, (attn.heads, -1))  # [B, L, H, D]
        img_key = img_key.unflatten(-1, (attn.heads, -1))
        img_value = img_value.unflatten(-1, (attn.heads, -1))

        txt_query = txt_query.unflatten(-1, (attn.heads, -1))
        txt_key = txt_key.unflatten(-1, (attn.heads, -1))
        txt_value = txt_value.unflatten(-1, (attn.heads, -1))

        # Apply QK normalization
        assert attn.norm_q is not None
        img_query = attn.norm_q(img_query)
        assert attn.norm_k is not None
        img_key = attn.norm_k(img_key)
        assert attn.norm_added_q is not None
        txt_query = attn.norm_added_q(txt_query)
        assert attn.norm_added_k is not None
        txt_key = attn.norm_added_k(txt_key)

        # Apply rotary embeddings
        if image_rotary_emb is not None:
            img_freqs, txt_freqs = image_rotary_emb
            img_query = apply_rotary_emb_qwen(img_query, img_freqs, use_real=False)
            img_key = apply_rotary_emb_qwen(img_key, img_freqs, use_real=False)
            txt_query = apply_rotary_emb_qwen(txt_query, txt_freqs, use_real=False)
            txt_key = apply_rotary_emb_qwen(txt_key, txt_freqs, use_real=False)

        # Concatenate for joint attention: [text, image]
        joint_query = torch.cat([txt_query, img_query], dim=1)
        joint_key = torch.cat([txt_key, img_key], dim=1)
        joint_value = torch.cat([txt_value, img_value], dim=1)

        # Compute joint attention
        joint_hidden_states = dispatch_attention_fn(
            joint_query,
            joint_key,
            joint_value,
            attn_mask=attention_mask,
            dropout_p=0.0,
            is_causal=False,
            backend=None,
        )

        # Reshape back
        joint_hidden_states = joint_hidden_states.flatten(2, 3)
        joint_hidden_states = joint_hidden_states.to(joint_query.dtype)

        # Split attention outputs back
        txt_attn_output = joint_hidden_states[:, :seq_txt, :]  # Text part
        img_attn_output = joint_hidden_states[:, seq_txt:, :]  # Image part

        # Apply output projections
        img_attn_output = attn.to_out[0](img_attn_output)
        if len(attn.to_out) > 1:
            img_attn_output = attn.to_out[1](img_attn_output)  # dropout

        txt_attn_output = attn.to_add_out(txt_attn_output)

        return img_attn_output, txt_attn_output
