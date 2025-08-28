"""
Caching utilities for V2 transformer models.

Implements first-block caching to accelerate transformer inference by reusing computations
when input changes are minimal. Supports Flux V2 architecture with double FB cache.

**Main Functions**

- :func:`cached_forward_v2` : Cached forward pass for V2 transformers.
- :func:`run_remaining_blocks_v2` : Process all remaining blocks (multi and single).
- :func:`run_remaining_multi_blocks_v2` : Process multi-head blocks only.
- :func:`run_remaining_single_blocks_v2` : Process single-head blocks only.

**Caching Strategy**

1. Compute the first transformer block.
2. Compare the residual with the cached residual.
3. If similar, reuse cached results for the remaining blocks; otherwise, recompute and update cache.
4. For double FB cache, repeat the process for single blocks.

.. note::
   V2 implementation with standalone functions for improved modularity.
"""

from typing import Any, Dict, Optional, Union

import torch
from diffusers.models.modeling_outputs import Transformer2DModelOutput

from nunchaku.caching.fbcache import check_and_apply_cache
from nunchaku.models.embeddings import pack_rotemb
from nunchaku.models.transformers.utils import pad_tensor


def cached_forward_v2(
    self,
    hidden_states: torch.Tensor,
    encoder_hidden_states: torch.Tensor = None,
    pooled_projections: torch.Tensor = None,
    timestep: torch.LongTensor = None,
    img_ids: torch.Tensor = None,
    txt_ids: torch.Tensor = None,
    guidance: torch.Tensor = None,
    joint_attention_kwargs: Optional[Dict[str, Any]] = None,
    controlnet_block_samples=None,
    controlnet_single_block_samples=None,
    return_dict: bool = True,
    controlnet_blocks_repeat: bool = False,
) -> Union[torch.Tensor, Transformer2DModelOutput]:
    """
    Cached forward function for V2 transformer with first-block caching.

    Replaces the transformer's forward method to enable caching optimizations.
    If residual_diff_threshold_multi < 0, caching is disabled.
    """

    # If caching disabled, use original forward
    if self.residual_diff_threshold_multi < 0.0:
        return self._original_forward(
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            pooled_projections=pooled_projections,
            timestep=timestep,
            img_ids=img_ids,
            txt_ids=txt_ids,
            guidance=guidance,
            joint_attention_kwargs=joint_attention_kwargs,
            controlnet_block_samples=controlnet_block_samples,
            controlnet_single_block_samples=controlnet_single_block_samples,
            return_dict=return_dict,
            controlnet_blocks_repeat=controlnet_blocks_repeat,
        )

    # Prepare inputs
    hidden_states = self.x_embedder(hidden_states)

    timestep = timestep.to(hidden_states.dtype) * 1000
    if guidance is not None:
        guidance = guidance.to(hidden_states.dtype) * 1000

    temb = (
        self.time_text_embed(timestep, pooled_projections)
        if guidance is None
        else self.time_text_embed(timestep, guidance, pooled_projections)
    )
    encoder_hidden_states = self.context_embedder(encoder_hidden_states)

    if txt_ids.ndim == 3:
        txt_ids = txt_ids[0]
    if img_ids.ndim == 3:
        img_ids = img_ids[0]

    ids = torch.cat((txt_ids, img_ids), dim=0)
    image_rotary_emb = self.pos_embed(ids)

    if joint_attention_kwargs is not None and "ip_adapter_image_embeds" in joint_attention_kwargs:
        ip_adapter_image_embeds = joint_attention_kwargs.pop("ip_adapter_image_embeds")
        ip_hidden_states = self.encoder_hid_proj(ip_adapter_image_embeds)
        joint_attention_kwargs.update({"ip_hidden_states": ip_hidden_states})

    txt_tokens = encoder_hidden_states.shape[1]
    img_tokens = hidden_states.shape[1]

    assert image_rotary_emb.ndim == 6
    assert image_rotary_emb.shape[0] == 1
    assert image_rotary_emb.shape[1] == 1
    assert image_rotary_emb.shape[2] == 1 * (txt_tokens + img_tokens)
    # [1, tokens, head_dim / 2, 1, 2] (sincos)
    image_rotary_emb = image_rotary_emb.reshape([1, txt_tokens + img_tokens, *image_rotary_emb.shape[3:]])
    rotary_emb_txt = image_rotary_emb[:, :txt_tokens, ...]  # .to(self.dtype)
    rotary_emb_img = image_rotary_emb[:, txt_tokens:, ...]  # .to(self.dtype)
    rotary_emb_single = image_rotary_emb

    rotary_emb_txt = pack_rotemb(pad_tensor(rotary_emb_txt, 256, 1))
    rotary_emb_img = pack_rotemb(pad_tensor(rotary_emb_img, 256, 1))
    rotary_emb_single = pack_rotemb(pad_tensor(rotary_emb_single, 256, 1))

    original_hidden_states = hidden_states

    # Process first block to get residual
    first_block = self.transformer_blocks[0]
    first_encoder_hidden_states, first_hidden_states = first_block(
        hidden_states=hidden_states,
        encoder_hidden_states=encoder_hidden_states,
        temb=temb,
        image_rotary_emb=(rotary_emb_img, rotary_emb_txt),
        joint_attention_kwargs=joint_attention_kwargs,
    )

    # Calculate residual for cache comparison
    hidden_states = first_hidden_states
    encoder_hidden_states = first_encoder_hidden_states
    first_hidden_states_residual_multi = hidden_states - original_hidden_states
    del original_hidden_states

    # Setup remaining blocks function and apply caching
    remaining_kwargs = {
        "temb": temb,
        "rotary_emb_img": rotary_emb_img,
        "rotary_emb_txt": rotary_emb_txt,
        "rotary_emb_single": rotary_emb_single,
        "joint_attention_kwargs": joint_attention_kwargs,
        "txt_tokens": txt_tokens,
    }
    if self.use_double_fb_cache:
        call_remaining_fn = run_remaining_multi_blocks_v2
    else:
        call_remaining_fn = run_remaining_blocks_v2

    hidden_states, encoder_hidden_states, _ = check_and_apply_cache(
        first_residual=first_hidden_states_residual_multi,
        hidden_states=hidden_states,
        encoder_hidden_states=encoder_hidden_states,
        threshold=self.residual_diff_threshold_multi,
        parallelized=False,
        mode="multi",
        verbose=self.verbose if hasattr(self, "verbose") else False,
        call_remaining_fn=lambda hidden_states, encoder_hidden_states, **kw: call_remaining_fn(
            self, hidden_states, encoder_hidden_states, **remaining_kwargs
        ),
        remaining_kwargs={},
    )
    if self.use_double_fb_cache:
        # Second stage caching for single blocks
        hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)
        original_cat = hidden_states

        # Process first single block
        first_block = self.single_transformer_blocks[0]
        hidden_states = first_block(
            hidden_states=hidden_states,
            temb=temb,
            image_rotary_emb=rotary_emb_single,
            joint_attention_kwargs=joint_attention_kwargs,
        )
        first_hidden_states_residual_single = hidden_states - original_cat
        del original_cat

        call_remaining_fn = run_remaining_single_blocks_v2

        original_dtype = hidden_states.dtype
        original_device = hidden_states.device

        hidden_states, _, _ = check_and_apply_cache(
            first_residual=first_hidden_states_residual_single,
            hidden_states=hidden_states,
            encoder_hidden_states=None,
            threshold=self.residual_diff_threshold_single,
            parallelized=False,
            mode="single",
            verbose=self.verbose,
            call_remaining_fn=lambda hidden_states, encoder_hidden_states, **kw: call_remaining_fn(
                self, hidden_states, encoder_hidden_states, **remaining_kwargs
            ),
            remaining_kwargs=remaining_kwargs,
        )

        hidden_states = hidden_states.to(original_dtype).to(original_device)

        hidden_states = hidden_states[:, txt_tokens:, ...]
        hidden_states = hidden_states.to(original_dtype).to(original_device)

    hidden_states = self.norm_out(hidden_states, temb)
    output = self.proj_out(hidden_states)

    if not return_dict:
        return (output,)

    return Transformer2DModelOutput(sample=output)


def run_remaining_blocks_v2(
    self,
    hidden_states,
    encoder_hidden_states,
    temb,
    rotary_emb_img,
    rotary_emb_txt,
    rotary_emb_single,
    joint_attention_kwargs,
    txt_tokens,
    **kwargs,
):
    """
    Process remaining transformer blocks (both multi and single).

    Called when cache is invalid. Processes all blocks after the first one.
    """
    original_dtype = hidden_states.dtype
    original_device = hidden_states.device
    original_h = hidden_states
    original_enc = encoder_hidden_states

    # Process remaining multi blocks
    for block in self.transformer_blocks[1:]:
        encoder_hidden_states, hidden_states = block(
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            temb=temb,
            image_rotary_emb=(rotary_emb_img, rotary_emb_txt),
            joint_attention_kwargs=joint_attention_kwargs,
        )

    # Concatenate encoder and decoder for single blocks
    hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)

    # Process all single blocks
    for block in self.single_transformer_blocks:
        hidden_states = block(
            hidden_states=hidden_states,
            temb=temb,
            image_rotary_emb=rotary_emb_single,
            joint_attention_kwargs=joint_attention_kwargs,
        )

    # Restore original dtype and device
    hidden_states = hidden_states.to(original_dtype).to(original_device)

    # Split concatenated result
    encoder_hidden_states = hidden_states[:, :txt_tokens, ...]
    hidden_states = hidden_states[:, txt_tokens:, ...]

    # Ensure contiguous memory layout
    hidden_states = hidden_states.contiguous()
    encoder_hidden_states = encoder_hidden_states.contiguous()

    # Calculate residuals
    hs_residual = hidden_states - original_h
    enc_residual = encoder_hidden_states - original_enc

    return hidden_states, encoder_hidden_states, hs_residual, enc_residual


def run_remaining_multi_blocks_v2(
    self,
    hidden_states,
    encoder_hidden_states,
    temb,
    rotary_emb_img,
    rotary_emb_txt,
    rotary_emb_single,
    joint_attention_kwargs,
    txt_tokens,
    **kwargs,
):
    """
    Process remaining multi-head transformer blocks only.

    Used when double FB cache is enabled. Skips single blocks.
    """
    original_h = hidden_states
    original_enc = encoder_hidden_states

    # Process remaining multi blocks
    for block in self.transformer_blocks[1:]:
        encoder_hidden_states, hidden_states = block(
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            temb=temb,
            image_rotary_emb=(rotary_emb_img, rotary_emb_txt),
            joint_attention_kwargs=joint_attention_kwargs,
        )

    # Ensure contiguous memory layout
    hidden_states = hidden_states.contiguous()
    encoder_hidden_states = encoder_hidden_states.contiguous()

    # Calculate residuals
    hs_residual = hidden_states - original_h
    enc_residual = encoder_hidden_states - original_enc

    return hidden_states, encoder_hidden_states, hs_residual, enc_residual


def run_remaining_single_blocks_v2(
    self,
    hidden_states,
    encoder_hidden_states,
    temb,
    rotary_emb_img,
    rotary_emb_txt,
    rotary_emb_single,
    joint_attention_kwargs,
    txt_tokens,
    **kwargs,
):
    """
    Process remaining single-head transformer blocks.

    Used for second stage of double FB cache.
    """
    # Save original for residual calculation
    original_hidden_states = hidden_states.clone()

    # Process remaining single blocks (skip first)
    for block in self.single_transformer_blocks[1:]:
        hidden_states = block(
            hidden_states=hidden_states,
            temb=temb,
            image_rotary_emb=rotary_emb_single,
            joint_attention_kwargs=joint_attention_kwargs,
        )

    hidden_states = hidden_states.contiguous()
    hs_residual = hidden_states - original_hidden_states

    return hidden_states, hs_residual
