"""
This module implements the PuLID forward function for the :class:`nunchaku.models.transformers.NunchakuFluxTransformer2dModel`,

.. note::
    This module is adapted from the original PuLID repository:
    https://github.com/ToTheBeginning/PuLID
"""

import logging
from typing import Any, Dict, Optional, Union

import torch
from diffusers.models.modeling_outputs import Transformer2DModelOutput

logger = logging.getLogger(__name__)


def pulid_forward(
    self,
    hidden_states: torch.Tensor,
    id_embeddings=None,
    id_weight=None,
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
    start_timestep: float | None = None,
    end_timestep: float | None = None,
) -> Union[torch.FloatTensor, Transformer2DModelOutput]:
    """
    Implements the forward pass for the PuLID transformer block.

    This function supports time and text conditioning, rotary embeddings, ControlNet integration,
    and joint attention. It is adapted from
    ``diffusers.models.flux.transformer_flux.py`` and the original PuLID repository.

    Parameters
    ----------
    self : nn.Module
        The :class:`nunchaku.models.transformers.NunchakuFluxTransformer2dModel` instance.
        This function is intended to be bound as a method.
    hidden_states : torch.Tensor
        Input hidden states of shape ``(batch_size, channels, height, width)``.
    id_embeddings : torch.Tensor, optional
        Optional PuLID ID embeddings for conditioning (default: None).
    id_weight : torch.Tensor, optional
        Optional PuLID ID weights for conditioning (default: None).
    encoder_hidden_states : torch.Tensor, optional
        Conditional embeddings (e.g., from text encoder) of shape ``(batch_size, sequence_len, embed_dim)``.
    pooled_projections : torch.Tensor, optional
        Embeddings projected from input conditions, shape ``(batch_size, projection_dim)``.
    timestep : torch.LongTensor, optional
        Timestep tensor indicating the denoising step.
    img_ids : torch.Tensor, optional
        Image token IDs for rotary embedding.
    txt_ids : torch.Tensor, optional
        Text token IDs for rotary embedding.
    guidance : torch.Tensor, optional
        Optional guidance tensor for classifier-free guidance or similar.
    joint_attention_kwargs : dict, optional
        Additional keyword arguments for joint attention, passed to the attention processor.
    controlnet_block_samples : Any, optional
        ControlNet block samples for multi-block conditioning (default: None).
    controlnet_single_block_samples : Any, optional
        ControlNet single block samples for single-block conditioning (default: None).
    return_dict : bool, optional
        If True (default), returns a :class:`~diffusers.models.modeling_outputs.Transformer2DModelOutput`.
        If False, returns a tuple containing the output tensor.
    controlnet_blocks_repeat : bool, optional
        Whether to repeat ControlNet blocks (default: False).
    start_timestep : float, optional
        If specified, disables ID embeddings for timesteps before this value.
    end_timestep : float, optional
        If specified, disables ID embeddings for timesteps after this value.

    Returns
    -------
    torch.FloatTensor or Transformer2DModelOutput
        If ``return_dict`` is True, returns a :class:`~diffusers.models.modeling_outputs.Transformer2DModelOutput`
        with the output sample. Otherwise, returns a tuple containing the output tensor.
    """
    hidden_states = self.x_embedder(hidden_states)

    if timestep.numel() > 1:
        timestep_float = timestep.flatten()[0].item()
    else:
        timestep_float = timestep.item()

    if start_timestep is not None and start_timestep > timestep_float:
        id_embeddings = None
    if end_timestep is not None and end_timestep < timestep_float:
        id_embeddings = None

    timestep = timestep.to(hidden_states.dtype) * 1000
    if guidance is not None:
        guidance = guidance.to(hidden_states.dtype) * 1000
    else:
        guidance = None

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

    nunchaku_block = self.transformer_blocks[0]
    encoder_hidden_states, hidden_states = nunchaku_block(
        hidden_states=hidden_states,
        id_embeddings=id_embeddings,
        id_weight=id_weight,
        encoder_hidden_states=encoder_hidden_states,
        temb=temb,
        image_rotary_emb=image_rotary_emb,
        joint_attention_kwargs=joint_attention_kwargs,
        controlnet_block_samples=controlnet_block_samples,
        controlnet_single_block_samples=controlnet_single_block_samples,
    )
    hidden_states = self.norm_out(hidden_states, temb)
    output = self.proj_out(hidden_states)

    if not return_dict:
        return (output,)

    return Transformer2DModelOutput(sample=output)
