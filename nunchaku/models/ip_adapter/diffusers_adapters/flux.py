"""
IP-Adapter integration for Flux pipelines in Diffusers.

This module provides functions to apply IP-Adapter modifications to
FluxTransformer2DModel and DiffusionPipeline objects, enabling image prompt
conditioning for generative models.
"""

import functools
import unittest

from diffusers import DiffusionPipeline, FluxTransformer2DModel
from torch import nn

from ....caching.fbcache import cache_context, create_cache_context
from ...ip_adapter import utils
from ...ip_adapter.utils import undo_all_mods_on_transformer


def apply_IPA_on_transformer(
    transformer: FluxTransformer2DModel,
    *,
    ip_adapter_scale: float = 1.0,
    repo_id: str,
):
    """
    Apply IP-Adapter modifications to a FluxTransformer2DModel.

    This function replaces the transformer's blocks with IP-Adapter-enabled blocks,
    loads per-layer IP-Adapter weights, and wraps the forward method to use the new blocks.

    Parameters
    ----------
    transformer : FluxTransformer2DModel
        The transformer model to modify.
    ip_adapter_scale : float, optional
        Scaling factor for the IP-Adapter (default is 1.0).
    repo_id : str
        HuggingFace Hub repository ID containing the IP-Adapter weights.

    Returns
    -------
    FluxTransformer2DModel
        The modified transformer with IP-Adapter support.
    """
    IPA_transformer_blocks = nn.ModuleList(
        [
            utils.IPA_TransformerBlocks(
                transformer=transformer,
                ip_adapter_scale=ip_adapter_scale,
                return_hidden_states_first=False,
                device=transformer.device,
            )
        ]
    )
    if getattr(transformer, "_is_cached", False):
        IPA_transformer_blocks[0].update_residual_diff_threshold(
            use_double_fb_cache=transformer.use_double_fb_cache,
            residual_diff_threshold_multi=transformer.residual_diff_threshold_multi,
            residual_diff_threshold_single=transformer.residual_diff_threshold_single,
        )
        undo_all_mods_on_transformer(transformer)
        if not hasattr(transformer, "_original_forward"):
            transformer._original_forward = transformer.forward
        if not hasattr(transformer, "_original_blocks"):
            transformer._original_blocks = transformer.transformer_blocks

    dummy_single_transformer_blocks = nn.ModuleList()

    IPA_transformer_blocks[0].load_ip_adapter_weights_per_layer(repo_id=repo_id)

    transformer.transformer_blocks = IPA_transformer_blocks
    transformer.single_transformer_blocks = dummy_single_transformer_blocks
    original_forward = transformer.forward

    @functools.wraps(original_forward)
    def new_forward(self, *args, **kwargs):
        with (
            unittest.mock.patch.object(self, "transformer_blocks", IPA_transformer_blocks),
            unittest.mock.patch.object(self, "single_transformer_blocks", dummy_single_transformer_blocks),
        ):
            return original_forward(*args, **kwargs)

    transformer.forward = new_forward.__get__(transformer)
    transformer._is_IPA = True

    return transformer


def apply_IPA_on_pipe(pipe: DiffusionPipeline, **kwargs):
    """
    Apply IP-Adapter modifications to a DiffusionPipeline.

    This function modifies the pipeline's transformer to support IP-Adapter
    conditioning. If the pipeline is cached, it also wraps the pipeline's
    __call__ method to ensure cache context is used.

    Parameters
    ----------
    pipe : DiffusionPipeline
        The pipeline to modify. Must contain a FluxTransformer2DModel as its transformer.
    **kwargs
        Additional keyword arguments passed to `apply_IPA_on_transformer`.

    Returns
    -------
    DiffusionPipeline
        The modified pipeline with IP-Adapter support.
    """
    if getattr(pipe, "_is_cached", False):
        original_call = pipe.__class__.__call__

        @functools.wraps(original_call)
        def new_call(self, *args, **kwargs):
            with cache_context(create_cache_context()):
                return original_call(self, *args, **kwargs)

        pipe.__class__.__call__ = new_call
        pipe.__class__._is_cached = True

    apply_IPA_on_transformer(pipe.transformer, **kwargs)

    return pipe
