"""
Flux Pipeline Caching Adapter
=============================

This module provides caching adapters specifically for Flux diffusion pipelines.
It implements both transformer-level and pipeline-level caching integration,
enabling efficient inference through advanced first-block caching strategies.

Overview
--------

The module supports both *single* and *double* first-block caching for Flux models,
with automatic context management to ensure proper cache lifecycle during
inference.

**Key Functions**

- :func:`apply_cache_on_transformer` -- Apply caching directly to a ``FluxTransformer2DModel``
- :func:`apply_cache_on_pipe` -- Apply caching to a complete Flux pipeline
"""

import functools
import unittest

from diffusers import DiffusionPipeline, FluxTransformer2DModel
from torch import nn

from ...caching import utils


def apply_cache_on_transformer(
    transformer: FluxTransformer2DModel,
    *,
    use_double_fb_cache: bool = False,
    residual_diff_threshold: float = 0.12,
    residual_diff_threshold_multi: float | None = None,
    residual_diff_threshold_single: float = 0.1,
):
    """
    Apply caching to a Flux transformer model.

    This function modifies a ``FluxTransformer2DModel`` to use cached transformer blocks
    for improved inference performance. It supports both single and double first-block
    caching strategies with configurable similarity thresholds.

    Parameters
    ----------
    transformer : FluxTransformer2DModel
        The Flux transformer model to apply caching to.
    use_double_fb_cache : bool, optional
        Whether to use double first-block caching. If True, caches both multi-head and
        single-head attention blocks. Defaults to False.
    residual_diff_threshold : float, optional
        Default similarity threshold for caching. Used for ``residual_diff_threshold_multi``
        if not explicitly provided. Defaults to 0.12.
    residual_diff_threshold_multi : float, optional
        Similarity threshold for Flux double blocks. Only used if ``use_double_fb_cache`` is True.
        If not provided, ``residual_diff_threshold`` is used.
    residual_diff_threshold_single : float, optional
        Similarity threshold for Flux single blocks. Only used if ``use_double_fb_cache`` is False.
        Defaults to 0.1.

    Returns
    -------
    FluxTransformer2DModel
        The same transformer instance with caching applied.

    Notes
    -----
    If the transformer is already cached, the function updates the thresholds
    instead of reapplying caching. The caching only activates when a cache
    context is present.
    """
    if residual_diff_threshold_multi is None:
        residual_diff_threshold_multi = residual_diff_threshold

    if getattr(transformer, "_is_cached", False):
        transformer.cached_transformer_blocks[0].update_residual_diff_threshold(
            use_double_fb_cache, residual_diff_threshold_multi, residual_diff_threshold_single
        )
        return transformer

    cached_transformer_blocks = nn.ModuleList(
        [
            utils.FluxCachedTransformerBlocks(
                transformer=transformer,
                use_double_fb_cache=use_double_fb_cache,
                residual_diff_threshold_multi=residual_diff_threshold_multi,
                residual_diff_threshold_single=residual_diff_threshold_single,
                return_hidden_states_first=False,
            )
        ]
    )
    dummy_single_transformer_blocks = nn.ModuleList()

    original_forward = transformer.forward

    @functools.wraps(original_forward)
    def new_forward(self, *args, **kwargs):
        cache_context = utils.get_current_cache_context()
        if cache_context is not None:
            with (
                unittest.mock.patch.object(self, "transformer_blocks", cached_transformer_blocks),
                unittest.mock.patch.object(self, "single_transformer_blocks", dummy_single_transformer_blocks),
            ):
                transformer._is_cached = True
                transformer.cached_transformer_blocks = cached_transformer_blocks
                transformer.single_transformer_blocks = dummy_single_transformer_blocks
                return original_forward(*args, **kwargs)
        else:
            return original_forward(*args, **kwargs)

    transformer.forward = new_forward.__get__(transformer)

    return transformer


def apply_cache_on_pipe(pipe: DiffusionPipeline, *, shallow_patch: bool = False, **kwargs):
    """
    Apply caching to a complete Flux diffusion pipeline.

    This function modifies a Flux diffusion pipeline to use caching during inference.
    It wraps the pipeline's ``__call__`` method to automatically create and manage cache
    contexts, and optionally applies transformer-level caching.

    Parameters
    ----------
    pipe : DiffusionPipeline
        The Flux diffusion pipeline to apply caching to.
    shallow_patch : bool, optional
        If True, only applies pipeline-level caching without modifying the transformer.
        Useful for testing. Defaults to False.
    **kwargs
        Additional keyword arguments passed to :func:`apply_cache_on_transformer`, including:

        - use_double_fb_cache (bool): Whether to use double first-block caching
        - residual_diff_threshold (float): Similarity threshold for caching
        - residual_diff_threshold_multi (float): Flux double blocks threshold
        - residual_diff_threshold_single (float): Flux single blocks threshold

    Returns
    -------
    DiffusionPipeline
        The same pipeline instance with caching applied.

    Notes
    -----
    The function modifies the pipeline class's ``__call__`` method, affecting all
    instances of the same pipeline class. If the pipeline is already cached,
    it skips the pipeline-level patching but still applies transformer caching
    unless ``shallow_patch`` is True.
    """
    if not getattr(pipe, "_is_cached", False):
        original_call = pipe.__class__.__call__

        @functools.wraps(original_call)
        def new_call(self, *args, **kwargs):
            with utils.cache_context(utils.create_cache_context()):
                return original_call(self, *args, **kwargs)

        pipe.__class__.__call__ = new_call
        pipe.__class__._is_cached = True

    if not shallow_patch:
        apply_cache_on_transformer(pipe.transformer, **kwargs)

    return pipe
