"""
Flux Pipeline Caching Adapter.

This module provides caching adapters specifically for Flux diffusion pipelines.
It implements both transformer-level and pipeline-level caching integration,
enabling efficient inference through advanced first-block caching strategies.

The module supports both single and double first-block caching for Flux models,
with automatic context management to ensure proper cache lifecycle during
inference.

Key Functions:
    apply_cache_on_transformer: Apply caching directly to a FluxTransformer2DModel
    apply_cache_on_pipe: Apply caching to a complete Flux pipeline

Caching Features:
    - Single first-block caching: Caches the first transformer block only
    - Double first-block caching: Caches both multi-head and single-head attention blocks
    - Dynamic threshold adjustment: Automatically adjusts similarity thresholds
    - Context management: Ensures proper cache setup and cleanup
    - Shallow patching: Optional lightweight patching for testing

Example:
    Apply caching to a Flux transformer::

        from diffusers import FluxTransformer2DModel
        from nunchaku.caching.diffusers_adapters.flux import apply_cache_on_transformer

        transformer = FluxTransformer2DModel.from_pretrained("model_name")
        cached_transformer = apply_cache_on_transformer(
            transformer,
            use_double_fb_cache=True,
            residual_diff_threshold_multi=0.12,
            residual_diff_threshold_single=0.09
        )

    Apply caching to a complete pipeline::

        from diffusers import FluxPipeline
        from nunchaku.caching.diffusers_adapters.flux import apply_cache_on_pipe

        pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-dev")
        cached_pipe = apply_cache_on_pipe(
            pipe,
            use_double_fb_cache=True,
            residual_diff_threshold=0.12
        )

Note:
    The caching is applied in-place and uses mock patching to temporarily replace
    transformer components during inference. The original functionality is preserved
    when not using caching context.
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
    
    This function modifies a FluxTransformer2DModel to use cached transformer blocks
    for improved inference performance. It supports both single and double first-block
    caching strategies with configurable similarity thresholds.
    
    Args:
        transformer (FluxTransformer2DModel): The Flux transformer model to apply caching to
        use_double_fb_cache (bool, optional): Whether to use double first-block caching.
            If True, caches both multi-head and single-head attention blocks. Defaults to False.
        residual_diff_threshold (float, optional): Default similarity threshold for caching.
            Used for residual_diff_threshold_multi if not explicitly provided. Defaults to 0.12.
        residual_diff_threshold_multi (float, optional): Similarity threshold for multi-head
            attention blocks. If None, uses residual_diff_threshold. Defaults to None.
        residual_diff_threshold_single (float, optional): Similarity threshold for single-head
            attention blocks. Defaults to 0.1.
            
    Returns:
        FluxTransformer2DModel: The same transformer instance with caching applied
        
    Example:
        Basic caching setup::
        
            transformer = FluxTransformer2DModel.from_pretrained("model_name")
            cached_transformer = apply_cache_on_transformer(
                transformer,
                use_double_fb_cache=True,
                residual_diff_threshold=0.12
            )
            
        Advanced configuration::
        
            cached_transformer = apply_cache_on_transformer(
                transformer,
                use_double_fb_cache=True,
                residual_diff_threshold_multi=0.15,
                residual_diff_threshold_single=0.08
            )
            
    Note:
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
    It wraps the pipeline's __call__ method to automatically create and manage cache
    contexts, and optionally applies transformer-level caching.
    
    Args:
        pipe (DiffusionPipeline): The Flux diffusion pipeline to apply caching to
        shallow_patch (bool, optional): If True, only applies pipeline-level caching
            without modifying the transformer. Useful for testing. Defaults to False.
        **kwargs: Additional keyword arguments passed to apply_cache_on_transformer,
            including:
            - use_double_fb_cache (bool): Whether to use double first-block caching
            - residual_diff_threshold (float): Similarity threshold for caching
            - residual_diff_threshold_multi (float): Multi-head attention threshold
            - residual_diff_threshold_single (float): Single-head attention threshold
            
    Returns:
        DiffusionPipeline: The same pipeline instance with caching applied
        
    Example:
        Basic usage::
        
            from diffusers import FluxPipeline
            pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-dev")
            cached_pipe = apply_cache_on_pipe(pipe)
            
            # Use normally - caching is transparent
            image = cached_pipe(prompt="A beautiful landscape")
            
        Advanced configuration::
        
            cached_pipe = apply_cache_on_pipe(
                pipe,
                use_double_fb_cache=True,
                residual_diff_threshold=0.1,
                residual_diff_threshold_single=0.05
            )
            
        Shallow patching for testing::
        
            cached_pipe = apply_cache_on_pipe(pipe, shallow_patch=True)
            
    Note:
        The function modifies the pipeline class's __call__ method, affecting all
        instances of the same pipeline class. If the pipeline is already cached,
        it skips the pipeline-level patching but still applies transformer caching
        unless shallow_patch is True.
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
