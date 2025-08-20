"""
Adapters for efficient caching in Flux V2 diffusion pipelines.

This module enables advanced first-block caching for Flux V2 models, supporting both single and double caching strategies. It provides:

- :func:`apply_cache_on_transformer` — Add caching to a ``NunchakuFluxTransformer2DModelV2``.
- :func:`apply_cache_on_pipe` — Add caching to a complete Flux V2 pipeline.

Caching is context-managed and only active within a cache context.
"""

import functools

from diffusers import DiffusionPipeline

from nunchaku.models.transformers.transformer_flux_v2 import NunchakuFluxTransformer2DModelV2

from ...caching.utils_v2 import (
    NunchakuCachedFluxTransformer2DModelV2,
    cache_context,
    create_cache_context,
    get_current_cache_context,
)


def apply_cache_on_transformer(
    transformer: NunchakuFluxTransformer2DModelV2,
    *,
    use_double_fb_cache: bool = False,
    residual_diff_threshold: float = 0.12,
    residual_diff_threshold_multi: float | None = None,
    residual_diff_threshold_single: float | None = None,
):
    """
    Enable caching for a ``NunchakuFluxTransformer2DModelV2``.

    This function wraps the transformer to use cached transformer blocks for faster inference.
    Supports both single and double first-block caching with configurable thresholds.

    Parameters
    ----------
    transformer : NunchakuFluxTransformer2DModelV2
        The transformer to modify.
    use_double_fb_cache : bool, optional
        If True, cache both multi-head and single-head attention blocks (default: False).
    residual_diff_threshold : float, optional
        Default similarity threshold for caching (default: 0.12).
    residual_diff_threshold_multi : float, optional
        Threshold for multi-head (double) blocks. If None, uses ``residual_diff_threshold``.
    residual_diff_threshold_single : float, optional
        Threshold for single-head blocks (default: None).

    Returns
    -------
    NunchakuFluxTransformer2DModelV2
        The transformer with caching enabled.

    Notes
    -----
    If already cached, only updates thresholds. Caching is only active within a cache context.
    """
    if not hasattr(transformer, "_original_forward"):
        transformer._original_forward = transformer.forward
    if not hasattr(transformer, "_original_blocks"):
        transformer._original_blocks = transformer.transformer_blocks

    if residual_diff_threshold_multi is None:
        residual_diff_threshold_multi = residual_diff_threshold

    if getattr(transformer, "_is_cached", False):
        transformer.cached_transformer_blocks[0].update_residual_diff_threshold(
            use_double_fb_cache, residual_diff_threshold_multi, residual_diff_threshold_single
        )
        return transformer

    # Create the cached model with caching parameters
    cached_model = NunchakuCachedFluxTransformer2DModelV2(
        x_embedder=transformer.x_embedder,
        time_text_embed=transformer.time_text_embed,
        context_embedder=transformer.context_embedder,
        pos_embed=transformer.pos_embed,
        encoder_hid_proj=getattr(transformer, "encoder_hid_proj", None),
        transformer_blocks=transformer.transformer_blocks,
        single_transformer_blocks=transformer.single_transformer_blocks,
        norm_out=transformer.norm_out,
        proj_out=getattr(transformer, "proj_out", None),
        use_double_fb_cache=use_double_fb_cache,
        residual_diff_threshold=residual_diff_threshold,
        residual_diff_threshold_multi=residual_diff_threshold_multi,
        residual_diff_threshold_single=residual_diff_threshold_single,
        verbose=False,
    )

    # Store original forward method
    original_forward = transformer.forward

    @functools.wraps(original_forward)
    def new_forward(self, *args, **kwargs):
        current_cache_context = get_current_cache_context()
        if current_cache_context is not None:
            # When cache context is active, use the cached model's forward method
            # Don't pass self since cached_model already has its own self
            return cached_model.forward(*args, **kwargs)
        else:
            # When no cache context, use original forward with self
            return original_forward(self, *args, **kwargs)

    # Replace the transformer's forward method
    transformer.forward = new_forward.__get__(transformer)

    # Set caching attributes on the transformer
    transformer._is_cached = True
    transformer.cached_model = cached_model
    transformer.use_double_fb_cache = use_double_fb_cache
    transformer.residual_diff_threshold_multi = residual_diff_threshold_multi
    transformer.residual_diff_threshold_single = residual_diff_threshold_single

    return transformer


def apply_cache_on_pipe(pipe: DiffusionPipeline, **kwargs):
    """
    Enable caching for a complete Flux diffusion pipeline.

    This function wraps the pipeline's ``__call__`` method to manage cache contexts,
    and optionally applies transformer-level caching.

    Parameters
    ----------
    pipe : DiffusionPipeline
        The Flux pipeline to modify.
    shallow_patch : bool, optional
        If True, only patch the pipeline (do not modify the transformer). Useful for testing (default: False).
    **kwargs
        Passed to :func:`apply_cache_on_transformer` (e.g., ``use_double_fb_cache``, ``residual_diff_threshold``, etc.).

    Returns
    -------
    DiffusionPipeline
        The pipeline with caching enabled.

    Notes
    -----
    The pipeline class's ``__call__`` is patched for all instances.
    """
    if not getattr(pipe, "_is_cached", False):
        original_call = pipe.__class__.__call__

        @functools.wraps(original_call)
        def new_call(self, *args, **kwargs):
            with cache_context(create_cache_context()):
                return original_call(self, *args, **kwargs)

        pipe.__class__.__call__ = new_call
        pipe.__class__._is_cached = True

    apply_cache_on_transformer(pipe.transformer, **kwargs)

    return pipe
