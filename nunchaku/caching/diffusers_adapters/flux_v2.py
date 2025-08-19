"""
V2 caching implementation using a separate forward function.
"""

import functools

from diffusers import DiffusionPipeline

from nunchaku.models.transformers.transformer_flux_v2 import NunchakuFluxTransformer2DModelV2

from ..FBCache import cache_context, create_cache_context
from ..utils_v2 import cached_forward_v2


def apply_cache_on_transformer(
    transformer: NunchakuFluxTransformer2DModelV2,
    *,
    use_double_fb_cache: bool = False,
    residual_diff_threshold: float = 0.12,
    residual_diff_threshold_multi: float | None = None,
    residual_diff_threshold_single: float | None = None,
):
    """
    Apply caching to transformer by replacing its forward method.
    """

    if residual_diff_threshold_multi is None:
        residual_diff_threshold_multi = residual_diff_threshold

    if getattr(transformer, "_is_cached", False):
        # Already cached, just update thresholds
        transformer.residual_diff_threshold_multi = residual_diff_threshold_multi
        transformer.residual_diff_threshold_single = residual_diff_threshold_single
        transformer.use_double_fb_cache = use_double_fb_cache
        return transformer

    # Store original forward method
    transformer._original_forward = transformer.forward

    # Set caching parameters
    transformer.residual_diff_threshold_multi = residual_diff_threshold_multi
    transformer.residual_diff_threshold_single = (
        residual_diff_threshold_single if residual_diff_threshold_single is not None else -1.0
    )
    transformer.use_double_fb_cache = use_double_fb_cache
    transformer.verbose = False

    transformer.forward = cached_forward_v2.__get__(transformer, transformer.__class__)
    transformer._is_cached = True

    return transformer


def apply_cache_on_pipe(pipe: DiffusionPipeline, **kwargs):
    """
    Apply caching to a Flux pipeline.
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
