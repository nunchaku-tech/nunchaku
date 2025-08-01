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
    residual_diff_threshold_single: float | None = None,
):
    """
    Enable caching for a ``FluxTransformer2DModel``.

    This function wraps the transformer to use cached transformer blocks for faster inference.
    Supports both single and double first-block caching with configurable thresholds.

    Parameters
    ----------
    transformer : FluxTransformer2DModel
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
    FluxTransformer2DModel
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
            with utils.cache_context(utils.create_cache_context()):
                return original_call(self, *args, **kwargs)

        pipe.__class__.__call__ = new_call
        pipe.__class__._is_cached = True

    apply_cache_on_transformer(pipe.transformer, **kwargs)

    return pipe
