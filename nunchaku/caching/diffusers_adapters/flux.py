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
        with (
            unittest.mock.patch.object(self, "transformer_blocks", cached_transformer_blocks),
            unittest.mock.patch.object(self, "single_transformer_blocks", dummy_single_transformer_blocks),
        ):
            return original_forward(*args, **kwargs)

    transformer.forward = new_forward.__get__(transformer)
    transformer._is_cached = True

    return transformer


def apply_cache_on_pipe(pipe: DiffusionPipeline, *, shallow_patch: bool = False, **kwargs):
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
