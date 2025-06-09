import functools
import unittest

import torch
from diffusers import DiffusionPipeline, SanaTransformer2DModel

from ...caching import utils


def apply_cache_on_transformer(transformer: SanaTransformer2DModel, *, residual_diff_threshold=0.12):
    if getattr(transformer, "_is_cached", False):
        return transformer

    cached_transformer_blocks = torch.nn.ModuleList(
        [
            utils.SanaCachedTransformerBlocks(
                transformer=transformer,
                residual_diff_threshold=residual_diff_threshold,
            )
        ]
    )
    original_forward = transformer.forward

    @functools.wraps(original_forward)
    def new_forward(self, *args, **kwargs):
        cache_context = utils.get_current_cache_context()
        if cache_context is not None:
            with unittest.mock.patch.object(self, "transformer_blocks", cached_transformer_blocks):
                return original_forward(*args, **kwargs)
        else:
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
