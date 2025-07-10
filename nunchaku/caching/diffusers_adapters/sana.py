"""
SANA Pipeline Caching Adapter.

This module provides caching adapters specifically for SANA diffusion pipelines.
It implements single first-block caching for SANA models, enabling efficient
inference through intelligent reuse of first transformer block computations.

The SANA adapter uses a simpler caching strategy compared to Flux, focusing
on single first-block caching with configurable similarity thresholds. This
approach is optimized for the specific architecture and usage patterns of
SANA models.

Key Functions:
    apply_cache_on_transformer: Apply caching directly to a SanaTransformer2DModel
    apply_cache_on_pipe: Apply caching to a complete SANA pipeline

Caching Features:
    - Single first-block caching: Caches the first transformer block's output
    - Configurable similarity thresholds: Adjust caching sensitivity
    - Context management: Automatic cache setup and cleanup
    - Batch size limitations: Optimized for batch sizes <= 2 (CFG support)

Example:
    Apply caching to a SANA transformer::

        from diffusers import SanaTransformer2DModel
        from nunchaku.caching.diffusers_adapters.sana import apply_cache_on_transformer

        transformer = SanaTransformer2DModel.from_pretrained("model_name")
        cached_transformer = apply_cache_on_transformer(
            transformer,
            residual_diff_threshold=0.12
        )

    Apply caching to a complete SANA pipeline::

        from diffusers import SanaPipeline
        from nunchaku.caching.diffusers_adapters.sana import apply_cache_on_pipe

        pipe = SanaPipeline.from_pretrained("Efficient-Large-Model/Sana_600M_512px")
        cached_pipe = apply_cache_on_pipe(
            pipe,
            residual_diff_threshold=0.1
        )

Note:
    SANA caching is specifically designed for the SANA architecture and uses
    mock patching to temporarily replace transformer blocks during inference.
    The caching is automatically disabled for batch sizes > 2 to ensure
    compatibility with classifier-free guidance.
"""

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


def apply_cache_on_pipe(pipe: DiffusionPipeline, **kwargs):
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
