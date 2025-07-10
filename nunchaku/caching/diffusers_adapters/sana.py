"""
Adapters for efficient caching in SANA diffusion pipelines.

This module enables single first-block caching for SANA models, providing:

- :func:`apply_cache_on_transformer` — Add caching to a ``SanaTransformer2DModel``.
- :func:`apply_cache_on_pipe` — Add caching to a complete SANA pipeline.

Caching is context-managed and only active within a cache context.
"""

import functools
import unittest

import torch
from diffusers import DiffusionPipeline, SanaTransformer2DModel

from ...caching import utils


def apply_cache_on_transformer(transformer: SanaTransformer2DModel, *, residual_diff_threshold=0.12):
    """
    Enable caching for a ``SanaTransformer2DModel``.

    This function wraps the transformer to use cached transformer blocks for faster inference.
    Uses single first-block caching with configurable similarity thresholds.

    Parameters
    ----------
    transformer : SanaTransformer2DModel
        The transformer to modify.
    residual_diff_threshold : float, optional
        Similarity threshold for caching (default: 0.12).

    Returns
    -------
    SanaTransformer2DModel
        The transformer with caching enabled.

    Notes
    -----
    If already cached, returns the transformer unchanged. Caching is only active within a cache context.
    """
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
    """
    Enable caching for a complete SANA diffusion pipeline.

    This function wraps the pipeline's ``__call__`` method to manage cache contexts,
    and applies transformer-level caching.

    Parameters
    ----------
    pipe : DiffusionPipeline
        The SANA pipeline to modify.
    **kwargs
        Passed to :func:`apply_cache_on_transformer` (e.g., ``residual_diff_threshold``).

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
