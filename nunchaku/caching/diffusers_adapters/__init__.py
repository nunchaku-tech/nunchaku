"""
Diffusers Pipeline Adapters for Nunchaku Caching.

This module provides adapter functions that integrate Nunchaku's caching capabilities
with diffusers pipelines. The adapters automatically detect pipeline types and apply
the appropriate caching strategy without requiring manual configuration.

The module serves as a unified interface for applying caching to different types of
diffusion pipelines, currently supporting:

- Flux pipelines (FluxPipeline and related variants)
- SANA pipelines (SanaPipeline and related variants)

Key Features:
    - Automatic pipeline type detection
    - Unified interface for different caching strategies
    - Seamless integration with existing diffusers workflows
    - Support for both shallow and deep caching patches

Example:
    Basic usage with automatic pipeline detection::

        from diffusers import FluxPipeline
        from nunchaku.caching.diffusers_adapters import apply_cache_on_pipe

        # Load any supported pipeline
        pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-dev")

        # Apply caching automatically based on pipeline type
        cached_pipe = apply_cache_on_pipe(
            pipe,
            residual_diff_threshold=0.1,
            use_double_fb_cache=True
        )

        # Use the cached pipeline normally
        image = cached_pipe(prompt="A beautiful landscape")

Note:
    The adapter functions modify the pipeline in-place, adding caching capabilities
    while preserving the original API. The caching behavior is transparent to the
    user and doesn't require changes to existing code.
"""

from diffusers import DiffusionPipeline


def apply_cache_on_pipe(pipe: DiffusionPipeline, *args, **kwargs):
    """
    Apply caching to a diffusers pipeline with automatic type detection.

    This function serves as a unified interface for applying Nunchaku caching
    to different types of diffusion pipelines. It automatically detects the
    pipeline type based on the class name and delegates to the appropriate
    caching implementation.

    Args:
        pipe (DiffusionPipeline): The diffusers pipeline to apply caching to
        *args: Variable positional arguments passed to the specific caching function
        **kwargs: Variable keyword arguments passed to the specific caching function.
            Common arguments include:
            - residual_diff_threshold (float): Similarity threshold for cache validity
            - use_double_fb_cache (bool): Whether to use double first-block caching
            - shallow_patch (bool): Whether to use shallow patching only
            - verbose (bool): Whether to enable verbose caching messages

    Returns:
        DiffusionPipeline: The same pipeline instance with caching applied

    Raises:
        ValueError: If the pipeline type is not supported (doesn't start with "Flux" or "Sana")
        AssertionError: If the input is not a DiffusionPipeline instance

    Example:
        With a Flux pipeline::

            from diffusers import FluxPipeline
            pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-dev")
            cached_pipe = apply_cache_on_pipe(
                pipe,
                residual_diff_threshold=0.12,
                use_double_fb_cache=True
            )

        With a SANA pipeline::

            from diffusers import SanaPipeline
            pipe = SanaPipeline.from_pretrained("Efficient-Large-Model/Sana_600M_512px")
            cached_pipe = apply_cache_on_pipe(
                pipe,
                residual_diff_threshold=0.1
            )

    Note:
        The function modifies the pipeline in-place and returns the same instance.
        Currently supported pipeline types are those with class names starting
        with "Flux" or "Sana".
    """
    assert isinstance(pipe, DiffusionPipeline)

    pipe_cls_name = pipe.__class__.__name__
    if pipe_cls_name.startswith("Flux"):
        from .flux import apply_cache_on_pipe as apply_cache_on_pipe_fn
    elif pipe_cls_name.startswith("Sana"):
        from .sana import apply_cache_on_pipe as apply_cache_on_pipe_fn
    else:
        raise ValueError(f"Unknown pipeline class name: {pipe_cls_name}")
    return apply_cache_on_pipe_fn(pipe, *args, **kwargs)
