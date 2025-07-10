"""
Pipeline Adapters for First-Block Caching in Nunchaku
"""

from diffusers import DiffusionPipeline


def apply_cache_on_pipe(pipe: DiffusionPipeline, *args, **kwargs):
    """
    Apply caching to a diffusers pipeline with automatic type detection.

    This function serves as a unified interface for applying first-block caching
    to different types of diffusion pipelines of nunchaku models. It automatically detects the
    pipeline type based on the class name and delegates to the appropriate
    caching implementation.

    Parameters
    ----------
    pipe : DiffusionPipeline
        The diffusers pipeline to apply caching to.
    *args
        Variable positional arguments passed to the specific caching function.
    **kwargs
        Variable keyword arguments passed to the specific caching function.
        Common arguments include:

            - residual_diff_threshold (float): Similarity threshold for cache validity
            - use_double_fb_cache (bool): Whether to use double first-block caching
            - shallow_patch (bool): Whether to use shallow patching only
            - verbose (bool): Whether to enable verbose caching messages

    Returns
    -------
    DiffusionPipeline
        The same pipeline instance with caching applied.

    Raises
    ------
    ValueError
        If the pipeline type is not supported (doesn't start with "Flux" or "Sana").
    AssertionError
        If the input is not a DiffusionPipeline instance.

    Examples
    --------
    With a Flux pipeline:

    .. code-block:: python

        from diffusers import FluxPipeline
        pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-dev")
        cached_pipe = apply_cache_on_pipe(
            pipe,
            residual_diff_threshold=0.12,
            use_double_fb_cache=True
        )

    With a SANA pipeline:

    .. code-block:: python

        from diffusers import SanaPipeline
        pipe = SanaPipeline.from_pretrained("Efficient-Large-Model/Sana_600M_512px")
        cached_pipe = apply_cache_on_pipe(
            pipe,
            residual_diff_threshold=0.1
        )

    Notes
    -----
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
