"""
IP-Adapter integration for Diffusers pipelines.

This module provides utilities to apply IP-Adapter modifications to compatible
Diffusers pipelines, such as Flux and PuLID pipelines.
"""

from diffusers import DiffusionPipeline


def apply_IPA_on_pipe(pipe: DiffusionPipeline, *args, **kwargs):
    """
    Apply IP-Adapter modifications to a supported Diffusers pipeline.

    Parameters
    ----------
    pipe : DiffusionPipeline
        The pipeline instance to modify. Must be a Flux or PuLID pipeline.
    *args
        Additional positional arguments passed to the underlying implementation.
    **kwargs
        Additional keyword arguments passed to the underlying implementation.

    Returns
    -------
    DiffusionPipeline
        The modified pipeline with IP-Adapter applied.

    Raises
    ------
    ValueError
        If the pipeline class is not supported.

    """
    assert isinstance(pipe, DiffusionPipeline)

    pipe_cls_name = pipe.__class__.__name__
    if pipe_cls_name.startswith("Flux") or pipe_cls_name.startswith("IPAFlux"):
        from .flux import apply_IPA_on_pipe as apply_IPA_on_pipe_fn
    else:
        raise ValueError(f"Unknown pipeline class name: {pipe_cls_name}")
    return apply_IPA_on_pipe_fn(pipe, *args, **kwargs)
