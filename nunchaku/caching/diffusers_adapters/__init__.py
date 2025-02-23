import importlib

from diffusers import DiffusionPipeline


def apply_cache_on_transformer(transformer, *args, **kwargs):
    transformer_cls_name = transformer.__class__.__name__
    if False:
        pass
    elif transformer_cls_name.startswith("Flux"):
        adapter_name = "flux"
    else:
        raise ValueError(f"Unknown transformer class name: {transformer_cls_name}")

    adapter_module = importlib.import_module(f".{adapter_name}", __package__)
    apply_cache_on_transformer_fn = getattr(adapter_module, "apply_cache_on_transformer")
    return apply_cache_on_transformer_fn(transformer, *args, **kwargs)


def apply_cache_on_pipe(pipe: DiffusionPipeline, *args, **kwargs):
    assert isinstance(pipe, DiffusionPipeline)

    pipe_cls_name = pipe.__class__.__name__
    if False:
        pass
    elif pipe_cls_name.startswith("Flux"):
        adapter_name = "flux"
    else:
        raise ValueError(f"Unknown pipeline class name: {pipe_cls_name}")
    print("Registering Flux")
    adapter_module = importlib.import_module(f".{adapter_name}", __package__)
    apply_cache_on_pipe_fn = getattr(adapter_module, "apply_cache_on_pipe")
    return apply_cache_on_pipe_fn(pipe, *args, **kwargs)
