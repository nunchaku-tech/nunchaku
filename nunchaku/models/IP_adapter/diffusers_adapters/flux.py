import functools
import unittest

from diffusers import DiffusionPipeline, FluxTransformer2DModel
from torch import nn

from nunchaku.caching.utils import cache_context, create_cache_context
from nunchaku.models.IP_adapter.utils import undo_all_mods_on_transformer

from ...IP_adapter import utils


def apply_IPA_on_transformer(transformer: FluxTransformer2DModel, *, ip_adapter_scale: float = 1.0, repo_id: str):
    IPA_transformer_blocks = nn.ModuleList(
        [
            utils.IPA_TransformerBlocks(
                transformer=transformer,
                ip_adapter_scale=ip_adapter_scale,
                return_hidden_states_first=False,
                device=transformer.device,
            )
        ]
    )
    if getattr(transformer, "_is_cached", False):
        IPA_transformer_blocks[0].update_residual_diff_threshold(
            use_double_fb_cache=transformer.use_double_fb_cache,
            residual_diff_threshold_multi=transformer.residual_diff_threshold_multi,
            residual_diff_threshold_single=transformer.residual_diff_threshold_single,
        )
        undo_all_mods_on_transformer(transformer)
        if not hasattr(transformer, "_original_forward"):
            transformer._original_forward = transformer.forward
        if not hasattr(transformer, "_original_blocks"):
            transformer._original_blocks = transformer.transformer_blocks

    dummy_single_transformer_blocks = nn.ModuleList()

    IPA_transformer_blocks[0].load_ip_adapter_weights_per_layer(repo_id=repo_id)

    transformer.transformer_blocks = IPA_transformer_blocks
    transformer.single_transformer_blocks = dummy_single_transformer_blocks
    original_forward = transformer.forward

    @functools.wraps(original_forward)
    def new_forward(self, *args, **kwargs):
        with (
            unittest.mock.patch.object(self, "transformer_blocks", IPA_transformer_blocks),
            unittest.mock.patch.object(self, "single_transformer_blocks", dummy_single_transformer_blocks),
        ):
            return original_forward(*args, **kwargs)

    transformer.forward = new_forward.__get__(transformer)
    transformer._is_IPA = True

    return transformer


def apply_IPA_on_pipe(pipe: DiffusionPipeline, *, shallow_patch: bool = False, **kwargs):
    if getattr(pipe, "_is_cached", False):
        original_call = pipe.__class__.__call__

        @functools.wraps(original_call)
        def new_call(self, *args, **kwargs):
            with cache_context(create_cache_context()):
                return original_call(self, *args, **kwargs)

        pipe.__class__.__call__ = new_call
        pipe.__class__._is_cached = True

    if not shallow_patch:
        apply_IPA_on_transformer(pipe.transformer, **kwargs)

    return pipe
