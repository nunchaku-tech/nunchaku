from diffusers import DiffusionPipeline, FluxTransformer2DModel
from torch import nn

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
    dummy_single_transformer_blocks = nn.ModuleList()

    IPA_transformer_blocks[0].load_ip_adapter_weights_per_layer(repo_id=repo_id)

    transformer.transformer_blocks = IPA_transformer_blocks
    transformer.single_transformer_blocks = dummy_single_transformer_blocks

    transformer._is_IPA = True

    return transformer


def apply_IPA_on_pipe(pipe: DiffusionPipeline, *, shallow_patch: bool = False, **kwargs):
    if not shallow_patch:
        apply_IPA_on_transformer(pipe.transformer, **kwargs)

    return pipe
