import torch
from diffusers.models.activations import GELU
from diffusers.models.attention import FeedForward
from torch import nn

from ..ops.fused import fused_gelu_mlp
from .linear import SVDQW4A4Linear


class NunchakuBaseAttention(nn.Module):
    def __init__(self, processor: str = "flashattn2", *args, **kwargs):
        super(NunchakuBaseAttention, self).__init__()
        self.processor = None
        self.set_processor(processor)

    def set_processor(self, processor: str):
        raise NotImplementedError("Subclass must implement this method")


def _patch_linear(module: nn.Module, linear_cls, **kwargs) -> nn.Module:
    for name, child in module.named_children():
        if isinstance(child, nn.Linear):
            setattr(module, name, linear_cls.from_linear(child, **kwargs))
        else:
            _patch_linear(child, linear_cls, **kwargs)
    return module


class NunchakuFeedForward(FeedForward):
    def __init__(self, ff: FeedForward, **kwargs):
        super(FeedForward, self).__init__()
        self.net = _patch_linear(ff.net, SVDQW4A4Linear, **kwargs)
        # for int4, we shift the activation of mlp_fc2 to make it unsigned
        self.net[2].act_unsigned = self.net[2].precision != "nvfp4"

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if isinstance(self.net[0], GELU):
            return fused_gelu_mlp(hidden_states, self.net[0].proj, self.net[2])
        else:
            # fallback to original implementation
            for module in self.net:
                hidden_states = module(hidden_states)
            return hidden_states
