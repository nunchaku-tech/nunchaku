"""
Nunchaku quantized attention-related modules.
"""

import torch
from diffusers.models.activations import GELU
from diffusers.models.attention import FeedForward
from torch import nn

from ..ops.fused import fused_gelu_mlp
from .linear import SVDQW4A4Linear


class NunchakuBaseAttention(nn.Module):
    """
    Base class for Nunchaku attention modules.

    Provides a common interface for attention modules with processor selection.

    Parameters
    ----------
    processor : str, optional
        Name of the attention processor to use. Default is "flashattn2".
    *args, **kwargs :
        Additional arguments for subclass initialization.
    """

    def __init__(self, processor: str = "flashattn2", *args, **kwargs):
        super(NunchakuBaseAttention, self).__init__()
        self.processor = None
        self.set_processor(processor)

    def set_processor(self, processor: str):
        """
        Set the attention processor. Must be implemented by subclasses.

        Parameters
        ----------
        processor : str
            Name of the processor to use.

        Raises
        ------
        NotImplementedError
            If not implemented in subclass.
        """
        raise NotImplementedError("Subclass must implement this method")


def _patch_linear(module: nn.Module, linear_cls, **kwargs) -> nn.Module:
    """
    Recursively replace all nn.Linear modules in a given module with a custom linear class.

    Parameters
    ----------
    module : nn.Module
        The module to patch.
    linear_cls : type
        The custom linear class to use for replacement.
    **kwargs :
        Additional arguments passed to ``from_linear``.

    Returns
    -------
    nn.Module
        The patched module with custom linear layers.
    """
    for name, child in module.named_children():
        if isinstance(child, nn.Linear):
            setattr(module, name, linear_cls.from_linear(child, **kwargs))
        else:
            _patch_linear(child, linear_cls, **kwargs)
    return module


class NunchakuFeedForward(FeedForward):
    """
    Quantized feed-forward (MLP) block with fused GELU support.

    Replaces linear layers in a FeedForward block with :class:`~nunchaku.models.linear.SVDQW4A4Linear` for quantized inference.
    Supports fused GELU-MLP computation for efficiency.

    Parameters
    ----------
    ff : FeedForward
        Source FeedForward block to quantize.
    **kwargs :
        Additional arguments for SVDQW4A4Linear.

    Notes
    -----
    For int4 quantization, the activation of the second MLP layer is shifted to be unsigned.
    """

    def __init__(self, ff: FeedForward, **kwargs):
        super(FeedForward, self).__init__()
        self.net = _patch_linear(ff.net, SVDQW4A4Linear, **kwargs)
        # For int4, shift the activation of mlp_fc2 to make it unsigned
        self.net[2].act_unsigned = self.net[2].precision != "nvfp4"

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the quantized feed-forward block.
        It will call :func:`~nunchaku.ops.fused.fused_gelu_mlp` if the first layer is GELU;
        otherwise, apply modules sequentially.

        Parameters
        ----------
        hidden_states : torch.Tensor, shape (B, D)
            Input tensor.

        Returns
        -------
        torch.Tensor, shape (B, D)
            Output tensor after feed-forward transformation.
        """
        if isinstance(self.net[0], GELU):
            return fused_gelu_mlp(hidden_states, self.net[0].proj, self.net[2])
        else:
            # Fallback to original implementation
            for module in self.net:
                hidden_states = module(hidden_states)
            return hidden_states
