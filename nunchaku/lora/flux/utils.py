"""
Utility functions for Flux LoRA operations.

This module provides utility functions for working with LoRA (Low-Rank Adaptation) weights
in Flux models, including format detection and tensor padding operations.
"""

import typing as tp

import torch

from ...utils import ceil_divide, load_state_dict_in_safetensors


def is_nunchaku_format(lora: str | dict[str, torch.Tensor]) -> bool:
    """
    Check if LoRA weights are in Nunchaku format.

    This function determines whether the provided LoRA weights are already in
    Nunchaku format by checking for specific key patterns that are characteristic
    of Nunchaku-formatted LoRA weights.

    Parameters
    ----------
    lora : str or dict[str, torch.Tensor]
        Either a path to a safetensors file containing LoRA weights, or a dictionary
        of LoRA weight tensors.

    Returns
    -------
    bool
        True if the LoRA weights are in Nunchaku format, False otherwise.

    Examples
    --------
    >>> # Check format from file path
    >>> is_nunchaku_format("path/to/lora.safetensors")
    True

    >>> # Check format from weight dictionary
    >>> weights = {"transformer_blocks.0.mlp_fc.weight": torch.randn(128, 256)}
    >>> is_nunchaku_format(weights)
    True
    """
    if isinstance(lora, str):
        tensors = load_state_dict_in_safetensors(lora, device="cpu", return_metadata=False)
        assert isinstance(tensors, dict), "Expected dict when return_metadata=False"
    else:
        tensors = lora

    for k in tensors.keys():
        if ".mlp_fc" in k or "mlp_context_fc1" in k:
            return True
    return False


def pad(
    tensor: tp.Optional[torch.Tensor],
    divisor: int | tp.Sequence[int],
    dim: int | tp.Sequence[int],
    fill_value: float | int = 0,
) -> torch.Tensor | None:
    """
    Pad a tensor to make its dimensions divisible by specified divisors.

    This function pads a tensor along specified dimensions to ensure that the
    tensor's shape is divisible by the given divisor(s). This is commonly used
    in quantization and matrix multiplication operations to ensure alignment
    with hardware requirements.

    Parameters
    ----------
    tensor : torch.Tensor or None
        The tensor to pad. If None, returns None.
    divisor : int or sequence of int
        The divisor(s) to pad the tensor dimensions to. If a single int is provided,
        it applies to all specified dimensions. If a sequence, it must match the
        length of the dim parameter.
    dim : int or sequence of int
        The dimension(s) to pad. If a single int is provided, pads that dimension.
        If a sequence, pads multiple dimensions.
    fill_value : float or int, optional
        The value to use for padding (default: 0).

    Returns
    -------
    torch.Tensor or None
        The padded tensor, or None if input tensor was None.

    Examples
    --------
    >>> # Pad a tensor to make dimension 0 divisible by 16
    >>> tensor = torch.randn(10, 20)
    >>> padded = pad(tensor, divisor=16, dim=0)
    >>> padded.shape
    torch.Size([16, 20])

    >>> # Pad multiple dimensions with different divisors
    >>> tensor = torch.randn(10, 20)
    >>> padded = pad(tensor, divisor=[16, 32], dim=[0, 1])
    >>> padded.shape
    torch.Size([16, 32])
    """
    if isinstance(divisor, int):
        if divisor <= 1:
            return tensor
    elif all(d <= 1 for d in divisor):
        return tensor
    if tensor is None:
        return None
    shape = list(tensor.shape)
    if isinstance(dim, int):
        assert isinstance(divisor, int)
        shape[dim] = ceil_divide(shape[dim], divisor) * divisor
    else:
        if isinstance(divisor, int):
            divisor = [divisor] * len(dim)
        for d, div in zip(dim, divisor, strict=True):
            shape[d] = ceil_divide(shape[d], div) * div
    result = torch.full(shape, fill_value, dtype=tensor.dtype, device=tensor.device)
    result[[slice(0, extent) for extent in tensor.shape]] = tensor
    return result
