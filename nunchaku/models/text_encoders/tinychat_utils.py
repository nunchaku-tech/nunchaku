# -*- coding: utf-8 -*-
"""
TinyChat Quantization Utilities
==============================

This module provides utility functions for quantized linear layers in the TinyChat backend.
"""

import torch

__all__ = ["ceil_num_groups", "convert_to_tinychat_w4x16y16_linear_weight"]


def ceil_divide(x: int, divisor: int) -> int:
    """
    Compute the ceiling of integer division.

    Parameters
    ----------
    x : int
        Dividend.
    divisor : int
        Divisor.

    Returns
    -------
    int
        The smallest integer greater than or equal to ``x / divisor``.
    """
    return (x + divisor - 1) // divisor


def ceil_num_groups(in_features: int, group_size: int, weight_bits: int = 4) -> int:
    """
    Calculate the padded number of quantization groups for TinyChat quantization.

    This ensures the number of groups is compatible with TinyChat's packing and kernel requirements.

    Parameters
    ----------
    in_features : int
        Input channel size (number of input features).
    group_size : int
        Quantization group size.
    weight_bits : int, optional
        Number of bits per quantized weight (default: 4).

    Returns
    -------
    int
        The padded number of quantization groups.

    Raises
    ------
    AssertionError
        If ``in_features`` is not divisible by ``group_size``, or if ``weight_bits`` is not 4, 2, or 1.
    NotImplementedError
        If ``group_size`` is not one of the supported values (>=128, 64, 32).
    """
    assert in_features % group_size == 0, "input channel size should be divisible by group size."
    num_groups = in_features // group_size
    assert weight_bits in (4, 2, 1), "weight bits should be 4, 2, or 1."
    pack_size = 32 // weight_bits  # one INT32 contains `pack_size` elements of weights
    num_packs = ceil_divide(num_groups, pack_size)
    if group_size >= 128:
        num_packs_factor = 1
    elif group_size == 64:
        num_packs_factor = 2
    elif group_size == 32:
        num_packs_factor = 4
    else:
        raise NotImplementedError("Unsupported group size for TinyChat quantization.")
    # make sure num_packs is a multiple of num_packs_factor
    num_packs = ceil_divide(num_packs, num_packs_factor) * num_packs_factor
    num_groups = num_packs * pack_size
    return num_groups


def pack_w4(weight: torch.Tensor) -> torch.Tensor:
    """
    Pack quantized 4-bit weights into TinyChat's int16 format.

    This function rearranges and packs 4-bit quantized weights (stored as int32) into
    the format expected by TinyChat CUDA kernels.

    Parameters
    ----------
    weight : torch.Tensor
        Quantized weight tensor of shape (out_features, in_features), dtype int32.
        The input channel dimension must be divisible by 32.

    Returns
    -------
    torch.Tensor
        Packed weight tensor of dtype int16.

    Raises
    ------
    AssertionError
        If input tensor is not int32 or input channel size is not divisible by 32.
    """
    assert weight.dtype == torch.int32, f"quantized weight should be torch.int32, but got {weight.dtype}."
    oc, ic = weight.shape
    assert ic % 32 == 0, "input channel size should be divisible by 32."
    # [0, 1, ..., 31] -> [0, 8, 16, 24, 1, 9, 17, 25, ..., 7, 15, 23, 31]
    weight = weight.view(-1, 4, 8)
    weight = weight[:, 0] | (weight[:, 1] << 4) | (weight[:, 2] << 8) | (weight[:, 3] << 12)
    weight = weight.view(oc // 4, 4, ic // 64, 16).permute(0, 2, 1, 3).reshape(oc // 4, ic)
    return weight.to(torch.int16)


def convert_to_tinychat_w4x16y16_linear_weight(
    weight: torch.Tensor,
    scale: torch.Tensor,
    zero: torch.Tensor,
    group_size: int = -1,
    zero_pre_scaled: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Convert a floating-point weight tensor to TinyChat W4-X16-Y16 quantized linear format.

    This function quantizes the input weights to 4 bits per value, applies group-wise
    scaling and zero-point, and packs the result into the format expected by TinyChat
    quantized linear layers.

    Parameters
    ----------
    weight : torch.Tensor
        Floating-point weight tensor of shape (out_features, in_features).
        Must be of dtype ``torch.float16`` or ``torch.bfloat16``.
    scale : torch.Tensor
        Per-group scale tensor (can be broadcastable).
    zero : torch.Tensor
        Per-group zero-point tensor (can be broadcastable).
    group_size : int, optional
        Quantization group size. If set to -1 (default), uses the full input dimension as a single group.
    zero_pre_scaled : bool, optional
        If True, the zero tensor is already scaled by the scale tensor (default: False).

    Returns
    -------
    tuple of torch.Tensor
        - packed_weight : torch.Tensor
            Packed quantized weight tensor (int16).
        - packed_scale : torch.Tensor
            Packed scale tensor (shape: [num_groups, out_features], dtype matches input).
        - packed_zero : torch.Tensor
            Packed zero-point tensor (shape: [num_groups, out_features], dtype matches input).

    Raises
    ------
    AssertionError
        If input types or shapes are invalid, or quantized values are out of range.

    Example
    -------
    .. code-block:: python

        qweight, qscale, qzero = convert_to_tinychat_w4x16y16_linear_weight(
            weight, scale, zero, group_size=128
        )
    """
    dtype, device = weight.dtype, weight.device
    assert dtype in (torch.float16, torch.bfloat16), "currently tinychat only supports fp16 and bf16."
    assert scale is not None, "scale tensor is required for quantization."
    assert zero is not None, "zero point tensor is required for quantization."
    weight = weight.to(dtype=torch.float32)
    scale = scale.to(dtype=torch.float32, device=device)
    zero = zero.to(dtype=torch.float32, device=device)
    if zero_pre_scaled:
        zero = zero * scale
    oc, ic = weight.shape
    group_size = ic if group_size <= 0 else group_size
    assert group_size <= ic, "group size should be less than or equal to input channel size."
    assert ic % group_size == 0, "input channel size should be divisible by group size."
    ng = ic // group_size
    if scale.numel() == 1:
        scale = scale.view(1, 1).expand(oc, ng)
    scale = scale.reshape(oc, ng).contiguous().view(oc, ng, 1)
    if zero.numel() == 1:
        zero = zero.view(1, 1).expand(oc, ng)
    zero = zero.reshape(oc, ng).contiguous().view(oc, ng, 1)
    weight = weight.view(oc, ng, -1).add_(zero).div_(scale).round_().view(oc, ic)
    assert weight.min() >= 0 and weight.max() <= 15, "quantized weight should be in [0, 15]."
    _weight = pack_w4(weight.to(torch.int32))
    _ng = ceil_num_groups(ic, group_size, weight_bits=4)
    _scale = torch.zeros((_ng, oc), dtype=dtype, device=device)
    _zero = torch.zeros((_ng, oc), dtype=dtype, device=device)
    _scale[:ng] = scale.view(oc, ng).t().to(dtype=dtype)
    _zero[:ng] = zero.view(oc, ng).t().to(dtype=dtype).neg_()
    return _weight, _scale, _zero
