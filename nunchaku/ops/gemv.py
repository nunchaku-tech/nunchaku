"""
This module provides Python wrappers for Nunchaku's high-performance AWQ W4A16 GEMV (General Matrix-Vector Multiplication) CUDA kernels.
"""

import torch

from .._C import ops


def awq_gemv_w4a16_cuda(
    in_feats: torch.Tensor,
    kernel: torch.Tensor,
    scaling_factors: torch.Tensor,
    zeros: torch.Tensor,
    m: int,
    n: int,
    k: int,
    group_size: int = 64,
) -> torch.Tensor:
    """
    Quantized GEMV using AWQ W4A16 format.

    This function wraps the CUDA kernel for performing quantized general matrix-vector multiplication
    with 4-bit weights and 16-bit activations (W4A16), using the AWQ quantization scheme.

    Parameters
    ----------
    in_feats : torch.Tensor
        Input feature vector of shape (k,) or (m, k), dtype: torch.float16 or torch.bfloat16.
    kernel : torch.Tensor
        Quantized weight matrix of shape (n // 4, k // 2), dtype: torch.int32.
    scaling_factors : torch.Tensor
        Per-group scaling factors, shape (k // group_size, n), dtype: torch.float16 or torch.bfloat16.
    zeros : torch.Tensor
        Per-group zero points, shape (k // group_size, n), dtype: torch.float16 or torch.bfloat16.
    m : int
        Batch size (number of input vectors).
    n : int
        Output feature dimension.
    k : int
        Input feature dimension.
    group_size : int, optional
        Number of input channels per quantization group. Default is 64.

    Returns
    -------
    torch.Tensor
        Output tensor of shape (m, n), dtype: torch.float16 or torch.bfloat16.
    """
    return ops.gemv_awq(in_feats, kernel, scaling_factors, zeros, m, n, k, group_size)
