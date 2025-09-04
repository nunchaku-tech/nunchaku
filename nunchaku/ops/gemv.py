"""
Python wrapper for Nunchaku's high-performance GEMV (General Matrix-Vector Multiplication) CUDA kernels.
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
    Performs quantized GEMV using the AWQ W4A16 format.

    Parameters
    ----------
    in_feats : torch.Tensor, shape (k,) or (m, k), dtype float16 or bfloat16
        Input feature vector or batch of vectors.
    kernel : torch.Tensor, shape (n // 4, k // 2), dtype int32
        Packed quantized weight matrix.
    scaling_factors : torch.Tensor, shape (k // group_size, n), dtype float16 or bfloat16
        Per-group scaling factors.
    zeros : torch.Tensor, shape (k // group_size, n), dtype float16 or bfloat16
        Per-group zero points.
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
    torch.Tensor, shape (m, n), dtype float16 or bfloat16
        Output tensor.

    Notes
    -----
    Notations:

    - m: batch size
    - n: output features
    - k: input features
    - group_size: quantization group size
    """
    return ops.gemv_awq(in_feats, kernel, scaling_factors, zeros, m, n, k, group_size)
