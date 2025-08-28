"""
Python wrappers for Nunchaku's quantized GEMV operations.
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
    return ops.gemv_awq(in_feats, kernel, scaling_factors, zeros, m, n, k, group_size)
