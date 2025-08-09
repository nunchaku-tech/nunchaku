"""
Python wrappers for Nunchaku's quantization operations.
"""

import torch

from .._C import ops


def svdq_quantize_w4a4_act_fuse_lora_cuda(
    input: torch.Tensor,
    output: torch.Tensor | None = None,
    oscales: torch.Tensor | None = None,
    lora_down: torch.Tensor | None = None,
    lora_act_out: torch.Tensor | None = None,
    smooth: torch.Tensor | None = None,
    fuse_glu: bool = False,
    fp4: bool = False,
) -> torch.Tensor:
    """
    This function wraps the high-performance CUDA kernel for SVDQuant W4A4 quantized GEMM.

    Notation
    --------
    M : int
        Batch size (number of input samples).
    K : int
        Number of input channels (feature dimension).
    N : int
        Number of output channels.
    G : int
        Number of groups. 64 for INT4 and 16 for NVFP4.
    R : int
        Rank of the low-rank branch.
    """
    batch_size, channels = input.shape
    rank = lora_down.shape[1]
    if output is None:
        output = torch.empty(batch_size, channels // 2, dtype=torch.uint8, device=input.device)
    if oscales is None:
        if fp4:
            oscales = torch.empty(channels // 16, batch_size, dtype=torch.float8_e4m3fn, device=input.device)
        else:
            oscales = torch.empty(channels // 64, batch_size, dtype=input.dtype, device=input.device)
    if lora_act_out is None:
        lora_act_out = torch.empty(batch_size, rank, dtype=torch.float32, device=input.device)

    ops.quantize_w4a4_act_fuse_lora(input, output, oscales, lora_down, lora_act_out, smooth, fuse_glu, fp4)
    return output, oscales, lora_act_out
