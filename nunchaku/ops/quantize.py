"""
This module provides Python wrappers for Nunchaku's high-performance SVDQuant quantization CUDA kernels.
"""

import torch

from .._C import ops
from ..utils import ceil_divide


def svdq_quantize_w4a4_act_fuse_lora_cuda(
    input: torch.Tensor,
    output: torch.Tensor | None = None,
    oscales: torch.Tensor | None = None,
    lora_down: torch.Tensor | None = None,
    lora_act_out: torch.Tensor | None = None,
    smooth: torch.Tensor | None = None,
    fuse_glu: bool = False,
    fp4: bool = False,
    pad_size: int = 256,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    r"""
    Quantize input activations and compute the down-projection of the low-rank branch.

    This function wraps the high-performance CUDA kernel for SVDQuant W4A4 quantized GEMM,
    supporting LoRA fusion and optional GLU activation fusion.

    Notations:

    - :math:`M`: Batch size (number of input tokens)
    - :math:`K`: Number of input channels (feature dimension)
    - :math:`N`: Number of output channels
    - :math:`G`: Number of groups (64 for INT4, 16 for NVFP4)
    - :math:`R`: Rank of the low-rank branch
    - :math:`M_\mathrm{pad}`: Padded batch size, computed as :math:`\lceil M / \mathrm{pad\_size} \rceil \times \mathrm{pad\_size}`

    Parameters
    ----------
    input : torch.Tensor
        Input tensor of shape :math:`(M, K)`.
    output : torch.Tensor or None, optional
        Output tensor to store quantized activations. If None, a tensor is allocated.
    oscales : torch.Tensor or None, optional
        Output scales tensor. If None, a tensor is allocated.
    lora_down : torch.Tensor or None, optional
        Down-projection weights of the low-rank branch of shape :math:`(K, R)`, where :math:`R` is the LoRA rank.
    lora_act_out : torch.Tensor or None, optional
        Output tensor for LoRA activations. If None, a tensor is allocated.
    smooth : torch.Tensor or None, optional
        Smoothing factor for quantization.
    fuse_glu : bool, default=False
        Whether to fuse GLU activation.
    fp4 : bool, default=False
        If True, use NVFP4 quantization (4-bit floating point); otherwise, use INT4.
    pad_size : int, default=256
        Pad batch size to a multiple of this value for efficient CUDA execution.

    Returns
    -------
    output : torch.Tensor
        Quantized output tensor of shape :math:`(M_\mathrm{pad}, K / 2)`, packed into dtype uint8.
    oscales : torch.Tensor
        Output scales tensor of shape :math:`(K / G, M_\mathrm{pad})`, dtype float8_e4m3fn (for NVFP4) or input dtype (for INT4).
    lora_act_out : torch.Tensor
        LoRA activation output tensor of shape :math:`(M_\mathrm{pad}, R)`, dtype float32.
    """
    batch_size, channels = input.shape
    rank = lora_down.shape[1]
    batch_size_pad = ceil_divide(batch_size, pad_size) * pad_size
    if output is None:
        output = torch.empty(batch_size_pad, channels // 2, dtype=torch.uint8, device=input.device)
    if oscales is None:
        if fp4:
            assert channels % 16 == 0
            oscales = torch.empty(channels // 16, batch_size_pad, dtype=torch.float8_e4m3fn, device=input.device)
        else:
            assert channels % 64 == 0
            oscales = torch.empty(channels // 64, batch_size_pad, dtype=input.dtype, device=input.device)
    if lora_act_out is None:
        lora_act_out = torch.empty(batch_size_pad, rank, dtype=torch.float32, device=input.device)

    ops.quantize_w4a4_act_fuse_lora(input, output, oscales, lora_down, lora_act_out, smooth, fuse_glu, fp4)
    return output, oscales, lora_act_out
