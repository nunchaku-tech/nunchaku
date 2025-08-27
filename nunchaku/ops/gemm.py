"""
This module provides Python wrappers for Nunchaku's high-performance quantized SVDQuant W4A4 GEMM (General Matrix-Matrix Multiplication) CUDA kernels.
"""

import math
import torch

from .._C import ops


def svdq_gemm_w4a4_cuda(
    act: torch.Tensor,
    wgt: torch.Tensor,
    out: torch.Tensor | None = None,
    qout: torch.Tensor | None = None,
    ascales: torch.Tensor | None = None,
    wscales: torch.Tensor | None = None,
    oscales: torch.Tensor | None = None,
    poolout: torch.Tensor | None = None,
    lora_act_in: torch.Tensor | None = None,
    lora_up: torch.Tensor | None = None,
    lora_down: torch.Tensor | None = None,
    lora_act_out: torch.Tensor | None = None,
    norm_q: torch.Tensor | None = None,
    norm_k: torch.Tensor | None = None,
    rotary_emb: torch.Tensor | None = None,
    bias: torch.Tensor | None = None,
    smooth_factor: torch.Tensor | None = None,
    out_vk: torch.Tensor | None = None,
    out_linearattn: torch.Tensor | None = None,
    act_unsigned: bool = False,
    lora_scales: list[float] | None = None,
    fuse_silu: bool = False,
    fp4: bool = False,
    alpha: float | None = 1.0,
    wcscales: torch.Tensor | None = None,
    out_q: torch.Tensor | None = None,
    out_k: torch.Tensor | None = None,
    out_v: torch.Tensor | None = None,
    attn_tokens: int = 0,
):
    """
    This function wraps the high-performance CUDA kernel for SVDQuant W4A4 quantized GEMM, supporting LoRA, rotary embeddings, normalization, and fused activations.

    Parameters
    ----------
    act : torch.Tensor
        Input activation tensor. Packed shape (M, K // 2), dtype: torch.int8.
    wgt : torch.Tensor
        Quantized weight tensor. Packed shape (N, K // 2), dtype: torch.int8.
    out : torch.Tensor or None, optional
        Output tensor for the linear layer, shape (M, N), dtype: torch.float16 or torch.bfloat16.
    qout : torch.Tensor or None, optional
        Quantized output tensor for the next layer, packed shape (M, N // 2), dtype: torch.int8.
    ascales : torch.Tensor or None, optional
        Activation scales, shape (K // G, M), dtype: torch.float16/torch.bfloat16 (INT4) or torch.float8_e4m3fn (NVFP4).
    wscales : torch.Tensor or None, optional
        Weight scales, shape (K // G, N), dtype: torch.float16/torch.bfloat16 (INT4) or torch.float8_e4m3fn (NVFP4).
    oscales : torch.Tensor or None, optional
        Output scales, shape (N // G, M), dtype: torch.float16/torch.bfloat16 (INT4) or torch.float8_e4m3fn (NVFP4).
    poolout : torch.Tensor or None, optional
        Reserved for future use. Leave as None.
    lora_act_in : torch.Tensor or None, optional
        LoRA down-projection activations, shape (M, R), dtype: torch.float32.
    lora_up : torch.Tensor or None, optional
        Packed LoRA up-projection weights, shape (N, R), dtype: torch.float16 or torch.bfloat16.
    lora_down : torch.Tensor or None, optional
        Packed LoRA down-projection weights for the next layer, shape (N, R), dtype: torch.float16 or torch.bfloat16.
    lora_act_out : torch.Tensor or None, optional
        Output tensor for LoRA down-projection in the next layer, shape (M, R), dtype: torch.float32.
    norm_q : torch.Tensor or None, optional
        Optional query RMS normalization tensor, shape (HEAD_DIM,), dtype: torch.float16 or torch.bfloat16.
    norm_k : torch.Tensor or None, optional
        Optional key RMS normalization tensor, shape (HEAD_DIM,), dtype: torch.float16 or torch.bfloat16.
    rotary_emb : torch.Tensor or None, optional
        Packed rotary embedding tensor, shape (M, HEAD_DIM // 2, 2, 2), dtype: torch.float32.
    bias : torch.Tensor or None, optional
        Bias tensor, shape (N,), dtype: torch.float16 or torch.bfloat16.
    smooth_factor : torch.Tensor or None, optional
        Smoothing factor for quantization in the next layer, shape (N,), dtype: torch.float16 or torch.bfloat16.
    out_vk : torch.Tensor or None, optional
        Used only in SANA. Leave as None.
    out_linearattn : torch.Tensor or None, optional
        Used only in SANA. Leave as None.
    act_unsigned : bool, default=False
        Whether activations are unsigned. This is only used for INT4 after the GeLU activation as we shift the activations to the positive range.
    lora_scales : list of float or None, optional
        Scaling factors for the LoRA branch of each group. Each group has 16 channels. If None, defaults to 1.0 for each group.
    fuse_silu : bool, default=False
        Whether to fuse SiLU activation.
    fp4 : bool, default=False
        Whether to use 4-bit floating point quantization (NVFP4).
    alpha : float or None, default=1.0
        Per-tensor scaling factor for NVFP4.
    wcscales : torch.Tensor or None, optional
        Per-channel scaling factors for NVFP4, shape (N,), dtype: torch.float8_e4m3fn.
    out_q : torch.Tensor or None, optional
        Packed output tensor for quantized Q (for attention), packed shape (B, H, M, D), dtype: torch.int8. Only used in nunchaku-fp16 attention.
    out_k : torch.Tensor or None, optional
        Packed output tensor for quantized K (for attention), packed shape (B, H, M, D), dtype: torch.int8. Only used in nunchaku-fp16 attention.
    out_v : torch.Tensor or None, optional
        Packed output tensor for quantized V (for attention), packed shape (B, H, M, D), dtype: torch.int8. Only used in nunchaku-fp16 attention.
    attn_tokens : int, default=0
        Number of attention tokens.

    Returns
    -------
    None
        Results are written in-place to the provided output tensors.

    Notes
    -----
    - M: batch size
    - K: input channels
    - N: output channels
    - G: number of groups (64 for INT4, 16 for NVFP4)
    - R: rank of the low-rank branch
    - All tensors must be CUDA tensors.
    - For LoRA, R is the rank of the low-rank adaptation.

    """
    if lora_scales is None:
        rank = lora_up.shape[1]
        lora_scales = [1.0] * math.ceil(rank / 16)
    if alpha is None:
        alpha = 1.0
    ops.gemm_w4a4(
        act,
        wgt,
        out,
        qout,
        ascales,
        wscales,
        oscales,
        poolout,
        lora_act_in,
        lora_up,
        lora_down,
        lora_act_out,
        norm_q,
        norm_k,
        rotary_emb,
        bias,
        smooth_factor,
        out_vk,
        out_linearattn,
        act_unsigned,
        lora_scales,
        fuse_silu,
        fp4,
        alpha,
        wcscales,
        out_q,
        out_k,
        out_v,
        attn_tokens,
    )
