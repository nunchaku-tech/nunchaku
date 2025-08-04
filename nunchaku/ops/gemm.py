"""
Python wrappers for Nunchaku's quantized GEMM operations.
"""

import torch

from .._C import ops


def svdq_gemm_w4a4(
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
    lora_scales: list[float] = [],
    fuse_silu: bool = False,
    fp4: bool = False,
    alpha: float = 1.0,
    wcscales: torch.Tensor | None = None,
    out_q: torch.Tensor | None = None,
    out_k: torch.Tensor | None = None,
    out_v: torch.Tensor | None = None,
    attn_tokens: int = 0,
):
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

    Parameters
    ----------
    act : torch.Tensor
        Input activation tensor. Packed shape (M, K // 2). Packed datatype: torch.int8
    wgt : torch.Tensor
        Quantized weight tensor. Packed shape (N, K // 2). Packed datatype: torch.int8
    out : torch.Tensor or None
        Output tensor for the linear layer. Shape (M, N). Datatype: torch.float16 or torch.bfloat16. If None, we will create a new tensor.
    qout : torch.Tensor or None
        Quantized output tensor for the next layer. Packed shape (M, N // 2). Packed datatype: torch.int8. If None, we will create a new tensor.
    ascales : torch.Tensor
        Activation scales tensor. Shape (K // G, M). Datatype: torch.float16 or torch.bfloat16 for INT4 and torch.float8_e4m3 for NVFP4.
    wscales : torch.Tensor
        Weight scales tensor. Shape (K // G, N). Datatype: torch.float16 or torch.bfloat16 for INT4 and torch.float8_e4m3 for NVFP4.
    oscales : torch.Tensor or None
        Output scales tensor. Shape (N // G, M). Datatype: torch.float16 or torch.bfloat16 for INT4 and torch.float8_e4m3 for NVFP4.
    poolout : torch.Tensor or None
        Not used for now. Just leave it as None.
    lora_act_in : torch.Tensor
        Low-rank down output tensor. Packed shape (M, R). Packed datatype: torch.float32.
    lora_up : torch.Tensor
        Low-rank up-projection weights. Packed shape (N, R). Packed datatype: torch.float16 or torch.bfloat16.
    lora_down : torch.Tensor or None
        Low-rank down-projection weights in the next layer. Packed shape (N, R). Packed datatype: torch.float16 or torch.bfloat16.
    lora_act_out : torch.Tensor or None
        Output tensor for low-rank down-projection in the next layer. Packed shape (M, R). Packed datatype: torch.float32.
    norm_q : torch.Tensor or None
        Query normalization tensor. Shape (HEAD_DIM,). Datatype: torch.float16 or torch.bfloat16.
    norm_k : torch.Tensor or None
        Key normalization tensor. Shape (HEAD_DIM,). Datatype: torch.float16 or torch.bfloat16.
    rotary_emb : torch.Tensor or None
        Rotary embedding tensor. Shape (M, HEAD_DIM // 2, 2, 2). Datatype: torch.float32. TODO: double check this.
    bias : torch.Tensor or None
        Bias tensor. Shape (N,). Datatype: torch.float16 or torch.bfloat16.
    smooth_factor : torch.Tensor or None
        Smoothing factor tensor for quantization in the next layer. Shape (N,). Datatype: torch.float16 or torch.bfloat16.
    out_vk : torch.Tensor or None
        Used only in SANA.
    out_linearattn : torch.Tensor or None
        Used only in SANA.
    act_unsigned : bool, default=False
        Whether activations are unsigned.
    lora_scales : list of float, default=[]
        Scaling factors for the low-rank branch.
    fuse_silu : bool, default=False
        Whether to fuse SiLU activation.
    fp4 : bool, default=False
        Whether to use 4-bit floating point quantization (NVFP4).
    alpha : float, default=1.0
        Per tensor scaling factor for NVFP4.
    wcscales : torch.Tensor or None, default=None
        Per channel scaling factors for NVFP4. Shape (N,). Datatype: torch.float8_e4m3.
    out_q : torch.Tensor or None, default=None
        Output tensor for quantized Q, used for Nunchaku attention. Packed shape (B, H, M, D). Datatype: torch.int8.
    out_k : torch.Tensor or None, default=None
        Output tensor for quantized K, used for Nunchaku attention. Packed shape (B, H, M, D). Datatype: torch.int8.
    out_v : torch.Tensor or None, default=None
        Output tensor for quantized V, used for Nunchaku attention. Packed shape (B, H, M, D). Datatype: torch.int8.
    attn_tokens : int, default=0
        Number of attention tokens.

    Returns
    -------
    None
        The results are written in-place to the provided output tensors.

    """
    if out is None:
        out = torch.empty(act.shape[0], wgt.shape[0], dtype=act.dtype, device=act.device)
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
    return out
