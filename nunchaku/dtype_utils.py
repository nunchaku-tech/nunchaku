from __future__ import annotations

import warnings
from typing import Optional

import torch


def pick_model_dtype(user_dtype: Optional[torch.dtype] = None, device: Optional[int] = None) -> torch.dtype:
    """
    Select appropriate dtype for the model based on user preference and hardware capabilities.

    Parameters
    ----------
    user_dtype : torch.dtype or None
        Explicitly requested dtype. If float16 or bfloat16, returns as-is.
    device : int or None
        CUDA device index for capability detection.

    Returns
    -------
    torch.dtype
        Selected dtype (float16 or bfloat16).
    """
    if user_dtype in (torch.float16, torch.bfloat16):
        return user_dtype
    dev_index = device if device is not None else (torch.cuda.current_device() if torch.cuda.is_available() else None)
    supports_bf16 = bool(
        torch.cuda.is_available()
        and (torch.cuda.get_device_capability(dev_index or 0)[0] >= 8)
        and torch.cuda.is_bf16_supported()
    )
    return torch.bfloat16 if supports_bf16 else torch.float16


def ensure_arch_compatible(dtype: torch.dtype, device: Optional[int] = None) -> torch.dtype:
    """
    Verify dtype compatibility with GPU architecture.

    Parameters
    ----------
    dtype : torch.dtype
        Requested dtype.
    device : int or None
        CUDA device index.

    Returns
    -------
    torch.dtype
        Compatible dtype (falls back to float16 if bfloat16 unsupported).
    """
    if dtype == torch.bfloat16 and torch.cuda.is_available():
        major, _ = torch.cuda.get_device_capability(device or 0)
        if major < 8:
            warnings.warn("[Nunchaku] Device does not support bfloat16; falling back to float16.")
            return torch.float16
    return dtype


def convert_awq_buffers_to_dtype(module: torch.nn.Module, dtype: torch.dtype) -> None:
    """
    Convert all floating-point buffers in quantized modules to target dtype.

    Parameters
    ----------
    module : torch.nn.Module
        Module containing quantized layers.
    dtype : torch.dtype
        Target dtype for floating-point buffers.

    Notes
    -----
    Preserves bitpacked weights (uint32) unchanged.
    Converts: wscales, wzeros, scales, zeros, smooth_factor, smooth_factor_orig, proj_down, proj_up.
    """

    def _to_if_attr(obj, name: str):
        if hasattr(obj, name):
            buf = getattr(obj, name)
            if isinstance(buf, torch.Tensor) and buf.is_floating_point():
                # Preserve Parameter status and gradient requirements
                if isinstance(buf, torch.nn.Parameter):
                    setattr(obj, name, torch.nn.Parameter(buf.data.to(dtype), requires_grad=buf.requires_grad))
                else:
                    setattr(obj, name, buf.to(dtype))

    # Convert AWQ quantization buffers
    _to_if_attr(module, "wscales")
    _to_if_attr(module, "wzeros")
    _to_if_attr(module, "scales")
    _to_if_attr(module, "zeros")

    # Convert SVDQ projection matrices
    _to_if_attr(module, "smooth_factor")
    _to_if_attr(module, "smooth_factor_orig")
    _to_if_attr(module, "proj_down")
    _to_if_attr(module, "proj_up")

    # Update module dtype attribute for consistency
    if hasattr(module, "torch_dtype"):
        module.torch_dtype = dtype

    for child in module.children():
        convert_awq_buffers_to_dtype(child, dtype)
