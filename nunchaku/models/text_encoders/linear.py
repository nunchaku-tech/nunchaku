# -*- coding: utf-8 -*-
"""TinyChat Quantized Linear Module"""

import torch
import torch.nn as nn

from ..._C.ops import gemm_awq, gemv_awq
from .tinychat_utils import ceil_num_groups, convert_to_tinychat_w4x16y16_linear_weight

__all__ = ["W4Linear"]


class W4Linear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        group_size: int = 128,
        dtype: torch.dtype = torch.float16,
        device: str | torch.device = "cuda",
    ):
        super().__init__()
        assert dtype in (torch.float16, torch.bfloat16), f"Unsupported dtype: {dtype}"

        self.in_features = in_features
        self.out_features = out_features
        self.group_size = group_size if group_size != -1 else in_features
        assert self.in_features % self.group_size == 0
        assert out_features % (32 // self.weight_bits) == 0
        self.ceil_num_groups = ceil_num_groups(
            in_features=self.in_features,
            group_size=self.group_size,
            weight_bits=self.weight_bits,
        )

        assert out_features % (self.interleave) == 0
        self.register_buffer(
            "qweight",
            torch.zeros(
                (
                    self.out_features // self.interleave,
                    self.in_features // (16 // self.weight_bits) * self.interleave,
                ),
                dtype=torch.int16,
                device=device,
            ),
        )
        self.register_buffer(
            "scales",
            torch.zeros((self.ceil_num_groups, self.out_features), dtype=dtype, device=device),
        )
        self.register_buffer(
            "scaled_zeros",
            torch.zeros((self.ceil_num_groups, self.out_features), dtype=dtype, device=device),
        )
        if bias:
            self.register_buffer("bias", torch.zeros((out_features), dtype=dtype, device=device))
        else:
            self.bias = None

    @property
    def weight_bits(self) -> int:
        return 4

    @property
    def interleave(self) -> int:
        return 4

    @torch.no_grad()
    def forward(self, x):
        if x.numel() / x.shape[-1] < 8:
            out = gemv_awq(
                x,
                self.qweight,
                self.scales,
                self.scaled_zeros,
                x.numel() // x.shape[-1],
                self.out_features,
                self.in_features,
                self.group_size,
            )
        else:
            if self.group_size != 128:
                raise NotImplementedError("Kernel currently only supports group_size=128.")
            out = gemm_awq(x, self.qweight, self.scales, self.scaled_zeros)
        out = out + self.bias if self.bias is not None else out
        return out

    @staticmethod
    def from_linear(
        linear: nn.Linear,
        group_size: int,
        init_only: bool = False,
        weight: torch.Tensor | None = None,
        scale: torch.Tensor | None = None,
        zero: torch.Tensor | None = None,
        zero_pre_scaled: bool = False,
    ) -> "W4Linear":
        """Convert a linear layer to a TinyChat 4-bit weight-only quantized linear layer.

        Args:
            linear (`nn.Linear`):
                linear layer to be converted.
            group_size (`int`):
                quantization group size.
            init_only (`bool`, *optional*, defaults to `False`):
                whether to only initialize the quantized linear layer.
            weight (`torch.Tensor`, *optional*, defaults to `None`):
                weight tensor for the quantized linear layer.
            scale (`torch.Tensor`, *optional*, defaults to `None`):
                scale tensor for the quantized linear layer.
            zero (`torch.Tensor`, *optional*, defaults to `None`):
                zero point tensor for the quantized linear layer.
            zero_pre_scaled (`bool`, *optional*, defaults to `False`):
                whether zero point tensor is pre-scaled.

        Returns:
            `W4Linear`:
                quantized linear layer.
        """
        assert isinstance(linear, nn.Linear)
        weight = linear.weight.data if weight is None else weight.data
        dtype, device = weight.dtype, weight.device
        oc, ic = linear.out_features, linear.in_features
        _linear = W4Linear(
            in_features=ic,
            out_features=oc,
            bias=linear.bias is not None,
            group_size=group_size,
            dtype=dtype,
            device=device,
        )
        if init_only:
            return _linear
        if linear.bias is not None:
            _linear.bias.data.copy_(linear.bias.data)
        if scale is None:
            assert zero is None, "scale and zero point tensors should be provided together."
            group_size = ic if group_size <= 0 else group_size
            assert group_size <= ic, "group size should be less than or equal to input channel size."
            assert ic % group_size == 0, "input channel size should be divisible by group size."
            ng, gs = ic // group_size, group_size
            weight = weight.to(dtype=torch.float32).view(oc, 1, ng, gs)
            vmin, vmax = weight.amin(dim=-1, keepdim=True), weight.amax(dim=-1, keepdim=True)
            scale = (vmax - vmin).div_(15)
            scale[scale == 0] = 1.0
            if zero_pre_scaled:
                zero = vmin.neg_().div_(scale).round_().clamp_(0, 15)
                weight = weight.div_(scale).add_(zero).round_().clamp_(0, 15).sub_(zero).mul_(scale)
            else:
                zero = vmin.neg_().clamp_min(0)
                weight = weight.add_(zero).div_(scale).round_().clamp_(0, 15).mul_(scale).sub_(zero)
            weight = weight.to(dtype=dtype).view(oc, ic)
            scale = scale.to(dtype=dtype)
            zero = zero.to(dtype=dtype)
        weight, scale, zero = convert_to_tinychat_w4x16y16_linear_weight(
            weight=weight,
            scale=scale,
            zero=zero,
            group_size=group_size,
            zero_pre_scaled=zero_pre_scaled,
        )
        _linear.qweight.data.copy_(weight)
        _linear.scales.data.copy_(scale)
        _linear.scaled_zeros.data.copy_(zero)
        return _linear

    def extra_repr(self) -> str:
        return "in_features={}, out_features={}, bias={}, weight_bits={}, group_size={}".format(
            self.in_features,
            self.out_features,
            self.bias is not None,
            self.weight_bits,
            self.group_size,
        )
