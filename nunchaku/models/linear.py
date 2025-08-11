import torch
from torch import nn

from ..ops.gemm import svdq_gemm_w4a4_cuda
from ..ops.gemv import awq_gemv_w4a16_cuda
from ..ops.quantize import svdq_quantize_w4a4_act_fuse_lora_cuda


class SVDQW4A4Linear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 32,
        bias: bool = True,
        precision: str = "int4",
        torch_dtype: torch.dtype = torch.bfloat16,
        device: str | torch.device = "cpu",
    ):
        super(SVDQW4A4Linear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank

        self.precision = precision
        self.torch_dtype = torch_dtype
        self.group_size = None

        if precision == "nvfp4":
            self.group_size = 16
        elif precision == "int4":
            self.group_size = 64
        else:
            raise ValueError(f"Invalid precision: {precision}")

        self.qweight = nn.Parameter(
            torch.empty(out_features, in_features // 2, dtype=torch.int8, device=device), requires_grad=False
        )
        self.bias = (
            nn.Parameter(torch.empty(out_features, dtype=torch_dtype, device=device), requires_grad=True)
            if bias
            else None
        )

        self.wscales = nn.Parameter(
            torch.empty(
                in_features // self.group_size,
                out_features,
                dtype=torch_dtype if precision == "int4" else torch.float8_e4m3fn,
                device=device,
            ),
            requires_grad=False,
        )
        self.smooth_factor = nn.Parameter(
            torch.empty(in_features, dtype=torch_dtype, device=device), requires_grad=False
        )
        self.smooth_factor_orig = nn.Parameter(
            torch.empty(in_features, dtype=torch_dtype, device=device), requires_grad=False
        )

        self.proj_down = nn.Parameter(torch.empty(in_features, rank, dtype=torch_dtype, device=device))
        self.proj_up = nn.Parameter(torch.empty(out_features, rank, dtype=torch_dtype, device=device))

        self.wtscale = None
        self.wcscales = None
        if precision == "nvfp4":
            self.wtscale = nn.Parameter(torch.ones(1, dtype=torch_dtype, device=device), requires_grad=False)
            self.wcscales = nn.Parameter(
                torch.ones(out_features, dtype=torch_dtype, device=device), requires_grad=False
            )

        self.act_unsigned = False

    @classmethod
    def from_linear(cls, linear: nn.Linear, **kwargs):
        in_features = kwargs.pop("in_features", linear.in_features)
        return cls(
            in_features=in_features,
            out_features=linear.out_features,
            bias=linear.bias is not None,
            torch_dtype=linear.weight.dtype,
            device=linear.weight.device,
            **kwargs,
        )

    def forward(self, x: torch.Tensor, output: torch.Tensor | None = None) -> torch.Tensor:
        # quantize the input run the down projection
        batch_size, seq_len, channels = x.shape
        x = x.view(batch_size * seq_len, channels)
        if output is None:
            output = torch.empty(batch_size * seq_len, self.out_features, dtype=x.dtype, device=x.device)
        quantized_x, ascales, lora_act_out = self.quantize(x)
        output = self.forward_quant(quantized_x, ascales, lora_act_out, output)
        output = output.view(batch_size, seq_len, -1)
        return output

    def quantize(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        quantized_x, ascales, lora_act_out = svdq_quantize_w4a4_act_fuse_lora_cuda(
            x, lora_down=self.proj_down, smooth=self.smooth_factor, fp4=self.precision == "nvfp4"
        )
        return quantized_x, ascales, lora_act_out

    def forward_quant(
        self,
        quantized_x: torch.Tensor,
        ascales: torch.Tensor,
        lora_act: torch.Tensor,
        output: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if output is None:
            output = torch.empty(
                quantized_x.shape[0], self.out_features, dtype=self.proj_up.dtype, device=quantized_x.device
            )

        svdq_gemm_w4a4_cuda(
            act=quantized_x,
            wgt=self.qweight,
            out=output,
            ascales=ascales,
            wscales=self.wscales,
            lora_act_in=lora_act,
            lora_up=self.proj_up,
            bias=self.bias,
            fp4=self.precision == "nvfp4",
            alpha=self.wtscale,
            wcscales=self.wcscales,
            act_unsigned=self.act_unsigned,
        )
        return output

    def __repr__(self):
        return f"SVDQW4A4Linear(in_features={self.in_features}, out_features={self.out_features}, rank={self.rank}, precision={self.precision}, act_unsigned={self.act_unsigned})"


class AWQW4A16Linear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        group_size: int = 64,
        torch_dtype: torch.dtype = torch.bfloat16,
        device: str | torch.device = "cuda",
    ):
        super(AWQW4A16Linear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.group_size = group_size

        self.qweight = nn.Parameter(
            torch.empty(out_features // 4, in_features // 2, dtype=torch.int32, device=device), requires_grad=False
        )
        self.bias = (
            nn.Parameter(torch.empty(out_features, dtype=torch_dtype, device=device), requires_grad=True)
            if bias
            else None
        )
        self.wscales = nn.Parameter(
            torch.empty(in_features // self.group_size, out_features, dtype=torch_dtype, device=device),
            requires_grad=False,
        )
        self.wzeros = nn.Parameter(
            torch.empty(in_features // self.group_size, out_features, dtype=torch_dtype, device=device),
            requires_grad=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = awq_gemv_w4a16_cuda(
            in_feats=x,
            kernel=self.qweight,
            scaling_factors=self.wscales,
            zeros=self.wzeros,
            m=x.shape[0],
            n=self.out_features,
            k=self.in_features,
            group_size=self.group_size,
        )
        if self.bias is not None:
            view_shape = [1] * (output.ndim - 1) + [-1]
            output.add_(self.bias.view(view_shape))
        return output

    @classmethod
    def from_linear(
        cls,
        linear: nn.Linear,
        group_size: int = 64,
        torch_dtype: torch.dtype = torch.bfloat16,
        device: str = "cpu",
        **kwargs,
    ):
        return cls(
            in_features=linear.in_features,
            out_features=linear.out_features,
            bias=linear.bias is not None,
            group_size=group_size,
            torch_dtype=torch_dtype,
            device=device,
        )

    def __repr__(self):
        return f"AWQW4A16Linear(in_features={self.in_features}, out_features={self.out_features}, group_size={self.group_size})"
