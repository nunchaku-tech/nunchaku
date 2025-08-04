import torch
from torch import nn

from ..ops.gemm import svdq_gemm_w4a4_cuda
from ..ops.gemv import awq_gemv_w4a16_cuda
from ..ops.quantize import svdq_w4a4_act_fuse_lora_cuda


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

        self.ascales = nn.Parameter(
            torch.empty(
                in_features // self.group_size,
                out_features,
                dtype=torch_dtype if precision == "int4" else torch.float8_e4m3fn,
                device=device,
            ),
            requires_grad=False,
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
        if precision == "nvfp4":
            self.wtscale = nn.Parameter(torch.empty(1, dtype=torch_dtype, device=device), requires_grad=False)

        self.wcscales = None

    @classmethod
    def from_linear(cls, linear: nn.Linear, **kwargs):
        return cls(
            in_features=kwargs.get("in_features", linear.in_features),
            out_features=linear.out_features,
            bias=linear.bias is not None,
            torch_dtype=linear.weight.dtype,
            device=linear.weight.device,
            **kwargs,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # quantize the input run the down projection
        quantized_x, lora_act_out = svdq_w4a4_act_fuse_lora_cuda(
            x,
            oscales=self.ascales,
            lora_down=self.proj_down_weight,
            smooth=self.smooth_factor,
            fp4=self.precision == "nvfp4",
        )

        output = svdq_gemm_w4a4_cuda(
            act=quantized_x,
            wgt=self.qweight,
            ascales=self.ascales,
            wscales=self.wscales,
            lora_act_in=lora_act_out,
            lora_up=self.proj_up_weight,
            lora_down=self.proj_down_weight,
            lora_act_out=lora_act_out,
            norm_q=None,
            norm_k=None,
            rotary_emb=None,
            bias=None,
            act_unsigned=False,  # TODO: check this.
            fp4=self.precision == "nvfp4",
        )
        return output

    def __repr__(self):
        return f"SVDQW4A4Linear(in_features={self.in_features}, out_features={self.out_features}, rank={self.rank}, precision={self.precision})"


class AWQW4A16Linear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        group_size: int = 128,
        torch_dtype: torch.dtype = torch.bfloat16,
        device: str | torch.device = "cuda",
    ):
        super(AWQW4A16Linear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.group_size = 128

        self.qweight = nn.Parameter(
            torch.empty(out_features, in_features // 2, dtype=torch.int8, device=device), requires_grad=False
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
        return awq_gemv_w4a16_cuda(
            act=x,
            wgt=self.qweight,
            ascales=self.ascales,
            wscales=self.wscales,
            m=x.shape[0],
            n=self.out_features,
            k=self.in_features,
            group_size=self.group_size,
        )

    @classmethod
    def from_linear(
        cls,
        linear: nn.Linear,
        group_size: int = 128,
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
