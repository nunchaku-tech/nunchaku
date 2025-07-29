import torch
from torch import nn


class SVDQW4A4Linear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 32,
        bias: bool = True,
        precision: str = "int4",
        torch_dtype: torch.dtype = torch.bfloat16,
        device: str = "cuda",
    ):
        super(SVDQW4A4Linear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.precision = precision
        self.torch_dtype = torch_dtype
        self.group_size = None

        if precision == "nvfp4":
            self.group_size = 16
        elif precision == "int4":
            self.group_size = 64
        else:
            raise ValueError(f"Invalid precision: {precision}")

        self.wgt = nn.Parameter(torch.empty(out_features, in_features // 2, dtype=torch.int8, device=device))
        self.ascales = nn.Parameter(
            torch.empty(
                in_features // self.group_size,
                out_features,
                dtype=torch_dtype if precision == "int4" else torch.float8_e4m3,
                device=device,
            )
        )
        self.wscales = nn.Parameter(
            torch.empty(
                in_features // self.group_size,
                out_features,
                dtype=torch_dtype if precision == "int4" else torch.float8_e4m3,
                device=device,
            )
        )

        self.down_weight = nn.Parameter(torch.empty(in_features, rank, dtype=torch_dtype, device=device))
        self.up_weight = nn.Parameter(torch.empty(out_features, rank, dtype=torch_dtype, device=device))

        self.alpha = None
        self.wcscales = None
