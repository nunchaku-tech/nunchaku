"""
Python wrappers for quantization operations.
"""

import torch

from ._C import ops


def svdq_w4a4_act_fuse_lora(
    input: torch.Tensor,
    output: torch.Tensor | None = None,
    oscales: torch.Tensor | None = None,
    lora_down: torch.Tensor | None = None,
    lora_act_out: torch.Tensor | None = None,
    smooth: torch.Tensor | None = None,
    fuse_glu: bool = False,
    fp4: bool = False,
) -> torch.Tensor:
    batch_size, channels = input.shape
    if output is None:
        output = torch.empty(batch_size, channels, dtype=torch.int8, device=input.device)
    if lora_act_out is None:
        lora_act_out = torch.empty(batch_size, channels, dtype=torch.int8, device=input.device)

    ops.svdq_w4a4_act_fuse_lora(input, output, oscales, lora_down, lora_act_out, smooth, fuse_glu, fp4)

    return output
