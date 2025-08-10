from typing import Optional, Tuple

import torch
from diffusers.models.normalization import AdaLayerNormZero, AdaLayerNormZeroSingle

from .linear import AWQW4A16Linear


class NunchakuAdaLayerNormZero(AdaLayerNormZero):
    def __init__(self, other: AdaLayerNormZero, scale_shift: float = 1.0):
        super(AdaLayerNormZero, self).__init__()

        self.scale_shift = scale_shift

        self.emb = other.emb
        self.silu = other.silu
        self.linear = AWQW4A16Linear.from_linear(other.linear)
        self.norm = other.norm

    def forward(
        self,
        x: torch.Tensor,
        timestep: Optional[torch.Tensor] = None,
        class_labels: Optional[torch.LongTensor] = None,
        hidden_dtype: Optional[torch.dtype] = None,
        emb: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.emb is not None:
            emb = self.emb(timestep, class_labels, hidden_dtype=hidden_dtype)
        emb = self.linear(self.silu(emb))

        # The weight layout has changed; use split_mod rather than chunk to separate the embedding.
        emb = emb.view(emb.shape[0], -1, 6).permute(2, 0, 1)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = emb

        norm_x = self.norm(x)

        if self.scale_shift != 0:
            scale_msa.add_(self.scale_shift)
            scale_mlp.add_(self.scale_shift)

        norm_x_scaled = norm_x * scale_msa[:, None] + shift_msa[:, None]
        # torch.addcmul(shift_msa[:, None], norm_x, scale_msa[:, None], value=1, out=norm_x)
        return norm_x_scaled, gate_msa, shift_mlp, scale_mlp, gate_mlp


class NunchakuAdaLayerNormZeroSingle(AdaLayerNormZeroSingle):

    def __init__(self, other: AdaLayerNormZeroSingle, scale_shift: float = 1.0):
        super(AdaLayerNormZeroSingle, self).__init__()

        self.scale_shift = scale_shift
        self.silu = other.silu
        self.linear = AWQW4A16Linear.from_linear(other.linear)
        self.norm = other.norm

    def forward(
        self,
        x: torch.Tensor,
        emb: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        emb = self.linear(self.silu(emb))

        # The weight layout has changed; use split_mod rather than chunk to separate the embedding.
        emb = emb.view(emb.shape[0], -1, 3).permute(2, 0, 1)
        shift_msa, scale_msa, gate_msa = emb

        if self.scale_shift != 0:
            scale_msa.add_(self.scale_shift)

        norm_x = self.norm(x)
        norm_x_scaled = norm_x * scale_msa[:, None] + shift_msa[:, None]
        return norm_x_scaled, gate_msa
