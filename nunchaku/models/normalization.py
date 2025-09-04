"""
Quantized normalization layers for efficient inference.
"""

from typing import Optional, Tuple

import torch
from diffusers.models.normalization import AdaLayerNormZero, AdaLayerNormZeroSingle

from .linear import AWQW4A16Linear


class NunchakuAdaLayerNormZero(AdaLayerNormZero):
    """
    Nunchaku quantized AdaLayerNormZero for diffusion models.

    Replaces the linear projection with AWQW4A16Linear for quantized inference.

    Parameters
    ----------
    other : AdaLayerNormZero
        Source AdaLayerNormZero instance to copy weights and structure from.
    scale_shift : float, optional
        Value to add to scale parameters. Default is 1.0.
        Nunchaku may have already fused the scale_shift into the linear weights, so you may want to set it to 0.

    Notes
    -----
    - B: batch size
    - D: hidden dimension
    """

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
        """
        Forward pass for quantized AdaLayerNormZero.

        Parameters
        ----------
        x : torch.Tensor, shape (B, D), dtype float32/float16
            Input tensor.
        timestep : Optional[torch.Tensor], shape (B,) or (1,), optional
            Timestep embedding input.
        class_labels : Optional[torch.LongTensor], shape (B,) or (1,), optional
            Class label input.
        hidden_dtype : Optional[torch.dtype], optional
            Dtype for embedding computation.
        emb : Optional[torch.Tensor], shape (B, E), optional
            Precomputed embedding. If None, computed from timestep and class_labels.

        Returns
        -------
        norm_x_scaled : torch.Tensor, shape (B, D)
            Normalized and scaled input.
        gate_msa : torch.Tensor, shape (B, D)
            Gate for MSA branch.
        shift_mlp : torch.Tensor, shape (B, D)
            Shift for MLP branch.
        scale_mlp : torch.Tensor, shape (B, D)
            Scale for MLP branch.
        gate_mlp : torch.Tensor, shape (B, D)
            Gate for MLP branch.

        Notes
        -----
        - B: batch size
        - D: hidden dimension
        """
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
        return norm_x_scaled, gate_msa, shift_mlp, scale_mlp, gate_mlp


class NunchakuAdaLayerNormZeroSingle(AdaLayerNormZeroSingle):
    """
    Nunchaku quantized AdaLayerNormZeroSingle.

    Uses AWQW4A16Linear for quantized embedding projection. Suitable for single-branch normalization.

    Parameters
    ----------
    other : AdaLayerNormZeroSingle
        Source AdaLayerNormZeroSingle instance to copy weights and structure from.
    scale_shift : float, optional
        Value to add to scale parameters. Default is 1.0.
        Nunchaku may have already fused the scale_shift into the linear weights, so you may want to set it to 0.

    Notes
    -----
    - B: batch size
    - D: hidden dimension
    """

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
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for quantized AdaLayerNormZeroSingle.

        Parameters
        ----------
        x : torch.Tensor, shape (B, D), dtype float32/float16
            Input tensor.
        emb : Optional[torch.Tensor], shape (B, E), optional
            Embedding tensor.

        Returns
        -------
        norm_x_scaled : torch.Tensor, shape (B, D)
            Normalized and scaled input.
        gate_msa : torch.Tensor, shape (B, D)
            Gate for MSA branch.

        Notes
        -----
        - B: batch size
        - D: hidden dimension
        """
        emb = self.linear(self.silu(emb))

        # The weight layout has changed; use split_mod rather than chunk to separate the embedding.
        emb = emb.view(emb.shape[0], -1, 3).permute(2, 0, 1)
        shift_msa, scale_msa, gate_msa = emb

        if self.scale_shift != 0:
            scale_msa.add_(self.scale_shift)

        norm_x = self.norm(x)
        norm_x_scaled = norm_x * scale_msa[:, None] + shift_msa[:, None]
        return norm_x_scaled, gate_msa
