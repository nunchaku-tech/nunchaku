import diffusers
import torch
from packaging.version import Version
from torch import nn


def rope(pos: torch.Tensor, dim: int, theta: int) -> torch.Tensor:
    """
    Rotary positional embedding function.
    Copied from https://github.com/huggingface/diffusers/blob/c9ff360966327ace3faad3807dc871a4e5447501/src/diffusers/models/transformers/transformer_flux.py#L38

    Parameters
    ----------
    pos : torch.Tensor
        Position tensor of shape (..., n).
    dim : int
        Embedding dimension (must be even).
    theta : int
        Rotary base.

    Returns
    -------
    torch.Tensor
        Rotary embedding tensor.
    """
    assert dim % 2 == 0, "The dimension must be even."

    scale = torch.arange(0, dim, 2, dtype=torch.float64, device=pos.device) / dim
    omega = 1.0 / (theta**scale)

    batch_size, seq_length = pos.shape
    out = torch.einsum("...n,d->...nd", pos, omega)

    USE_SINCOS = True
    if USE_SINCOS:
        cos_out = torch.cos(out)
        sin_out = torch.sin(out)
        stacked_out = torch.stack([sin_out, cos_out], dim=-1)
        out = stacked_out.view(batch_size, -1, dim // 2, 1, 2)
    else:
        out = out.view(batch_size, -1, dim // 2, 1, 1)

    return out.float()


class NunchakuFluxPosEmbed(nn.Module):
    """
    Multi-dimensional rotary embedding module.
    Adapted from https://github.com/huggingface/diffusers/blob/c9ff360966327ace3faad3807dc871a4e5447501/src/diffusers/models/transformers/transformer_flux.py#L55

    Parameters
    ----------
    dim : int
        Embedding dimension.
    theta : int
        Rotary base.
    axes_dim : list[int]
        List of axis dimensions for each spatial axis.
    """

    def __init__(self, dim: int, theta: int, axes_dim: list[int]):
        super(NunchakuFluxPosEmbed, self).__init__()
        self.dim = dim
        self.theta = theta
        self.axes_dim = axes_dim

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        """
        Computes rotary embeddings for multi-dimensional positions.

        Parameters
        ----------
        ids : torch.Tensor
            Position indices tensor of shape (..., n_axes).

        Returns
        -------
        torch.Tensor
            Rotary embedding tensor.
        """
        if Version(diffusers.__version__) >= Version("0.31.0"):
            ids = ids[None, ...]
        n_axes = ids.shape[-1]
        emb = torch.cat([rope(ids[..., i], self.axes_dim[i], self.theta) for i in range(n_axes)], dim=-3)
        return emb.unsqueeze(1)


def pack_rotemb(rotemb: torch.Tensor) -> torch.Tensor:
    """
    Packs rotary embeddings for efficient computation.

    Parameters
    ----------
    rotemb : torch.Tensor
        Rotary embedding tensor of shape (B, M, D//2, 1, 2), dtype float32.

    Returns
    -------
    torch.Tensor
        Packed rotary embedding tensor of shape (B, M, D).
    """
    assert rotemb.dtype == torch.float32
    B = rotemb.shape[0]
    M = rotemb.shape[1]
    D = rotemb.shape[2] * 2
    assert rotemb.shape == (B, M, D // 2, 1, 2)
    assert M % 16 == 0
    assert D % 8 == 0
    rotemb = rotemb.reshape(B, M // 16, 16, D // 8, 8)
    rotemb = rotemb.permute(0, 1, 3, 2, 4)
    # 16*8 pack, FP32 accumulator (C) format
    # https://docs.nvidia.com/cuda/parallel-thread-execution/#mma-16816-c
    ##########################################|--M--|--D--|
    ##########################################|-3--4--5--6|
    ##########################################  :  :  :  :
    rotemb = rotemb.reshape(*rotemb.shape[0:3], 2, 8, 4, 2)
    rotemb = rotemb.permute(0, 1, 2, 4, 5, 3, 6)
    rotemb = rotemb.contiguous()
    rotemb = rotemb.view(B, M, D)
    return rotemb
