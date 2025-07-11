"""
This module implements the encoders for PuLID.

.. note::
    This module is adapted from https://github.com/ToTheBeginning/PuLID.
"""

import math

import torch
from torch import nn


def FeedForward(dim, mult=4):
    """
    Feed-forward network (FFN) block with LayerNorm and GELU activation.

    Parameters
    ----------
    dim : int
        Input and output feature dimension.
    mult : int, optional
        Expansion multiplier for the hidden dimension (default: 4).

    Returns
    -------
    nn.Sequential
        A sequential FFN block: LayerNorm -> Linear -> GELU -> Linear.
    """
    inner_dim = int(dim * mult)
    return nn.Sequential(
        nn.LayerNorm(dim),
        nn.Linear(dim, inner_dim, bias=False),
        nn.GELU(),
        nn.Linear(inner_dim, dim, bias=False),
    )


def reshape_tensor(x, heads):
    """
    Reshape a tensor for multi-head attention.

    Parameters
    ----------
    x : torch.Tensor
        Input tensor of shape (batch_size, seq_len, width).
    heads : int
        Number of attention heads.

    Returns
    -------
    torch.Tensor
        Reshaped tensor of shape (batch_size, heads, seq_len, dim_per_head).
    """
    bs, length, width = x.shape
    x = x.view(bs, length, heads, -1)
    x = x.transpose(1, 2)
    x = x.reshape(bs, heads, length, -1)
    return x


class PerceiverAttentionCA(nn.Module):
    """
    Perceiver-style cross-attention module.

    Parameters
    ----------
    dim : int, optional
        Input feature dimension for queries (default: 3072).
    dim_head : int, optional
        Dimension per attention head (default: 128).
    heads : int, optional
        Number of attention heads (default: 16).
    kv_dim : int, optional
        Input feature dimension for keys/values (default: 2048).
    """

    def __init__(self, *, dim=3072, dim_head=128, heads=16, kv_dim=2048):
        super().__init__()
        self.scale = dim_head**-0.5
        self.dim_head = dim_head
        self.heads = heads
        inner_dim = dim_head * heads

        self.norm1 = nn.LayerNorm(dim if kv_dim is None else kv_dim)
        self.norm2 = nn.LayerNorm(dim)

        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim if kv_dim is None else kv_dim, inner_dim * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)

    def forward(self, x, latents):
        """
        Forward pass for cross-attention.

        Parameters
        ----------
        x : torch.Tensor
            Image features of shape (batch_size, n1, D).
        latents : torch.Tensor
            Latent features of shape (batch_size, n2, D).

        Returns
        -------
        torch.Tensor
            Output tensor of shape (batch_size, n2, D).
        """
        x = self.norm1(x)
        latents = self.norm2(latents)

        b, seq_len, _ = latents.shape

        q = self.to_q(latents)
        k, v = self.to_kv(x).chunk(2, dim=-1)

        q = reshape_tensor(q, self.heads)
        k = reshape_tensor(k, self.heads)
        v = reshape_tensor(v, self.heads)

        # attention
        scale = 1 / math.sqrt(math.sqrt(self.dim_head))
        weight = (q * scale) @ (k * scale).transpose(-2, -1)
        weight = torch.softmax(weight.float(), dim=-1).type(weight.dtype)
        out = weight @ v

        out = out.permute(0, 2, 1, 3).reshape(b, seq_len, -1)

        return self.to_out(out)


class PerceiverAttention(nn.Module):
    """
    Perceiver-style self-attention module with optional cross-attention.

    Parameters
    ----------
    dim : int
        Input feature dimension for queries.
    dim_head : int, optional
        Dimension per attention head (default: 64).
    heads : int, optional
        Number of attention heads (default: 8).
    kv_dim : int, optional
        Input feature dimension for keys/values (default: None).
    """

    def __init__(self, *, dim, dim_head=64, heads=8, kv_dim=None):
        super().__init__()
        self.scale = dim_head**-0.5
        self.dim_head = dim_head
        self.heads = heads
        inner_dim = dim_head * heads

        self.norm1 = nn.LayerNorm(dim if kv_dim is None else kv_dim)
        self.norm2 = nn.LayerNorm(dim)

        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim if kv_dim is None else kv_dim, inner_dim * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)

    def forward(self, x, latents):
        """
        Forward pass for (cross-)attention.

        Parameters
        ----------
        x : torch.Tensor
            Image features of shape (batch_size, n1, D).
        latents : torch.Tensor
            Latent features of shape (batch_size, n2, D).

        Returns
        -------
        torch.Tensor
            Output tensor of shape (batch_size, n2, D).
        """
        x = self.norm1(x)
        latents = self.norm2(latents)

        b, seq_len, _ = latents.shape

        q = self.to_q(latents)
        kv_input = torch.cat((x, latents), dim=-2)
        k, v = self.to_kv(kv_input).chunk(2, dim=-1)

        q = reshape_tensor(q, self.heads)
        k = reshape_tensor(k, self.heads)
        v = reshape_tensor(v, self.heads)

        # attention
        scale = 1 / math.sqrt(math.sqrt(self.dim_head))
        weight = (q * scale) @ (k * scale).transpose(-2, -1)
        weight = torch.softmax(weight.float(), dim=-1).type(weight.dtype)
        out = weight @ v

        out = out.permute(0, 2, 1, 3).reshape(b, seq_len, -1)

        return self.to_out(out)


class IDFormer(nn.Module):
    """
    IDFormer: Perceiver-style transformer encoder for identity and vision features.

    This module fuses identity embeddings (e.g., from ArcFace) and multi-scale ViT features
    using a stack of PerceiverAttention and FeedForward layers.

    The architecture:
      - Concatenates ID embedding tokens and query tokens as latents.
      - Latents attend to each other and interact with ViT features via cross-attention.
      - Multi-scale ViT features are inserted in order, each scale processed by a block of layers.

    Parameters
    ----------
    dim : int, optional
        Embedding dimension for all tokens (default: 1024).
    depth : int, optional
        Total number of transformer layers (must be divisible by 5, default: 10).
    dim_head : int, optional
        Dimension per attention head (default: 64).
    heads : int, optional
        Number of attention heads (default: 16).
    num_id_token : int, optional
        Number of ID embedding tokens (default: 5).
    num_queries : int, optional
        Number of query tokens (default: 32).
    output_dim : int, optional
        Output projection dimension (default: 2048).
    ff_mult : int, optional
        Feed-forward expansion multiplier (default: 4).
    """

    def __init__(
        self,
        dim=1024,
        depth=10,
        dim_head=64,
        heads=16,
        num_id_token=5,
        num_queries=32,
        output_dim=2048,
        ff_mult=4,
    ):
        super().__init__()

        self.num_id_token = num_id_token
        self.dim = dim
        self.num_queries = num_queries
        assert depth % 5 == 0
        self.depth = depth // 5
        scale = dim**-0.5

        self.latents = nn.Parameter(torch.randn(1, num_queries, dim) * scale)
        self.proj_out = nn.Parameter(scale * torch.randn(dim, output_dim))

        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        PerceiverAttention(dim=dim, dim_head=dim_head, heads=heads),
                        FeedForward(dim=dim, mult=ff_mult),
                    ]
                )
            )

        for i in range(5):
            setattr(
                self,
                f"mapping_{i}",
                nn.Sequential(
                    nn.Linear(1024, 1024),
                    nn.LayerNorm(1024),
                    nn.LeakyReLU(),
                    nn.Linear(1024, 1024),
                    nn.LayerNorm(1024),
                    nn.LeakyReLU(),
                    nn.Linear(1024, dim),
                ),
            )

        self.id_embedding_mapping = nn.Sequential(
            nn.Linear(1280, 1024),
            nn.LayerNorm(1024),
            nn.LeakyReLU(),
            nn.Linear(1024, 1024),
            nn.LayerNorm(1024),
            nn.LeakyReLU(),
            nn.Linear(1024, dim * num_id_token),
        )

    def forward(self, x, y):
        """
        Forward pass for IDFormer.

        Parameters
        ----------
        x : torch.Tensor
            ID embedding tensor of shape (batch_size, 1280) or (batch_size, N, 1280).
        y : list of torch.Tensor
            List of 5 ViT feature tensors, each of shape (batch_size, feature_dim).

        Returns
        -------
        torch.Tensor
            Output tensor of shape (batch_size, num_queries, output_dim).
        """
        latents = self.latents.repeat(x.size(0), 1, 1)

        num_duotu = x.shape[1] if x.ndim == 3 else 1

        x = self.id_embedding_mapping(x)
        x = x.reshape(-1, self.num_id_token * num_duotu, self.dim)

        latents = torch.cat((latents, x), dim=1)

        for i in range(5):
            vit_feature = getattr(self, f"mapping_{i}")(y[i])
            ctx_feature = torch.cat((x, vit_feature), dim=1)
            for attn, ff in self.layers[i * self.depth : (i + 1) * self.depth]:
                latents = attn(ctx_feature, latents) + latents
                latents = ff(latents) + latents

        latents = latents[:, : self.num_queries]
        latents = latents @ self.proj_out
        return latents
