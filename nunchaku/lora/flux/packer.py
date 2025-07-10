"""
Weight packing utilities for Nunchaku models and LoRAs.

This module provides classes and functions for packing and unpacking weight tensors
for efficient GPU computation using Matrix Multiply and Accumulate (MMA) operations.

Classes
-------
- MmaWeightPackerBase
- NunchakuWeightPacker
"""

# Copy the packer from https://github.com/mit-han-lab/deepcompressor/
import torch

from ...utils import ceil_divide
from .utils import pad


class MmaWeightPackerBase:
    """
    Base class for MMA weight packing.

    Provides methods and attributes for packing weight tensors for efficient
    GPU computation using MMA operations.

    Parameters
    ----------
    bits : int
        Quantization bits (1, 4, 8, 16, or 32).
    warp_n : int
        Warp size in the n dimension.
    comp_n : int, optional
        Computation tile size in n (default: 16).
    comp_k : int, optional
        Computation tile size in k (default: 256 // bits).

    Attributes
    ----------
    bits : int
        Quantization bits.
    comp_n : int
        Tile size in n for MMA computation.
    comp_k : int
        Tile size in k for MMA computation.
    insn_n : int
        Tile size in n for MMA instruction (always 8).
    insn_k : int
        Tile size in k for MMA instruction.
    num_lanes : int
        Number of lanes in a warp (always 32).
    num_k_lanes : int
        Number of lanes in k (always 4).
    num_n_lanes : int
        Number of lanes in n (always 8).
    warp_n : int
        Warp size in n.
    reg_k : int
        Elements in a register in k.
    reg_n : int
        Elements in a register in n (always 1).
    k_pack_size : int
        Elements in a pack in k.
    n_pack_size : int
        Elements in a pack in n.
    pack_size : int
        Total elements in a pack.
    mem_k : int
        Tile size in k for memory access.
    mem_n : int
        Tile size in n for memory access.
    num_k_packs : int
        Packs in k for memory access.
    num_n_packs : int
        Packs in n for memory access.

    Raises
    ------
    AssertionError
        If parameters are not valid.
    """

    def __init__(self, bits: int, warp_n: int, comp_n: int = None, comp_k: int = None):
        self.bits = bits
        assert self.bits in (1, 4, 8, 16, 32), "weight bits should be 1, 4, 8, 16, or 32."

        # region compute tile size
        self.comp_n = comp_n if comp_n is not None else 16
        self.comp_k = comp_k if comp_k is not None else 256 // self.bits
        self.insn_n = 8
        self.insn_k = self.comp_k
        assert self.insn_k * self.bits in (128, 256), f"insn_k ({self.insn_k}) * bits ({self.bits}) should be 128 or 256."
        assert self.comp_n % self.insn_n == 0, f"comp_n ({self.comp_n}) should be divisible by insn_n ({self.insn_n})."
        self.num_lanes = 32
        self.num_k_lanes = 4
        self.num_n_lanes = 8
        assert (
            warp_n >= self.comp_n and warp_n % self.comp_n == 0
        ), f"warp_n ({warp_n}) should be divisible by comp_n({self.comp_n})."
        self.warp_n = warp_n
        # endregion
        # region memory
        self.reg_k = 32 // self.bits
        self.reg_n = 1
        self.k_pack_size = self.comp_k // (self.num_k_lanes * self.reg_k)
        self.n_pack_size = self.comp_n // (self.num_n_lanes * self.reg_n)
        self.pack_size = self.k_pack_size * self.n_pack_size
        assert 1 <= self.pack_size <= 4, "pack size should be less than or equal to 4."
        assert self.k_pack_size * self.num_k_lanes * self.reg_k == self.comp_k
        assert self.n_pack_size * self.num_n_lanes * self.reg_n == self.comp_n
        self.mem_k = self.comp_k
        self.mem_n = warp_n
        self.num_k_packs = self.mem_k // (self.k_pack_size * self.num_k_lanes * self.reg_k)
        self.num_n_packs = self.mem_n // (self.n_pack_size * self.num_n_lanes * self.reg_n)
        # endregion

    def get_view_shape(self, n: int, k: int) -> tuple[int, int, int, int, int, int, int, int, int, int]:
        """
        Compute the tensor view shape for MMA operations.

        Parameters
        ----------
        n : int
            Output channel size (must be divisible by mem_n).
        k : int
            Input channel size (must be divisible by mem_k).

        Returns
        -------
        tuple of int
            (n_tiles, num_n_packs, n_pack_size, num_n_lanes, reg_n,
             k_tiles, num_k_packs, k_pack_size, num_k_lanes, reg_k)
        """
        assert n % self.mem_n == 0, "output channel size should be divisible by mem_n."
        assert k % self.mem_k == 0, "input channel size should be divisible by mem_k."
        return (
            n // self.mem_n,
            self.num_n_packs,
            self.n_pack_size,
            self.num_n_lanes,
            self.reg_n,
            k // self.mem_k,
            self.num_k_packs,
            self.k_pack_size,
            self.num_k_lanes,
            self.reg_k,
        )


class NunchakuWeightPacker(MmaWeightPackerBase):
    """
    Nunchaku weight packer for quantized neural networks.

    Extends :class:`MmaWeightPackerBase` to provide Nunchaku-specific packing
    for quantized weights, scales, and low-rank weights.

    Parameters
    ----------
    bits : int
        Quantization bits (1, 4, 8, 16, or 32).
    warp_n : int, optional
        Warp size in n (default: 128).

    Attributes
    ----------
    num_k_unrolls : int
        Number of k unrolls (always 2).

    Methods
    -------
    pack_weight(weight)
        Pack quantized weights.
    pack_scale(scale, group_size)
        Pack scale tensors.
    pack_micro_scale(scale, group_size)
        Pack micro-scale tensors.
    pack_lowrank_weight(weight, down)
        Pack low-rank weights.
    unpack_lowrank_weight(weight, down)
        Unpack low-rank weights.
    check_if_micro_scale(group_size)
        Check if micro-scale packing is used.
    pad_weight(weight)
        Pad weight tensor.
    pad_scale(scale, group_size, fill_value)
        Pad scale tensor.
    pad_lowrank_weight(weight, down)
        Pad low-rank weight tensor.
    """

    def __init__(self, bits: int, warp_n: int = 128):
        super().__init__(bits=bits, warp_n=warp_n)
        self.num_k_unrolls = 2

    def pack_weight(self, weight: torch.Tensor) -> torch.Tensor:
        assert weight.dtype == torch.int32, f"quantized weight should be torch.int32, but got {weight.dtype}."
        n, k = weight.shape
        assert n % self.mem_n == 0, f"output channel size ({n}) should be divisible by mem_n ({self.mem_n})."
        assert k % (self.mem_k * self.num_k_unrolls) == 0, (
            f"input channel size ({k}) should be divisible by "
            f"mem_k ({self.mem_k}) * num_k_unrolls ({self.num_k_unrolls})."
        )
        n_tiles, k_tiles = n // self.mem_n, k // self.mem_k
        weight = weight.reshape(
            n_tiles,
            self.num_n_packs,
            self.n_pack_size,
            self.num_n_lanes,
            self.reg_n,
            k_tiles,
            self.num_k_packs,
            self.k_pack_size,
            self.num_k_lanes,
            self.reg_k,
        )
        weight = weight.permute(0, 5, 6, 1, 3, 8, 2, 7, 4, 9).contiguous()
        assert weight.shape[4:-2] == (8, 4, 2, 2)
        if self.bits == 4:
            weight = weight.bitwise_and_(0xF)
            shift = torch.arange(0, 32, 4, dtype=torch.int32, device=weight.device)
            weight = weight.bitwise_left_shift_(shift)
            weight = weight.sum(dim=-1, dtype=torch.int32)
        elif self.bits == 8:
            weight = weight.bitwise_and_(0xFF)
            shift = torch.arange(0, 32, 8, dtype=torch.int32, device=weight.device)
            weight = weight.bitwise_left_shift_(shift)
            weight = weight.sum(dim=-1, dtype=torch.int32)
        else:
            raise NotImplementedError(f"weight bits {self.bits} is not supported.")
        return weight.view(dtype=torch.int8).view(n, -1)

    def pack_scale(self, scale: torch.Tensor, group_size: int) -> torch.Tensor:
        if self.check_if_micro_scale(group_size=group_size):
            return self.pack_micro_scale(scale, group_size=group_size)
        assert scale.dtype in (torch.float16, torch.bfloat16), "currently nunchaku only supports fp16 and bf16."
        n = scale.shape[0]
        s_pack_size = min(max(self.warp_n // self.num_lanes, 2), 8)
        num_s_lanes = min(self.num_lanes, self.warp_n // s_pack_size)
        num_s_packs = self.warp_n // (s_pack_size * num_s_lanes)
        warp_s = num_s_packs * num_s_lanes * s_pack_size
        assert warp_s == self.warp_n, "warp_n for scales should be equal to warp_n for weights."
        scale = scale.reshape(n // warp_s, num_s_packs, num_s_lanes // 4, s_pack_size // 2, 4, 2, -1)
        scale = scale.permute(0, 6, 1, 2, 4, 3, 5).contiguous()
        return scale.view(-1) if group_size == -1 else scale.view(-1, n)

    def pack_micro_scale(self, scale: torch.Tensor, group_size: int) -> torch.Tensor:
        assert scale.dtype in (torch.float16, torch.bfloat16), "currently nunchaku only supports fp16 and bf16."
        assert scale.max() <= 448, "scale should be less than 448."
        assert scale.min() >= -448, "scale should be greater than -448."
        assert group_size == 16, "currently only support group size 16."
        assert self.insn_k == 64, "insn_k should be 64."
        scale = scale.to(dtype=torch.float8_e4m3fn)
        n = scale.shape[0]
        assert self.warp_n >= 32, "currently only support warp_n >= 32."
        s_pack_size = min(max(self.warp_n // self.num_lanes, 1), 4)
        num_s_lanes = 4 * 8
        num_s_packs = ceil_divide(self.warp_n, s_pack_size * num_s_lanes)
        warp_s = num_s_packs * num_s_lanes * s_pack_size
        assert warp_s == self.warp_n, "warp_n for scales should be equal to warp_n for weights."
        scale = scale.view(n // warp_s, num_s_packs, s_pack_size, 4, 8, -1, self.insn_k // group_size)
        scale = scale.permute(0, 5, 1, 4, 3, 2, 6).contiguous()
        return scale.view(-1, n)

    def pack_lowrank_weight(self, weight: torch.Tensor, down: bool) -> torch.Tensor:
        """
        Pack low-rank weight.

        Parameters
        ----------
        weight : torch.Tensor
            Low-rank weight tensor.
        down : bool
            If True, for down projection in low-rank branch.

        Returns
        -------
        torch.Tensor
            Packed low-rank weight.
        """
        assert weight.dtype in (torch.float16, torch.bfloat16), f"Unsupported weight dtype {weight.dtype}."
        reg_n, reg_k = 1, 2
        pack_n = self.n_pack_size * self.num_n_lanes * reg_n
        pack_k = self.k_pack_size * self.num_k_lanes * reg_k
        weight = pad(weight, divisor=(pack_n, pack_k), dim=(0, 1))
        if down:
            r, c = weight.shape
            r_packs, c_packs = r // pack_n, c // pack_k
            weight = weight.view(r_packs, pack_n, c_packs, pack_k).permute(2, 0, 1, 3)
        else:
            c, r = weight.shape
            c_packs, r_packs = c // pack_n, r // pack_k
            weight = weight.view(c_packs, pack_n, r_packs, pack_k).permute(0, 2, 1, 3)
        weight = weight.reshape(
            c_packs, r_packs, self.n_pack_size, self.num_n_lanes, reg_n, self.k_pack_size, self.num_k_lanes, reg_k
        )
        weight = weight.permute(0, 1, 3, 6, 2, 5, 4, 7).contiguous()
        return weight.view(c, r)

    def unpack_lowrank_weight(self, weight: torch.Tensor, down: bool) -> torch.Tensor:
        """
        Unpack low-rank weight.

        Parameters
        ----------
        weight : torch.Tensor
            Packed low-rank weight tensor.
        down : bool
            If True, for down projection in low-rank branch.

        Returns
        -------
        torch.Tensor
            Unpacked low-rank weight.
        """
        c, r = weight.shape
        assert weight.dtype in (torch.float16, torch.bfloat16), f"Unsupported weight dtype {weight.dtype}."
        reg_n, reg_k = 1, 2
        pack_n = self.n_pack_size * self.num_n_lanes * reg_n
        pack_k = self.k_pack_size * self.num_k_lanes * reg_k
        if down:
            r_packs, c_packs = r // pack_n, c // pack_k
        else:
            c_packs, r_packs = c // pack_n, r // pack_k
        weight = weight.view(
            c_packs, r_packs, self.num_n_lanes, self.num_k_lanes, self.n_pack_size, self.k_pack_size, reg_n, reg_k
        )
        weight = weight.permute(0, 1, 4, 2, 6, 5, 3, 7).contiguous()
        weight = weight.view(c_packs, r_packs, pack_n, pack_k)
        if down:
            weight = weight.permute(1, 2, 0, 3).contiguous().view(r, c)
        else:
            weight = weight.permute(0, 2, 1, 3).contiguous().view(c, r)
        return weight

    def check_if_micro_scale(self, group_size: int) -> bool:
        """
        Check if micro-scale packing is used.

        Parameters
        ----------
        group_size : int

        Returns
        -------
        bool
        """
        return self.insn_k == group_size * 4

    def pad_weight(self, weight: torch.Tensor) -> torch.Tensor:
        """
        Pad weight tensor to required dimensions.

        Parameters
        ----------
        weight : torch.Tensor

        Returns
        -------
        torch.Tensor
        """
        assert weight.ndim == 2, "weight tensor should be 2D."
        return pad(weight, divisor=(self.mem_n, self.mem_k * self.num_k_unrolls), dim=(0, 1))

    def pad_scale(self, scale: torch.Tensor, group_size: int, fill_value: float = 0) -> torch.Tensor:
        """
        Pad scale tensor to required dimensions.

        Parameters
        ----------
        scale : torch.Tensor
        group_size : int
        fill_value : float, optional

        Returns
        -------
        torch.Tensor
        """
        if group_size > 0 and scale.numel() > scale.shape[0]:
            scale = scale.view(scale.shape[0], 1, -1, 1)
            if self.check_if_micro_scale(group_size=group_size):
                scale = pad(scale, divisor=(self.warp_n, self.insn_k // group_size), dim=(0, 2), fill_value=fill_value)
            else:
                scale = pad(scale, divisor=(self.warp_n, self.num_k_unrolls), dim=(0, 2), fill_value=fill_value)
        else:
            scale = pad(scale, divisor=self.warp_n, dim=0, fill_value=fill_value)
        return scale

    def pad_lowrank_weight(self, weight: torch.Tensor, down: bool) -> torch.Tensor:
        """
        Pad low-rank weight tensor to required dimensions.

        Parameters
        ----------
        weight : torch.Tensor
        down : bool

        Returns
        -------
        torch.Tensor
        """
        assert weight.ndim == 2, "weight tensor should be 2D."
        return pad(weight, divisor=self.warp_n, dim=1 if down else 0)
