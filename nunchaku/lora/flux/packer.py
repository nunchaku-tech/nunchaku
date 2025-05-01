# Copy the packer from https://github.com/mit-han-lab/deepcompressor/
import torch

from ...utils import ceil_divide
from .utils import pad


class MmaWeightPackerBase:
    def __init__(self, bits: int, warp_n: int, comp_n: int = None, comp_k: int = None):
        self.bits = bits
        assert self.bits in (1, 4, 8, 16, 32), "weight bits should be 1, 4, 8, 16, or 32."

        # region compute tile size
        self.comp_n = comp_n if comp_n is not None else 16
        """smallest tile size in `n` dimension for MMA computation."""
        self.comp_k = comp_k if comp_k is not None else 256 // self.bits
        """smallest tile size in `k` dimension for MMA computation."""
        # the smallest MMA computation may contain several MMA instructions
        self.insn_n = 8  # mma instruction tile size in `n` dimension
        """tile size in `n` dimension for MMA instruction."""
        self.insn_k = self.comp_k
        """tile size in `k` dimension for MMA instruction."""
        assert self.insn_k * self.bits in (
            128,
            256,
        ), f"insn_k ({self.insn_k}) * bits ({self.bits}) should be 128 or 256."
        assert self.comp_n % self.insn_n == 0, f"comp_n ({self.comp_n}) should be divisible by insn_n ({self.insn_n})."
        self.num_lanes = 32
        """there are 32 lanes (or threds) in a warp."""
        self.num_k_lanes = 4
        self.num_n_lanes = 8
        assert (
            warp_n >= self.comp_n and warp_n % self.comp_n == 0
        ), f"warp_n ({warp_n}) should be divisible by comp_n({self.comp_n})."
        self.warp_n = warp_n
        # endregion
        # region memory
        self.reg_k = 32 // self.bits
        """number of elements in a register in `k` dimension."""
        self.reg_n = 1
        """number of elements in a register in `n` dimension (always 1)."""
        self.k_pack_size = self.comp_k // (self.num_k_lanes * self.reg_k)
        """number of elements in a pack in `k` dimension."""
        self.n_pack_size = self.comp_n // (self.num_n_lanes * self.reg_n)
        """number of elements in a pack in `n` dimension."""
        self.pack_size = self.k_pack_size * self.n_pack_size
        """number of elements in a pack accessed by a lane at a time."""
        assert 1 <= self.pack_size <= 4, "pack size should be less than or equal to 4."
        assert self.k_pack_size * self.num_k_lanes * self.reg_k == self.comp_k
        assert self.n_pack_size * self.num_n_lanes * self.reg_n == self.comp_n
        self.mem_k = self.comp_k
        """the tile size in `k` dimension for one tensor memory access."""
        self.mem_n = warp_n
        """the tile size in `n` dimension for one tensor memory access."""
        self.num_k_packs = self.mem_k // (self.k_pack_size * self.num_k_lanes * self.reg_k)
        """number of packs in `k` dimension for one tensor memory access."""
        self.num_n_packs = self.mem_n // (self.n_pack_size * self.num_n_lanes * self.reg_n)
        """number of packs in `n` dimension for one tensor memory access."""
        # endregion

    def get_view_shape(self, n: int, k: int) -> tuple[int, int, int, int, int, int, int, int, int, int]:
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
    def __init__(self, bits: int, warp_n: int = 128):
        super().__init__(bits=bits, warp_n=warp_n)
        self.num_k_unrolls = 2

    def pack_weight(self, weight: torch.Tensor) -> torch.Tensor:
        assert weight.dtype == torch.int32, f"quantized weight should be torch.int32, but got {weight.dtype}."
        n, k = weight.shape
        assert n % self.mem_n == 0, f"output channel size ({n}) should be divisible by mem_n ({self.mem_n})."
        # currently, Nunchaku did not check the boundry of unrolled `k` dimension
        assert k % (self.mem_k * self.num_k_unrolls) == 0, (
            f"input channel size ({k}) should be divisible by "
            f"mem_k ({self.mem_k}) * num_k_unrolls ({self.num_k_unrolls})."
        )
        n_tiles, k_tiles = n // self.mem_n, k // self.mem_k
        weight = weight.reshape(
            n_tiles,
            self.num_n_packs,  # 8 when warp_n = 128
            self.n_pack_size,  # always 2 in nunchaku
            self.num_n_lanes,  # constant 8
            self.reg_n,  # constant 1
            k_tiles,
            self.num_k_packs,  # 1
            self.k_pack_size,  # always 2 in nunchaku
            self.num_k_lanes,  # constant 4
            self.reg_k,  # always 8 = 32 bits / 4 bits
        )
        # (n_tiles, num_n_packs, n_pack_size, num_n_lanes, reg_n, k_tiles, num_k_packs, k_pack_size, num_k_lanes, reg_k)
        # =>
        # (n_tiles, k_tiles, num_k_packs, num_n_packs, num_n_lanes, num_k_lanes, n_pack_size, k_pack_size, reg_n, reg_k)
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
        return weight.view(dtype=torch.int8).view(n, -1)  # assume little-endian

    def pack_scale(self, scale: torch.Tensor, group_size: int) -> torch.Tensor:
        if self.check_if_micro_scale(group_size=group_size):
            return self.pack_micro_scale(scale, group_size=group_size)
        # note: refer to https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#mma-16864-c
        assert scale.dtype in (torch.float16, torch.bfloat16), "currently nunchaku only supports fp16 and bf16."
        n = scale.shape[0]
        # nunchaku load scales all in one access
        # for `[warp_n, warp_k]` weights, we load `[warp_n, warp_k / group_size]` scales
        # scale loading is parallelized in `n` dimension, that is,
        #     `num_s_lanes` in a warp load `num_s_packs` of `s_pack_size` elements, in total `warp_s` elements
        # each element in `n` dimension is 16 bit as it contains 1 fp16
        # min `s_pack_size` set to 2 element, since each lane at least holds 2 accumulator results in `n` dimension
        # max `s_pack_size` set to 128b/16b = 8 elements
        # for `warp_n = 8`, we have
        #     `s_pack_size = 2`, `num_s_lanes = 4`,  `num_s_packs = 1`
        # for `warp_n = 128`, we have
        #     `s_pack_size = 4`, `num_s_lanes = 32`, `num_s_packs = 1`
        # for `warp_n = 512`, we have
        #     `s_pack_size = 8`, `num_s_lanes = 32`, `num_s_packs = 2`
        s_pack_size = min(max(self.warp_n // self.num_lanes, 2), 8)
        num_s_lanes = min(self.num_lanes, self.warp_n // s_pack_size)
        num_s_packs = self.warp_n // (s_pack_size * num_s_lanes)
        warp_s = num_s_packs * num_s_lanes * s_pack_size
        assert warp_s == self.warp_n, "warp_n for scales should be equal to warp_n for weights."
        # `num_n_lanes = 8 (constant)` generates 8 elements consecutive in `n` dimension
        # however, they are held by 4 lanes, each lane holds 2 elements in `n` dimension
        # thus, we start from first 4 lanes, assign 2 elements to each lane, until all 8 elements are assigned
        #       we then repeat the process for the same 4 lanes, until each lane holds `s_pack_size` elements
        #       finally, we move to next 4 lanes, and repeat the process until all `num_s_lanes` lanes are assigned
        #       the process is repeated for `num_s_packs` times
        # here is an example for `warp_n = 128, s_pack_size = 4, num_s_lanes = 32, num_s_packs = 1`
        # wscales store order:
        #  0   1   8   9   <-- load by lane 0, broadcast to lane {0, 4, 8, ..., 28} (8x)
        #  2   3   10  11  <-- load by lane 1, broadcast to lane {1, 5, 9, ..., 29} (8x)
        #  4   5   12  13  <-- load by lane 2, broadcast to lane {2, 6, 10, ..., 30} (8x)
        #  6   7   14  15  <-- load by lane 3, broadcast to lane {3, 7, 11, ..., 31} (8x)
        #  16  17  24  25  <-- load by lane 4, broadcast to lane {0, 4, 8, ..., 28} (8x)
        #  ...
        #  22  23  30  31  <-- load by lane 7, broadcast to lane {3, 7, 11, ..., 31} (8x)
        #  ... ...
        #  112 113 120 121 <-- load by lane 28, broadcast to lane {0, 4, 8, ..., 28} (8x)
        #  ...
        #  118 119 126 127 <-- load by lane 31, broadcast to lane {3, 7, 11, ..., 31} (8x)
        scale = scale.reshape(n // warp_s, num_s_packs, num_s_lanes // 4, s_pack_size // 2, 4, 2, -1)
        scale = scale.permute(0, 6, 1, 2, 4, 3, 5).contiguous()
        return scale.view(-1) if group_size == -1 else scale.view(-1, n)  # the shape is just used for validation

    def pack_micro_scale(self, scale: torch.Tensor, group_size: int) -> torch.Tensor:
        assert scale.dtype in (torch.float16, torch.bfloat16), "currently nunchaku only supports fp16 and bf16."
        assert scale.max() <= 448, "scale should be less than 448."
        assert scale.min() >= -448, "scale should be greater than -448."
        assert group_size == 16, "currently only support group size 16."
        assert self.insn_k == 64, "insn_k should be 64."
        scale = scale.to(dtype=torch.float8_e4m3fn)
        n = scale.shape[0]
        assert self.warp_n >= 32, "currently only support warp_n >= 32."
        # for `[warp_n, warp_k]` weights, we load `[warp_n, warp_k / group_size]` scales
        # scale loading is parallelized in `n` dimension, that is,
        #     `num_s_lanes` in a warp load `num_s_packs` of `s_pack_size` elements, in total `warp_s` elements
        # each element in `n` dimension is 32 bit as it contains 4 fp8 in `k` dimension
        # min `s_pack_size` set to 1 element
        # max `s_pack_size` set to 128b/32b = 4 elements
        # for `warp_n = 128`, we have
        #     `s_pack_size = 4`, `num_s_lanes = 32`, `num_s_packs = 1`
        # for `warp_n = 512`, we have
        #     `s_pack_size = 8`, `num_s_lanes = 32`, `num_s_packs = 2`
        s_pack_size = min(max(self.warp_n // self.num_lanes, 1), 4)
        num_s_lanes = 4 * 8  # 32 lanes is divided into 4 pieces, each piece has 8 lanes at a stride of 4
        num_s_packs = ceil_divide(self.warp_n, s_pack_size * num_s_lanes)
        warp_s = num_s_packs * num_s_lanes * s_pack_size
        assert warp_s == self.warp_n, "warp_n for scales should be equal to warp_n for weights."
        # note: refer to https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#mma-scaling-thread-id-b-selection
        # we start from first 8 lines at a stride of 4, assign 1 element to each lane, until all 8 elements are assigned
        #    we then move to next 8 lines at a stride of 4, and repeat the process until all 32 lanes are assigned
        # here is an example for `warp_n = 128, s_pack_size = 4, num_s_lanes = 32, num_s_packs = 1`
        # wscales store order:
        #  0   32  64  96   <-- load by lane 0
        #  8   40  72  104  <-- load by lane 1
        #  16  48  80  112  <-- load by lane 2
        #  24  56  88  120  <-- load by lane 3
        #  1   33  65  97   <-- load by lane 4
        #  ...
        #  25  57  81  113  <-- load by lane 7
        #  ...
        #  7   39  71  103  <-- load by lane 28
        #  ...
        #  31  63  95  127  <-- load by lane 31
        scale = scale.view(n // warp_s, num_s_packs, s_pack_size, 4, 8, -1, self.insn_k // group_size)
        scale = scale.permute(0, 5, 1, 4, 3, 2, 6).contiguous()
        return scale.view(-1, n)  # the shape is just used for validation

    def pack_lowrank_weight(self, weight: torch.Tensor, down: bool) -> torch.Tensor:
        """Pack Low-Rank Weight.

        Args:
            weight (`torch.Tensor`):
                low-rank weight tensor.
            down (`bool`):
                whether the weight is for down projection in low-rank branch.
        """
        assert weight.dtype in (torch.float16, torch.bfloat16), f"Unsupported weight dtype {weight.dtype}."
        reg_n, reg_k = 1, 2  # reg_n is always 1, reg_k is 32 bits // 16 bits = 2
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
        # (c_packs, r_packs, n_pack_size, num_n_lanes, reg_n, k_pack_size, num_k_lanes, reg_k)
        # =>
        # (c_packs, r_packs, num_n_lanes, num_k_lanes, n_pack_size, k_pack_size, reg_n, reg_k)
        weight = weight.permute(0, 1, 3, 6, 2, 5, 4, 7).contiguous()
        return weight.view(c, r)

    def unpack_lowrank_weight(self, weight: torch.Tensor, down: bool) -> torch.Tensor:
        """Unpack Low-Rank Weight.

        Args:
            weight (`torch.Tensor`):
                low-rank weight tensor.
            down (`bool`):
                whether the weight is for down projection in low-rank branch.
        """
        c, r = weight.shape
        assert weight.dtype in (torch.float16, torch.bfloat16), f"Unsupported weight dtype {weight.dtype}."
        reg_n, reg_k = 1, 2  # reg_n is always 1, reg_k is 32 bits // 16 bits = 2
        pack_n = self.n_pack_size * self.num_n_lanes * reg_n
        pack_k = self.k_pack_size * self.num_k_lanes * reg_k
        if down:
            r_packs, c_packs = r // pack_n, c // pack_k
        else:
            c_packs, r_packs = c // pack_n, r // pack_k
        weight = weight.view(
            c_packs, r_packs, self.num_n_lanes, self.num_k_lanes, self.n_pack_size, self.k_pack_size, reg_n, reg_k
        )
        # (c_packs, r_packs, num_n_lanes, num_k_lanes, n_pack_size, k_pack_size, reg_n, reg_k)
        # =>
        # (c_packs, r_packs, n_pack_size, num_n_lanes, reg_n, k_pack_size, num_k_lanes, reg_k)
        weight = weight.permute(0, 1, 4, 2, 6, 5, 3, 7).contiguous()
        weight = weight.view(c_packs, r_packs, pack_n, pack_k)
        if down:
            weight = weight.permute(1, 2, 0, 3).contiguous().view(r, c)
        else:
            weight = weight.permute(0, 2, 1, 3).contiguous().view(c, r)
        return weight

    def check_if_micro_scale(self, group_size: int) -> bool:
        return self.insn_k == group_size * 4

    def pad_weight(self, weight: torch.Tensor) -> torch.Tensor:
        assert weight.ndim == 2, "weight tensor should be 2D."
        return pad(weight, divisor=(self.mem_n, self.mem_k * self.num_k_unrolls), dim=(0, 1))

    def pad_scale(self, scale: torch.Tensor, group_size: int, fill_value: float = 0) -> torch.Tensor:
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
        assert weight.ndim == 2, "weight tensor should be 2D."
        return pad(weight, divisor=self.warp_n, dim=1 if down else 0)
