import copy

import torch
from torch import nn

from ..utils import copy_params_into


def fuse_linears(linears: list[nn.Linear]) -> nn.Linear:
    assert len(linears) > 0
    if len(linears) == 1:
        return linears[0]
    else:
        assert all(linear.in_features == linears[0].in_features for linear in linears)
        out_features = sum(linear.out_features for linear in linears)
        bias = all(linear.bias is not None for linear in linears)
        return nn.Linear(
            linears[0].in_features,
            out_features,
            bias=bias,
            dtype=linears[0].weight.dtype,
            device=linears[0].weight.device,
        )


class CPUOffloadManager:
    """Generic manager for per-transformer-block CPU offloading with async memory operations.

    This class can be used with any transformer model that has a list of transformer blocks.
    It provides memory-efficient processing by keeping only the current block on GPU.
    """

    def __init__(
        self,
        blocks: list[nn.Module],
        device: str | torch.device = torch.device("cuda"),
        use_pin_memory: bool = True,
        on_gpu_modules: list[nn.Module] = [],
        num_blocks_on_gpu: int = 1,
        empty_cache_freq: int = 0,
    ):
        self.blocks = blocks
        self.use_pin_memory = use_pin_memory
        self.on_gpu_modules = on_gpu_modules
        self.num_blocks_on_gpu = num_blocks_on_gpu
        assert self.num_blocks_on_gpu > 0

        # Two streams: one for compute, one for memory operations, will be initialized in set_device
        self.compute_stream = None
        self.memory_stream = None

        self.compute_done = torch.cuda.Event(blocking=False)
        self.memory_done = torch.cuda.Event(blocking=False)

        self.buffer_blocks = [copy.deepcopy(blocks[0]), copy.deepcopy(blocks[0])]

        self.device = None
        self.set_device(device)

        self.current_block_idx = 0
        self.forward_counter = 0
        self.empty_cache_freq = empty_cache_freq

    def set_device(self, device: torch.device | str, force: bool = False):
        if isinstance(device, str):
            device = torch.device(device)
        assert device.type == "cuda"
        if self.device == device and not force:
            return
        self.device = device
        self.compute_stream = torch.cuda.Stream(device=device)
        self.memory_stream = torch.cuda.Stream(device=device)
        for block in self.buffer_blocks:
            block.to(device)
        for module in self.on_gpu_modules:
            module.to(device)
        for i, block in enumerate(self.blocks):
            if i < self.num_blocks_on_gpu:
                block.to(device)
            else:
                block.to("cpu")
                if self.use_pin_memory:
                    for p in block.parameters(recurse=True):
                        p.data = p.data.pin_memory()
                    for b in block.buffers(recurse=True):
                        b.data = b.data.pin_memory()

    def load_block(self, block_idx: int, non_blocking: bool = True):
        """Move a transformer block to GPU."""
        # if the block is already on GPU, don't load it to the buffer
        if block_idx < self.num_blocks_on_gpu:
            return
        # if there are blocks on GPU, don't load the first block to the buffer again
        if block_idx >= len(self.blocks):
            return

        block = self.blocks[block_idx]
        copy_params_into(block, self.buffer_blocks[block_idx % 2], non_blocking=non_blocking)

    def step(self, next_stream: torch.cuda.Stream | None = None):
        """Move to the next block, triggering preload operations."""
        next_compute_done = torch.cuda.Event()
        next_compute_done.record(self.compute_stream)
        with torch.cuda.stream(self.memory_stream):
            self.memory_stream.wait_event(self.compute_done)
            self.load_block(self.current_block_idx + 1)  # if the current block is the last block, load the first block
            next_memory_done = torch.cuda.Event()
            next_memory_done.record(self.memory_stream)
        self.memory_done = next_memory_done
        self.compute_done = next_compute_done
        self.current_block_idx += 1
        if self.current_block_idx < len(self.blocks):
            # get ready for the next compute
            self.compute_stream.wait_event(self.memory_done)
        else:
            # ready to finish
            if next_stream is None:
                torch.cuda.current_stream().wait_event(self.compute_done)
            else:
                next_stream.wait_event(self.compute_done)
            self.current_block_idx = 0
            self.forward_counter += 1
            if self.empty_cache_freq > 0 and self.forward_counter % self.empty_cache_freq == 0:
                torch.cuda.empty_cache()

    def get_block(self, block_idx: int | None = None) -> nn.Module:
        if block_idx is None:
            block_idx = self.current_block_idx
        if block_idx < self.num_blocks_on_gpu:
            return self.blocks[block_idx]
        else:
            return self.buffer_blocks[block_idx % 2]

    def initialize(self, stream: torch.cuda.Stream | None = None):
        if stream is None:
            stream = torch.cuda.current_stream()
        self.compute_done.record(stream)
        self.memory_done.record(stream)
