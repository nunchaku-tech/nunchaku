from typing import Dict, List

import torch
from torch import nn


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


class BlockOffloadManager:
    """Generic manager for per-transformer-block CPU offloading with async memory operations.

    This class can be used with any transformer model that has a list of transformer blocks.
    It provides memory-efficient processing by keeping only the current block on GPU.
    """

    def __init__(
        self,
        blocks: List[nn.Module],
        device: torch.device = torch.device("cuda"),
        use_pin_memory: bool = False,
    ):
        self.blocks = blocks
        self.device = device
        self.cpu_device = torch.device("cpu")
        self.use_pin_memory = use_pin_memory

        # Two streams: one for compute, one for memory operations
        self.compute_stream = torch.cuda.Stream(device=device)
        self.memory_stream = torch.cuda.Stream(device=device)

        self.compute_done = torch.cuda.Event(blocking=False)
        self.memory_done = torch.cuda.Event(blocking=False)

        self.next_compute_done = torch.cuda.Event(blocking=False)
        self.next_memory_done = torch.cuda.Event(blocking=False)

        # Current state tracking
        self.current_block_idx = 0

        # Initialize: first block on GPU, others on CPU
        self._initialize_blocks()

    def set_device(self, device: torch.device | str):
        if isinstance(device, str):
            device = torch.device(device)
        assert device.type == "cuda"
        self.device = device
        self.compute_stream = torch.cuda.Stream(device=device)
        self.memory_stream = torch.cuda.Stream(device=device)

    def _initialize_blocks(self):
        """Initialize blocks: first on GPU, others on CPU."""
        p = next(self.blocks[0].parameters())
        if p.device != self.device:
            self.blocks[0].to(self.device)

    def _move_block_to_cpu(self, block_idx: int):
        """Move a transformer block to CPU."""
        if block_idx >= len(self.blocks):
            return

        block = self.blocks[block_idx]

        # Move all parameters to CPU
        for name, param in block.named_parameters():
            if param.device != self.cpu_device:
                param.data = param.data.to(self.cpu_device, non_blocking=True)

        # Move all buffers to CPU
        for name, buffer in block.named_buffers():
            if buffer.device != self.cpu_device:
                buffer.data = buffer.data.to(self.cpu_device, non_blocking=True)

    def _move_block_to_gpu(self, block_idx: int):
        """Move a transformer block to GPU."""
        if block_idx >= len(self.blocks):
            return

        block = self.blocks[block_idx]

        # Move all parameters to GPU
        for name, param in block.named_parameters():
            if param.device != self.device:
                param.data = param.data.to(self.device, non_blocking=True)

        # Move all buffers to GPU
        for name, buffer in block.named_buffers():
            if buffer.device != self.device:
                buffer.data = buffer.data.to(self.device, non_blocking=True)

    def load_block(self, idx: int):
        if idx >= len(self.blocks):
            return

        self._move_block_to_gpu(idx)

    def offload_block(self, idx: int):
        """Offload the current block to CPU using memory stream."""
        if idx <= 0:
            return

        self._move_block_to_cpu(idx)

    def step(self):
        """Move to the next block, triggering preload and offload operations."""
        self.memory_done.wait(self.compute_stream)
        with torch.cuda.stream(self.memory_stream):
            self.offload_block(self.current_block_idx - 1)  # offload the previous block
            self.load_block(self.current_block_idx + 1)  # preload the next block
            self.next_memory_done.record(self.memory_stream)

        self.current_block_idx += 1
        self.memory_done = self.next_memory_done
        self.compute_done = self.next_compute_done

    def wait_for_block(self):
        """Wait until a block is loaded on GPU"""
        self.compute_done.wait(self.memory_stream)

    def get_memory_usage(self) -> Dict[str, float]:
        """Get memory usage statistics."""
        gpu_memory = 0.0
        cpu_memory = 0.0

        for i, block in enumerate(self.blocks):
            block_memory = 0.0
            for name, param in block.named_parameters():
                if param.device == self.device:
                    block_memory += param.numel() * param.element_size()
                elif param.device == self.cpu_device:
                    cpu_memory += param.numel() * param.element_size()

            if i in self.blocks_on_gpu:
                gpu_memory += block_memory
            else:
                cpu_memory += block_memory

        return {
            "gpu_memory_mb": gpu_memory / (1024 * 1024),
            "cpu_memory_mb": cpu_memory / (1024 * 1024),
            "total_memory_mb": (gpu_memory + cpu_memory) / (1024 * 1024),
        }
