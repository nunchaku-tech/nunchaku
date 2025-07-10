"""
Caching utilities for transformer models.

Implements first-block caching to accelerate transformer inference by reusing computations
when input changes are minimal. Supports SANA and Flux architectures.

Main Classes
------------
- :class:`CacheContext` : Manages cache buffers and incremental naming.
- :class:`SanaCachedTransformerBlocks` : Cached transformer blocks for SANA models.
- :class:`FluxCachedTransformerBlocks` : Cached transformer blocks for Flux models.

Key Functions
-------------
- :func:`get_buffer`, :func:`set_buffer` : Cache buffer management.
- :func:`cache_context` : Context manager for cache operations.
- :func:`are_two_tensors_similar` : Tensor similarity check.
- :func:`apply_prev_hidden_states_residual` : Applies cached residuals.
- :func:`get_can_use_cache` : Checks cache usability.
- :func:`check_and_apply_cache` : Main cache logic.

Caching Strategy
----------------
1. Compute the first transformer block.
2. Compare the residual with the cached residual.
3. If similar, reuse cached results for the remaining blocks; otherwise, recompute and update cache.

.. note::
   Adapted from ParaAttention:
   https://github.com/chengzeyi/ParaAttention/src/para_attn/first_block_cache/
"""

import contextlib
import dataclasses
from collections import defaultdict
from typing import DefaultDict, Dict, Optional, Tuple

import torch
from torch import nn

from nunchaku.models.transformers.utils import pad_tensor

num_transformer_blocks = 19  # FIXME
num_single_transformer_blocks = 38  # FIXME


@dataclasses.dataclass
class CacheContext:
    """
    Manages cache buffers and incremental naming for transformer model inference.

    Attributes
    ----------
    buffers : Dict[str, torch.Tensor]
        Stores cached tensor buffers.
    incremental_name_counters : DefaultDict[str, int]
        Counters for generating unique incremental cache entry names.
    """

    buffers: Dict[str, torch.Tensor] = dataclasses.field(default_factory=dict)
    incremental_name_counters: DefaultDict[str, int] = dataclasses.field(default_factory=lambda: defaultdict(int))

    def get_incremental_name(self, name=None):
        """
        Generate an incremental cache entry name.

        Parameters
        ----------
        name : str, optional
            Base name for the counter. If None, uses "default".

        Returns
        -------
        str
            Incremental name in the format ``"{name}_{counter}"``.
        """
        if name is None:
            name = "default"
        idx = self.incremental_name_counters[name]
        self.incremental_name_counters[name] += 1
        return f"{name}_{idx}"

    def reset_incremental_name(self):
        """
        Reset all incremental name counters.

        After calling this, :meth:`get_incremental_name` will start from 0 for each name.
        """
        self.incremental_name_counters.clear()

    # @torch.compiler.disable # This is a torchscript feature
    def get_buffer(self, name=str):
        """
        Retrieve a cached tensor buffer by name.

        Parameters
        ----------
        name : str
            Name of the buffer to retrieve.

        Returns
        -------
        torch.Tensor or None
            The cached tensor if found, otherwise None.
        """
        return self.buffers.get(name)

    def set_buffer(self, name, buffer):
        """
        Store a tensor buffer in the cache.

        Args:
            name (str): The name to associate with the buffer
            buffer (torch.Tensor): The tensor to cache
        """
        self.buffers[name] = buffer

    def clear_buffers(self):
        """
        Clear all cached buffers.

        This removes all stored tensors from the cache, freeing up memory.
        """
        self.buffers.clear()


@torch.compiler.disable
def get_buffer(name):
    """
    Retrieve a cached tensor buffer from the current cache context.

    This is a convenience function that gets the buffer from the currently active
    cache context. The cache context must be set before calling this function.

    Args:
        name (str): The name of the buffer to retrieve

    Returns:
        torch.Tensor or None: The cached tensor if found, None otherwise

    Raises:
        AssertionError: If no cache context is currently active

    Example:
        >>> with cache_context(create_cache_context()):
        ...     set_buffer("my_tensor", torch.randn(2, 3))
        ...     cached = get_buffer("my_tensor")
    """
    cache_context = get_current_cache_context()
    assert cache_context is not None, "cache_context must be set before"
    return cache_context.get_buffer(name)


@torch.compiler.disable
def set_buffer(name, buffer):
    """
    Store a tensor buffer in the current cache context.

    This is a convenience function that sets the buffer in the currently active
    cache context. The cache context must be set before calling this function.

    Args:
        name (str): The name to associate with the buffer
        buffer (torch.Tensor): The tensor to cache

    Raises:
        AssertionError: If no cache context is currently active

    Example:
        >>> with cache_context(create_cache_context()):
        ...     set_buffer("my_tensor", torch.randn(2, 3))
        ...     cached = get_buffer("my_tensor")
    """
    cache_context = get_current_cache_context()
    assert cache_context is not None, "cache_context must be set before"
    cache_context.set_buffer(name, buffer)


_current_cache_context = None


def create_cache_context():
    """
    Create a new cache context for managing cached computations.

    Returns:
        CacheContext: A new cache context instance

    Example:
        >>> context = create_cache_context()
        >>> with cache_context(context):
        ...     # Cached operations here
        ...     pass
    """
    return CacheContext()


def get_current_cache_context():
    """
    Get the currently active cache context.

    Returns:
        CacheContext or None: The current cache context if one is active, None otherwise

    Example:
        >>> with cache_context(create_cache_context()):
        ...     current = get_current_cache_context()
        ...     assert current is not None
    """
    return _current_cache_context


@contextlib.contextmanager
def cache_context(cache_context):
    """
    Context manager for setting the active cache context.

    This context manager sets the global cache context for the duration of the
    with block, then restores the previous context when exiting.

    Args:
        cache_context (CacheContext): The cache context to use

    Yields:
        None: The context manager yields nothing

    Example:
        >>> context = create_cache_context()
        >>> with cache_context(context):
        ...     set_buffer("key", torch.tensor([1, 2, 3]))
        ...     cached = get_buffer("key")
    """
    global _current_cache_context
    old_cache_context = _current_cache_context
    _current_cache_context = cache_context
    try:
        yield
    finally:
        _current_cache_context = old_cache_context


@torch.compiler.disable
def are_two_tensors_similar(t1, t2, *, threshold, parallelized=False):
    """
    Check if two tensors are similar based on relative L1 distance.

    Computes the relative L1 distance between two tensors and compares it to a threshold.
    The relative distance is calculated as ``mean(abs(t1 - t2)) / mean(abs(t1))``.

    Args:
        t1 (torch.Tensor): First tensor for comparison
        t2 (torch.Tensor): Second tensor for comparison
        threshold (float): Similarity threshold. Tensors are considered similar if
            relative distance < threshold
        parallelized (bool, optional): Whether computation is parallelized.
            Currently unused. Defaults to False.

    Returns:
        tuple[bool, float]: A tuple containing:

        - bool: True if tensors are similar (diff < threshold), False otherwise
        - float: The computed relative L1 distance
    """
    mean_diff = (t1 - t2).abs().mean()
    mean_t1 = t1.abs().mean()
    diff = (mean_diff / mean_t1).item()
    return diff < threshold, diff


@torch.compiler.disable
def apply_prev_hidden_states_residual(
    hidden_states: torch.Tensor,
    encoder_hidden_states: torch.Tensor = None,
    mode: str = "multi",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply previously cached residual states to current hidden states.

    This function retrieves cached residual computations and applies them to the current
    hidden states, avoiding the need to recompute transformer blocks when the cache
    is valid.

    Args:
        hidden_states (torch.Tensor): Current hidden states to apply residuals to
        encoder_hidden_states (torch.Tensor, optional): Encoder hidden states.
            Required for "multi" mode. Defaults to None.
        mode (str, optional): Caching mode, either "multi" or "single".
            Defaults to "multi".

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: A tuple containing:

        - torch.Tensor: Updated hidden states with residuals applied
        - torch.Tensor: Updated encoder hidden states (for "multi" mode) or
          original hidden states (for "single" mode)

    Raises:
        AssertionError: If required cached residuals are not found
        ValueError: If mode is not "multi" or "single"

    Example:
        >>> # In multi mode
        >>> hidden_states = torch.randn(2, 256, 512)
        >>> encoder_states = torch.randn(2, 128, 512)
        >>> # Assume residuals are cached
        >>> updated_h, updated_e = apply_prev_hidden_states_residual(
        ...     hidden_states, encoder_states, mode="multi"
        ... )
    """
    if mode == "multi":
        hidden_states_residual = get_buffer("multi_hidden_states_residual")
        assert hidden_states_residual is not None, "multi_hidden_states_residual must be set before"
        hidden_states = hidden_states + hidden_states_residual
        hidden_states = hidden_states.contiguous()

        if encoder_hidden_states is not None:
            enc_hidden_res = get_buffer("multi_encoder_hidden_states_residual")
            msg = "multi_encoder_hidden_states_residual must be set before"
            assert enc_hidden_res is not None, msg
            encoder_hidden_states = encoder_hidden_states + enc_hidden_res
            encoder_hidden_states = encoder_hidden_states.contiguous()

        return hidden_states, encoder_hidden_states

    elif mode == "single":
        single_residual = get_buffer("single_hidden_states_residual")
        msg = "single_hidden_states_residual must be set before"
        assert single_residual is not None, msg
        hidden_states = hidden_states + single_residual
        hidden_states = hidden_states.contiguous()

        return hidden_states

    else:
        raise ValueError(f"Unknown mode {mode}; expected 'multi' or 'single'")


@torch.compiler.disable
def get_can_use_cache(
    first_hidden_states_residual: torch.Tensor, threshold: float, parallelized: bool = False, mode: str = "multi"
):
    """
    Determine if cached computations can be used based on residual similarity.

    Compares the current first hidden states residual with the previously cached
    residual to determine if the cache is still valid. If the residuals are similar
    enough (below threshold), cached computations can be reused.

    Args:
        first_hidden_states_residual (torch.Tensor): Current first block residual
        threshold (float): Similarity threshold for cache validity
        parallelized (bool, optional): Whether computation is parallelized.
            Defaults to False.
        mode (str, optional): Caching mode, either "multi" or "single".
            Defaults to "multi".

    Returns:
        tuple[bool, float]: A tuple containing:

        - bool: True if cache can be used (residuals are similar), False otherwise
        - float: The computed similarity difference, or threshold if no cache exists

    Raises:
        ValueError: If mode is not "multi" or "single"

    Example:
        >>> residual = torch.randn(2, 256, 512)
        >>> can_use, diff = get_can_use_cache(residual, threshold=0.1)
        >>> if can_use:
        ...     print(f"Cache hit! Difference: {diff:.4f}")
        ... else:
        ...     print(f"Cache miss. Difference: {diff:.4f}")
    """
    if mode == "multi":
        buffer_name = "first_multi_hidden_states_residual"
    elif mode == "single":
        buffer_name = "first_single_hidden_states_residual"
    else:
        raise ValueError(f"Unknown mode {mode}; expected 'multi' or 'single'")

    prev_res = get_buffer(buffer_name)

    if prev_res is None:
        return False, threshold

    is_similar, diff = are_two_tensors_similar(
        prev_res,
        first_hidden_states_residual,
        threshold=threshold,
        parallelized=parallelized,
    )
    return is_similar, diff


def check_and_apply_cache(
    *,
    first_residual: torch.Tensor,
    hidden_states: torch.Tensor,
    encoder_hidden_states: Optional[torch.Tensor] = None,
    threshold: float,
    parallelized: bool,
    mode: str,
    verbose: bool,
    call_remaining_fn,
    remaining_kwargs,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], float]:
    """
    Main caching logic: check if cache can be used and apply accordingly.

    This function implements the core caching decision logic. It checks if the cache
    can be used based on residual similarity, and either applies cached computations
    or computes new values and caches them for future use.

    Args:
        first_residual (torch.Tensor): First block residual for similarity comparison
        hidden_states (torch.Tensor): Current hidden states
        encoder_hidden_states (torch.Tensor, optional): Encoder hidden states.
            Required for "multi" mode. Defaults to None.
        threshold (float): Similarity threshold for cache validity
        parallelized (bool): Whether computation is parallelized
        mode (str): Caching mode, either "multi" or "single"
        verbose (bool): Whether to print caching status messages
        call_remaining_fn (callable): Function to call remaining transformer blocks
        remaining_kwargs (dict): Additional keyword arguments for call_remaining_fn

    Returns:
        Tuple[torch.Tensor, Optional[torch.Tensor], float]: A tuple containing:

        - torch.Tensor: Updated hidden states
        - torch.Tensor or None: Updated encoder hidden states (for "multi" mode)
        - float: Current threshold value

    Example:
        >>> def remaining_fn(hidden_states, **kwargs):
        ...     # Process remaining transformer blocks
        ...     return processed_states
        >>>
        >>> updated_h, updated_e, new_threshold = check_and_apply_cache(
        ...     first_residual=residual,
        ...     hidden_states=hidden_states,
        ...     encoder_hidden_states=encoder_states,
        ...     threshold=0.1,
        ...     parallelized=False,
        ...     mode="multi",
        ...     verbose=True,
        ...     call_remaining_fn=remaining_fn,
        ...     remaining_kwargs={}
        ... )
    """
    can_use_cache, diff = get_can_use_cache(
        first_residual,
        threshold=threshold,
        parallelized=parallelized,
        mode=mode,
    )
    torch._dynamo.graph_break()

    if can_use_cache:
        if verbose:
            print(f"[{mode.upper()}] Cache hit! diff={diff:.4f}, " f"new threshold={threshold:.4f}")

        out = apply_prev_hidden_states_residual(hidden_states, encoder_hidden_states, mode=mode)
        updated_h, updated_enc = out if isinstance(out, tuple) else (out, None)
        return updated_h, updated_enc, threshold

    old_threshold = threshold

    if verbose:
        print(f"[{mode.upper()}] Cache miss. diff={diff:.4f}, " f"was={old_threshold:.4f} => now={threshold:.4f}")

    if mode == "multi":
        set_buffer("first_multi_hidden_states_residual", first_residual)
    else:
        set_buffer("first_single_hidden_states_residual", first_residual)

    result = call_remaining_fn(
        hidden_states=hidden_states, encoder_hidden_states=encoder_hidden_states, **remaining_kwargs
    )

    if mode == "multi":
        updated_h, updated_enc, hs_res, enc_res = result
        set_buffer("multi_hidden_states_residual", hs_res)
        set_buffer("multi_encoder_hidden_states_residual", enc_res)
        return updated_h, updated_enc, threshold

    elif mode == "single":
        updated_cat_states, cat_res = result
        set_buffer("single_hidden_states_residual", cat_res)
        return updated_cat_states, None, threshold

    raise ValueError(f"Unknown mode {mode}")


class SanaCachedTransformerBlocks(nn.Module):
    """
    Cached transformer blocks implementation for SANA models.

    This class wraps SANA transformer blocks to enable caching of first-block
    computations. It implements a single-block caching strategy where the first
    transformer block's residual is cached and reused when input similarity
    is above the threshold.

    Args:
        transformer (nn.Module): The original transformer model to wrap
        residual_diff_threshold (float): Similarity threshold for cache validity
        verbose (bool, optional): Whether to print caching status messages.
            Defaults to False.

    Attributes:
        transformer (nn.Module): Reference to the original transformer
        transformer_blocks (nn.ModuleList): The transformer blocks to cache
        residual_diff_threshold (float): Current similarity threshold
        verbose (bool): Verbosity flag for debugging

    Example:
        >>> from diffusers import SanaTransformer2DModel
        >>> transformer = SanaTransformer2DModel.from_pretrained("model_name")
        >>> cached_blocks = SanaCachedTransformerBlocks(
        ...     transformer=transformer,
        ...     residual_diff_threshold=0.1,
        ...     verbose=True
        ... )
        >>> # Use cached_blocks in place of original transformer blocks
    """

    def __init__(
        self,
        *,
        transformer=None,
        residual_diff_threshold,
        verbose: bool = False,
    ):
        super().__init__()
        self.transformer = transformer
        self.transformer_blocks = transformer.transformer_blocks
        self.residual_diff_threshold = residual_diff_threshold
        self.verbose = verbose

    def forward(
        self,
        hidden_states,
        attention_mask,
        encoder_hidden_states,
        encoder_attention_mask=None,
        timestep=None,
        post_patch_height=None,
        post_patch_width=None,
    ):
        """
        Forward pass with caching for SANA transformer blocks.

        This method implements the cached forward pass for SANA models. It computes
        the first transformer block and checks if the residual is similar enough to
        the cached version to reuse previous computations.

        Args:
            hidden_states (torch.Tensor): Input hidden states
            attention_mask (torch.Tensor): Attention mask for the input
            encoder_hidden_states (torch.Tensor): Encoder hidden states
            encoder_attention_mask (torch.Tensor, optional): Encoder attention mask.
                Defaults to None.
            timestep (torch.Tensor, optional): Timestep tensor for conditioning.
                Defaults to None.
            post_patch_height (int, optional): Height after patch embedding.
                Defaults to None.
            post_patch_width (int, optional): Width after patch embedding.
                Defaults to None.

        Returns:
            torch.Tensor: Processed hidden states from transformer blocks

        Note:
            If batch size > 2 or residual_diff_threshold <= 0, caching is disabled
            and the original forward pass is used.
        """
        batch_size = hidden_states.shape[0]
        if self.residual_diff_threshold <= 0.0 or batch_size > 2:
            if batch_size > 2:
                print("Batch size > 2 (for SANA CFG)" " currently not supported")

            first_transformer_block = self.transformer_blocks[0]
            hidden_states = first_transformer_block(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                timestep=timestep,
                height=post_patch_height,
                width=post_patch_width,
                skip_first_layer=False,
            )
            return hidden_states

        original_hidden_states = hidden_states
        first_transformer_block = self.transformer_blocks[0]

        hidden_states = first_transformer_block.forward_layer_at(
            0,
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            timestep=timestep,
            height=post_patch_height,
            width=post_patch_width,
        )
        first_hidden_states_residual = hidden_states - original_hidden_states
        del original_hidden_states

        can_use_cache, _ = get_can_use_cache(
            first_hidden_states_residual,
            threshold=self.residual_diff_threshold,
            parallelized=self.transformer is not None and getattr(self.transformer, "_is_parallelized", False),
        )

        torch._dynamo.graph_break()
        if can_use_cache:
            del first_hidden_states_residual
            if self.verbose:
                print("Cache hit!!!")
            hidden_states, _ = apply_prev_hidden_states_residual(hidden_states, None)
        else:
            if self.verbose:
                print("Cache miss!!!")
            set_buffer("first_multi_hidden_states_residual", first_hidden_states_residual)
            del first_hidden_states_residual

            hidden_states, hidden_states_residual = self.call_remaining_transformer_blocks(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                timestep=timestep,
                post_patch_height=post_patch_height,
                post_patch_width=post_patch_width,
            )
            set_buffer("multi_hidden_states_residual", hidden_states_residual)
        torch._dynamo.graph_break()

        return hidden_states

    def call_remaining_transformer_blocks(
        self,
        hidden_states,
        attention_mask,
        encoder_hidden_states,
        encoder_attention_mask=None,
        timestep=None,
        post_patch_height=None,
        post_patch_width=None,
    ):
        """
        Process remaining transformer blocks after the first block.

        This method is called when the cache is invalid and remaining transformer
        blocks need to be computed. It skips the first layer and processes the
        remaining blocks to generate the final output.

        Args:
            hidden_states (torch.Tensor): Hidden states from the first block
            attention_mask (torch.Tensor): Attention mask for the input
            encoder_hidden_states (torch.Tensor): Encoder hidden states
            encoder_attention_mask (torch.Tensor, optional): Encoder attention mask.
                Defaults to None.
            timestep (torch.Tensor, optional): Timestep tensor for conditioning.
                Defaults to None.
            post_patch_height (int, optional): Height after patch embedding.
                Defaults to None.
            post_patch_width (int, optional): Width after patch embedding.
                Defaults to None.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: A tuple containing:

            - torch.Tensor: Final hidden states after processing all blocks
            - torch.Tensor: Residual difference for caching
        """
        first_transformer_block = self.transformer_blocks[0]
        original_hidden_states = hidden_states
        hidden_states = first_transformer_block(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            timestep=timestep,
            height=post_patch_height,
            width=post_patch_width,
            skip_first_layer=True,
        )
        hidden_states_residual = hidden_states - original_hidden_states

        return hidden_states, hidden_states_residual


class FluxCachedTransformerBlocks(nn.Module):
    """
    Cached transformer blocks implementation for Flux models.

    This class wraps Flux transformer blocks to enable advanced caching strategies
    including single and double first-block caching. It supports both multi-head
    attention and single-head attention transformer blocks with independent
    caching thresholds.

    Args:
        transformer (nn.Module): The original Flux transformer model to wrap
        use_double_fb_cache (bool, optional): Whether to use double first-block caching.
            If True, caches both multi and single transformer blocks. Defaults to True.
        residual_diff_threshold_multi (float): Similarity threshold for multi-head
            attention blocks cache validity
        residual_diff_threshold_single (float): Similarity threshold for single-head
            attention blocks cache validity
        return_hidden_states_first (bool, optional): Whether to return hidden states
            first in the output tuple. Defaults to True.
        return_hidden_states_only (bool, optional): Whether to return only hidden
            states (not encoder states). Defaults to False.
        verbose (bool, optional): Whether to print caching status messages.
            Defaults to False.

    Attributes:
        transformer (nn.Module): Reference to the original transformer
        transformer_blocks (nn.ModuleList): Multi-head attention transformer blocks
        single_transformer_blocks (nn.ModuleList): Single-head attention transformer blocks
        use_double_fb_cache (bool): Double first-block caching flag
        residual_diff_threshold_multi (float): Multi-block similarity threshold
        residual_diff_threshold_single (float): Single-block similarity threshold
        return_hidden_states_first (bool): Output order flag
        return_hidden_states_only (bool): Output type flag
        verbose (bool): Verbosity flag for debugging
        m: Model interface for optimized operations
        dtype (torch.dtype): Data type for computations
        device (torch.device): Device for computations

    Example:
        >>> from diffusers import FluxTransformer2DModel
        >>> transformer = FluxTransformer2DModel.from_pretrained("model_name")
        >>> cached_blocks = FluxCachedTransformerBlocks(
        ...     transformer=transformer,
        ...     use_double_fb_cache=True,
        ...     residual_diff_threshold_multi=0.12,
        ...     residual_diff_threshold_single=0.09,
        ...     verbose=True
        ... )
    """

    def __init__(
        self,
        *,
        transformer: nn.Module = None,
        use_double_fb_cache: bool = True,
        residual_diff_threshold_multi: float,
        residual_diff_threshold_single: float,
        return_hidden_states_first: bool = True,
        return_hidden_states_only: bool = False,
        verbose: bool = False,
    ):
        super().__init__()
        self.transformer = transformer
        self.transformer_blocks = transformer.transformer_blocks
        self.single_transformer_blocks = transformer.single_transformer_blocks

        self.use_double_fb_cache = use_double_fb_cache
        self.residual_diff_threshold_multi = residual_diff_threshold_multi
        self.residual_diff_threshold_single = residual_diff_threshold_single

        self.return_hidden_states_first = return_hidden_states_first
        self.return_hidden_states_only = return_hidden_states_only
        self.verbose = verbose

        self.m = self.transformer_blocks[0].m
        self.dtype = torch.bfloat16 if self.m.isBF16() else torch.float16
        self.device = transformer.device

    @staticmethod
    def pack_rotemb(rotemb: torch.Tensor) -> torch.Tensor:
        """
        Pack rotary embedding tensor for optimized GPU computation.

        This method reshapes and permutes the rotary embedding tensor to match
        the optimal memory layout for GPU matrix multiplication operations,
        specifically for 16x8 MMA (Matrix Multiply-Accumulate) operations.

        Args:
            rotemb (torch.Tensor): Input rotary embedding tensor of shape
                (B, M, D//2, 1, 2) where B is batch size, M is sequence length,
                and D is the embedding dimension.

        Returns:
            torch.Tensor: Packed rotary embedding tensor of shape (B, M, D)

        Raises:
            AssertionError: If input tensor doesn't meet the required shape or
                dimension constraints (M % 16 == 0, D % 8 == 0, dtype == float32)

        Note:
            This function is optimized for CUDA GPU execution and follows
            the NVIDIA PTX MMA 16x8x16 format for FP32 accumulation.
            Reference: https://docs.nvidia.com/cuda/parallel-thread-execution/#mma-16816-c
        """
        assert rotemb.dtype == torch.float32
        B = rotemb.shape[0]
        M = rotemb.shape[1]
        D = rotemb.shape[2] * 2
        msg_shape = "rotemb shape must be (B, M, D//2, 1, 2)"
        assert rotemb.shape == (B, M, D // 2, 1, 2), msg_shape
        assert M % 16 == 0
        assert D % 8 == 0
        rotemb = rotemb.reshape(B, M // 16, 16, D // 8, 8)
        rotemb = rotemb.permute(0, 1, 3, 2, 4)
        # 16*8 pack, FP32 accumulator (C) format
        # https://docs.nvidia.com/cuda/parallel-thread-execution/#mma-16816-c
        rotemb = rotemb.reshape(*rotemb.shape[0:3], 2, 8, 4, 2)
        rotemb = rotemb.permute(0, 1, 2, 4, 5, 3, 6)
        rotemb = rotemb.contiguous()
        rotemb = rotemb.view(B, M, D)
        return rotemb

    def update_residual_diff_threshold(
        self, use_double_fb_cache=True, residual_diff_threshold_multi=0.12, residual_diff_threshold_single=0.09
    ):
        """
        Update caching configuration parameters.

        This method allows dynamic updating of the caching thresholds and
        double caching behavior during runtime.

        Args:
            use_double_fb_cache (bool, optional): Whether to use double first-block
                caching. Defaults to True.
            residual_diff_threshold_multi (float, optional): New similarity threshold
                for multi-head attention blocks. Defaults to 0.12.
            residual_diff_threshold_single (float, optional): New similarity threshold
                for single-head attention blocks. Defaults to 0.09.

        Example:
            >>> cached_blocks.update_residual_diff_threshold(
            ...     use_double_fb_cache=False,
            ...     residual_diff_threshold_multi=0.15
            ... )
        """
        self.use_double_fb_cache = use_double_fb_cache
        self.residual_diff_threshold_multi = residual_diff_threshold_multi
        self.residual_diff_threshold_single = residual_diff_threshold_single

    def forward(
        self,
        hidden_states: torch.Tensor,
        temb: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        image_rotary_emb: torch.Tensor,
        joint_attention_kwargs=None,
        controlnet_block_samples=None,
        controlnet_single_block_samples=None,
        skip_first_layer=False,
    ):
        """
        Forward pass with advanced caching for Flux transformer blocks.

        This method implements the cached forward pass for Flux models with support
        for both single and double first-block caching strategies. It handles both
        multi-head and single-head attention blocks with independent caching.

        Args:
            hidden_states (torch.Tensor): Input hidden states tensor
            temb (torch.Tensor): Time embedding tensor
            encoder_hidden_states (torch.Tensor): Encoder hidden states tensor
            image_rotary_emb (torch.Tensor): Rotary position embedding for images
            joint_attention_kwargs (dict, optional): Additional attention parameters.
                Defaults to None.
            controlnet_block_samples (list, optional): ControlNet block samples.
                Defaults to None.
            controlnet_single_block_samples (list, optional): ControlNet single block
                samples. Defaults to None.
            skip_first_layer (bool, optional): Whether to skip the first layer.
                Defaults to False.

        Returns:
            torch.Tensor or tuple: Depending on configuration:

            - If return_hidden_states_only=True: hidden_states only
            - If return_hidden_states_first=True: (hidden_states, encoder_hidden_states)
            - Otherwise: (encoder_hidden_states, hidden_states)

        Note:
            If batch_size > 1 or residual_diff_threshold_multi < 0, caching is disabled
            and the original forward pass is used. The method supports both single
            and double first-block caching depending on use_double_fb_cache flag.
        """
        batch_size = hidden_states.shape[0]
        txt_tokens = encoder_hidden_states.shape[1]
        img_tokens = hidden_states.shape[1]

        original_dtype = hidden_states.dtype
        original_device = hidden_states.device

        hidden_states = hidden_states.to(self.dtype).to(original_device)
        encoder_hidden_states = encoder_hidden_states.to(self.dtype).to(original_device)
        temb = temb.to(self.dtype).to(original_device)
        image_rotary_emb = image_rotary_emb.to(original_device)

        if controlnet_block_samples is not None:
            controlnet_block_samples = (
                torch.stack(controlnet_block_samples).to(original_device) if len(controlnet_block_samples) > 0 else None
            )
        if controlnet_single_block_samples is not None:
            controlnet_single_block_samples = (
                torch.stack(controlnet_single_block_samples).to(original_device)
                if len(controlnet_single_block_samples) > 0
                else None
            )

        assert image_rotary_emb.ndim == 6
        assert image_rotary_emb.shape[0] == 1
        assert image_rotary_emb.shape[1] == 1
        # [1, tokens, head_dim/2, 1, 2] (sincos)
        total_tokens = txt_tokens + img_tokens
        assert image_rotary_emb.shape[2] == 1 * total_tokens

        image_rotary_emb = image_rotary_emb.reshape([1, txt_tokens + img_tokens, *image_rotary_emb.shape[3:]])
        rotary_emb_txt = image_rotary_emb[:, :txt_tokens, ...]
        rotary_emb_img = image_rotary_emb[:, txt_tokens:, ...]
        rotary_emb_single = image_rotary_emb

        rotary_emb_txt = self.pack_rotemb(pad_tensor(rotary_emb_txt, 256, 1))
        rotary_emb_img = self.pack_rotemb(pad_tensor(rotary_emb_img, 256, 1))
        rotary_emb_single = self.pack_rotemb(pad_tensor(rotary_emb_single, 256, 1))

        if (self.residual_diff_threshold_multi < 0.0) or (batch_size > 1):
            if batch_size > 1 and self.verbose:
                print("Batch size > 1 currently not supported")

            hidden_states = self.m.forward(
                hidden_states,
                encoder_hidden_states,
                temb,
                rotary_emb_img,
                rotary_emb_txt,
                rotary_emb_single,
                controlnet_block_samples,
                controlnet_single_block_samples,
                skip_first_layer,
            )

            hidden_states = hidden_states.to(original_dtype).to(original_device)

            encoder_hidden_states = hidden_states[:, :txt_tokens, ...]
            hidden_states = hidden_states[:, txt_tokens:, ...]

            if self.return_hidden_states_only:
                return hidden_states
            if self.return_hidden_states_first:
                return hidden_states, encoder_hidden_states
            return encoder_hidden_states, hidden_states

        remaining_kwargs = {
            "temb": temb,
            "rotary_emb_img": rotary_emb_img,
            "rotary_emb_txt": rotary_emb_txt,
            "rotary_emb_single": rotary_emb_single,
            "controlnet_block_samples": controlnet_block_samples,
            "controlnet_single_block_samples": controlnet_single_block_samples,
            "txt_tokens": txt_tokens,
        }

        original_hidden_states = hidden_states
        first_hidden_states, first_encoder_hidden_states = self.m.forward_layer(
            0,
            hidden_states,
            encoder_hidden_states,
            temb,
            rotary_emb_img,
            rotary_emb_txt,
            controlnet_block_samples,
            controlnet_single_block_samples,
        )
        hidden_states = first_hidden_states
        encoder_hidden_states = first_encoder_hidden_states
        first_hidden_states_residual_multi = hidden_states - original_hidden_states
        del original_hidden_states

        if self.use_double_fb_cache:
            call_remaining_fn = self.call_remaining_multi_transformer_blocks
        else:
            call_remaining_fn = self.call_remaining_FBCache_transformer_blocks

        torch._dynamo.graph_break()
        updated_h, updated_enc, threshold = check_and_apply_cache(
            first_residual=first_hidden_states_residual_multi,
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            threshold=self.residual_diff_threshold_multi,
            parallelized=(self.transformer is not None and getattr(self.transformer, "_is_parallelized", False)),
            mode="multi",
            verbose=self.verbose,
            call_remaining_fn=call_remaining_fn,
            remaining_kwargs=remaining_kwargs,
        )
        self.residual_diff_threshold_multi = threshold
        if not self.use_double_fb_cache:
            if self.return_hidden_states_only:
                return updated_h
            if self.return_hidden_states_first:
                return updated_h, updated_enc
            return updated_enc, updated_h

        # DoubleFBCache
        cat_hidden_states = torch.cat([updated_enc, updated_h], dim=1)
        original_cat = cat_hidden_states
        cat_hidden_states = self.m.forward_single_layer(0, cat_hidden_states, temb, rotary_emb_single)

        first_hidden_states_residual_single = cat_hidden_states - original_cat
        del original_cat

        call_remaining_fn_single = self.call_remaining_single_transformer_blocks

        updated_cat, _, threshold = check_and_apply_cache(
            first_residual=first_hidden_states_residual_single,
            hidden_states=cat_hidden_states,
            encoder_hidden_states=None,
            threshold=self.residual_diff_threshold_single,
            parallelized=(self.transformer is not None and getattr(self.transformer, "_is_parallelized", False)),
            mode="single",
            verbose=self.verbose,
            call_remaining_fn=call_remaining_fn_single,
            remaining_kwargs=remaining_kwargs,
        )
        self.residual_diff_threshold_single = threshold

        # torch._dynamo.graph_break()

        final_enc = updated_cat[:, :txt_tokens, ...]
        final_h = updated_cat[:, txt_tokens:, ...]

        final_h = final_h.to(original_dtype).to(original_device)
        final_enc = final_enc.to(original_dtype).to(original_device)

        if self.return_hidden_states_only:
            return final_h
        if self.return_hidden_states_first:
            return final_h, final_enc
        return final_enc, final_h

    def call_remaining_FBCache_transformer_blocks(
        self,
        hidden_states: torch.Tensor,
        temb: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        rotary_emb_img: torch.Tensor,
        rotary_emb_txt: torch.Tensor,
        rotary_emb_single: torch.Tensor,
        controlnet_block_samples=None,
        controlnet_single_block_samples=None,
        skip_first_layer=True,
        txt_tokens=None,
    ):
        """
        Process remaining blocks using single first-block cache strategy.

        This method processes all remaining transformer blocks when using single
        first-block caching (not double caching). It handles both multi-head and
        single-head attention blocks in sequence.

        Args:
            hidden_states (torch.Tensor): Input hidden states
            temb (torch.Tensor): Time embedding tensor
            encoder_hidden_states (torch.Tensor): Encoder hidden states
            rotary_emb_img (torch.Tensor): Image rotary embeddings
            rotary_emb_txt (torch.Tensor): Text rotary embeddings
            rotary_emb_single (torch.Tensor): Single-head rotary embeddings
            controlnet_block_samples (list, optional): ControlNet block samples
            controlnet_single_block_samples (list, optional): ControlNet single block samples
            skip_first_layer (bool, optional): Whether to skip first layer. Defaults to True.
            txt_tokens (int, optional): Number of text tokens

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
                (hidden_states, encoder_hidden_states, hidden_states_residual, enc_residual)
        """
        original_dtype = hidden_states.dtype
        original_device = hidden_states.device
        original_hidden_states = hidden_states
        original_encoder_hidden_states = encoder_hidden_states
        hidden_states = self.m.forward(
            hidden_states,
            encoder_hidden_states,
            temb,
            rotary_emb_img,
            rotary_emb_txt,
            rotary_emb_single,
            controlnet_block_samples,
            controlnet_single_block_samples,
            skip_first_layer,
        )

        hidden_states = hidden_states.to(original_dtype).to(original_device)

        encoder_hidden_states = hidden_states[:, :txt_tokens, ...]
        hidden_states = hidden_states[:, txt_tokens:, ...]

        hidden_states = hidden_states.contiguous()
        encoder_hidden_states = encoder_hidden_states.contiguous()

        hidden_states_residual = hidden_states - original_hidden_states
        enc_residual = encoder_hidden_states - original_encoder_hidden_states

        return hidden_states, encoder_hidden_states, hidden_states_residual, enc_residual

    def call_remaining_multi_transformer_blocks(
        self,
        hidden_states: torch.Tensor,
        temb: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        rotary_emb_img: torch.Tensor,
        rotary_emb_txt: torch.Tensor,
        rotary_emb_single: torch.Tensor,
        controlnet_block_samples=None,
        controlnet_single_block_samples=None,
        skip_first_layer=False,
        txt_tokens=None,
    ):
        """
        Process remaining multi-head attention transformer blocks.

        This method processes the remaining multi-head attention blocks when using
        double first-block caching strategy. It's called after the first block
        cache decision.

        Args:
            hidden_states (torch.Tensor): Input hidden states
            temb (torch.Tensor): Time embedding tensor
            encoder_hidden_states (torch.Tensor): Encoder hidden states
            rotary_emb_img (torch.Tensor): Image rotary embeddings
            rotary_emb_txt (torch.Tensor): Text rotary embeddings
            rotary_emb_single (torch.Tensor): Single-head rotary embeddings
            controlnet_block_samples (list, optional): ControlNet block samples
            controlnet_single_block_samples (list, optional): ControlNet single block samples
            skip_first_layer (bool, optional): Whether to skip first layer. Defaults to False.
            txt_tokens (int, optional): Number of text tokens

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
                (hidden_states, encoder_hidden_states, hidden_states_residual, enc_residual)
        """
        start_idx = 1
        original_hidden_states = hidden_states.clone()
        original_encoder_hidden_states = encoder_hidden_states.clone()

        for idx in range(start_idx, num_transformer_blocks):
            hidden_states, encoder_hidden_states = self.m.forward_layer(
                idx,
                hidden_states,
                encoder_hidden_states,
                temb,
                rotary_emb_img,
                rotary_emb_txt,
                controlnet_block_samples,
                controlnet_single_block_samples,
            )

        hidden_states = hidden_states.contiguous()
        encoder_hidden_states = encoder_hidden_states.contiguous()

        hs_res = hidden_states - original_hidden_states
        enc_res = encoder_hidden_states - original_encoder_hidden_states
        return hidden_states, encoder_hidden_states, hs_res, enc_res

    def call_remaining_single_transformer_blocks(
        self,
        hidden_states: torch.Tensor,
        temb: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        rotary_emb_img: torch.Tensor,
        rotary_emb_txt: torch.Tensor,
        rotary_emb_single: torch.Tensor,
        controlnet_block_samples=None,
        controlnet_single_block_samples=None,
        skip_first_layer=False,
        txt_tokens=None,
    ):
        """
        Process remaining single-head attention transformer blocks.

        This method processes the remaining single-head attention blocks in the
        double first-block caching strategy. It's called after the multi-head
        blocks have been processed.

        Args:
            hidden_states (torch.Tensor): Input hidden states (concatenated)
            temb (torch.Tensor): Time embedding tensor
            encoder_hidden_states (torch.Tensor): Encoder hidden states (unused in single blocks)
            rotary_emb_img (torch.Tensor): Image rotary embeddings (unused in single blocks)
            rotary_emb_txt (torch.Tensor): Text rotary embeddings (unused in single blocks)
            rotary_emb_single (torch.Tensor): Single-head rotary embeddings
            controlnet_block_samples (list, optional): ControlNet block samples (unused)
            controlnet_single_block_samples (list, optional): ControlNet single block samples (unused)
            skip_first_layer (bool, optional): Whether to skip first layer. Defaults to False.
            txt_tokens (int, optional): Number of text tokens (unused)

        Returns:
            tuple[torch.Tensor, torch.Tensor]: (hidden_states, hidden_states_residual)
        """
        start_idx = 1
        original_hidden_states = hidden_states.clone()

        for idx in range(start_idx, num_single_transformer_blocks):
            hidden_states = self.m.forward_single_layer(
                idx,
                hidden_states,
                temb,
                rotary_emb_single,
            )

        hidden_states = hidden_states.contiguous()
        hs_res = hidden_states - original_hidden_states
        return hidden_states, hs_res
