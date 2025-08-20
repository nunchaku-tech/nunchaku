"""
Caching utilities for V2 transformer models.

Implements first-block caching to accelerate transformer inference by reusing computations
when input changes are minimal. Supports Flux V2 architecture with double FB cache.

**Main Classes**

- :class:`CacheContext` : Manages cache buffers and incremental naming.
- :class:`NunchakuCachedFluxTransformer2DModelV2` : Cached transformer for Flux V2 models.

**Key Functions**

- :func:`get_buffer`, :func:`set_buffer` : Cache buffer management.
- :func:`cache_context` : Context manager for cache operations.
- :func:`are_two_tensors_similar` : Tensor similarity check.
- :func:`apply_prev_hidden_states_residual` : Applies cached residuals.
- :func:`get_can_use_cache` : Checks cache usability.
- :func:`check_and_apply_cache` : Main cache logic.

**Caching Strategy**

1. Compute the first transformer block.
2. Compare the residual with the cached residual.
3. If similar, reuse cached results for the remaining blocks; otherwise, recompute and update cache.
4. For double FB cache, repeat the process for single blocks.

.. note::
   V2 implementation with enhanced double FB cache support.
   Adapted from ParaAttention:
   https://github.com/chengzeyi/ParaAttention/src/para_attn/first_block_cache/
"""

import contextlib
import dataclasses
from collections import defaultdict
from typing import Any, DefaultDict, Dict, Optional, Tuple, Union

import torch
from diffusers.models.modeling_outputs import Transformer2DModelOutput

from nunchaku.models.embeddings import pack_rotemb
from nunchaku.models.transformers.transformer_flux_v2 import NunchakuFluxTransformer2DModelV2
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
    def get_buffer(self, name: str) -> Optional[torch.Tensor]:
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

    def set_buffer(self, name: str, buffer: torch.Tensor):
        """
        Store a tensor buffer in the cache.

        Parameters
        ----------
        name : str
            The name to associate with the buffer.
        buffer : torch.Tensor
            The tensor to cache.
        """
        self.buffers[name] = buffer

    def clear_buffers(self):
        """
        Clear all cached tensor buffers.

        Removes all stored tensors from the cache.
        """
        self.buffers.clear()


@torch.compiler.disable
def get_buffer(name: str) -> torch.Tensor:
    """
    Retrieve a cached tensor buffer from the current cache context.

    Parameters
    ----------
    name : str
        The name of the buffer to retrieve.

    Returns
    -------
    torch.Tensor or None
        The cached tensor if found, otherwise None.

    Raises
    ------
    AssertionError
        If no cache context is currently active.

    Examples
    --------
    >>> with cache_context(create_cache_context()):
    ...     set_buffer("my_tensor", torch.randn(2, 3))
    ...     cached = get_buffer("my_tensor")
    """
    cache_context = get_current_cache_context()
    assert cache_context is not None, "cache_context must be set before"
    return cache_context.get_buffer(name)


@torch.compiler.disable
def set_buffer(name: str, buffer: torch.Tensor):
    """
    Store a tensor buffer in the current cache context.

    Parameters
    ----------
    name : str
        The name to associate with the buffer.
    buffer : torch.Tensor
        The tensor to cache.

    Raises
    ------
    AssertionError
        If no cache context is currently active.

    Examples
    --------
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
    Create a new :class:`CacheContext` for managing cached computations.

    Returns
    -------
    CacheContext
        A new cache context instance.

    Examples
    --------
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
    Context manager to set the active cache context.

    Sets the global cache context for the duration of the ``with`` block, restoring the previous context on exit.

    Parameters
    ----------
    cache_context : CacheContext
        The cache context to activate.

    Yields
    ------
    None

    Examples
    --------
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
def are_two_tensors_similar(t1: torch.Tensor, t2: torch.Tensor, *, threshold: float, parallelized: bool = False):
    """
    Check if two tensors are similar based on relative L1 distance.

    The relative distance is computed as
    ``mean(abs(t1 - t2)) / mean(abs(t1))`` and compared to ``threshold``.

    Parameters
    ----------
    t1 : torch.Tensor
        First tensor.
    t2 : torch.Tensor
        Second tensor.
    threshold : float
        Similarity threshold. Tensors are similar if relative distance < threshold.
    parallelized : bool, optional
        Unused. For API compatibility.

    Returns
    -------
    tuple of (bool, float)
        - bool: True if tensors are similar, False otherwise.
        - float: The computed relative L1 distance.
    """
    mean_diff = (t1 - t2).abs().mean()
    mean_t1 = t1.abs().mean()
    diff = (mean_diff / mean_t1).item()
    return diff < threshold, diff


@torch.compiler.disable
def apply_prev_hidden_states_residual(
    hidden_states: torch.Tensor,
    encoder_hidden_states: torch.Tensor | None = None,
    mode: str = "multi",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply cached residuals to hidden states.

    Parameters
    ----------
    hidden_states : torch.Tensor
        Current hidden states.
    encoder_hidden_states : torch.Tensor, optional
        Encoder hidden states (required for ``mode="multi"``).
    mode : {"multi", "single"}, default: "multi"
        Whether to apply residuals for Flux double blocks or single blocks.

    Returns
    -------
    tuple or torch.Tensor
        - If ``mode="multi"``: (updated_hidden_states, updated_encoder_hidden_states)
        - If ``mode="single"``: updated_hidden_states

    Raises
    ------
    AssertionError
        If required cached residuals are not found.
    ValueError
        If mode is not "multi" or "single".
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
    Check if cached computations can be reused based on residual similarity.

    Parameters
    ----------
    first_hidden_states_residual : torch.Tensor
        Current first block residual.
    threshold : float
        Similarity threshold for cache validity.
    parallelized : bool, optional
        Whether computation is parallelized. Default is False.
    mode : {"multi", "single"}, optional
        Caching mode. Default is "multi".

    Returns
    -------
    tuple of (bool, float)
        - bool: True if cache can be used (residuals are similar), False otherwise.
        - float: The computed similarity difference, or threshold if no cache exists.

    Raises
    ------
    ValueError
        If mode is not "multi" or "single".
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
    Check and apply cache based on residual similarity.

    This function determines whether cached results can be used by comparing the
    first block residuals. If the cache is valid, it applies cached computations;
    otherwise, it computes new values and updates the cache.

    Parameters
    ----------
    first_residual : torch.Tensor
        First block residual for similarity comparison.
    hidden_states : torch.Tensor
        Current hidden states.
    encoder_hidden_states : torch.Tensor, optional
        Encoder hidden states (required for "multi" mode).
    threshold : float
        Similarity threshold for cache validity.
    parallelized : bool
        Whether computation is parallelized.
    mode : {"multi", "single"}
        Caching mode.
    verbose : bool
        Whether to print caching status messages.
    call_remaining_fn : callable
        Function to call remaining transformer blocks.
    remaining_kwargs : dict
        Additional keyword arguments for `call_remaining_fn`.

    Returns
    -------
    tuple
        (updated_hidden_states, updated_encoder_hidden_states, threshold)
        - updated_hidden_states (torch.Tensor)
        - updated_encoder_hidden_states (torch.Tensor or None)
        - threshold (float)
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


class NunchakuCachedFluxTransformer2DModelV2(NunchakuFluxTransformer2DModelV2):
    """
    Cached Flux V2 transformer with double first-block caching support.

    This class extends the base Flux V2 transformer to add caching capabilities
    for both multi-head and single-head transformer blocks, significantly improving
    inference speed for similar consecutive inputs.
    """

    def __init__(
        self,
        use_double_fb_cache: bool = False,
        residual_diff_threshold: float = 0.12,
        residual_diff_threshold_multi: Optional[float] = None,
        residual_diff_threshold_single: Optional[float] = None,
        verbose: bool = False,
        **components,
    ):
        """
        Initialize the cached Flux V2 transformer model.

        Two ways to use this:
        1. Pass all components as keyword arguments:
           NunchakuCachedFluxTransformer2DModelV2(
               x_embedder=transformer.x_embedder,
               time_text_embed=transformer.time_text_embed,
               ...
           )

        2. Use from_transformer class method:
           NunchakuCachedFluxTransformer2DModelV2.from_transformer(transformer)

        Parameters
        ----------
        use_double_fb_cache : bool, optional
            If True, cache both multi-head and single-head attention blocks (default: False).
        residual_diff_threshold : float, optional
            Default similarity threshold for caching (default: 0.12).
        residual_diff_threshold_multi : float, optional
            Threshold for multi-head (double) blocks. If None, uses `residual_diff_threshold`.
        residual_diff_threshold_single : float, optional
            Threshold for single-head blocks. If None, disables single block caching.
        verbose : bool, optional
            If True, print cache hit/miss information (default: False).
        **components
            Model components (x_embedder, time_text_embed, etc.)
        """
        # Initialize nn.Module properly with all necessary internal structures
        torch.nn.Module.__init__(self)

        # Initialize the internal module dictionaries that PyTorch needs for proper memory management
        # This ensures efficient tensor handling without calling the expensive parent __init__
        self._modules = {}
        self._parameters = {}
        self._buffers = {}
        self._non_persistent_buffers_set = set()
        self._backward_hooks = {}
        self._forward_hooks = {}
        self._forward_pre_hooks = {}
        self._state_dict_hooks = {}
        self._load_state_dict_pre_hooks = {}

        # Set all passed components as attributes using proper module registration
        for key, value in components.items():
            if isinstance(value, torch.nn.Module):
                self._modules[key] = value
            else:
                setattr(self, key, value)

        # Verify required components are present
        required_components = [
            "x_embedder",
            "time_text_embed",
            "context_embedder",
            "pos_embed",
            "transformer_blocks",
            "single_transformer_blocks",
            "norm_out",
        ]

        for component in required_components:
            if not hasattr(self, component):
                raise ValueError(f"Missing required component: {component}")

        # Set optional components with defaults if not provided
        if not hasattr(self, "proj_out"):
            if hasattr(self.norm_out, "linear"):
                self.proj_out = self.norm_out.linear
            else:
                self.proj_out = None

        if not hasattr(self, "encoder_hid_proj"):
            self.encoder_hid_proj = None

        # Get inner_dim from pos_embed if available
        if hasattr(self.pos_embed, "dim"):
            self.inner_dim = self.pos_embed.dim
        elif hasattr(self.pos_embed, "inner_dim"):
            self.inner_dim = self.pos_embed.inner_dim

        # Set caching-specific attributes
        self.use_double_fb_cache = use_double_fb_cache
        self.verbose = verbose

        # Set residual difference thresholds
        if residual_diff_threshold_multi is None:
            residual_diff_threshold_multi = residual_diff_threshold

        self.residual_diff_threshold = residual_diff_threshold
        self.residual_diff_threshold_multi = residual_diff_threshold_multi
        self.residual_diff_threshold_single = (
            residual_diff_threshold_single if residual_diff_threshold_single is not None else -1.0
        )

        # Initialize cache-related flags
        self._is_cached = True
        self._original_forward = None
        self._original_blocks = None

    @classmethod
    def from_transformer(
        cls,
        transformer: NunchakuFluxTransformer2DModelV2,
        use_double_fb_cache: bool = False,
        residual_diff_threshold: float = 0.12,
        residual_diff_threshold_multi: Optional[float] = None,
        residual_diff_threshold_single: Optional[float] = None,
        verbose: bool = False,
    ):
        """
        Create a cached transformer from an existing transformer instance.
        This is a convenience method that automatically extracts all components.

        Parameters
        ----------
        transformer : NunchakuFluxTransformer2DModelV2
            The existing transformer model to wrap with caching.
        use_double_fb_cache : bool, optional
            If True, cache both multi-head and single-head attention blocks (default: False).
        residual_diff_threshold : float, optional
            Default similarity threshold for caching (default: 0.12).
        residual_diff_threshold_multi : float, optional
            Threshold for multi-head (double) blocks. If None, uses `residual_diff_threshold`.
        residual_diff_threshold_single : float, optional
            Threshold for single-head blocks. If None, disables single block caching.
        verbose : bool, optional
            If True, print cache hit/miss information (default: False).

        Returns
        -------
        NunchakuCachedFluxTransformer2DModelV2
            The transformer with caching enabled.
        """
        # Extract all relevant components from transformer
        components = {}

        # Get all model components
        for attr_name in dir(transformer):
            attr = getattr(transformer, attr_name)
            # Skip private attributes and methods
            if not attr_name.startswith("_") and not callable(attr):
                # Keep nn.Module attributes and important non-module attributes
                if isinstance(attr, torch.nn.Module) or attr_name in ["inner_dim", "dtype"]:
                    components[attr_name] = attr

        return cls(
            use_double_fb_cache=use_double_fb_cache,
            residual_diff_threshold=residual_diff_threshold,
            residual_diff_threshold_multi=residual_diff_threshold_multi,
            residual_diff_threshold_single=residual_diff_threshold_single,
            verbose=verbose,
            **components,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor = None,
        pooled_projections: torch.Tensor = None,
        timestep: torch.LongTensor = None,
        img_ids: torch.Tensor = None,
        txt_ids: torch.Tensor = None,
        guidance: torch.Tensor = None,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        controlnet_block_samples=None,
        controlnet_single_block_samples=None,
        return_dict: bool = True,
        controlnet_blocks_repeat: bool = False,
    ) -> Union[torch.Tensor, Transformer2DModelOutput]:
        """
        Forward pass with V2 caching support.

        Implements the cached forward pass with optional double FB cache for both
        multi-head and single-head transformer blocks.

        Parameters
        ----------
        hidden_states : torch.Tensor
            Input hidden states of shape `(batch_size, image_sequence_length, in_channels)`.
        encoder_hidden_states : torch.Tensor, optional
            Conditional embeddings of shape `(batch_size, text_sequence_length, joint_attention_dim)`.
        pooled_projections : torch.Tensor, optional
            Pooled embeddings of shape `(batch_size, projection_dim)`.
        timestep : torch.LongTensor, optional
            Denoising timestep.
        img_ids : torch.Tensor, optional
            Image position IDs.
        txt_ids : torch.Tensor, optional
            Text position IDs.
        guidance : torch.Tensor, optional
            Guidance scale tensor.
        joint_attention_kwargs : dict, optional
            Additional keyword arguments for attention processor.
        controlnet_block_samples : list, optional
            ControlNet block samples (not yet supported in V2).
        controlnet_single_block_samples : list, optional
            ControlNet single block samples (not yet supported in V2).
        return_dict : bool, optional
            Whether to return a Transformer2DModelOutput (default: True).
        controlnet_blocks_repeat : bool, optional
            ControlNet blocks repeat flag.

        Returns
        -------
        Union[torch.Tensor, Transformer2DModelOutput]
            Output tensor or Transformer2DModelOutput if return_dict is True.

        Notes
        -----
        When caching is enabled (residual_diff_threshold_multi >= 0), the forward pass
        uses first-block caching to potentially skip computation of remaining blocks
        when inputs are similar to cached values.
        """
        hidden_states = self.x_embedder(hidden_states)

        timestep = timestep.to(hidden_states.dtype) * 1000
        if guidance is not None:
            guidance = guidance.to(hidden_states.dtype) * 1000

        temb = (
            self.time_text_embed(timestep, pooled_projections)
            if guidance is None
            else self.time_text_embed(timestep, guidance, pooled_projections)
        )
        encoder_hidden_states = self.context_embedder(encoder_hidden_states)

        if txt_ids.ndim == 3:
            txt_ids = txt_ids[0]
        if img_ids.ndim == 3:
            img_ids = img_ids[0]

        ids = torch.cat((txt_ids, img_ids), dim=0)
        image_rotary_emb = self.pos_embed(ids)

        if joint_attention_kwargs is not None and "ip_adapter_image_embeds" in joint_attention_kwargs:
            ip_adapter_image_embeds = joint_attention_kwargs.pop("ip_adapter_image_embeds")
            ip_hidden_states = self.encoder_hid_proj(ip_adapter_image_embeds)
            joint_attention_kwargs.update({"ip_hidden_states": ip_hidden_states})

        txt_tokens = encoder_hidden_states.shape[1]
        img_tokens = hidden_states.shape[1]

        assert image_rotary_emb.ndim == 6
        assert image_rotary_emb.shape[0] == 1
        assert image_rotary_emb.shape[1] == 1
        assert image_rotary_emb.shape[2] == 1 * (txt_tokens + img_tokens)
        # [1, tokens, head_dim / 2, 1, 2] (sincos)
        image_rotary_emb = image_rotary_emb.reshape([1, txt_tokens + img_tokens, *image_rotary_emb.shape[3:]])
        rotary_emb_txt = image_rotary_emb[:, :txt_tokens, ...]  # .to(self.dtype)
        rotary_emb_img = image_rotary_emb[:, txt_tokens:, ...]  # .to(self.dtype)
        rotary_emb_single = image_rotary_emb

        rotary_emb_txt = pack_rotemb(pad_tensor(rotary_emb_txt, 256, 1))
        rotary_emb_img = pack_rotemb(pad_tensor(rotary_emb_img, 256, 1))
        rotary_emb_single = pack_rotemb(pad_tensor(rotary_emb_single, 256, 1))

        if self.residual_diff_threshold_multi < 0.0:
            for index_block, block in enumerate(self.transformer_blocks):
                # Note: NunchakuFluxTransformerBlock returns (encoder_hidden_states, hidden_states)
                encoder_hidden_states, hidden_states = block(
                    hidden_states=hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    temb=temb,
                    image_rotary_emb=(rotary_emb_img, rotary_emb_txt),
                    joint_attention_kwargs=joint_attention_kwargs,
                )

                # controlnet residual
                if controlnet_block_samples is not None:
                    raise NotImplementedError("Controlnet is not supported for FluxTransformer2DModelV2 for now")

            hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)
            for index_block, block in enumerate(self.single_transformer_blocks):
                hidden_states = block(
                    hidden_states=hidden_states,
                    temb=temb,
                    image_rotary_emb=rotary_emb_single,
                    joint_attention_kwargs=joint_attention_kwargs,
                )

                # controlnet residual
                if controlnet_single_block_samples is not None:
                    raise NotImplementedError("Controlnet is not supported for FluxTransformer2DModelV2 for now")
        else:
            # FBCache
            remaining_kwargs = {
                "temb": temb,
                "rotary_emb_img": rotary_emb_img,
                "rotary_emb_txt": rotary_emb_txt,
                "rotary_emb_single": rotary_emb_single,
                "controlnet_block_samples": controlnet_block_samples,
                "controlnet_single_block_samples": controlnet_single_block_samples,
                "joint_attention_kwargs": joint_attention_kwargs,
                "txt_tokens": txt_tokens,
            }
            original_hidden_states = hidden_states

            # Note: NunchakuFluxTransformerBlock returns (encoder_hidden_states, hidden_states)
            first_encoder_hidden_states, first_hidden_states = self.transformer_blocks[0](
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                temb=temb,
                image_rotary_emb=(rotary_emb_img, rotary_emb_txt),
                joint_attention_kwargs=joint_attention_kwargs,
            )
            hidden_states = first_hidden_states
            encoder_hidden_states = first_encoder_hidden_states
            first_hidden_states_residual_multi = hidden_states - original_hidden_states
            del original_hidden_states

            # Determine which function to call based on double FB cache setting
            if self.use_double_fb_cache:
                call_remaining_fn = self.call_remaining_multi_blocks
            else:
                call_remaining_fn = self.call_remaining_blocks

            torch._dynamo.graph_break()
            hidden_states, encoder_hidden_states, _ = check_and_apply_cache(
                first_residual=first_hidden_states_residual_multi,
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                threshold=self.residual_diff_threshold_multi,
                parallelized=False,
                mode="multi",
                verbose=self.verbose,
                call_remaining_fn=lambda hidden_states, encoder_hidden_states, **kw: call_remaining_fn(
                    hidden_states, encoder_hidden_states, **remaining_kwargs
                ),
                remaining_kwargs={},
            )

            if self.use_double_fb_cache:
                # Second stage caching for single blocks
                hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)
                original_cat = hidden_states

                # Process first single block
                first_block = self.single_transformer_blocks[0]
                hidden_states = first_block(
                    hidden_states=hidden_states,
                    temb=temb,
                    image_rotary_emb=rotary_emb_single,
                    joint_attention_kwargs=joint_attention_kwargs,
                )
                first_hidden_states_residual_single = hidden_states - original_cat
                del original_cat

                original_dtype = hidden_states.dtype
                original_device = hidden_states.device

                hidden_states, _, _ = check_and_apply_cache(
                    first_residual=first_hidden_states_residual_single,
                    hidden_states=hidden_states,
                    encoder_hidden_states=None,
                    threshold=self.residual_diff_threshold_single,
                    parallelized=False,
                    mode="single",
                    verbose=self.verbose,
                    call_remaining_fn=lambda hidden_states, encoder_hidden_states, **kw: self.call_remaining_single_blocks(
                        hidden_states, **remaining_kwargs
                    ),
                    remaining_kwargs={},
                )

                hidden_states = hidden_states.to(original_dtype).to(original_device)
                hidden_states = hidden_states[:, txt_tokens:, ...]

        hidden_states = self.norm_out(hidden_states, temb)
        output = self.proj_out(hidden_states)

        if not return_dict:
            return (output,)

        return Transformer2DModelOutput(sample=output)

    def call_remaining_blocks(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb: torch.Tensor,
        rotary_emb_img: torch.Tensor,
        rotary_emb_txt: torch.Tensor,
        rotary_emb_single: torch.Tensor,
        controlnet_block_samples,
        controlnet_single_block_samples,
        joint_attention_kwargs,
        txt_tokens,
    ):
        """
        Process remaining transformer blocks (both multi and single).

        Called when cache is invalid. Processes all blocks after the first one.

        Parameters
        ----------
        hidden_states : torch.Tensor
            Input hidden states.
        encoder_hidden_states : torch.Tensor
            Encoder hidden states.
        temb : torch.Tensor
            Time embedding tensor.
        rotary_emb_img : torch.Tensor
            Image rotary embeddings.
        rotary_emb_txt : torch.Tensor
            Text rotary embeddings.
        rotary_emb_single : torch.Tensor
            Single-head rotary embeddings.
        controlnet_block_samples : list, optional
            ControlNet block samples.
        controlnet_single_block_samples : list, optional
            ControlNet single block samples.
        joint_attention_kwargs : dict, optional
            Joint attention kwargs.
        txt_tokens : int
            Number of text tokens.

        Returns
        -------
        tuple
            (updated_hidden_states, updated_encoder_hidden_states,
             hidden_states_residual, encoder_hidden_states_residual)
        """
        original_dtype = hidden_states.dtype
        original_device = hidden_states.device

        # Store original states for residual calculation
        original_hidden_states = hidden_states  # .clone()
        original_encoder_hidden_states = encoder_hidden_states  # .clone()

        for index_block, block in enumerate(self.transformer_blocks[1:]):
            # Note: NunchakuFluxTransformerBlock returns (encoder_hidden_states, hidden_states)
            encoder_hidden_states, hidden_states = block(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                temb=temb,
                image_rotary_emb=(rotary_emb_img, rotary_emb_txt),
                joint_attention_kwargs=joint_attention_kwargs,
            )
            # controlnet residual
            if controlnet_block_samples is not None:
                raise NotImplementedError("Controlnet is not supported for FluxTransformer2DModelV2 for now")

        hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)

        # Process all single blocks
        for block in self.single_transformer_blocks:
            hidden_states = block(
                hidden_states=hidden_states,
                temb=temb,
                image_rotary_emb=rotary_emb_single,
                joint_attention_kwargs=joint_attention_kwargs,
            )

        # Restore original dtype and device
        hidden_states = hidden_states.to(original_dtype).to(original_device)

        # Split concatenated result
        encoder_hidden_states = hidden_states[:, :txt_tokens, ...]
        hidden_states = hidden_states[:, txt_tokens:, ...]

        # Ensure contiguous memory layout
        hidden_states = hidden_states.contiguous()
        encoder_hidden_states = encoder_hidden_states.contiguous()

        # Calculate residuals
        hs_residual = hidden_states - original_hidden_states
        enc_residual = encoder_hidden_states - original_encoder_hidden_states

        return hidden_states, encoder_hidden_states, hs_residual, enc_residual

    def call_remaining_multi_blocks(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb: torch.Tensor,
        rotary_emb_img: torch.Tensor,
        rotary_emb_txt: torch.Tensor,
        rotary_emb_single: torch.Tensor,
        joint_attention_kwargs,
        txt_tokens,
        **kwargs,
    ):
        """
        Process remaining multi-head transformer blocks only.

        Used when double FB cache is enabled. Skips single blocks.

        Parameters
        ----------
        hidden_states : torch.Tensor
            Input hidden states.
        encoder_hidden_states : torch.Tensor
            Encoder hidden states.
        temb : torch.Tensor
            Time embedding tensor.
        rotary_emb_img : torch.Tensor
            Image rotary embeddings.
        rotary_emb_txt : torch.Tensor
            Text rotary embeddings.
        rotary_emb_single : torch.Tensor
            Single-head rotary embeddings.
        joint_attention_kwargs : dict
            Joint attention kwargs.
        txt_tokens : int
            Number of text tokens.

        Returns
        -------
        tuple
            (updated_hidden_states, updated_encoder_hidden_states,
             hidden_states_residual, encoder_hidden_states_residual)
        """
        original_h = hidden_states
        original_enc = encoder_hidden_states

        # Process remaining multi blocks
        for block in self.transformer_blocks[1:]:
            encoder_hidden_states, hidden_states = block(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                temb=temb,
                image_rotary_emb=(rotary_emb_img, rotary_emb_txt),
                joint_attention_kwargs=joint_attention_kwargs,
            )

        # Ensure contiguous memory layout
        hidden_states = hidden_states.contiguous()
        encoder_hidden_states = encoder_hidden_states.contiguous()

        # Calculate residuals
        hs_residual = hidden_states - original_h
        enc_residual = encoder_hidden_states - original_enc

        return hidden_states, encoder_hidden_states, hs_residual, enc_residual

    def call_remaining_single_blocks(
        self,
        hidden_states: torch.Tensor,
        temb: torch.Tensor = None,
        rotary_emb_single: torch.Tensor = None,
        joint_attention_kwargs=None,
        **kwargs,
    ):
        """
        Process remaining single-head transformer blocks.

        Used for second stage of double FB cache.

        Parameters
        ----------
        hidden_states : torch.Tensor
            Input hidden states (concatenated encoder and decoder).
        temb : torch.Tensor, optional
            Time embedding tensor.
        rotary_emb_single : torch.Tensor, optional
            Single-head rotary embeddings.
        joint_attention_kwargs : dict, optional
            Joint attention kwargs.

        Returns
        -------
        tuple
            (updated_hidden_states, hidden_states_residual)
        """
        # Save original for residual calculation
        original_hidden_states = hidden_states.clone()

        # Process remaining single blocks (skip first)
        for block in self.single_transformer_blocks[1:]:
            hidden_states = block(
                hidden_states=hidden_states,
                temb=temb,
                image_rotary_emb=rotary_emb_single,
                joint_attention_kwargs=joint_attention_kwargs,
            )

        hidden_states = hidden_states.contiguous()
        hs_residual = hidden_states - original_hidden_states

        return hidden_states, hs_residual
