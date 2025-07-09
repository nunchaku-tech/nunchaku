"""
TeaCache: Temporal Embedding Analysis Caching for Flux Transformers.

This module implements TeaCache, a temporal caching mechanism that optimizes
transformer model inference by skipping computation when input changes are
below a threshold. The approach is based on Temporal Embedding Analysis (TEA)
that tracks the relative L1 distance of modulated inputs across timesteps.

The TeaCache system works by:
1. Analyzing the modulated input from the first transformer block
2. Computing a relative L1 distance compared to the previous timestep
3. Applying a rescaling function to the distance metric
4. Skipping transformer computation when accumulated distance is below threshold
5. Reusing previous residual computations for efficiency

Key Components:
    TeaCache: Context manager for applying temporal caching to transformer models
    make_teacache_forward: Factory function that creates a cached forward method

The caching strategy is particularly effective for diffusion models during
inference where consecutive timesteps often have similar inputs, allowing
significant computational savings without meaningful quality loss.

Example:
    Basic usage with a Flux transformer::

        from nunchaku.caching.teacache import TeaCache
        from diffusers import FluxTransformer2DModel

        model = FluxTransformer2DModel.from_pretrained("black-forest-labs/FLUX.1-dev")
        
        with TeaCache(model, num_steps=50, rel_l1_thresh=0.6, skip_steps=10):
            # Model forward passes will use temporal caching
            for step in range(50):
                output = model(inputs_for_step)

Note:
    The rescaling function uses polynomial coefficients optimized for Flux models:
    [4.98651651e02, -2.83781631e02, 5.58554382e01, -3.82021401e00, 2.64230861e-01]
"""

from types import MethodType
from typing import Any, Callable, Optional, Union

import numpy as np
import torch
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.models.transformers.transformer_flux import FluxTransformer2DModel
from diffusers.utils import logging
from diffusers.utils.constants import USE_PEFT_BACKEND
from diffusers.utils.import_utils import is_torch_version
from diffusers.utils.peft_utils import scale_lora_layers, unscale_lora_layers

from ..models.transformers import NunchakuFluxTransformer2dModel

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


def make_teacache_forward(num_steps: int = 50, rel_l1_thresh: float = 0.6, skip_steps: int = 0) -> Callable:
    """
    Create a cached forward method for Flux transformers using TeaCache.
    
    This factory function creates a modified forward method that implements temporal
    caching based on the relative L1 distance of modulated inputs. The caching
    decision is made by analyzing the first transformer block's modulated input
    and comparing it to the previous timestep.
    
    Args:
        num_steps (int, optional): Total number of inference steps. Used to determine
            when to reset the counter. Defaults to 50.
        rel_l1_thresh (float, optional): Relative L1 distance threshold for caching.
            Lower values mean more aggressive caching. Defaults to 0.6.
        skip_steps (int, optional): Number of initial steps to skip caching.
            Useful for allowing the model to stabilize. Defaults to 0.
            
    Returns:
        Callable: A cached forward method that can be bound to a transformer model
        
    Example:
        >>> model = FluxTransformer2DModel.from_pretrained("model_name")
        >>> cached_forward = make_teacache_forward(num_steps=50, rel_l1_thresh=0.6)
        >>> model.forward = cached_forward.__get__(model, type(model))
        
    Note:
        The rescaling function uses polynomial coefficients optimized for Flux models.
        The accumulated distance is reset when it exceeds the threshold or at the
        beginning/end of the inference sequence.
    """
    def teacache_forward(
        self: Union[FluxTransformer2DModel, NunchakuFluxTransformer2dModel],
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        pooled_projections: torch.Tensor,
        timestep: torch.LongTensor,
        img_ids: torch.Tensor,
        txt_ids: torch.Tensor,
        guidance: torch.Tensor,
        joint_attention_kwargs: Optional[dict[str, Any]] = None,
        controlnet_block_samples: Optional[torch.Tensor] = None,
        controlnet_single_block_samples: Optional[torch.Tensor] = None,
        return_dict: bool = True,
        controlnet_blocks_repeat: bool = False,
    ) -> Union[torch.FloatTensor, Transformer2DModelOutput]:
        """
        The [`FluxTransformer2DModel`] forward method.

        Args:
            hidden_states (`torch.FloatTensor` of shape `(batch size, channel, height, width)`):
                Input `hidden_states`.
            encoder_hidden_states (`torch.FloatTensor` of shape `(batch size, sequence_len, embed_dims)`):
                Conditional embeddings (embeddings computed from the input conditions such as prompts) to use.
            pooled_projections (`torch.FloatTensor` of shape `(batch_size, projection_dim)`): Embeddings projected
                from the embeddings of input conditions.
            timestep ( `torch.LongTensor`):
                Used to indicate denoising step.
            block_controlnet_hidden_states: (`list` of `torch.Tensor`):
                A list of tensors that if specified are added to the residuals of transformer blocks.
            joint_attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
                `self.processor` in
                [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~models.transformer_2d.Transformer2DModelOutput`] instead of a plain
                tuple.

        Returns:
            If `return_dict` is True, an [`~models.transformer_2d.Transformer2DModelOutput`] is returned, otherwise a
            `tuple` where the first element is the sample tensor.
        """
        if joint_attention_kwargs is not None:
            joint_attention_kwargs = joint_attention_kwargs.copy()
            lora_scale = joint_attention_kwargs.pop("scale", 1.0)
        else:
            lora_scale = 1.0

        if USE_PEFT_BACKEND:
            # weight the lora layers by setting `lora_scale` for each PEFT layer
            scale_lora_layers(self, lora_scale)
        else:
            if joint_attention_kwargs is not None and joint_attention_kwargs.get("scale", None) is not None:
                logger.warning(
                    "Passing `scale` via `joint_attention_kwargs` when not using the PEFT backend is ineffective."
                )

        hidden_states = self.x_embedder(hidden_states)

        timestep = timestep.to(hidden_states.dtype) * 1000  # type: ignore
        if guidance is not None:
            guidance = guidance.to(hidden_states.dtype) * 1000
        else:
            guidance = None

        temb = (
            self.time_text_embed(timestep, pooled_projections)
            if guidance is None
            else self.time_text_embed(timestep, guidance, pooled_projections)
        )
        encoder_hidden_states = self.context_embedder(encoder_hidden_states)

        if txt_ids.ndim == 3:
            logger.warning(
                "Passing `txt_ids` 3d torch.Tensor is deprecated."
                "Please remove the batch dimension and pass it as a 2d torch Tensor"
            )
            txt_ids = txt_ids[0]
        if img_ids.ndim == 3:
            logger.warning(
                "Passing `img_ids` 3d torch.Tensor is deprecated."
                "Please remove the batch dimension and pass it as a 2d torch Tensor"
            )
            img_ids = img_ids[0]

        ids = torch.cat((txt_ids, img_ids), dim=0)
        image_rotary_emb = self.pos_embed(ids)

        if joint_attention_kwargs is not None and "ip_adapter_image_embeds" in joint_attention_kwargs:
            ip_adapter_image_embeds = joint_attention_kwargs.pop("ip_adapter_image_embeds")
            ip_hidden_states = self.encoder_hid_proj(ip_adapter_image_embeds)  # type: ignore
            joint_attention_kwargs.update({"ip_hidden_states": ip_hidden_states})

        inp = hidden_states.clone()
        temb_ = temb.clone()
        modulated_inp, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.transformer_blocks[0].norm1(inp, emb=temb_)  # type: ignore
        if self.cnt == 0 or self.cnt == num_steps - 1:
            should_calc = True
            self.accumulated_rel_l1_distance = 0
        else:
            coefficients = [
                4.98651651e02,
                -2.83781631e02,
                5.58554382e01,
                -3.82021401e00,
                2.64230861e-01,
            ]
            rescale_func = np.poly1d(coefficients)
            self.accumulated_rel_l1_distance += rescale_func(
                (
                    (modulated_inp - self.previous_modulated_input).abs().mean()
                    / self.previous_modulated_input.abs().mean()
                )
                .cpu()
                .item()
            )
            if self.accumulated_rel_l1_distance < rel_l1_thresh:
                should_calc = False
            else:
                should_calc = True
                self.accumulated_rel_l1_distance = 0
        self.previous_modulated_input = modulated_inp
        self.cnt += 1
        if self.cnt == num_steps:
            self.cnt = 0

        ckpt_kwargs: dict[str, Any]

        if self.cnt > skip_steps:
            if not should_calc:
                hidden_states += self.previous_residual
            else:
                ori_hidden_states = hidden_states.clone()
                for index_block, block in enumerate(self.transformer_blocks):
                    if torch.is_grad_enabled() and self.gradient_checkpointing:

                        def create_custom_forward(module, return_dict=None):  # type: ignore
                            def custom_forward(*inputs):  # type: ignore
                                if return_dict is not None:
                                    return module(*inputs, return_dict=return_dict)
                                else:
                                    return module(*inputs)

                            return custom_forward

                        ckpt_kwargs = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
                        encoder_hidden_states, hidden_states = torch.utils.checkpoint.checkpoint(
                            create_custom_forward(block),
                            hidden_states,
                            encoder_hidden_states,
                            temb,
                            image_rotary_emb,
                            **ckpt_kwargs,
                        )

                    else:
                        encoder_hidden_states, hidden_states = block(
                            hidden_states=hidden_states,
                            encoder_hidden_states=encoder_hidden_states,
                            temb=temb,
                            image_rotary_emb=image_rotary_emb,
                            joint_attention_kwargs=joint_attention_kwargs,
                        )

                    # controlnet residual
                    if controlnet_block_samples is not None:
                        interval_control = len(self.transformer_blocks) / len(controlnet_block_samples)
                        interval_control = int(np.ceil(interval_control))
                        # For Xlabs ControlNet.
                        if controlnet_blocks_repeat:
                            hidden_states = (
                                hidden_states + controlnet_block_samples[index_block % len(controlnet_block_samples)]
                            )
                        else:
                            hidden_states = hidden_states + controlnet_block_samples[index_block // interval_control]
                hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)

                for index_block, block in enumerate(self.single_transformer_blocks):
                    if torch.is_grad_enabled() and self.gradient_checkpointing:

                        def create_custom_forward(module, return_dict=None):  # type: ignore
                            def custom_forward(*inputs):  # type: ignore
                                if return_dict is not None:
                                    return module(*inputs, return_dict=return_dict)
                                else:
                                    return module(*inputs)

                            return custom_forward

                        ckpt_kwargs = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
                        hidden_states = torch.utils.checkpoint.checkpoint(
                            create_custom_forward(block),
                            hidden_states,
                            temb,
                            image_rotary_emb,
                            **ckpt_kwargs,
                        )

                    else:
                        hidden_states = block(
                            hidden_states=hidden_states,
                            temb=temb,
                            image_rotary_emb=image_rotary_emb,
                            joint_attention_kwargs=joint_attention_kwargs,
                        )

                    # controlnet residual
                    if controlnet_single_block_samples is not None:
                        interval_control = len(self.single_transformer_blocks) / len(controlnet_single_block_samples)
                        interval_control = int(np.ceil(interval_control))
                        hidden_states[:, encoder_hidden_states.shape[1] :, ...] = (
                            hidden_states[:, encoder_hidden_states.shape[1] :, ...]
                            + controlnet_single_block_samples[index_block // interval_control]
                        )

                hidden_states = hidden_states[:, encoder_hidden_states.shape[1] :, ...]
                self.previous_residual = hidden_states - ori_hidden_states
        else:
            for index_block, block in enumerate(self.transformer_blocks):
                if torch.is_grad_enabled() and self.gradient_checkpointing:

                    def create_custom_forward(module, return_dict=None):  # type: ignore
                        def custom_forward(*inputs):  # type: ignore
                            if return_dict is not None:
                                return module(*inputs, return_dict=return_dict)
                            else:
                                return module(*inputs)

                        return custom_forward

                    ckpt_kwargs = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
                    encoder_hidden_states, hidden_states = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(block),
                        hidden_states,
                        encoder_hidden_states,
                        temb,
                        image_rotary_emb,
                        **ckpt_kwargs,
                    )

                else:
                    encoder_hidden_states, hidden_states = block(
                        hidden_states=hidden_states,
                        encoder_hidden_states=encoder_hidden_states,
                        temb=temb,
                        image_rotary_emb=image_rotary_emb,
                        joint_attention_kwargs=joint_attention_kwargs,
                    )

                # controlnet residual
                if controlnet_block_samples is not None:
                    interval_control = len(self.transformer_blocks) / len(controlnet_block_samples)
                    interval_control = int(np.ceil(interval_control))
                    # For Xlabs ControlNet.
                    if controlnet_blocks_repeat:
                        hidden_states = (
                            hidden_states + controlnet_block_samples[index_block % len(controlnet_block_samples)]
                        )
                    else:
                        hidden_states = hidden_states + controlnet_block_samples[index_block // interval_control]
            hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)

            for index_block, block in enumerate(self.single_transformer_blocks):
                if torch.is_grad_enabled() and self.gradient_checkpointing:

                    def create_custom_forward(module, return_dict=None):  # type: ignore
                        def custom_forward(*inputs):  # type: ignore
                            if return_dict is not None:
                                return module(*inputs, return_dict=return_dict)
                            else:
                                return module(*inputs)

                        return custom_forward

                    ckpt_kwargs = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
                    hidden_states = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(block),
                        hidden_states,
                        temb,
                        image_rotary_emb,
                        **ckpt_kwargs,
                    )

                else:
                    hidden_states = block(
                        hidden_states=hidden_states,
                        temb=temb,
                        image_rotary_emb=image_rotary_emb,
                        joint_attention_kwargs=joint_attention_kwargs,
                    )

                # controlnet residual
                if controlnet_single_block_samples is not None:
                    interval_control = len(self.single_transformer_blocks) / len(controlnet_single_block_samples)
                    interval_control = int(np.ceil(interval_control))
                    hidden_states[:, encoder_hidden_states.shape[1] :, ...] = (
                        hidden_states[:, encoder_hidden_states.shape[1] :, ...]
                        + controlnet_single_block_samples[index_block // interval_control]
                    )

            hidden_states = hidden_states[:, encoder_hidden_states.shape[1] :, ...]

        hidden_states = self.norm_out(hidden_states, temb)
        output: torch.FloatTensor = self.proj_out(hidden_states)

        if USE_PEFT_BACKEND:
            # remove `lora_scale` from each PEFT layer
            unscale_lora_layers(self, lora_scale)

        if not return_dict:
            return output

        return Transformer2DModelOutput(sample=output)

    return teacache_forward


class TeaCache:
    """
    Context manager for applying TeaCache temporal caching to transformer models.
    
    This class provides a context manager that temporarily modifies a Flux transformer
    model to use TeaCache temporal caching. When entering the context, the model's
    forward method is replaced with a cached version that tracks temporal changes
    and skips computation when appropriate.
    
    Args:
        model (Union[FluxTransformer2DModel, NunchakuFluxTransformer2dModel]): 
            The transformer model to apply caching to
        num_steps (int, optional): Total number of inference steps. Defaults to 50.
        rel_l1_thresh (float, optional): Relative L1 distance threshold for caching.
            Lower values enable more aggressive caching. Defaults to 0.6.
        skip_steps (int, optional): Number of initial steps to skip caching.
            Useful for model stabilization. Defaults to 0.
        enabled (bool, optional): Whether caching is enabled. If False, the model
            behaves normally. Defaults to True.
            
    Attributes:
        model: Reference to the transformer model
        num_steps (int): Total number of inference steps
        rel_l1_thresh (float): Caching threshold
        skip_steps (int): Number of steps to skip caching
        enabled (bool): Caching enabled flag
        previous_model_forward: Original forward method (for restoration)
        
    Example:
        Basic usage::
        
            with TeaCache(model, num_steps=50, rel_l1_thresh=0.6):
                for step in range(50):
                    output = model(inputs[step])
                    
        Disabling caching conditionally::
        
            with TeaCache(model, enabled=use_caching):
                # Model will use caching only if use_caching is True
                output = model(inputs)
                
    Note:
        The context manager automatically restores the original forward method
        when exiting, ensuring the model can be used normally afterward.
    """
    def __init__(
        self,
        model: Union[FluxTransformer2DModel, NunchakuFluxTransformer2dModel],
        num_steps: int = 50,
        rel_l1_thresh: float = 0.6,
        skip_steps: int = 0,
        enabled: bool = True,
    ) -> None:
        self.model = model
        self.num_steps = num_steps
        self.rel_l1_thresh = rel_l1_thresh
        self.skip_steps = skip_steps
        self.enabled = enabled
        self.previous_model_forward = self.model.forward

    def __enter__(self) -> "TeaCache":
        """
        Enter the TeaCache context and apply caching to the model.
        
        This method is called when entering the 'with' block. It replaces the
        model's forward method with a cached version and initializes the
        necessary state variables for tracking temporal changes.
        
        Returns:
            TeaCache: Self reference for context manager protocol
            
        Note:
            If caching is disabled (enabled=False), the model is left unchanged.
        """
        if self.enabled:
            # self.model.__class__.forward = make_teacache_forward(self.num_steps, self.rel_l1_thresh, self.skip_steps)  # type: ignore
            self.model.forward = MethodType(
                make_teacache_forward(self.num_steps, self.rel_l1_thresh, self.skip_steps), self.model
            )
            self.model.cnt = 0
            self.model.accumulated_rel_l1_distance = 0
            self.model.previous_modulated_input = None
            self.model.previous_residual = None
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        """
        Exit the TeaCache context and restore the original model.
        
        This method is called when exiting the 'with' block. It restores the
        model's original forward method and cleans up the state variables
        that were added for caching.
        
        Args:
            exc_type: Exception type (if any occurred)
            exc_value: Exception value (if any occurred)  
            traceback: Exception traceback (if any occurred)
            
        Note:
            If caching was disabled (enabled=False), no cleanup is performed.
        """
        if self.enabled:
            self.model.forward = self.previous_model_forward
            del self.model.cnt
            del self.model.accumulated_rel_l1_distance
            del self.model.previous_modulated_input
            del self.model.previous_residual
