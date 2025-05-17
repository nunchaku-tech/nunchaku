import inspect
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from diffusers import AutoencoderKL, FlowMatchEulerDiscreteScheduler
from diffusers.image_processor import PipelineImageInput, VaeImageProcessor
from diffusers.models.transformers.transformer_flux import FluxTransformer2DModel
from diffusers.pipelines.flux.pipeline_flux import FluxPipeline
from diffusers.pipelines.flux.pipeline_output import FluxPipelineOutput
from transformers import (
    CLIPImageProcessor,
    CLIPTextModel,
    CLIPTokenizer,
    CLIPVisionModelWithProjection,
    T5EncoderModel,
    T5TokenizerFast,
)


def randn_tensor(
    shape: Union[Tuple, List],
    generator: Optional[Union[List["torch.Generator"], "torch.Generator"]] = None,
    device: Optional["torch.device"] = None,
    dtype: Optional["torch.dtype"] = None,
    layout: Optional["torch.layout"] = None,
):
    """A helper function to create random tensors on the desired `device` with the desired `dtype`. When
    passing a list of generators, you can seed each batch size individually. If CPU generators are passed, the tensor
    is always created on the CPU.
    """
    # device on which tensor is created defaults to device
    rand_device = device
    batch_size = shape[0]

    layout = layout or torch.strided
    device = device or torch.device("cpu")

    if generator is not None:
        gen_device_type = generator.device.type if not isinstance(generator, list) else generator[0].device.type
        if gen_device_type != device.type and gen_device_type == "cpu":
            rand_device = "cpu"
            if device != "mps":
                logger.info(
                    f"The passed generator was created on 'cpu' even though a tensor on {device} was expected."
                    f" Tensors will be created on 'cpu' and then moved to {device}. Note that one can probably"
                    f" slighly speed up this function by passing a generator that was created on the {device} device."
                )
        elif gen_device_type != device.type and gen_device_type == "cuda":
            raise ValueError(f"Cannot generate a {device} tensor from a generator of type {gen_device_type}.")

    if isinstance(generator, list):
        shape = (1,) + shape[1:]
        latents = [
            torch.randn(
                shape,
                generator=generator[i],
                device=rand_device,
                dtype=dtype,
                layout=layout,
            )
            for i in range(batch_size)
        ]
        latents = torch.cat(latents, dim=0).to(device)
    else:
        latents = torch.randn(shape, generator=generator, device=rand_device, dtype=dtype, layout=layout).to(device)

    return latents


# from xvst.utils import setup_#logger

# from .omini_flux import tranformer_forward

#
# logger = setup_#logger("OminiFlux")


# Calculates a shift value for the scheduler's timesteps based on image sequence length.
# This is part of the Flux model's specific timestep scheduling mechanism.
def calculate_shift(
    image_seq_len,  # Sequence length of the image latents (e.g., H_lat * W_lat)
    base_seq_len: int = 256,  # Reference base sequence length
    max_seq_len: int = 4096,  # Reference maximum sequence length
    base_shift: float = 0.5,  # Shift value at base_seq_len
    max_shift: float = 1.16,  # Shift value at max_seq_len
):
    # Linear interpolation/extrapolation of the shift value based on image_seq_len
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    mu = image_seq_len * m + b
    return mu


# Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.retrieve_timesteps
# Utility function to prepare timesteps for the scheduler.
# Handles custom timesteps, sigmas, or a specified number of inference steps.
def retrieve_timesteps(
    scheduler,
    num_inference_steps: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
    timesteps: Optional[List[int]] = None,
    sigmas: Optional[List[float]] = None,
    **kwargs,  # Additional arguments for scheduler.set_timesteps
):
    r"""
    Calls the scheduler's `set_timesteps` method and retrieves timesteps from the scheduler after the call. Handles
    custom timesteps. Any kwargs will be supplied to `scheduler.set_timesteps`.

    Args:
        scheduler (`SchedulerMixin`):
            The scheduler to get timesteps from.
        num_inference_steps (`int`):
            The number of diffusion steps used when generating samples with a pre-trained model. If used, `timesteps`
            must be `None`.
        device (`str` or `torch.device`, *optional*):
            The device to which the timesteps should be moved to. If `None`, the timesteps are not moved.
        timesteps (`List[int]`, *optional*):
            Custom timesteps used to override the timestep spacing strategy of the scheduler. If `timesteps` is passed,
            `num_inference_steps` and `sigmas` must be `None`.
        sigmas (`List[float]`, *optional*):
            Custom sigmas used to override the timestep spacing strategy of the scheduler. If `sigmas` is passed,
            `num_inference_steps` and `timesteps` must be `None`.

    Returns:
        `Tuple[torch.Tensor, int]`: A tuple where the first element is the timestep schedule from the scheduler and the
        second element is the number of inference steps.
    """
    if timesteps is not None and sigmas is not None:
        raise ValueError("Only one of `timesteps` or `sigmas` can be passed. Please choose one to set custom values")
    if timesteps is not None:
        accepts_timesteps = "timesteps" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accepts_timesteps:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" timestep schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    elif sigmas is not None:
        accept_sigmas = "sigmas" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accept_sigmas:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" sigmas schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps
    return timesteps, num_inference_steps


# Custom pipeline for OminiFlux with specific handling for downsized image conditioning.
# Inherits from the base FluxPipeline and overrides or adds methods for:
# - VAE image encoding with shift and scale factors.
# - Preparation of latent image IDs for positional encoding, including sub-IDs for downsized/patched inputs.
# - Preparation of image latents from input images, potentially with downsampling.
# - The main call (`__call__`) method adapted for OminiFlux specific inputs like `cond_latents` and `cond_ids`.
class OminiFluxDownsizeAllInOnePipeline(FluxPipeline):
    r"""
    The Flux pipeline for text-to-image generation.

    Reference: https://blackforestlabs.ai/announcing-black-forest-labs/

    Args:
        transformer ([`FluxTransformer2DModel`]):
            Conditional Transformer (MMDiT) architecture to denoise the encoded image latents.
        scheduler ([`FlowMatchEulerDiscreteScheduler`]):
            A scheduler to be used in combination with `transformer` to denoise the encoded image latents.
        vae ([`AutoencoderKL`]):
            Variational Auto-Encoder (VAE) Model to encode and decode images to and from latent representations.
        text_encoder ([`CLIPTextModel`]):
            [CLIP](https://huggingface.co/docs/transformers/model_doc/clip#transformers.CLIPTextModel), specifically
            the [clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14) variant.
        text_encoder_2 ([`T5EncoderModel`]):
            [T5](https://huggingface.co/docs/transformers/en/model_doc/t5#transformers.T5EncoderModel), specifically
            the [google/t5-v1_1-xxl](https://huggingface.co/google/t5-v1_1-xxl) variant.
        tokenizer (`CLIPTokenizer`):
            Tokenizer of class
            [CLIPTokenizer](https://huggingface.co/docs/transformers/en/model_doc/clip#transformers.CLIPTokenizer).
        tokenizer_2 (`T5TokenizerFast`):
            Second Tokenizer of class
            [T5TokenizerFast](https://huggingface.co/docs/transformers/en/model_doc/t5#transformers.T5TokenizerFast).
    """

    def __init__(
        self,
        scheduler: FlowMatchEulerDiscreteScheduler,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        text_encoder_2: T5EncoderModel,
        tokenizer_2: T5TokenizerFast,
        transformer: FluxTransformer2DModel,
        image_encoder: CLIPVisionModelWithProjection = None,
        feature_extractor: CLIPImageProcessor = None,
    ):
        super().__init__(
            vae=vae,
            text_encoder=text_encoder,
            text_encoder_2=text_encoder_2,
            tokenizer=tokenizer,
            tokenizer_2=tokenizer_2,
            transformer=transformer,
            scheduler=scheduler,
            image_encoder=image_encoder,
            feature_extractor=feature_extractor,
        )

        # Flux latents are turned into 2x2 patches and packed. This means the latent width and height has to be divisible
        # by the patch size. So the vae scale factor is multiplied by the patch size to account for this
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor * 2)

    # Encodes an image into the latent space using the VAE.
    # Applies the VAE's shift and scaling factors to the latents.
    def _encode_vae_image(self, images: torch.Tensor, generator: torch.Generator = None):
        if isinstance(generator, list):
            image_latents = [
                self.vae.encode(images[i : i + 1]).latent_dist.sample(generator[i]) for i in range(images.shape[0])
            ]
            image_latents = torch.cat(image_latents, dim=0)
        else:
            image_latents = self.vae.encode(images).latent_dist.sample(generator)

        image_latents = (image_latents - self.vae.config.shift_factor) * self.vae.config.scaling_factor

        return image_latents

    @staticmethod
    # Prepares a tensor of IDs for positional encoding of image latents.
    # Each "pixel" in the latent grid gets a unique ID based on its (0, H, W) coordinates.
    # The first dimension (channel 0) is often 0, H is in channel 1, W is in channel 2.
    def _prepare_latent_image_ids(batch_size, height, width, device, dtype):
        latent_image_ids = torch.zeros(height, width, 3)
        latent_image_ids[..., 1] = latent_image_ids[..., 1] + torch.arange(0, height)[:, None]
        latent_image_ids[..., 2] = latent_image_ids[..., 2] + torch.arange(0, width)[None, :]

        latent_image_id_height, latent_image_id_width, latent_image_id_channels = latent_image_ids.shape

        latent_image_ids = latent_image_ids.reshape(
            latent_image_id_height * latent_image_id_width, latent_image_id_channels
        )

        return latent_image_ids.to(device=device, dtype=dtype)

    @staticmethod
    # Prepares latent image IDs for sub-regions or patches of the full latent image.
    # This is used when `downsize=True`, where the conditional image might be processed in patches.
    # `step` defines the stride, `w_offset` and `h_offset` define the starting corner of the patch.
    def _prepare_latent_image_sub_ids(batch_size, height, width, device, dtype, step=1, w_offset=0, h_offset=0):
        latent_image_ids = torch.zeros(height, width, 3)
        latent_image_ids[..., 1] = latent_image_ids[..., 1] + torch.arange(0, height * step, step)[:, None] + h_offset
        latent_image_ids[..., 2] = latent_image_ids[..., 2] + torch.arange(0, width * step, step)[None, :] + w_offset

        latent_image_id_height, latent_image_id_width, latent_image_id_channels = latent_image_ids.shape

        latent_image_ids = latent_image_ids.reshape(
            latent_image_id_height * latent_image_id_width, latent_image_id_channels
        )

        return latent_image_ids.to(device=device, dtype=dtype)

    # Prepares conditional image latents when `downsize=True`.
    # The image is encoded, and then sub-IDs are generated for four quadrants/patches
    # of the downsized latent space. This suggests the model might process these patches
    # or attend to them differently during the denoising steps.
    def prepare_sub_image_latents(
        self,
        image,  # The input conditional image
        batch_size,
        num_channels_latents,
        height,
        width,
        dtype,
        device,
        generator,
        latents=None,
    ):
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        # VAE applies 8x compression on images but we must also account for packing which requires
        # latent height and width to be divisible by 2.
        # 2x downsize
        height = 2 * (int(height) // (self.vae_scale_factor * 2)) // 2  # 2x downsize
        width = 2 * (int(width) // (self.vae_scale_factor * 2)) // 2  # 2x downsize
        # sub pixels ids
        latent_image_ids = []
        latent_image_ids.append(
            self._prepare_latent_image_sub_ids(batch_size, height // 2, width // 2, device, dtype, step=2)
        )
        latent_image_ids.append(
            self._prepare_latent_image_sub_ids(batch_size, height // 2, width // 2, device, dtype, step=2, h_offset=1)
        )
        latent_image_ids.append(
            self._prepare_latent_image_sub_ids(
                batch_size,
                height // 2,
                width // 2,
                device,
                dtype,
                step=2,
                h_offset=1,
                w_offset=1,
            )
        )
        latent_image_ids.append(
            self._prepare_latent_image_sub_ids(batch_size, height // 2, width // 2, device, dtype, step=2, w_offset=1)
        )

        if latents is not None:
            return latents.to(device=device, dtype=dtype), latent_image_ids

        image = image.to(device=device, dtype=dtype)
        image_latents = self._encode_vae_image(images=image, generator=generator)
        if batch_size > image_latents.shape[0] and batch_size % image_latents.shape[0] == 0:
            # expand init_latents for batch_size
            additional_image_per_prompt = batch_size // image_latents.shape[0]
            image_latents = torch.cat([image_latents] * additional_image_per_prompt, dim=0)
        elif batch_size > image_latents.shape[0] and batch_size % image_latents.shape[0] != 0:
            raise ValueError(
                f"Cannot duplicate `image` of batch size {image_latents.shape[0]} to {batch_size} text prompts."
            )
        else:
            image_latents = torch.cat([image_latents], dim=0)
        latents = self._pack_latents(image_latents, batch_size, num_channels_latents, height, width)
        return latents, latent_image_ids

    # Prepares image latents from an input image (e.g., for image-to-image or conditioning).
    # Handles VAE encoding and packing of latents into the format expected by the Flux transformer.
    # Also generates latent image IDs for positional encoding.
    # The `downsize` flag is present but its effect on `step` seems unused here, as `step` is fixed to 1 or 2 earlier.
    def prepare_image_latents(
        self,
        image,  # The input image
        batch_size,
        num_channels_latents,
        height,
        width,
        dtype,
        device,
        generator,
        latents=None,
        downsize=False,
    ):
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        # VAE applies 8x compression on images but we must also account for packing which requires
        # latent height and width to be divisible by 2.
        height = 2 * (int(height) // (self.vae_scale_factor * 2))
        width = 2 * (int(width) // (self.vae_scale_factor * 2))
        latent_image_ids = self._prepare_latent_image_ids(batch_size, height // 2, width // 2, device, dtype)

        if latents is not None:
            return latents.to(device=device, dtype=dtype), latent_image_ids

        image = image.to(device=device, dtype=dtype)
        image_latents = self._encode_vae_image(images=image, generator=generator)
        if batch_size > image_latents.shape[0] and batch_size % image_latents.shape[0] == 0:
            # expand init_latents for batch_size
            additional_image_per_prompt = batch_size // image_latents.shape[0]
            image_latents = torch.cat([image_latents] * additional_image_per_prompt, dim=0)
        elif batch_size > image_latents.shape[0] and batch_size % image_latents.shape[0] != 0:
            raise ValueError(
                f"Cannot duplicate `image` of batch size {image_latents.shape[0]} to {batch_size} text prompts."
            )
        else:
            image_latents = torch.cat([image_latents], dim=0)

        latents = self._pack_latents(image_latents, batch_size, num_channels_latents, height, width)
        return latents, latent_image_ids

    # Prepares initial noisy latents for the generation process (if no initial `latents` are provided).
    # Also generates latent image IDs for positional encoding.
    # The height and width are adjusted for VAE and patch packing.
    def prepare_latents(
        self,
        batch_size,
        num_channels_latents,
        height,
        width,
        dtype,
        device,
        generator,
        latents=None,
    ):
        # VAE applies 8x compression on images but we must also account for packing which requires
        # latent height and width to be divisible by 2.
        height = 2 * (int(height) // (self.vae_scale_factor * 2))
        width = 2 * (int(width) // (self.vae_scale_factor * 2))

        shape = (batch_size, num_channels_latents, height, width)

        if latents is not None:
            latent_image_ids = self._prepare_latent_image_ids(batch_size, height // 2, width // 2, device, dtype)
            return latents.to(device=device, dtype=dtype), latent_image_ids

        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        latents = self._pack_latents(latents, batch_size, num_channels_latents, height, width)

        latent_image_ids = self._prepare_latent_image_ids(batch_size, height // 2, width // 2, device, dtype)

        return latents, latent_image_ids

    @torch.no_grad()
    # Main entry point for the OminiFlux pipeline.
    # Orchestrates prompt encoding, latent preparation (initial noise and conditional), and the denoising loop.
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,  # Main text prompt
        prompt_2: Optional[Union[str, List[str]]] = None,  # Secondary text prompt (e.g., for T5)
        image: PipelineImageInput = None,  # Conditional input image
        height: Optional[int] = None,  # Target output height
        width: Optional[int] = None,  # Target output width
        num_inference_steps: int = 28,  # Number of denoising steps
        sigmas: Optional[List[float]] = None,  # Optional custom sigmas for scheduler
        guidance_scale: float = 3.5,  # Classifier-free guidance strength
        num_images_per_prompt: Optional[int] = 1,  # Number of images to generate per prompt
        generator: Optional[
            Union[torch.Generator, List[torch.Generator]]
        ] = None,  # Random generator for reproducibility
        latents: Optional[torch.FloatTensor] = None,  # Optional pre-defined initial latents (noise)
        cond_latents: Optional[torch.FloatTensor] = None,  # Optional pre-defined conditional latents
        prompt_embeds: Optional[torch.FloatTensor] = None,  # Optional pre-computed prompt embeddings
        pooled_prompt_embeds: Optional[torch.FloatTensor] = None,  # Optional pre-computed pooled prompt embeddings
        output_type: Optional[str] = "pil",  # Desired output format ("pil", "latent", etc.)
        return_dict: bool = True,  # Whether to return a FluxPipelineOutput object
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,  # Kwargs for joint attention (e.g., IP-Adapter)
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,  # Optional callback after each step
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],  # Tensors to pass to the callback
        max_sequence_length: int = 512,  # Max sequence length for tokenizers
        downsize: bool = True,  # If true, uses downsized conditioning logic (prepare_sub_image_latents)
    ):

        height = height or self.default_sample_size * self.vae_scale_factor
        width = width or self.default_sample_size * self.vae_scale_factor

        self._guidance_scale = guidance_scale
        self._joint_attention_kwargs = joint_attention_kwargs
        self._current_timestep = None
        self._interrupt = False

        print(prompt)
        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device

        if downsize:
            condition_image = self.image_processor.preprocess(image, height=height // 2, width=width // 2)
        else:
            condition_image = self.image_processor.preprocess(image, height=height, width=width)

        condition_image = condition_image.to(dtype=torch.float32)

        # logger.debug(f"Condition image shape: {condition_image.shape}")

        lora_scale = self.joint_attention_kwargs.get("scale", None) if self.joint_attention_kwargs is not None else None

        (prompt_embeds, pooled_prompt_embeds, text_ids) = self.encode_prompt(
            prompt=prompt,
            prompt_2=prompt_2,
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            device=device,
            num_images_per_prompt=num_images_per_prompt,
            max_sequence_length=max_sequence_length,
            lora_scale=lora_scale,
        )

        # logger.debug(f"prompt embeds shape: {prompt_embeds.shape}")
        # logger.debug(f"pooled prompt embeds image shape: {pooled_prompt_embeds.shape}")
        # logger.debug(f"text ids shape: {text_ids.shape}")
        # 4.Prepare timesteps
        # sigmas = (
        #     np.linspace(1.0, 1 / num_inference_steps, num_inference_steps)
        #     if sigmas is None
        #     else sigmas
        # )
        # image_seq_len = (int(height) // self.vae_scale_factor // 2) * (
        #     int(width) // self.vae_scale_factor // 2
        # )
        # mu = calculate_shift(
        #     image_seq_len,
        #     self.scheduler.config.get("base_image_seq_len", 256),
        #     self.scheduler.config.get("max_image_seq_len", 4096),
        #     self.scheduler.config.get("base_shift", 0.5),
        #     self.scheduler.config.get("max_shift", 1.16),
        # )
        # timesteps, num_inference_steps = retrieve_timesteps(
        #     self.scheduler, num_inference_steps, device, sigmas=sigmas, mu=mu
        # )

        # num_warmup_steps = max(
        #     len(timesteps) - num_inference_steps * self.scheduler.order, 0
        # )
        # self._num_timesteps = len(timesteps)

        timesteps = torch.linspace(1.0, 0, num_inference_steps + 1).to(device=self.device, dtype=torch.float32)
        shift = 3.0
        timesteps = timesteps * shift / (1 + (shift - 1) * timesteps)

        # 5. Prepare latent variables
        num_channels_latents = self.transformer.config.in_channels // 4

        if downsize:
            cond_latents, sub_cond_latent_image_ids = self.prepare_sub_image_latents(
                condition_image,
                batch_size * num_images_per_prompt,
                num_channels_latents,
                height,
                width,
                prompt_embeds.dtype,
                device,
                generator,
                cond_latents,
            )
        else:
            cond_latents, cond_latent_image_ids = self.prepare_image_latents(
                condition_image,
                batch_size * num_images_per_prompt,
                num_channels_latents,
                height,
                width,
                prompt_embeds.dtype,
                device,
                generator,
                cond_latents,
            )
        # #logger.debug(f"Condition latents shape: {cond_latents.shape}")

        latents, latent_image_ids = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )
        # #logger.debug(f"latents shape: {latents.shape}")

        num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)
        num_warmup_steps = 0
        # print(f"num_warmup_steps: {num_warmup_steps}")
        self._num_timesteps = len(timesteps)

        # handle guidance
        if self.transformer.config.guidance_embeds:
            guidance = torch.full([1], guidance_scale, device=device, dtype=torch.float32)
            guidance = guidance.expand(latents.shape[0])
        else:
            guidance = None

        if self.joint_attention_kwargs is None:
            self._joint_attention_kwargs = {}

        # 6. Denoising loop
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps[:-1]):
                print(f"i: {i}, t: {t}")
                if self.interrupt:
                    continue

                self._current_timestep = t
                # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
                timestep = t.expand(latents.shape[0]).to(latents.dtype)

                if downsize:
                    cond_latent_image_ids = sub_cond_latent_image_ids[i % 4]

                noise_pred = self.transformer(
                    hidden_states=latents,
                    cond_hidden_states=cond_latents,
                    encoder_hidden_states=prompt_embeds,
                    pooled_projections=pooled_prompt_embeds,
                    timestep=timestep,
                    cond_ids=cond_latent_image_ids,
                    img_ids=latent_image_ids,
                    txt_ids=text_ids,
                    guidance=guidance,
                    joint_attention_kwargs=self.joint_attention_kwargs,
                    return_dict=False,
                )[0]

                # compute the previous noisy sample x_t -> x_t-1
                latents_dtype = latents.dtype
                # latents = self.scheduler.step(
                #     noise_pred, t, latents, return_dict=False
                # )[0]
                # print(f"latents shape: {latents.shape}")
                # print(f"noise_pred shape: {noise_pred.shape}")
                # print(f"timesteps: {timesteps}")
                latents = latents + (timesteps[i + 1] - timesteps[i]) * noise_pred

                if latents.dtype != latents_dtype:
                    if torch.backends.mps.is_available():
                        # some platforms (eg. apple mps) misbehave due to a pytorch bug: https://github.com/pytorch/pytorch/pull/99272
                        latents = latents.to(latents_dtype)

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                    latents = callback_outputs.pop("latents", latents)
                    prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()
        self._current_timestep = None

        if output_type == "latent":
            image = latents
        else:
            # Unpack latents from the [B, C*patch_size*patch_size, H_patch, W_patch] format
            # back to [B, C, H, W] for VAE decoding.
            latents = self._unpack_latents(latents, height, width, self.vae_scale_factor)
            # Apply inverse scaling and shift before VAE decoding
            latents = (latents / self.vae.config.scaling_factor) + self.vae.config.shift_factor
            image = self.vae.decode(latents, return_dict=False)[0]
            image = self.image_processor.postprocess(image, output_type=output_type)

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (image,)

        return FluxPipelineOutput(images=image)
