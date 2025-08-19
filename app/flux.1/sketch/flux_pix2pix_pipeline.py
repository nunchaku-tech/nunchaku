from typing import Any, Callable

import torch
from diffusers.pipelines.flux.pipeline_flux import FluxPipeline, FluxPipelineOutput, FluxTransformer2DModel
from einops import rearrange
from peft.tuners import lora
from PIL import Image
from torch import nn
from torchvision.transforms import functional as F


class FluxPix2pixTurboPipeline(FluxPipeline):
    def update_alpha(self, alpha: float) -> None:
        self._alpha = alpha
        transformer = self.transformer

        if self.precision == "bf16":
            for m in transformer.modules():
                if isinstance(m, lora.LoraLayer):
                    m.scaling["default_0"] = alpha
        else:
            assert self.precision in ["int4", "fp4"]
            transformer.set_lora_strength(alpha)

    def load_control_module(self, pretrained_model_name_or_path: str, weight_name: str, alpha: float = 1):
        state_dict, _ = self.lora_state_dict(pretrained_model_name_or_path, weight_name=weight_name, return_alphas=True)

        transformer = self.transformer
        assert isinstance(transformer, FluxTransformer2DModel)

        if self.precision == "bf16":
            self.load_lora_into_transformer(state_dict, {}, transformer=transformer)
        else:
            self.transformer.update_lora_params(state_dict)
        self.update_alpha(alpha)

    @torch.no_grad()
    def __call__(
        self,
        image: str or Image,
        image_type: str = "sketch",
        alpha: float = 1.0,
        prompt: str | None = None,
        prompt_2: str | None = None,
        height: int | None = 1024,
        width: int | None = 1024,
        timesteps: list[int] = None,
        generator: torch.Generator | None = None,
        prompt_embeds: torch.FloatTensor | None = None,
        pooled_prompt_embeds: torch.FloatTensor | None = None,
        output_type: str | None = "pil",
        return_dict: bool = True,
        joint_attention_kwargs: dict[str, Any] | None = None,
        callback_on_step_end: Callable[[int, int, dict], None] | None = None,
        callback_on_step_end_tensor_inputs: list[str] = ["latents"],
        max_sequence_length: int = 512,
    ):
        if alpha != self._alpha:
            self.update_alpha(alpha)

        height = height or self.default_sample_size * self.vae_scale_factor
        width = width or self.default_sample_size * self.vae_scale_factor
        guidance_scale = 0

        num_images_per_prompt = 1

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt,
            prompt_2,
            height,
            width,
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
            max_sequence_length=max_sequence_length,
        )

        self._guidance_scale = guidance_scale
        self._joint_attention_kwargs = joint_attention_kwargs
        self._interrupt = False

        # 2. Define call parameters
        batch_size = 1

        device = self._execution_device

        lora_scale = self.joint_attention_kwargs.get("scale", None) if self.joint_attention_kwargs is not None else None
        (
            prompt_embeds,
            pooled_prompt_embeds,
            text_ids,
        ) = self.encode_prompt(
            prompt=prompt,
            prompt_2=prompt_2,
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            device=device,
            num_images_per_prompt=num_images_per_prompt,
            max_sequence_length=max_sequence_length,
            lora_scale=lora_scale,
        )

        if isinstance(image, str):
            image = Image.open(image).convert("RGB").resize((width, height), Image.LANCZOS)
        else:
            image = image.resize((width, height), Image.LANCZOS)
        image_t = F.to_tensor(image) < 0.5
        image_t = image_t.unsqueeze(0).to(self.dtype).to(device)

        kernel_size = 4
        if hasattr(self, "erosion_kernel"):
            erosion_kernel = self.erosion_kernel
        else:
            erosion_kernel = torch.ones(1, 1, kernel_size, kernel_size, dtype=self.dtype, device=device)
            self.erosion_kernel = erosion_kernel

        image_t = nn.functional.conv2d(image_t[:, :1], erosion_kernel, padding=kernel_size // 2) > kernel_size**2 - 0.1
        image_t = torch.concat([image_t, image_t, image_t], dim=1).to(self.dtype)

        image_t = (image_t - 0.5) * 2

        # 4. Prepare latent variables
        encoded_image = self.vae.encode(image_t, return_dict=False)[0].sample(generator=generator)
        encoded_image = (encoded_image - self.vae.config.shift_factor) * self.vae.config.scaling_factor
        if generator is None:
            z = torch.randn_like(encoded_image)
        else:
            z = torch.randn(
                encoded_image.shape, device=generator.device, dtype=encoded_image.dtype, generator=generator
            ).to(device)
        noisy_latent = z * (1 - alpha) + encoded_image * alpha
        noisy_latent = rearrange(noisy_latent, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=2, pw=2)
        num_channels_latents = self.transformer.config.in_channels // 4
        _, latent_image_ids = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents=None,
        )

        # 5. Denoising
        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        t = torch.full((batch_size,), 1.0, dtype=self.dtype, device=device)

        # handle guidance
        if self.transformer.config.guidance_embeds:
            guidance = torch.tensor([guidance_scale], device=device)
            guidance = guidance.expand(noisy_latent.shape[0])
        else:
            guidance = None

        pred = self.transformer(
            hidden_states=noisy_latent,
            timestep=t,
            guidance=guidance,
            pooled_projections=pooled_prompt_embeds,
            encoder_hidden_states=prompt_embeds,
            txt_ids=text_ids,
            img_ids=latent_image_ids,
            joint_attention_kwargs=self.joint_attention_kwargs,
            return_dict=False,
        )[0]

        encoded_output = noisy_latent - pred

        if output_type == "latent":
            image = encoded_output

        else:
            encoded_output = self._unpack_latents(encoded_output, height, width, self.vae_scale_factor)
            encoded_output = (encoded_output / self.vae.config.scaling_factor) + self.vae.config.shift_factor
            image = self.vae.decode(encoded_output, return_dict=False)[0]
            image = self.image_processor.postprocess(image, output_type=output_type)

        if not return_dict:
            return (image,)

        return FluxPipelineOutput(images=image)
