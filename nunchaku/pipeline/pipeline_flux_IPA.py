from typing import Any, List, Optional

import torch
from diffusers import FluxPipeline


class FluxPipelineWrapper(FluxPipeline):
    @torch.no_grad()
    def get_image_embeds(
        self,
        num_images_per_prompt: int = 1,
        ip_adapter_image: Optional[Any] = None,  # PipelineImageInput
        ip_adapter_image_embeds: Optional[List[torch.Tensor]] = None,
        negative_ip_adapter_image: Optional[Any] = None,  # PipelineImageInput
        negative_ip_adapter_image_embeds: Optional[List[torch.Tensor]] = None,
    ) -> (Optional[torch.Tensor], Optional[torch.Tensor]):
        batch_size = 1

        device = self.transformer.device

        image_embeds = None
        if ip_adapter_image is not None or ip_adapter_image_embeds is not None:
            image_embeds = self.prepare_ip_adapter_image_embeds(
                ip_adapter_image=ip_adapter_image,
                ip_adapter_image_embeds=ip_adapter_image_embeds,
                device=device,
                num_images_per_prompt=batch_size * num_images_per_prompt,
            )
            image_embeds = self.transformer.encoder_hid_proj(image_embeds)

        negative_image_embeds = None
        if negative_ip_adapter_image is not None or negative_ip_adapter_image_embeds is not None:
            negative_image_embeds = self.prepare_ip_adapter_image_embeds(
                ip_adapter_image=negative_ip_adapter_image,
                ip_adapter_image_embeds=negative_ip_adapter_image_embeds,
                device=device,
                num_images_per_prompt=batch_size * num_images_per_prompt,
            )
            negative_image_embeds = self.transformer.encoder_hid_proj(negative_image_embeds)

        return image_embeds, negative_image_embeds
