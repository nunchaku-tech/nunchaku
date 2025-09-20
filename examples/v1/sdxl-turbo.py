import torch
from diffusers import StableDiffusionXLPipeline

from nunchaku.models.unets.unet_sdxl import NunchakuSDXLUNet2DConditionModel

if __name__ == "__main__":
    unet = NunchakuSDXLUNet2DConditionModel.from_pretrained(
        "nunchaku-tech/nunchaku-sdxl-turbo/svdq-int4_r32-sdxl-turbo.safetensors"
    )
    pipeline = StableDiffusionXLPipeline.from_pretrained(
        "stabilityai/sdxl-turbo", unet=unet, torch_dtype=torch.bfloat16, variant="fp16"
    ).to("cuda")
    prompt = "A cinematic shot of a baby racoon wearing an intricate italian priest robe."

    image = pipeline(prompt=prompt, guidance_scale=0.0, num_inference_steps=4).images[0]

    image.save("sdxl-turbo.png")
