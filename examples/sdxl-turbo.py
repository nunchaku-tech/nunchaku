import torch
from diffusers import AutoPipelineForText2Image
from nunchaku.models.unets.unet_sdxl import NunchakuSDXLUNet2DConditionModel


if __name__ == "__main__":
    
    # TODO figure out how to initialize an original SDXL pipeline without loading everything from huggingface hub.
    pipeline = AutoPipelineForText2Image.from_pretrained("stabilityai/sdxl-turbo", torch_dtype=torch.bfloat16, variant="fp16")
    prompt = "A cinematic shot of a baby racoon wearing an intricate italian priest robe."
    quantized_unet = NunchakuSDXLUNet2DConditionModel.from_pretrained("/path/to/sdxl-turbo-quantized.safetensors")
    pipeline.unet = quantized_unet
    pipeline = pipeline.to("cuda")
    
    image = pipeline(prompt=prompt, guidance_scale=0.0, num_inference_steps=4).images[0]

    image.save("racoon.png")
    
