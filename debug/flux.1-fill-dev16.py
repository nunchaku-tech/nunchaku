import torch
from diffusers import FluxFillPipeline
from diffusers.utils import load_image

from nunchaku import NunchakuFluxTransformer2dModel
from nunchaku.utils import get_precision

image = load_image("./removal_image.png")
mask = load_image("./removal_mask.png")

precision = get_precision()  # auto-detect your precision is 'int4' or 'fp4' based on your GPU
pipe = FluxFillPipeline.from_pretrained("black-forest-labs/FLUX.1-Fill-dev", torch_dtype=torch.bfloat16)
pipe.load_lora_weights(
    "./loras/removalV2.safetensors"
)  # Path to your LoRA safetensors, can also be a remote HuggingFace path
pipe.fuse_lora(lora_scale=5)
pipe.enable_model_cpu_offload()

image = pipe(
    prompt="",
    image=image,
    mask_image=mask,
    height=720,
    width=1280,
    guidance_scale=30,
    num_inference_steps=20,
    max_sequence_length=512,
    generator=torch.Generator().manual_seed(42),
).images[0]
image.save(f"flux.1-fill-dev-bf16.png")
