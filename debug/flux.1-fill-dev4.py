import torch
from diffusers import FluxFillPipeline
from diffusers.utils import load_image

from nunchaku import NunchakuFluxTransformer2dModel
from nunchaku.utils import get_precision

image = load_image("./removal_image.png")
mask = load_image("./removal_mask.png")

precision = get_precision()  # auto-detect your precision is 'int4' or 'fp4' based on your GPU
transformer = NunchakuFluxTransformer2dModel.from_pretrained(f"mit-han-lab/svdq-{precision}-flux.1-fill-dev")
### LoRA Related Code ###
transformer.update_lora_params(
    "loras/removalV2.safetensors"
)  # Path to your LoRA safetensors, can also be a remote HuggingFace path
transformer.set_lora_strength(1)  # Your LoRA strength here
### End of LoRA Related Code ###

pipe = FluxFillPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-Fill-dev", transformer=transformer, torch_dtype=torch.bfloat16
).to("cuda")
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
image.save(f"flux.1-fill-dev-{precision}.png")
