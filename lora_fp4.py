import torch
from diffusers import FluxPipeline

from nunchaku import NunchakuFluxTransformer2dModel
from nunchaku.utils import get_precision

precision = get_precision()  # auto-detect your precision is 'int4' or 'fp4' based on your GPU
transformer = NunchakuFluxTransformer2dModel.from_pretrained(
    f"mit-han-lab/nunchaku-flux.1-dev/svdq-{precision}_r32-flux.1-dev.safetensors"
)
pipeline = FluxPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-dev", transformer=transformer, torch_dtype=torch.bfloat16
).to("cuda")

### LoRA Related Code ###
transformer.update_lora_params(
    "./loras/skin_texture.safetensors"
)  # Path to your LoRA safetensors, can also be a remote HuggingFace path
transformer.set_lora_strength(10)  # Your LoRA strength here
### End of LoRA Related Code ###

prompt = "Hyperrealistic art of  <lora:skin texture style v5:1> a closeup of a man's hand holding a colt pistol in his hand, perfect skin, detailed skin pore, realism style, perfect image, perfect body, perfect anatomy, sharp image, detailed image, high quality photography, skin texture style, solo, long sleeves, holding, weapon, holding weapon, gun, blue background, holding gun, handgun, m1911, photorealistic, hand focus, Extremely high-resolution details, photographic, realism pushed to extreme, fine texture, incredibly lifelike"  # noqa: E501

image = pipeline(
    prompt,
    num_inference_steps=20,
    guidance_scale=3.5,
    generator=torch.Generator().manual_seed(42),
).images[0]
image.save(f"flux.1-dev-{precision}-10.png")
