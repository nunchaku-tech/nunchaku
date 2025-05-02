import torch
from diffusers import FluxPipeline

from nunchaku import NunchakuFluxTransformer2dModel
from nunchaku.utils import get_precision

precision = get_precision()  # auto-detect your precision is 'int4' or 'fp4' based on your GPU
transformer = NunchakuFluxTransformer2dModel.from_pretrained(f"mit-han-lab/svdq-{precision}-flux.1-dev")
pipeline = FluxPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-dev", transformer=transformer, torch_dtype=torch.bfloat16
).to("cuda")

image = pipeline(
    "cozy mountain cabin covered in snow, with smoke curling from the chimney and a warm, inviting light spilling through the windows",  # noqa: E501
    num_inference_steps=8,
    guidance_scale=3.5,
    generator=torch.Generator().manual_seed(23),
).images[0]
image.save("a.png")

### LoRA Related Code ###
transformer.update_lora_params(
    "alimama-creative/FLUX.1-Turbo-Alpha/diffusion_pytorch_model.safetensors"
)  # Path to your LoRA safetensors, can also be a remote HuggingFace path
transformer.set_lora_strength(10)  # Your LoRA strength here
### End of LoRA Related Code ###

image = pipeline(
    "cozy mountain cabin covered in snow, with smoke curling from the chimney and a warm, inviting light spilling through the windows",  # noqa: E501
    num_inference_steps=8,
    guidance_scale=3.5,
    generator=torch.Generator().manual_seed(23),
).images[0]
image.save("b.png")

transformer.reset_lora()
image = pipeline(
    "cozy mountain cabin covered in snow, with smoke curling from the chimney and a warm, inviting light spilling through the windows",  # noqa: E501
    num_inference_steps=8,
    guidance_scale=3.5,
    generator=torch.Generator().manual_seed(23),
).images[0]
image.save("c.png")
