import torch
from diffusers import FluxKontextPipeline
from diffusers.utils import load_image
from nunchaku import NunchakuFluxTransformer2dModel

transformer = NunchakuFluxTransformer2dModel.from_pretrained(
    "nunchaku-models/nunchaku-flux.1-kontext-dev/svdq-fp4_r32-flux.1-kontext-dev.safetensors"
)

pipe = FluxKontextPipeline.from_pretrained("black-forest-labs/FLUX.1-Kontext-dev", transformer=transformer, torch_dtype=torch.bfloat16)
pipe.to("cuda")

image = load_image(
    "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/yarn-art-pikachu.png"
).convert("RGB")

prompt = "Make Pikachu hold a sign that says 'Nunchaku is awesome', yarn art style, detailed, vibrant colors"
image = pipe(
    image=image,
    prompt=prompt,
    guidance_scale=2.5
).images[0]
image.save("flux-kontext-dev.png")
