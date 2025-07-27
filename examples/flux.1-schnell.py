import torch
from diffusers import FluxPipeline

from nunchaku import NunchakuFluxTransformer2dModel
from nunchaku.utils import get_precision

precision = get_precision()  # auto-detect your precision is 'int4' or 'fp4' based on your GPU
transformer = NunchakuFluxTransformer2dModel.from_pretrained(
    f"nunchaku-tech/nunchaku-flux.1-schnell/svdq-{precision}_r32-flux.1-schnell.safetensors"
)
pipeline = FluxPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-schnell", transformer=transformer, torch_dtype=torch.bfloat16
).to("cuda")
image = pipeline(
    "A cat holding a sign that says hello world", width=1024, height=1024, num_inference_steps=4, guidance_scale=0
).images[0]
image.save(f"flux.1-schnell-{precision}.png")
