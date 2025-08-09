import torch
from diffusers import FluxPipeline

from nunchaku import NunchakuFluxTransformer2dModel
from nunchaku.utils import get_precision

precision = get_precision()  # auto-detect your precision is 'int4' or 'fp4' based on your GPU
transformer = NunchakuFluxTransformer2dModel.from_pretrained(
    f"nunchaku-tech/nunchaku-flux.1-dev-colossus/svdq-{precision}_r32-flux.1-dev-colossusv12.safetensors"
)
pipeline = FluxPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-dev", transformer=transformer, torch_dtype=torch.bfloat16
).to("cuda")
image = pipeline(
    "face portrait of a 20 years old woman , gothic short blue hair, ruby collar, she is touch her cheek",
    num_inference_steps=50,
    guidance_scale=3.5,
).images[0]
image.save(f"flux.1-dev-colossus-{precision}.png")
