import torch
from diffusers import FluxPipeline

from nunchaku.models.transformers.transformer_flux_v2 import NunchakuFluxTransformer2DModelV2
from nunchaku.utils import get_precision

precision = get_precision()  # auto-detect your precision is 'int4' or 'fp4' based on your GPU

torch_dtype = torch.float16  # Auto-selects bfloat16 on Ampere+ GPUs, float16 otherwise

transformer = NunchakuFluxTransformer2DModelV2.from_pretrained(
    f"nunchaku-tech/nunchaku-flux.1-dev/svdq-{precision}_r32-flux.1-dev.safetensors", torch_dtype=torch_dtype
)
pipeline = FluxPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-dev", transformer=transformer, torch_dtype=torch_dtype
).to("cuda")
image = pipeline("A cat holding a sign that says hello world", num_inference_steps=50, guidance_scale=3.5).images[0]
image.save(f"flux.1-dev-{precision}.png")
