import torch
from diffusers import FluxPipeline

from nunchaku.caching.diffusers_adapters.flux_v2 import apply_cache_on_pipe
from nunchaku.models.transformers.transformer_flux_v2 import NunchakuFluxTransformer2DModelV2
from nunchaku.utils import get_precision

precision = get_precision()  # auto-detect your precision is 'int4' or 'fp4' based on your GPU
transformer = NunchakuFluxTransformer2DModelV2.from_pretrained(
    f"nunchaku-tech/nunchaku-flux.1-dev/svdq-{precision}_r32-flux.1-dev.safetensors"
)
pipeline = FluxPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-dev", transformer=transformer, torch_dtype=torch.bfloat16
).to("cuda")

apply_cache_on_pipe(
    pipeline,
    use_double_fb_cache=True,
    residual_diff_threshold_multi=0.12,
    residual_diff_threshold_single=0.20,
)


image = pipeline("A cat holding a sign that says hello world", num_inference_steps=50, guidance_scale=3.5).images[0]
image.save(f"flux.1-dev-cache-{precision}.png")
