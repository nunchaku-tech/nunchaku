import torch
from diffusers import FluxPipeline

from nunchaku import NunchakuFluxTransformer2dModel
from nunchaku.caching.diffusers_adapters import apply_cache_on_pipe
from nunchaku.utils import get_precision

precision = get_precision()

transformer = NunchakuFluxTransformer2dModel.from_pretrained(f"mit-han-lab/svdq-{precision}-flux.1-dev")

pipeline = FluxPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-dev", transformer=transformer, torch_dtype=torch.bfloat16
).to("cuda")

apply_cache_on_pipe(
    pipeline,
    use_double_fb_cache=True,
    residual_diff_threshold_multi=0.09,
    residual_diff_threshold_single=0.12,
)

image = pipeline(["A cat holding a sign that says hello world"], num_inference_steps=50).images[0]

image.save(f"flux.1-dev-cache-{precision}.png")
