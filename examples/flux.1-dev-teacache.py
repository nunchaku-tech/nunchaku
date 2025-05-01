import time

import torch
from diffusers.pipelines.flux.pipeline_flux import FluxPipeline

from nunchaku import NunchakuFluxTransformer2dModel
from nunchaku.caching.teacache import TeaCache
from nunchaku.utils import get_precision

precision = get_precision()  # auto-detect your precision is 'int4' or 'fp4' based on your GPU
transformer = NunchakuFluxTransformer2dModel.from_pretrained(f"mit-han-lab/svdq-{precision}-flux.1-dev")
pipeline = FluxPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-dev", transformer=transformer, torch_dtype=torch.bfloat16
).to("cuda")
start_time = time.time()
with TeaCache(model=transformer, num_steps=50, rel_l1_thresh=0.3, enabled=True):
    image = pipeline(
        "A cat holding a sign that says hello world",
        num_inference_steps=50,
        guidance_scale=3.5,
        height=1024,
        width=1024,
        generator=torch.Generator(device="cuda").manual_seed(0),
    ).images[0]
end_time = time.time()
print(f"Time taken: {(end_time - start_time)} seconds")
image.save(f"flux.1-dev-{precision}-tc.png")
