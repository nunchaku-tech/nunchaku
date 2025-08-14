import time

import torch
from diffusers.pipelines.flux.pipeline_flux import FluxPipeline

from nunchaku import NunchakuFluxTransformer2dModel
from nunchaku.caching.teacache import TeaCache
from nunchaku.utils import get_precision

precision = get_precision()  # auto-detect your precision is 'int4' or 'fp4' based on your GPU
transformer = NunchakuFluxTransformer2dModel.from_pretrained(
    f"nunchaku-tech/nunchaku-flux.1-dev/svdq-{precision}_r32-flux.1-dev.safetensors"
)
pipeline = FluxPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-dev", transformer=transformer, torch_dtype=torch.bfloat16
).to("cuda")
start_time = time.time()

prompts = [
    "A cheerful woman in a pastel dress, holding a basket of colorful Easter eggs with a sign that says 'Happy Easter'",
    "A young peace activist with a gentle smile, holding a handmade sign that says 'Peace'",
    "A friendly chef wearing a tall white hat, holding a wooden spoon with a sign that says 'Let's Cook!",
]

with TeaCache(model=transformer, num_steps=50, rel_l1_thresh=0.3, enabled=True):
    image = pipeline(
        prompts,
        num_inference_steps=50,
        guidance_scale=3.5,
        height=1024,
        width=1024,
        generator=torch.Generator(device="cuda").manual_seed(0),
    ).images

end_time = time.time()
print(f"Time taken: {(end_time - start_time)} seconds")
image[0].save(f"flux.1-dev-{precision}1-tc.png")
image[1].save(f"flux.1-dev-{precision}2-tc.png")
image[2].save(f"flux.1-dev-{precision}3-tc.png")
