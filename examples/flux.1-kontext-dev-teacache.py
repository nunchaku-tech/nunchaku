import time
from nunchaku.caching.teacache import TeaCache
import torch
from diffusers import FluxKontextPipeline
from diffusers.utils import load_image

from nunchaku import NunchakuFluxTransformer2dModel
from nunchaku.utils import get_precision

transformer = NunchakuFluxTransformer2dModel.from_pretrained(
    f"nunchaku-tech/nunchaku-flux.1-kontext-dev/svdq-{get_precision()}_r32-flux.1-kontext-dev.safetensors"
)

pipeline = FluxKontextPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-Kontext-dev", transformer=transformer, torch_dtype=torch.bfloat16
).to("cuda")

image = load_image(
    "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/yarn-art-pikachu.png"
).convert("RGB")

prompt = "Make Pikachu hold a sign that says 'Nunchaku is awesome', yarn art style, detailed, vibrant colors"

start_time = time.time()
with TeaCache(model=transformer, num_steps=50, rel_l1_thresh=0.3, enabled=True, model_name="flux-kontext"):
    image = pipeline(image=image, prompt=prompt, guidance_scale=2.5).images[0]
end_time = time.time()
print(f"Time taken: {(end_time - start_time)} seconds")
image.save(f"flux-kontext-dev-{get_precision()}-tc.png")
