import torch
from diffusers import FluxPipeline

from nunchaku import NunchakuFluxTransformer2dModel
from nunchaku.utils import get_precision

precision = get_precision()  # auto-detect your precision is 'int4' or 'fp4' based on your GPU
transformer = NunchakuFluxTransformer2dModel.from_pretrained(
    f"nunchaku-tech/nunchaku-flux.1-krea-dev/svdq-{precision}_r32-flux.1-krea-dev.safetensors"
)
pipeline = FluxPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-krea-dev", torch_dtype=torch.bfloat16, transformer=transformer
).to("cuda")
prompt = (
    "Tiny paper origami kingdom, a river flowing through a lush valley, bright saturated image,"
    "a fox to the left, deer to the right, birds in the sky, bushes and tress all around"
)
image = pipeline(prompt, height=1024, width=1024, guidance_scale=4.5).images[0]
image.save("flux-krea-dev.png")
