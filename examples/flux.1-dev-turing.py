import torch
from diffusers import FluxPipeline

from nunchaku import NunchakuFluxTransformer2dModel
from nunchaku.utils import get_precision

precision = get_precision()  # auto-detect your precision is 'int4' or 'fp4' based on your GPU
transformer = NunchakuFluxTransformer2dModel.from_pretrained(
    f"nunchaku-tech/nunchaku-flux.1-dev/svdq-{precision}_r32-flux.1-dev.safetensors",
    offload=True,
    torch_dtype=torch.float16,  # Turing GPUs only support fp16 precision
)  # set offload to False if you want to disable offloading
transformer.set_attention_impl("nunchaku-fp16")  # Turing GPUs only support fp16 attention
pipeline = FluxPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-dev", transformer=transformer, torch_dtype=torch.float16
)  # no need to set the device here
pipeline.enable_sequential_cpu_offload()  # diffusers' offloading
image = pipeline("A cat holding a sign that says hello world", num_inference_steps=50, guidance_scale=3.5).images[0]
image.save(f"flux.1-dev-{precision}.png")
