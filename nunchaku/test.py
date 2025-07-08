"""Test script for generating an image using the Nunchaku FLUX.1-schnell.

This script demonstrates how to load a quantized Nunchaku Flux transformer model and
use it with the Diffusers FluxPipeline to generate an image from a text prompt.

Example:
    $ python -m nunchaku.test

The generated image will be saved as "flux.1-schnell.png" in the current directory.
"""

import torch
from diffusers import FluxPipeline

from nunchaku.models.transformers.transformer_flux import NunchakuFluxTransformer2dModel
from nunchaku.utils import get_precision, is_turing

if __name__ == "__main__":
    precision = get_precision()
    torch_dtype = torch.float16 if is_turing() else torch.bfloat16

    transformer = NunchakuFluxTransformer2dModel.from_pretrained(
        f"mit-han-lab/svdq-{precision}-flux.1-schnell", torch_dtype=torch_dtype, offload=True
    )
    pipeline = FluxPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-schnell", transformer=transformer, torch_dtype=torch_dtype
    )
    pipeline.enable_sequential_cpu_offload()
    image = pipeline(
        "A cat holding a sign that says hello world", width=1024, height=1024, num_inference_steps=4, guidance_scale=0
    ).images[0]
    image.save("flux.1-schnell.png")
