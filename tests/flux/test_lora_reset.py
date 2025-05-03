import os

import torch
from diffusers import FluxPipeline

from nunchaku import NunchakuFluxTransformer2dModel
from nunchaku.utils import get_precision

from ..utils import compute_lpips


def test_lora_reset():
    precision = get_precision()  # auto-detect your precision is 'int4' or 'fp4' based on your GPU
    transformer = NunchakuFluxTransformer2dModel.from_pretrained(
        f"mit-han-lab/svdq-{precision}-flux.1-dev", offload=True
    )
    pipeline = FluxPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-dev", transformer=transformer, torch_dtype=torch.bfloat16
    )
    pipeline.enable_sequential_cpu_offload()

    save_dir = os.path.join("test_results", "bf16", "flux", "lora_reset")
    os.makedirs(save_dir, exist_ok=True)

    image = pipeline(
        "cozy mountain cabin covered in snow, with smoke curling from the chimney and a warm, inviting light spilling through the windows",  # noqa: E501
        num_inference_steps=8,
        guidance_scale=3.5,
        generator=torch.Generator().manual_seed(23),
    ).images[0]
    image.save(os.path.join(save_dir, "before.png"))

    transformer.update_lora_params("alimama-creative/FLUX.1-Turbo-Alpha/diffusion_pytorch_model.safetensors")
    transformer.set_lora_strength(50)
    transformer.reset_lora()

    image = pipeline(
        "cozy mountain cabin covered in snow, with smoke curling from the chimney and a warm, inviting light spilling through the windows",  # noqa: E501
        num_inference_steps=8,
        guidance_scale=3.5,
        generator=torch.Generator().manual_seed(23),
    ).images[0]
    image.save(os.path.join(save_dir, "after.png"))

    lpips = compute_lpips(os.path.join(save_dir, "before.png"), os.path.join(save_dir, "after.png"))
    print(f"LPIPS: {lpips}")
    assert lpips < 0.158 * 1.1
