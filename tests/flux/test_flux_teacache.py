import gc
import os
import pytest

import torch
from diffusers.pipelines.flux.pipeline_flux import FluxPipeline

from examples.teacache import TeaCache
from nunchaku.utils import get_precision, is_turing
from nunchaku import NunchakuFluxTransformer2dModel
from tests.utils import already_generate, compute_lpips

@pytest.mark.skipif(is_turing(), reason="Skip tests due to using Turing GPUs")
@pytest.mark.parametrize(
    "height,width,num_inference_steps,prompt,name,seed,threshold,expected_lpips",
    [
        (1024, 1024, 30, "A cat holding a sign that says hello world", "cat", 0, 0.6, 0.226),
        (512, 2048, 25, "The brown fox jumps over the lazy dog", "fox", 1234, 0.7, 0.089),
        (1024, 768, 50, "A scene from the Titanic movie featuring the Muppets", "muppets", 42, 0.3, 0.393),
        (1024, 768, 50, "A crystal ball showing a waterfall", "waterfall", 42, 0.6, 0.091),
    ],
)
def test_flux_teacache(
    height: int, width: int, num_inference_steps: int, prompt: str, name: str, seed: int, threshold: float, expected_lpips: float
):
    device = torch.device("cuda")
    precision = get_precision()
    results_dir_16_bit = os.path.join("test_results", "bfloat16", "flux.1-dev", "teacache", name)
    results_dir_4_bit = os.path.join("test_results", precision, "flux.1-dev", "teacache", name)

    os.makedirs(results_dir_16_bit, exist_ok=True)
    os.makedirs(results_dir_4_bit, exist_ok=True)

    # First, generate results with the 16-bit model
    if not already_generate(results_dir_16_bit, 1):
        pipeline = FluxPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-dev", torch_dtype=torch.bfloat16
        ).to(device)
        with torch.inference_mode():
            with TeaCache(
                model=pipeline.transformer,
                num_steps=num_inference_steps,
                rel_l1_thresh=threshold,
                enabled=True,
            ):
                result = pipeline(
                    prompt=prompt,
                    num_inference_steps=num_inference_steps,
                    height=height,
                    width=width,
                    generator=torch.Generator(device=device).manual_seed(seed)
                ).images[0]
                result.save(os.path.join(results_dir_16_bit, f"{name}_{seed}.png"))

        # Clean up the 16-bit model
        del pipeline
        del result
        gc.collect()
        torch.cuda.empty_cache()

    # Then, generate results with the 4-bit model
    if not already_generate(results_dir_4_bit, 1):
        transformer = NunchakuFluxTransformer2dModel.from_pretrained(f"mit-han-lab/svdq-{precision}-flux.1-dev")
        pipeline = FluxPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-dev", transformer=transformer, torch_dtype=torch.bfloat16
        ).to(device)
        with torch.inference_mode():
            with TeaCache(
                model=pipeline.transformer,
                num_steps=num_inference_steps,
                rel_l1_thresh=threshold,
                enabled=True,
            ):
                result = pipeline(
                    prompt=prompt,
                    num_inference_steps=num_inference_steps,
                    height=height,
                    width=width,
                    generator=torch.Generator(device=device).manual_seed(seed)
                ).images[0]
                result.save(os.path.join(results_dir_4_bit, f"{name}_{seed}.png"))

        # Clean up the 4-bit model
        del pipeline
        del transformer
        del result
        gc.collect()
        torch.cuda.empty_cache()

    lpips = compute_lpips(results_dir_16_bit, results_dir_4_bit)
    print(f"lpips: {lpips}")
    assert lpips < expected_lpips * 1.1
