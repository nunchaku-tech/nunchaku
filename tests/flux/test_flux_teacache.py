import gc
import os

import pytest
import torch
from diffusers.pipelines.flux.pipeline_flux import FluxPipeline

from nunchaku import NunchakuFluxTransformer2dModel
from nunchaku.caching.teacache import TeaCache
from nunchaku.utils import get_precision, is_turing

from .utils import already_generate, compute_lpips, offload_pipeline


@pytest.mark.skipif(is_turing(), reason="Skip tests due to using Turing GPUs")
@pytest.mark.parametrize(
    "height,width,num_inference_steps,prompt,name,seed,threshold,expected_lpips",
    [
        (
            1024,
            1024,
            30,
            "A cat holding a sign that says hello world",
            "cat",
            0,
            0.6,
            0.363 if get_precision() == "int4" else 0.363,
        ),
        (
            512,
            2048,
            25,
            "The brown fox jumps over the lazy dog",
            "fox",
            1234,
            0.7,
            0.417 if get_precision() == "int4" else 0.349,
        ),
        (
            1024,
            768,
            50,
            "A scene from the Titanic movie featuring the Muppets",
            "muppets",
            42,
            0.3,
            0.360 if get_precision() == "int4" else 0.495,
        ),
        (
            1024,
            768,
            50,
            "A crystal ball showing a waterfall",
            "waterfall",
            23,
            0.6,
            0.253 if get_precision() == "int4" else 0.226,
        ),
    ],
)
def test_flux_teacache(
    height: int,
    width: int,
    num_inference_steps: int,
    prompt: str,
    name: str,
    seed: int,
    threshold: float,
    expected_lpips: float,
):
    gc.collect()
    torch.cuda.empty_cache()

    device = torch.device("cuda")
    precision = get_precision()

    ref_root = os.environ.get("NUNCHAKU_TEST_CACHE_ROOT", os.path.join("test_results", "ref"))
    results_dir_16_bit = os.path.join(ref_root, "bf16", "flux.1-dev", "teacache", name)
    results_dir_4_bit = os.path.join("test_results", precision, "flux.1-dev", "teacache", name)

    os.makedirs(results_dir_16_bit, exist_ok=True)
    os.makedirs(results_dir_4_bit, exist_ok=True)

    # First, generate results with the 16-bit model
    if not already_generate(results_dir_16_bit, 1):
        pipeline = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-dev", torch_dtype=torch.bfloat16)

        # Possibly offload the model to CPU when GPU memory is scarce
        pipeline = offload_pipeline(pipeline)
        result = pipeline(
            prompt=prompt,
            num_inference_steps=num_inference_steps,
            height=height,
            width=width,
            generator=torch.Generator(device=device).manual_seed(seed),
        ).images[0]
        result.save(os.path.join(results_dir_16_bit, f"{name}_{seed}.png"))

        # Clean up the 16-bit model
        del pipeline.transformer
        del pipeline.text_encoder
        del pipeline.text_encoder_2
        del pipeline.vae
        del pipeline
        del result
        gc.collect()
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

    free, total = torch.cuda.mem_get_info()  # bytes
    print(f"After 16-bit generation: Free: {free/1024**2:.0f} MB  /  Total: {total/1024**2:.0f} MB")

    # Then, generate results with the 4-bit model
    if not already_generate(results_dir_4_bit, 1):
        transformer = NunchakuFluxTransformer2dModel.from_pretrained(f"mit-han-lab/svdq-{precision}-flux.1-dev")
        pipeline = FluxPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-dev", transformer=transformer, torch_dtype=torch.bfloat16
        ).to("cuda")
        with torch.inference_mode():
            with TeaCache(
                model=pipeline.transformer, num_steps=num_inference_steps, rel_l1_thresh=threshold, enabled=True
            ):
                result = pipeline(
                    prompt=prompt,
                    num_inference_steps=num_inference_steps,
                    height=height,
                    width=width,
                    generator=torch.Generator(device=device).manual_seed(seed),
                ).images[0]
        result.save(os.path.join(results_dir_4_bit, f"{name}_{seed}.png"))

        # Clean up the 4-bit model
        del pipeline
        del transformer
        gc.collect()
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

    free, total = torch.cuda.mem_get_info()  # bytes
    print(f"After 4-bit generation: Free: {free/1024**2:.0f} MB  /  Total: {total/1024**2:.0f} MB")

    lpips = compute_lpips(results_dir_16_bit, results_dir_4_bit)
    print(f"lpips: {lpips}")
    assert lpips < expected_lpips * 1.1
