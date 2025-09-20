import gc
import os
from pathlib import Path

import pytest
import torch
from diffusers import StableDiffusionXLPipeline

from nunchaku.models.unets.unet_sdxl import NunchakuSDXLUNet2DConditionModel
from nunchaku.utils import get_precision, is_turing

from ...flux.utils import already_generate, compute_lpips, hash_str_to_int
from .test_sdxl_turbo import plot, run_benchmark


@pytest.mark.skipif(
    is_turing() or get_precision() == "fp4", reason="Skip tests due to using Turing GPUs or FP4 precision"
)
@pytest.mark.parametrize("expected_lpips", [0.25 if get_precision() == "int4" else 0.18])
def test_sdxl_lpips(expected_lpips: float):
    gc.collect()
    torch.cuda.empty_cache()

    precision = get_precision()

    ref_root = Path(os.environ.get("NUNCHAKU_TEST_CACHE_ROOT", os.path.join("test_results", "ref")))
    results_dir_original = ref_root / "fp16" / "sdxl"
    results_dir_nunchaku = ref_root / precision / "sdxl"

    os.makedirs(results_dir_original, exist_ok=True)
    os.makedirs(results_dir_nunchaku, exist_ok=True)

    prompts = [
        "Ilya Repin, Moebius, Yoshitaka Amano, 1980s nubian punk rock glam core fashion shoot, closeup, 35mm ",
        "A honeybee sitting on a flower in a garden full of yellow flowers",
        "Vibrant, tropical rainforest, teeming with wildlife, nature photography ",
        "very realistic photo of barak obama in a wing eating contest",
        "oil paint of colorful wildflowers in a meadow, Paul Signac divisionism style ",
    ]

    if not already_generate(results_dir_original, 5):
        pipeline = StableDiffusionXLPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.bfloat16, use_safetensors=True, variant="fp16"
        ).to("cuda")

        for prompt in prompts:
            seed = hash_str_to_int(prompt)
            result = pipeline(
                prompt=prompt, guidance_scale=5.0, num_inference_steps=50, generator=torch.Generator().manual_seed(seed)
            ).images[0]
            result.save(os.path.join(results_dir_original, f"{seed}.png"))

        del pipeline.unet
        del pipeline.text_encoder
        del pipeline.text_encoder_2
        del pipeline.vae
        del pipeline
        del result
        gc.collect()
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

    free, total = torch.cuda.mem_get_info()
    print(f"After original generation: Free: {free/1024**2:.0f} MB  /  Total: {total/1024**2:.0f} MB")

    if not already_generate(results_dir_nunchaku, 5):
        quantized_unet = NunchakuSDXLUNet2DConditionModel.from_pretrained(
            "nunchaku-tech/nunchaku-sdxl/svdq-int4_r32-sdxl.safetensors"
        )
        pipeline = StableDiffusionXLPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            unet=quantized_unet,
            torch_dtype=torch.bfloat16,
            use_safetensors=True,
            variant="fp16",
        )
        pipeline.unet = quantized_unet
        pipeline = pipeline.to("cuda")
        for prompt in prompts:
            seed = hash_str_to_int(prompt)
            result = pipeline(
                prompt=prompt, guidance_scale=5.0, num_inference_steps=50, generator=torch.Generator().manual_seed(seed)
            ).images[0]
            result.save(os.path.join(results_dir_nunchaku, f"{seed}.png"))

        del pipeline
        del quantized_unet
        gc.collect()
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

    free, total = torch.cuda.mem_get_info()
    print(f"After Nunchaku generation: Free: {free/1024**2:.0f} MB  /  Total: {total/1024**2:.0f} MB")

    lpips = compute_lpips(results_dir_original, results_dir_nunchaku)
    print(f"lpips: {lpips}")
    assert lpips < expected_lpips * 1.15


@pytest.mark.skipif(is_turing(), reason="Skip tests due to using Turing GPUs")
def test_sdxl_time_cost():
    batch_sizes = [2, 4, 8]
    runs = 20
    inference_steps = 50
    guidance_scale = 5.0
    device_name = torch.cuda.get_device_name(0)
    results = {"Original FP16": [], "Nunchaku INT4": []}

    pipeline_original = StableDiffusionXLPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.bfloat16, use_safetensors=True, variant="fp16"
    ).to("cuda")

    for batch_size in batch_sizes:
        benchmark_original = run_benchmark(
            pipeline_original, batch_size, guidance_scale, device_name, runs, inference_steps
        )
        results["Original FP16"].append(benchmark_original.mean() * inference_steps)

    pipeline_original = None
    gc.collect()
    torch.cuda.empty_cache()

    quantized_unet = NunchakuSDXLUNet2DConditionModel.from_pretrained(
        "nunchaku-tech/nunchaku-sdxl/svdq-int4_r32-sdxl.safetensors"
    )
    pipeline_quantized = StableDiffusionXLPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        unet=quantized_unet,
        torch_dtype=torch.bfloat16,
        use_safetensors=True,
        variant="fp16",
    )

    pipeline_quantized.unet = quantized_unet
    pipeline_quantized = pipeline_quantized.to("cuda")

    for batch_size in batch_sizes:
        benchmark_quantized = run_benchmark(
            pipeline_quantized, batch_size, guidance_scale, device_name, runs, inference_steps
        )
        results["Nunchaku INT4"].append(benchmark_quantized.mean() * inference_steps)

    ref_root = Path(os.environ.get("NUNCHAKU_TEST_CACHE_ROOT", os.path.join("test_results", "ref")))
    plot_save_path = ref_root / "time_cost" / "sdxl"
    os.makedirs(plot_save_path, exist_ok=True)

    plot(batch_sizes, results, device_name, runs, inference_steps, plot_save_path, "SDXL")
