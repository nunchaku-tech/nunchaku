import gc
import os
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest
import torch
from diffusers import StableDiffusionXLPipeline

from nunchaku.models.unets.unet_sdxl import NunchakuSDXLUNet2DConditionModel
from nunchaku.utils import get_precision, is_turing

from ...flux.utils import already_generate, compute_lpips, hash_str_to_int


@pytest.mark.skipif(
    is_turing() or get_precision() == "fp4", reason="Skip tests due to using Turing GPUs or FP4 precision"
)
@pytest.mark.parametrize("expected_lpips", [0.25 if get_precision() == "int4" else 0.18])
def test_sdxl_turbo_lpips(expected_lpips: float):
    gc.collect()
    torch.cuda.empty_cache()

    precision = get_precision()

    ref_root = Path(os.environ.get("NUNCHAKU_TEST_CACHE_ROOT", os.path.join("test_results", "ref")))
    results_dir_original = ref_root / "fp16" / "sdxl-turbo"
    results_dir_nunchaku = ref_root / precision / "sdxl-turbo"

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
            "stabilityai/sdxl-turbo", torch_dtype=torch.bfloat16, variant="fp16"
        ).to("cuda")

        for prompt in prompts:
            seed = hash_str_to_int(prompt)
            result = pipeline(
                prompt=prompt, guidance_scale=0.0, num_inference_steps=4, generator=torch.Generator().manual_seed(seed)
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
            "nunchaku-tech/nunchaku-sdxl-turbo/svdq-int4_r32-sdxl-turbo.safetensors"
        )
        pipeline = StableDiffusionXLPipeline.from_pretrained(
            "stabilityai/sdxl-turbo", unet=quantized_unet, torch_dtype=torch.bfloat16, variant="fp16"
        )
        pipeline = pipeline.to("cuda")
        for prompt in prompts:
            seed = hash_str_to_int(prompt)
            result = pipeline(
                prompt=prompt, guidance_scale=0.0, num_inference_steps=4, generator=torch.Generator().manual_seed(seed)
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


class PerfHook:
    def __init__(self):
        self.start = []
        self.end = []

    def pre_hook(self, module, input):
        self.start.append(time.perf_counter())

    def post_hook(self, module, input, output):
        self.end.append(time.perf_counter())


def run_benchmark(pipeline, batch_size, guidance_scale, device, runs, inference_steps):

    prompt = "A cinematic shot of a baby racoon wearing an intricate italian priest robe."
    # warmup
    _ = pipeline(
        prompt=prompt,
        guidance_scale=guidance_scale,
        num_inference_steps=inference_steps,
        num_images_per_prompt=batch_size,
    ).images
    time_cost = []

    unet = pipeline.unet

    perf_hook = PerfHook()

    handle_pre = unet.register_forward_pre_hook(perf_hook.pre_hook)
    handle_post = unet.register_forward_hook(perf_hook.post_hook)

    # run
    for _ in range(runs):
        _ = pipeline(
            prompt=prompt,
            guidance_scale=guidance_scale,
            num_inference_steps=inference_steps,
            num_images_per_prompt=batch_size,
        ).images
    time_cost = [perf_hook.end[i] - perf_hook.start[i] for i in range(len(perf_hook.start))]

    # to numpy for stats
    time_cost = np.array(time_cost)
    print(f"device: {device}")
    print(f"runs :{runs}")
    print(f"batch_size: {batch_size}")
    print(f"max :{time_cost.max():.4f}")
    print(f"min :{time_cost.min():.4f}")
    print(f"avg :{time_cost.mean():.4f}")
    print(f"std :{time_cost.std():.4f}")

    handle_pre.remove()
    handle_post.remove()

    return time_cost


def plot(batch_sizes, results, device_name, runs, inference_steps, plot_save_path, title):
    x = np.arange(len(batch_sizes))
    width = 0.35

    fig, ax = plt.subplots()
    rects2 = ax.bar(x + width / 2, results["Nunchaku INT4"], width, label="Nunchaku INT4")

    ax.set_ylabel(f"Average time cost (seconds)\n{runs} runs of {inference_steps} inference steps each.")
    ax.set_xlabel("Batch size")
    ax.set_title(f"{title} diffusion time cost\n(GPU: {device_name})")
    ax.set_xticks(x)
    ax.set_xticklabels(batch_sizes)
    ax.legend()

    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(
                f"{height:.3f}",
                xy=(rect.get_x() + rect.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
            )

    autolabel(rects2)

    plt.tight_layout()
    plt.savefig(plot_save_path / "plot.png", dpi=300, bbox_inches="tight")


@pytest.mark.skipif(
    is_turing() or get_precision() == "fp4", reason="Skip tests due to using Turing GPUs or FP4 precision"
)
@pytest.mark.parametrize("expected_latency", [0.306])
def test_sdxl_turbo_time_cost(expected_latency: float):
    batch_size = 8
    runs = 5
    guidance_scale = 0.0
    inference_steps = 4
    device_name = torch.cuda.get_device_name(0)
    results = {"Nunchaku INT4": []}

    quantized_unet = NunchakuSDXLUNet2DConditionModel.from_pretrained(
        "nunchaku-tech/nunchaku-sdxl-turbo/svdq-int4_r32-sdxl-turbo.safetensors"
    )
    pipeline_quantized = StableDiffusionXLPipeline.from_pretrained(
        "stabilityai/sdxl-turbo", unet=quantized_unet, torch_dtype=torch.bfloat16, variant="fp16"
    )

    pipeline_quantized = pipeline_quantized.to("cuda")

    benchmark_quantized = run_benchmark(
        pipeline_quantized, batch_size, guidance_scale, device_name, runs, inference_steps
    )
    avg_latency = benchmark_quantized.mean() * inference_steps
    results["Nunchaku INT4"].append(avg_latency)

    ref_root = Path(os.environ.get("NUNCHAKU_TEST_CACHE_ROOT", os.path.join("test_results", "ref")))
    plot_save_path = ref_root / "time_cost" / "sdxl-turbo"
    os.makedirs(plot_save_path, exist_ok=True)

    plot([batch_size], results, device_name, runs, inference_steps, plot_save_path, "SDXL-Turbo")

    assert avg_latency < expected_latency * 1.1
