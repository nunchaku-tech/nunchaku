"""
Test for V2 Flux double FB cache implementation.
Tests the NunchakuFluxTransformer2DModelV2 with double FB cache enabled.
"""

import gc
import os
import sys

import pytest
import torch

if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from diffusers import FluxPipeline

from nunchaku.caching.diffusers_adapters.flux_v2 import apply_cache_on_pipe
from nunchaku.models.transformers.transformer_flux_v2 import NunchakuFluxTransformer2DModelV2
from nunchaku.utils import get_precision, is_turing


@pytest.mark.skipif(is_turing(), reason="Skip tests due to using Turing GPUs")
@pytest.mark.parametrize(
    "use_double_fb_cache,residual_diff_threshold_multi,residual_diff_threshold_single,height,width,num_inference_steps,expected_lpips",
    [
        (True, 0.09, 0.20, 1024, 1024, 30, 0.24 if get_precision() == "int4" else 0.165),
        (True, 0.09, 0.15, 1024, 1024, 50, 0.24 if get_precision() == "int4" else 0.161),
    ],
)
def test_flux_dev_double_fb_cache_v2(
    use_double_fb_cache: bool,
    residual_diff_threshold_multi: float,
    residual_diff_threshold_single: float,
    height: int,
    width: int,
    num_inference_steps: int,
    expected_lpips: float,
):
    gc.collect()
    torch.cuda.empty_cache()

    precision = get_precision()

    transformer = NunchakuFluxTransformer2DModelV2.from_pretrained(
        f"mit-han-lab/nunchaku-flux.1-dev/svdq-{precision}_r32-flux.1-dev.safetensors"
    )

    pipeline = FluxPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-dev", transformer=transformer, torch_dtype=torch.bfloat16
    ).to("cuda")

    apply_cache_on_pipe(
        pipeline,
        use_double_fb_cache=use_double_fb_cache,
        residual_diff_threshold_multi=residual_diff_threshold_multi,
        residual_diff_threshold_single=residual_diff_threshold_single,
    )

    prompt = "A cat holding a sign that says hello world"
    generator = torch.Generator("cuda").manual_seed(42)

    image = pipeline(
        prompt,
        num_inference_steps=num_inference_steps,
        height=height,
        width=width,
        guidance_scale=3.5,
        generator=generator,
    ).images[0]

    assert image is not None
    assert image.size == (width, height)

    del pipeline, transformer
    gc.collect()
    torch.cuda.empty_cache()


@pytest.mark.skipif(is_turing(), reason="Skip tests due to using Turing GPUs")
def test_v2_cache_verbose_logging():
    """Test V2 cache with verbose logging enabled."""

    gc.collect()
    torch.cuda.empty_cache()

    precision = get_precision()

    transformer = NunchakuFluxTransformer2DModelV2.from_pretrained(
        f"mit-han-lab/nunchaku-flux.1-dev/svdq-{precision}_r32-flux.1-dev.safetensors"
    )

    transformer.verbose = True

    pipeline = FluxPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-dev", transformer=transformer, torch_dtype=torch.bfloat16
    ).to("cuda")

    apply_cache_on_pipe(
        pipeline,
        use_double_fb_cache=True,
        residual_diff_threshold_multi=0.09,
        residual_diff_threshold_single=0.12,
    )

    prompt = "A simple test image"
    generator = torch.Generator("cuda").manual_seed(42)

    image = pipeline(
        prompt, num_inference_steps=5, height=512, width=512, guidance_scale=3.5, generator=generator
    ).images[0]

    assert image is not None
    assert image.size == (512, 512)

    # Clean up
    del pipeline, transformer
    gc.collect()
    torch.cuda.empty_cache()


@pytest.mark.skipif(is_turing(), reason="Skip tests due to using Turing GPUs")
@pytest.mark.parametrize(
    "threshold_single",
    [0.08, 0.10, 0.12, 0.15, 0.20],  # Test different thresholds
)
def test_v2_threshold_variations(threshold_single: float):
    """Test V2 with different threshold_single values."""

    gc.collect()
    torch.cuda.empty_cache()

    precision = get_precision()

    transformer = NunchakuFluxTransformer2DModelV2.from_pretrained(
        f"mit-han-lab/nunchaku-flux.1-dev/svdq-{precision}_r32-flux.1-dev.safetensors"
    )

    pipeline = FluxPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-dev", transformer=transformer, torch_dtype=torch.bfloat16
    ).to("cuda")

    apply_cache_on_pipe(
        pipeline,
        use_double_fb_cache=True,
        residual_diff_threshold_multi=0.09,
        residual_diff_threshold_single=threshold_single,
    )

    prompt = "A beautiful landscape"
    generator = torch.Generator("cuda").manual_seed(42)

    image = pipeline(
        prompt, num_inference_steps=10, height=512, width=512, guidance_scale=3.5, generator=generator
    ).images[0]

    assert image is not None
    assert image.size == (512, 512)

    del pipeline, transformer
    gc.collect()
    torch.cuda.empty_cache()


@pytest.mark.skipif(is_turing(), reason="Skip tests due to using Turing GPUs")
def test_v2_memory_usage():
    """Test V2 memory usage with cache enabled."""

    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    precision = get_precision()

    transformer = NunchakuFluxTransformer2DModelV2.from_pretrained(
        f"mit-han-lab/nunchaku-flux.1-dev/svdq-{precision}_r32-flux.1-dev.safetensors"
    )

    pipeline = FluxPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-dev", transformer=transformer, torch_dtype=torch.bfloat16
    ).to("cuda")

    apply_cache_on_pipe(
        pipeline,
        use_double_fb_cache=True,
        residual_diff_threshold_multi=0.09,
        residual_diff_threshold_single=0.15,
    )

    # Measure memory before generation
    torch.cuda.synchronize()
    memory_before = torch.cuda.memory_allocated() / (1024**3)  # GB

    # Generate image
    prompt = "Memory test image"
    generator = torch.Generator("cuda").manual_seed(42)

    image = pipeline(
        prompt, num_inference_steps=20, height=1024, width=1024, guidance_scale=3.5, generator=generator
    ).images[0]

    # Measure peak memory
    torch.cuda.synchronize()
    peak_memory = torch.cuda.max_memory_allocated() / (1024**3)  # GB

    print(f"Memory before: {memory_before:.2f} GB")
    print(f"Peak memory: {peak_memory:.2f} GB")
    print(f"Memory increase: {peak_memory - memory_before:.2f} GB")

    # V2 typically uses ~18GB based on our tests
    assert peak_memory < 20, f"Peak memory {peak_memory:.2f} GB exceeds expected limit"

    assert image is not None
    assert image.size == (1024, 1024)

    # Clean up
    del pipeline, transformer
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    print("Running V2 double FB cache tests...")

    test_flux_dev_double_fb_cache_v2(
        use_double_fb_cache=True,
        residual_diff_threshold_multi=0.09,
        residual_diff_threshold_single=0.12,
        height=512,
        width=512,
        num_inference_steps=10,
        expected_lpips=0.24 if get_precision() == "int4" else 0.165,
    )
    print("✓ Basic test passed")

    test_v2_cache_verbose_logging()
    print("✓ Verbose logging test passed")

    test_v2_memory_usage()
    print("✓ Memory usage test passed")

    print("\nAll V2 double FB cache tests passed!")
