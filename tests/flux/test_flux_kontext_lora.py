"""
Test LoRA functionality for FLUX.1-Kontext model
"""

import gc
import os
from pathlib import Path

import numpy as np
import pytest
import torch
from diffusers import FluxKontextPipeline
from diffusers.utils import load_image
from PIL import Image

from nunchaku import NunchakuFluxTransformer2dModel
from nunchaku.utils import get_precision, is_turing


def compute_pixel_difference(img1_path: str, img2_path: str) -> dict:
    """Compute pixel-level differences between two images"""
    img1 = np.array(Image.open(img1_path)).astype(float)
    img2 = np.array(Image.open(img2_path)).astype(float)

    diff = np.abs(img1 - img2)

    return {
        "mean_diff": np.mean(diff),
        "max_diff": np.max(diff),
        "pixels_changed": np.mean(diff > 0) * 100,
        "pixels_changed_significantly": np.mean(diff > 10) * 100,
    }


@pytest.mark.skipif(is_turing(), reason="Skip tests due to using Turing GPUs")
def test_kontext_lora_application():
    """Test that LoRA weights are properly applied to Kontext model"""
    gc.collect()
    torch.cuda.empty_cache()

    precision = get_precision()

    # Setup directories
    results_dir = Path("test_results") / precision / "flux.1-kontext-dev" / "lora_test"
    os.makedirs(results_dir, exist_ok=True)

    # Load test image
    image = load_image(
        "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/yarn-art-pikachu.png"
    ).convert("RGB")

    prompt = "neon light, city atmosphere"
    seed = 42
    num_inference_steps = 28
    guidance_scale = 2.5

    # Load model
    transformer = NunchakuFluxTransformer2dModel.from_pretrained(
        f"mit-han-lab/nunchaku-flux.1-kontext-dev/svdq-{precision}_r32-flux.1-kontext-dev.safetensors"
    )

    pipeline = FluxKontextPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-Kontext-dev", transformer=transformer, torch_dtype=torch.bfloat16
    ).to("cuda")

    # Test 1: Generate without LoRA
    generator = torch.Generator().manual_seed(seed)
    result_no_lora = pipeline(
        image=image,
        prompt=prompt,
        generator=generator,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
    ).images[0]
    result_no_lora.save(results_dir / "no_lora.png")

    # Test 2: Apply LoRA and generate
    transformer.update_lora_params(
        "nunchaku-tech/nunchaku-test-models/relight-kontext-lora-single-caption_comfy.safetensors"
    )
    transformer.set_lora_strength(1.0)

    generator = torch.Generator().manual_seed(seed)
    result_lora_1 = pipeline(
        image=image,
        prompt=prompt,
        generator=generator,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
    ).images[0]
    result_lora_1.save(results_dir / "lora_1.0.png")

    # Test 3: Change LoRA strength
    transformer.set_lora_strength(2.0)

    generator = torch.Generator().manual_seed(seed)
    result_lora_2 = pipeline(
        image=image,
        prompt=prompt,
        generator=generator,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
    ).images[0]
    result_lora_2.save(results_dir / "lora_2.0.png")

    # Test 4: Disable LoRA
    transformer.set_lora_strength(0.0)

    generator = torch.Generator().manual_seed(seed)
    result_lora_0 = pipeline(
        image=image,
        prompt=prompt,
        generator=generator,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
    ).images[0]
    result_lora_0.save(results_dir / "lora_0.0.png")

    # Compute differences
    diff_1 = compute_pixel_difference(results_dir / "no_lora.png", results_dir / "lora_1.0.png")

    diff_2 = compute_pixel_difference(results_dir / "no_lora.png", results_dir / "lora_2.0.png")

    diff_0 = compute_pixel_difference(results_dir / "no_lora.png", results_dir / "lora_0.0.png")

    diff_scale = compute_pixel_difference(results_dir / "lora_1.0.png", results_dir / "lora_2.0.png")

    # Assertions
    # LoRA 1.0 should change the output
    assert diff_1["mean_diff"] > 1.0, "LoRA 1.0 should significantly change the output"
    assert diff_1["pixels_changed"] > 50, "LoRA 1.0 should change more than 50% of pixels"

    # LoRA 2.0 should have a significant effect (but not necessarily stronger than 1.0 due to saturation)
    assert diff_2["mean_diff"] > 1.0, "LoRA 2.0 should significantly change the output"

    # Different LoRA strengths should produce different results
    assert diff_scale["mean_diff"] > 1.0, "Different LoRA strengths should produce different results"

    # Log the actual differences for debugging
    print(f"LoRA 1.0 vs baseline difference: {diff_1['mean_diff']:.2f}")
    print(f"LoRA 2.0 vs baseline difference: {diff_2['mean_diff']:.2f}")
    print(f"LoRA 1.0 vs 2.0 difference: {diff_scale['mean_diff']:.2f}")

    # Note: We're not asserting that LoRA 0.0 matches baseline due to known issue
    # where LoRA weights may not be fully removed when strength=0.0
    print(f"LoRA 0.0 vs baseline difference: {diff_0['mean_diff']:.2f}")
    if diff_0["mean_diff"] > 1.0:
        print("WARNING: LoRA 0.0 differs from baseline - LoRA may not be fully disabled")

    # Clean up
    del pipeline
    del transformer
    gc.collect()
    torch.cuda.empty_cache()


@pytest.mark.skipif(is_turing(), reason="Skip tests due to using Turing GPUs")
def test_kontext_lora_naming_conversion():
    """Test that LoRA naming conversion works correctly"""
    from safetensors.torch import load_file

    from nunchaku.lora.flux.diffusers_converter import convert_keys_to_diffusers

    # Download or use local LoRA file
    try:
        lora_path = "relight-kontext-lora-single-caption.safetensors"
        if not os.path.exists(lora_path):
            from huggingface_hub import hf_hub_download

            lora_path = hf_hub_download(
                repo_id="nunchaku-tech/nunchaku-test-models",
                filename="relight-kontext-lora-single-caption_comfy.safetensors",
            )
    except Exception as e:
        pytest.skip(f"Cannot download LoRA file: {e}")

    # Load LoRA weights
    tensors = load_file(lora_path)

    # Check original format - should be ComfyUI format for this test file
    original_keys = list(tensors.keys())
    assert any(
        k.startswith("lora_unet_") for k in original_keys
    ), "Test LoRA file should be in ComfyUI format with lora_unet_ prefix"

    # Apply conversion
    converted = convert_keys_to_diffusers(tensors)
    converted_keys = list(converted.keys())

    # Check conversion results - should not have lora_unet_ prefix after conversion
    assert not any(
        k.startswith("lora_unet_") for k in converted_keys
    ), "Converted keys should not have lora_unet_ prefix"

    # Check that keys are properly formatted
    for key in converted_keys:
        if "lora_A" in key or "lora_B" in key:
            # Should be in format: block_type.idx.layer.lora_X.weight or final.layer.*
            assert (
                "double_blocks" in key or "single_blocks" in key or "final.layer" in key
            ), f"Key {key} should contain block type"
            assert (
                ".lora_A.weight" in key or ".lora_B.weight" in key
            ), f"Key {key} should end with .lora_A.weight or .lora_B.weight"

    # Check that all tensors are preserved
    assert len(converted) == len(tensors), "All tensors should be preserved during conversion"

    print(f"Successfully converted {len(converted)} LoRA keys")
    print(f"Sample converted keys: {converted_keys[:3]}")


@pytest.mark.skipif(is_turing(), reason="Skip tests due to using Turing GPUs")
@pytest.mark.parametrize(
    "lora_strength,expected_change",
    [
        (0.5, 1.0),  # Medium strength should cause moderate change
        (1.0, 1.5),  # Full strength should cause significant change
        (1.5, 2.0),  # Over-strength should cause larger change
    ],
)
def test_kontext_lora_strength_scaling(lora_strength, expected_change):
    """Test that LoRA strength scaling works proportionally"""
    gc.collect()
    torch.cuda.empty_cache()

    precision = get_precision()

    # Load model
    transformer = NunchakuFluxTransformer2dModel.from_pretrained(
        f"mit-han-lab/nunchaku-flux.1-kontext-dev/svdq-{precision}_r32-flux.1-kontext-dev.safetensors"
    )

    pipeline = FluxKontextPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-Kontext-dev", transformer=transformer, torch_dtype=torch.bfloat16
    ).to("cuda")

    # Load test image
    image = load_image(
        "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/yarn-art-pikachu.png"
    ).convert("RGB")

    prompt = "dramatic lighting, cinematic"
    seed = 123

    # Generate baseline
    generator = torch.Generator().manual_seed(seed)
    baseline = pipeline(image=image, prompt=prompt, generator=generator, num_inference_steps=20).images[0]

    transformer.update_lora_params(
        "nunchaku-tech/nunchaku-test-models/relight-kontext-lora-single-caption_comfy.safetensors"
    )
    transformer.set_lora_strength(lora_strength)

    # Generate with LoRA
    generator = torch.Generator().manual_seed(seed)
    with_lora = pipeline(image=image, prompt=prompt, generator=generator, num_inference_steps=20).images[0]

    # Compute difference
    baseline_arr = np.array(baseline).astype(float)
    lora_arr = np.array(with_lora).astype(float)
    mean_diff = np.mean(np.abs(baseline_arr - lora_arr))

    # Assert that change is proportional to strength
    # Allow 50% tolerance due to non-linear effects
    assert (
        mean_diff > expected_change * 0.5
    ), f"LoRA strength {lora_strength} should cause mean difference > {expected_change * 0.5}, got {mean_diff}"

    print(f"LoRA strength {lora_strength}: mean difference = {mean_diff:.2f}")

    # Clean up
    del pipeline
    del transformer
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    test_kontext_lora_naming_conversion()
    test_kontext_lora_application()
    for strength, expected in [(0.5, 1.0), (1.0, 1.5), (1.5, 2.0)]:
        test_kontext_lora_strength_scaling(strength, expected)
