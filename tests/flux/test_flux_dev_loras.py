import pytest

from nunchaku.utils import get_precision, is_turing

from .utils import run_test


@pytest.mark.skipif(is_turing(), reason="Skip tests due to using Turing GPUs")
@pytest.mark.parametrize(
    "num_inference_steps,lora_name,lora_strength,cpu_offload,expected_lpips",
    [
        (25, "realism", 0.9, True, 0.136 if get_precision() == "int4" else 0.112),
        # (25, "ghibsky", 1, False, 0.186),
        # (28, "anime", 1, False, 0.284),
        (24, "sketch", 1, True, 0.291 if get_precision() == "int4" else 0.182),
        # (28, "yarn", 1, False, 0.211),
        # (25, "haunted_linework", 1, True, 0.317),
    ],
)
def test_flux_dev_loras(num_inference_steps, lora_name, lora_strength, cpu_offload, expected_lpips):
    run_test(
        precision=get_precision(),
        model_name="flux.1-dev",
        dataset_name=lora_name,
        height=1024,
        width=1024,
        num_inference_steps=num_inference_steps,
        guidance_scale=3.5,
        use_qencoder=False,
        attention_impl="nunchaku-fp16",
        cpu_offload=cpu_offload,
        lora_names=lora_name,
        lora_strengths=lora_strength,
        cache_threshold=0,
        expected_lpips=expected_lpips,
    )


# lora composition & large rank loras
@pytest.mark.skipif(is_turing(), reason="Skip tests due to using Turing GPUs")
def test_flux_dev_turbo8_ghibsky_1024x1024():
    run_test(
        precision=get_precision(),
        model_name="flux.1-dev",
        dataset_name="haunted_linework",
        height=1024,
        width=1024,
        num_inference_steps=8,
        guidance_scale=3.5,
        use_qencoder=False,
        cpu_offload=True,
        lora_names=["realism", "ghibsky", "anime", "sketch", "yarn", "haunted_linework", "turbo8"],
        lora_strengths=[0, 1, 0, 0, 0, 0, 1],
        cache_threshold=0,
        expected_lpips=0.310 if get_precision() == "int4" else 0.168,
    )
