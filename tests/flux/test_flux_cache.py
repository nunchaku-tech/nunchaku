import pytest

from nunchaku.utils import get_precision, is_turing

from .utils import run_test


@pytest.mark.skipif(is_turing(), reason="Skip tests due to using Turing GPUs")
@pytest.mark.parametrize(
    "cache_threshold,height,width,num_inference_steps,lora_name,lora_strength,expected_lpips",
    [
        (0.12, 1024, 1024, 30, None, 1, 0.212 if get_precision() == "int4" else 0.161),
    ],
)
def test_flux_dev_cache(
    cache_threshold: float,
    height: int,
    width: int,
    num_inference_steps: int,
    lora_name: str,
    lora_strength: float,
    expected_lpips: float,
):
    run_test(
        precision=get_precision(),
        model_name="flux.1-dev",
        dataset_name="MJHQ" if lora_name is None else lora_name,
        height=height,
        width=width,
        num_inference_steps=num_inference_steps,
        guidance_scale=3.5,
        use_qencoder=False,
        cpu_offload=False,
        lora_names=lora_name,
        lora_strengths=lora_strength,
        cache_threshold=cache_threshold,
        expected_lpips=expected_lpips,
    )
