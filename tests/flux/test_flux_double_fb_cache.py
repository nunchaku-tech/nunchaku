import pytest

from nunchaku.utils import get_precision, is_turing

from .utils import run_test


@pytest.mark.skipif(is_turing(), reason="Skip tests due to using Turing GPUs")
@pytest.mark.parametrize(
    "use_double_fb_cache,residual_diff_threshold_multi,residual_diff_threshold_single,height,width,num_inference_steps,lora_name,lora_strength,expected_lpips",
    [
        (True, 0.09, 0.12, 1024, 1024, 30, None, 1, 0.24 if get_precision() == "int4" else 0.165),
        (True, 0.09, 0.12, 1024, 1024, 50, None, 1, 0.24 if get_precision() == "int4" else 0.161),
    ],
)
def test_flux_dev_double_fb_cache(
    use_double_fb_cache: bool,
    residual_diff_threshold_multi: float,
    residual_diff_threshold_single: float,
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
        use_double_fb_cache=use_double_fb_cache,
        residual_diff_threshold_multi=residual_diff_threshold_multi,
        residual_diff_threshold_single=residual_diff_threshold_single,
        expected_lpips=expected_lpips,
    )
