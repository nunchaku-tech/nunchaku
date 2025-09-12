import pytest
import torch

from nunchaku.utils import get_precision, is_turing

from .utils import run_test


@pytest.mark.skipif(is_turing(), reason="Skip tests due to using Turing GPUs")
@pytest.mark.parametrize(
    "height,width,num_inference_steps,attention_impl,cpu_offload,expected_lpips",
    [
        (1024, 1024, 50, "flashattn2", False, 0.139 if get_precision() == "int4" else 0.146),
        (2048, 512, 25, "nunchaku-fp16", False, 0.168 if get_precision() == "int4" else 0.156),
    ],
)
def test_flux_dev(
    height: int, width: int, num_inference_steps: int, attention_impl: str, cpu_offload: bool, expected_lpips: float
):
    run_test(
        precision=get_precision(),
        model_name="flux.1-dev",
        height=height,
        width=width,
        num_inference_steps=num_inference_steps,
        attention_impl=attention_impl,
        cpu_offload=cpu_offload,
        expected_lpips=expected_lpips,
    )


@pytest.mark.skipif(is_turing(), reason="Skip tests due to using Turing GPUs")
@pytest.mark.parametrize(
    "dtype,height,width,num_inference_steps,attention_impl,cpu_offload,expected_lpips",
    [
        # Test with float16
        (torch.float16, 1024, 1024, 50, "flashattn2", False, 0.139 if get_precision() == "int4" else 0.146),
        (torch.float16, 2048, 512, 25, "nunchaku-fp16", False, 0.168 if get_precision() == "int4" else 0.156),
        # Test with bfloat16
        (torch.bfloat16, 1024, 1024, 50, "flashattn2", False, 0.139 if get_precision() == "int4" else 0.146),
        (torch.bfloat16, 2048, 512, 25, "nunchaku-fp16", False, 0.168 if get_precision() == "int4" else 0.156),
    ],
)
def test_flux_dev_with_dtype(
    dtype: torch.dtype,
    height: int,
    width: int,
    num_inference_steps: int,
    attention_impl: str,
    cpu_offload: bool,
    expected_lpips: float,
):
    """Test flux.1-dev with both float16 and bfloat16 dtypes."""
    run_test(
        precision=get_precision(),
        model_name="flux.1-dev",
        dtype=dtype,
        height=height,
        width=width,
        num_inference_steps=num_inference_steps,
        attention_impl=attention_impl,
        cpu_offload=cpu_offload,
        expected_lpips=expected_lpips,
    )
