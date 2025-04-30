import pytest

from nunchaku.utils import get_precision, is_turing

from .utils import run_test


@pytest.mark.skipif(is_turing(), reason="Skip tests due to using Turing GPUs")
@pytest.mark.parametrize(
    "height,width,attention_impl,cpu_offload,expected_lpips",
    [(1024, 1024, "nunchaku-fp16", False, 0.209 if get_precision() == "int4" else 0.148)],
)
def test_shuttle_jaguar(height: int, width: int, attention_impl: str, cpu_offload: bool, expected_lpips: float):
    run_test(
        precision=get_precision(),
        model_name="shuttle-jaguar",
        height=height,
        width=width,
        attention_impl=attention_impl,
        cpu_offload=cpu_offload,
        expected_lpips=expected_lpips,
    )
