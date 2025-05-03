# skip this test
import pytest

from nunchaku.utils import get_precision, is_turing

from .utils import run_test


@pytest.mark.skipif(is_turing(), reason="Skip tests due to using Turing GPUs")
@pytest.mark.parametrize(
    "height,width,attention_impl,cpu_offload,expected_lpips,batch_size",
    [
        (1024, 1024, "nunchaku-fp16", False, 0.140 if get_precision() == "int4" else 0.135, 2),
        (1920, 1080, "flashattn2", True, 0.177 if get_precision() == "int4" else 0.123, 4),
    ],
)
def test_flux_schnell(
    height: int, width: int, attention_impl: str, cpu_offload: bool, expected_lpips: float, batch_size: int
):
    run_test(
        precision=get_precision(),
        height=height,
        width=width,
        attention_impl=attention_impl,
        cpu_offload=cpu_offload,
        expected_lpips=expected_lpips,
        batch_size=batch_size,
    )
