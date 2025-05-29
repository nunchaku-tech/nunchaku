import pytest
import torch
from diffusers import FluxPipeline

from nunchaku import NunchakuFluxTransformer2dModel
from nunchaku.utils import get_precision, is_turing


@pytest.mark.skipif(
    is_turing() or torch.cuda.device_count() <= 1, reason="Skip tests due to using Turing GPUs or single GPU"
)
def test_device_id():
    precision = get_precision()  # auto-detect your precision is 'int4' or 'fp4' based on your GPU
    torch_dtype = torch.float16 if is_turing("cuda:1") else torch.bfloat16
    transformer = NunchakuFluxTransformer2dModel.from_pretrained(
        f"mit-han-lab/nunchaku-flux.1-schnell/svdq-{precision}_r32-flux.1-schnell.safetensors",
        torch_dtype=torch_dtype,
        device="cuda:1",
    )
    pipeline = FluxPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-schnell", transformer=transformer, torch_dtype=torch_dtype
    ).to("cuda:1")
    pipeline(
        "A cat holding a sign that says hello world", width=1024, height=1024, num_inference_steps=4, guidance_scale=0
    )
