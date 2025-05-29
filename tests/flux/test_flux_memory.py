import pytest
import torch
from diffusers import FluxPipeline

from nunchaku import NunchakuFluxTransformer2dModel, NunchakuT5EncoderModel
from nunchaku.utils import get_precision, is_turing


@pytest.mark.skipif(is_turing(), reason="Skip tests due to using Turing GPUs")
@pytest.mark.parametrize(
    "use_qencoder,cpu_offload,memory_limit",
    [
        (False, False, 17),
        (False, True, 13),
        (True, False, 12),
        (True, True, 6),
    ],
)
def test_flux_schnell_memory(use_qencoder: bool, cpu_offload: bool, memory_limit: float):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    precision = get_precision()
    pipeline_init_kwargs = {
        "transformer": NunchakuFluxTransformer2dModel.from_pretrained(
            f"mit-han-lab/nunchaku-flux.1-schnell/svdq-{precision}_r32-flux.1-schnell.safetensors", offload=cpu_offload
        )
    }
    if use_qencoder:
        text_encoder_2 = NunchakuT5EncoderModel.from_pretrained(
            "mit-han-lab/nunchaku-t5/awq-int4-flux.1-t5xxl.safetensors"
        )
        pipeline_init_kwargs["text_encoder_2"] = text_encoder_2
    pipeline = FluxPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-schnell", torch_dtype=torch.bfloat16, **pipeline_init_kwargs
    )

    if cpu_offload:
        pipeline.enable_sequential_cpu_offload()
    else:
        pipeline = pipeline.to("cuda")

    pipeline(
        "A cat holding a sign that says hello world", width=1024, height=1024, num_inference_steps=4, guidance_scale=0
    )
    memory = torch.cuda.max_memory_reserved(0) / 1024**3
    assert memory < memory_limit
    del pipeline
    # release the gpu memory
    torch.cuda.empty_cache()
