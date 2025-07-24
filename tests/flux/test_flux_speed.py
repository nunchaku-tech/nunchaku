# single case for expect latency
# original: bf16
# use quantized text encoder, see doc
# run fp4 on 5090

import time

import torch
import pytest

from diffusers import FluxPipeline

from nunchaku import NunchakuFluxTransformer2dModel, NunchakuT5EncoderModel
from nunchaku.utils import get_precision, is_turing

@pytest.mark.skipif(is_turing(), reason="Skip tests due to using Turing GPUs")
@pytest.mark.parametrize(
    "warmup_time,test_times,num_inference_steps,guidance_scale,use_qencoder,cpu_offload,expected_latency_ms",
    [
        (2, 5, 30, 3.5, True, False, 100),
    ],
)
def test_flux_speed(warmup_times: int, test_times: int, num_inference_steps: int, guidance_scale: float, 
                    use_qencoder: bool, cpu_offload: bool, expected_latency_ms: float):
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
        "A cat holding a sign that says hello world", 
        width=1024, height=1024, 
        num_inference_steps=num_inference_steps, guidance_scale=guidance_scale
    )

    latency_list = []
    dummy_prompt = "A cat holding a sign that says hello world"

    for _ in range(warmup_times):
        pipeline(
            prompt=dummy_prompt, num_inference_steps=num_inference_steps, guidance_scale=guidance_scale
        )
        torch.cuda.synchronize()
    for _ in range(test_times):
        start_time = time.time()
        pipeline(
            prompt=dummy_prompt, num_inference_steps=num_inference_steps, guidance_scale=guidance_scale
        )
        torch.cuda.synchronize()
        end_time = time.time()
        latency_list.append(end_time - start_time)
        
    print(f"Latency: {sum(latency_list) / len(latency_list):.5f} s")
    
    
