import gc

import torch
from diffusers import (
    AutoencoderKL,
    FlowMatchEulerDiscreteScheduler,
    FluxControlNetModel,
    FluxControlNetPipeline,
    FluxPipeline,
)
from diffusers.models import FluxMultiControlNetModel
from diffusers.utils import load_image
from transformers import CLIPTextModel, CLIPTokenizer, T5EncoderModel, T5TokenizerFast

from nunchaku import NunchakuT5EncoderModel
from nunchaku.caching.diffusers_adapters import apply_cache_on_pipe
from nunchaku.models.transformers.transformer_flux import NunchakuFluxTransformer2dModel
from nunchaku.utils import get_precision


def test_flux_txt2img_cache_controlnet():
    bfl_repo = "black-forest-labs/FLUX.1-dev"
    dtype = torch.bfloat16  # or torch.float16, or torch.float32
    device = "cuda"  # or "cpu" if you want to run on CPU

    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(bfl_repo, subfolder="scheduler", torch_dtype=dtype)
    text_encoder = CLIPTextModel.from_pretrained(bfl_repo, subfolder="text_encoder", torch_dtype=dtype)
    text_encoder_2 = T5EncoderModel.from_pretrained(bfl_repo, subfolder="text_encoder_2", torch_dtype=dtype)
    tokenizer = CLIPTokenizer.from_pretrained(
        bfl_repo, subfolder="tokenizer", torch_dtype=dtype, clean_up_tokenization_spaces=True
    )
    tokenizer_2 = T5TokenizerFast.from_pretrained(
        bfl_repo, subfolder="tokenizer_2", torch_dtype=dtype, clean_up_tokenization_spaces=True
    )
    vae = AutoencoderKL.from_pretrained(bfl_repo, subfolder="vae", torch_dtype=dtype)
    precision = get_precision()
    transformer = NunchakuFluxTransformer2dModel.from_pretrained(
        f"mit-han-lab/nunchaku-flux.1-dev/svdq-{precision}_r32-flux.1-dev.safetensors",
        # offload=True
    )
    transformer.set_attention_impl("nunchaku-fp16")

    # qencoder
    text_encoder_2 = NunchakuT5EncoderModel.from_pretrained("mit-han-lab/nunchaku-t5/awq-int4-flux.1-t5xxl.safetensors")
    controlnet_union = FluxControlNetModel.from_pretrained(
        "Shakker-Labs/FLUX.1-dev-ControlNet-Union-Pro-2.0", torch_dtype=torch.bfloat16
    )
    controlnet = FluxMultiControlNetModel(
        [controlnet_union]
    )  # we always recommend loading via FluxMultiControlNetModel

    params = {
        "scheduler": scheduler,
        "vae": vae,
        "tokenizer": tokenizer,
        "tokenizer_2": tokenizer_2,
        "text_encoder": text_encoder,
        "text_encoder_2": text_encoder_2,
        "transformer": transformer,
    }
    # pipe
    pipe = FluxPipeline(**params).to(device, dtype=dtype)
    pipe_cn = FluxControlNetPipeline(**params, controlnet=controlnet).to(device, dtype)

    # offload
    pipe.enable_sequential_cpu_offload(device=device)
    pipe_cn.enable_sequential_cpu_offload(device=device)

    # cache
    apply_cache_on_pipe(
        pipe_cn,
        use_double_fb_cache=True,
        residual_diff_threshold_multi=0.09,
        residual_diff_threshold_single=0.12,
    )

    params = {
        "prompt": "A bohemian-style female travel blogger with sun-kissed skin and messy beach waves.",
        "height": 1152,
        "width": 768,
        "num_inference_steps": 30,
        "guidance_scale": 3.5,
    }

    # pipe
    txt2img_res = pipe(
        **params,
    ).images[0]
    txt2img_res.save("flux.1-dev-txt2img.jpg")

    gc.collect()
    torch.cuda.empty_cache()

    # cache
    apply_cache_on_pipe(
        pipe_cn,
        use_double_fb_cache=True,
        residual_diff_threshold_multi=0.09,
        residual_diff_threshold_single=0.12,
    )

    # pipe_cn
    control_iamge = load_image(
        "https://huggingface.co/Shakker-Labs/FLUX.1-dev-ControlNet-Union-Pro/resolve/main/assets/openpose.jpg"
    )
    params["control_image"] = [control_iamge]
    params["controlnet_conditioning_scale"] = [0.9]
    params["control_guidance_end"] = [0.65]
    cn_res = pipe_cn(
        **params,
    ).images[0]
    cn_res.save("flux.1-dev-cn-txt2img.jpg")
