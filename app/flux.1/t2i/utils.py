import torch
from diffusers import FluxPipeline
from peft.tuners import lora
from vars import LORA_PATHS, SVDQ_LORA_PATHS

from nunchaku import NunchakuFluxTransformer2dModel
from nunchaku.models.transformers.transformer_flux_v2 import NunchakuFluxTransformer2DModelV2


def hash_str_to_int(s: str) -> int:
    """Hash a string to an integer."""
    modulus = 10**9 + 7  # Large prime modulus
    hash_int = 0
    for char in s:
        hash_int = (hash_int * 31 + ord(char)) % modulus
    return hash_int


def get_pipeline(
    model_name: str,
    precision: str,
    use_qencoder: bool = False,
    lora_name: str = "None",
    lora_weight: float = 1,
    device: str | torch.device = "cuda",
    pipeline_init_kwargs: dict = {},
) -> FluxPipeline:
    if model_name == "schnell":
        if precision in ["int4", "fp4"]:
            assert torch.device(device).type == "cuda", "int4 only supported on CUDA devices"
            if precision == "int4":
                transformer = NunchakuFluxTransformer2dModel.from_pretrained(
                    "mit-han-lab/nunchaku-flux.1-schnell/svdq-int4_r32-flux.1-schnell.safetensors"
                )
            else:
                assert precision == "fp4"
                transformer = NunchakuFluxTransformer2dModel.from_pretrained(
                    "mit-han-lab/nunchaku-flux.1-schnell/svdq-fp4_r32-flux.1-schnell.safetensors", precision="fp4"
                )
            pipeline_init_kwargs["transformer"] = transformer
            if use_qencoder:
                from nunchaku.models.text_encoders.t5_encoder import NunchakuT5EncoderModel

                text_encoder_2 = NunchakuT5EncoderModel.from_pretrained(
                    "mit-han-lab/nunchaku-t5/awq-int4-flux.1-t5xxl.safetensors"
                )
                pipeline_init_kwargs["text_encoder_2"] = text_encoder_2
        else:
            assert precision == "bf16"
        pipeline = FluxPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-schnell", torch_dtype=torch.bfloat16, **pipeline_init_kwargs
        )
    elif model_name == "schnell_v2":
        transformer = NunchakuFluxTransformer2DModelV2.from_pretrained(
            f"mit-han-lab/nunchaku-flux.1-schnell/svdq-{precision}_r32-flux.1-schnell.safetensors"
        )
        pipeline = FluxPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-schnell",
            transformer=transformer,
            torch_dtype=torch.bfloat16,
            **pipeline_init_kwargs,
        )
    elif model_name == "dev":
        if precision == "int4":
            transformer = NunchakuFluxTransformer2dModel.from_pretrained(
                "mit-han-lab/nunchaku-flux.1-dev/svdq-int4_r32-flux.1-dev.safetensors"
            )
            if lora_name not in ["All", "None"]:
                transformer.update_lora_params(SVDQ_LORA_PATHS[lora_name])
                transformer.set_lora_strength(lora_weight)
            pipeline_init_kwargs["transformer"] = transformer
            if use_qencoder:
                from nunchaku.models.text_encoders.t5_encoder import NunchakuT5EncoderModel

                text_encoder_2 = NunchakuT5EncoderModel.from_pretrained(
                    "mit-han-lab/nunchaku-t5/awq-int4-flux.1-t5xxl.safetensors"
                )
                pipeline_init_kwargs["text_encoder_2"] = text_encoder_2
            pipeline = FluxPipeline.from_pretrained(
                "black-forest-labs/FLUX.1-dev", torch_dtype=torch.bfloat16, **pipeline_init_kwargs
            )
        else:
            assert precision == "bf16"
            pipeline = FluxPipeline.from_pretrained(
                "black-forest-labs/FLUX.1-dev", torch_dtype=torch.bfloat16, **pipeline_init_kwargs
            )
            if lora_name == "All":
                # Pre-load all the LoRA weights for demo use
                for name, path in LORA_PATHS.items():
                    pipeline.load_lora_weights(path["name_or_path"], weight_name=path["weight_name"], adapter_name=name)
                for m in pipeline.transformer.modules():
                    if isinstance(m, lora.LoraLayer):
                        m.set_adapter(m.scaling.keys())
                        for name in m.scaling.keys():
                            m.scaling[name] = 0
            elif lora_name != "None":
                path = LORA_PATHS[lora_name]
                pipeline.load_lora_weights(
                    path["name_or_path"], weight_name=path["weight_name"], adapter_name=lora_name
                )
                for m in pipeline.transformer.modules():
                    if isinstance(m, lora.LoraLayer):
                        for name in m.scaling.keys():
                            m.scaling[name] = lora_weight
    else:
        raise NotImplementedError(f"Model {model_name} not implemented")
    if precision == "bf16":
        pipeline.enable_model_cpu_offload()
    else:
        pipeline = pipeline.to(device)

    return pipeline
