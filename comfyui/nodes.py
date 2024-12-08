import os
import types

import comfy.model_base
import comfy.model_patcher
import comfy.sd
import folder_paths
import GPUtil
import torch
from comfy.ldm.common_dit import pad_to_patch_size
from comfy.supported_models import Flux, FluxSchnell
from diffusers import FluxTransformer2DModel
from einops import rearrange, repeat
from torch import nn
from transformers import T5EncoderModel

from nunchaku.models.transformer_flux import NunchakuFluxTransformer2dModel


class ComfyUIFluxForwardWrapper(nn.Module):
    def __init__(self, model: NunchakuFluxTransformer2dModel, config):
        super(ComfyUIFluxForwardWrapper, self).__init__()
        self.model = model
        self.dtype = next(model.parameters()).dtype
        self.config = config

    def forward(self, x, timestep, context, y, guidance, control=None, transformer_options={}, **kwargs):
        assert control is None  # for now
        bs, c, h, w = x.shape
        patch_size = self.config["patch_size"]
        x = pad_to_patch_size(x, (patch_size, patch_size))

        img = rearrange(x, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=patch_size, pw=patch_size)

        h_len = (h + (patch_size // 2)) // patch_size
        w_len = (w + (patch_size // 2)) // patch_size
        img_ids = torch.zeros((h_len, w_len, 3), device=x.device, dtype=x.dtype)
        img_ids[:, :, 1] = img_ids[:, :, 1] + torch.linspace(
            0, h_len - 1, steps=h_len, device=x.device, dtype=x.dtype
        ).unsqueeze(1)
        img_ids[:, :, 2] = img_ids[:, :, 2] + torch.linspace(
            0, w_len - 1, steps=w_len, device=x.device, dtype=x.dtype
        ).unsqueeze(0)
        img_ids = repeat(img_ids, "h w c -> b (h w) c", b=bs)

        txt_ids = torch.zeros((bs, context.shape[1], 3), device=x.device, dtype=x.dtype)
        out = self.model(
            hidden_states=img,
            encoder_hidden_states=context,
            pooled_projections=y,
            timestep=timestep,
            img_ids=img_ids,
            txt_ids=txt_ids,
            guidance=guidance if self.config["guidance_embed"] else None,
        ).sample

        out = rearrange(out, "b (h w) (c ph pw) -> b c (h ph) (w pw)", h=h_len, w=w_len, ph=2, pw=2)[:, :, :h, :w]
        return out


class SVDQuantFluxDiTLoader:
    @classmethod
    def INPUT_TYPES(s):
        model_paths = ["mit-han-lab/svdq-int4-flux.1-schnell", "mit-han-lab/svdq-int4-flux.1-dev"]
        ngpus = len(GPUtil.getGPUs())
        return {
            "required": {
                "model_path": (model_paths,),
                "device_id": (
                    "INT",
                    {"default": 0, "min": 0, "max": ngpus, "step": 1, "display": "number", "lazy": True},
                ),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "load_model"
    CATEGORY = "SVDQuant"
    TITLE = "SVDQuant Flux DiT Loader"

    def load_model(self, model_path: str, device_id: int, **kwargs) -> tuple[FluxTransformer2DModel]:
        device = f"cuda:{device_id}"
        transformer = NunchakuFluxTransformer2dModel.from_pretrained(model_path).to(device)
        dit_config = {
            "image_model": "flux",
            "in_channels": 16,
            "patch_size": 2,
            "out_channels": 16,
            "vec_in_dim": 768,
            "context_in_dim": 4096,
            "hidden_size": 3072,
            "mlp_ratio": 4.0,
            "num_heads": 24,
            "depth": 19,
            "depth_single_blocks": 38,
            "axes_dim": [16, 56, 56],
            "theta": 10000,
            "qkv_bias": True,
            "disable_unet_model_creation": True,
        }
        if "schnell" in model_path:
            dit_config["guidance_embed"] = False
            model_config = FluxSchnell(dit_config)
        else:
            assert "dev" in model_path
            dit_config["guidance_embed"] = True
            model_config = Flux(dit_config)

        model_config.set_inference_dtype(torch.bfloat16, None)
        model_config.custom_operations = None

        model = model_config.get_model({})
        model.diffusion_model = ComfyUIFluxForwardWrapper(transformer, config=dit_config)
        model = comfy.model_patcher.ModelPatcher(model, device, device_id)
        return (model,)


def svdquant_t5_forward(
    self: T5EncoderModel,
    input_ids: torch.LongTensor,
    attention_mask,
    intermediate_output=None,
    final_layer_norm_intermediate=True,
    dtype: str | torch.dtype = torch.bfloat16,
):
    assert attention_mask is None
    assert intermediate_output is None
    assert final_layer_norm_intermediate
    outputs = self.encoder(input_ids, attention_mask=attention_mask)
    hidden_states = outputs["last_hidden_state"]
    hidden_states = hidden_states.to(dtype=dtype)
    return hidden_states, None


class SVDQuantTextEncoderLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model_type": (["flux"],),
                "text_encoder1": (folder_paths.get_filename_list("text_encoders"),),
                "text_encoder2": (folder_paths.get_filename_list("text_encoders"),),
                "t5_min_length": (
                    "INT",
                    {"default": 512, "min": 256, "max": 1024, "step": 128, "display": "number", "lazy": True},
                ),
                "t5_precision": (["BF16", "INT4"],),
            }
        }

    RETURN_TYPES = ("CLIP",)
    FUNCTION = "load_text_encoder"

    CATEGORY = "SVDQuant"

    TITLE = "SVDQuant Text Encoder Loader"

    def load_text_encoder(
        self, model_type: str, text_encoder1: str, text_encoder2: str, t5_min_length: int, t5_precision: str
    ):
        text_encoder_path1 = folder_paths.get_full_path_or_raise("text_encoders", text_encoder1)
        text_encoder_path2 = folder_paths.get_full_path_or_raise("text_encoders", text_encoder2)
        if model_type == "flux":
            clip_type = comfy.sd.CLIPType.FLUX
        else:
            raise ValueError(f"Unknown type {model_type}")

        clip = comfy.sd.load_clip(
            ckpt_paths=[text_encoder_path1, text_encoder_path2],
            embedding_directory=folder_paths.get_folder_paths("embeddings"),
            clip_type=clip_type,
        )

        if model_type == "flux":
            clip.tokenizer.t5xxl.min_length = t5_min_length

        if t5_precision == "INT4":
            from nunchaku.models.text_encoder import NunchakuT5EncoderModel

            transformer = clip.cond_stage_model.t5xxl.transformer
            param = next(transformer.parameters())
            dtype = param.dtype
            device = param.device
            transformer = NunchakuT5EncoderModel.from_pretrained("mit-han-lab/svdq-flux.1-t5")
            transformer.forward = types.MethodType(svdquant_t5_forward, transformer)
            clip.cond_stage_model.t5xxl.transformer = (
                transformer.to(device=device, dtype=dtype) if device.type == "cuda" else transformer
            )

        return (clip,)


class SVDQuantLoraLoader:
    def __init__(self):
        self.cur_lora_name = "None"

    @classmethod
    def INPUT_TYPES(s):
        hf_lora_names = ["anime", "ghibsky", "realism", "yarn", "sketch"]
        lora_name_list = [
            "None",
            *folder_paths.get_filename_list("loras"),
            *[f"mit-han-lab/svdquant-models/svdq-flux.1-dev-lora-{n}.safetensors" for n in hf_lora_names],
        ]
        return {
            "required": {
                "model": ("MODEL", {"tooltip": "The diffusion model the LoRA will be applied to."}),
                "lora_name": (lora_name_list, {"tooltip": "The name of the LoRA."}),
                "lora_strength": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": -100.0,
                        "max": 100.0,
                        "step": 0.01,
                        "tooltip": "How strongly to modify the diffusion model. This value can be negative.",
                    },
                ),
            }
        }

    RETURN_TYPES = ("MODEL",)
    OUTPUT_TOOLTIPS = ("The modified diffusion model.",)
    FUNCTION = "load_lora"
    TITLE = "SVDQuant LoRA Loader"

    CATEGORY = "SVDQuant"
    DESCRIPTION = (
        "LoRAs are used to modify the diffusion model, "
        "altering the way in which latents are denoised such as applying styles. "
        "Currently, only one LoRA nodes can be applied."
    )

    def load_lora(self, model, lora_name: str, lora_strength: float):
        if self.cur_lora_name == lora_name:
            if self.cur_lora_name == "None":
                pass  # Do nothing since the lora is None
            else:
                model.model.diffusion_model.model.set_lora_strength(lora_strength)
        else:
            if lora_name == "None":
                model.model.diffusion_model.model.set_lora_strength(0)
            else:
                try:
                    lora_path = folder_paths.get_full_path_or_raise("loras", lora_name)
                except FileNotFoundError:
                    lora_path = lora_name

                model.model.diffusion_model.model.update_lora_params(lora_path)
                model.model.diffusion_model.model.set_lora_strength(lora_strength)
            self.cur_lora_name = lora_name

        return (model,)


NODE_CLASS_MAPPINGS = {
    "SVDQuantFluxDiTLoader": SVDQuantFluxDiTLoader,
    "SVDQuantTextEncoderLoader": SVDQuantTextEncoderLoader,
    "SVDQuantLoRALoader": SVDQuantLoraLoader,
}
