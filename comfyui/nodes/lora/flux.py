import os
import tempfile

import folder_paths
from safetensors.torch import save_file

from nunchaku.lora.flux import comfyui2diffusers, convert_to_nunchaku_flux_lowrank_dict, detect_format, xlab2diffusers

class SVDQuantFluxLoraLoader:
    def __init__(self):
        self.cur_lora_name = "None"

    @classmethod
    def INPUT_TYPES(s):
        lora_name_list = [
            "None",
            *folder_paths.get_filename_list("loras"),
            "aleksa-codes/flux-ghibsky-illustration/lora.safetensors",
        ]

        base_model_paths = [
            "mit-han-lab/svdq-int4-flux.1-dev",
            "mit-han-lab/svdq-int4-flux.1-schnell",
            "mit-han-lab/svdq-fp4-flux.1-dev",
            "mit-han-lab/svdq-fp4-flux.1-schnell",
            "mit-han-lab/svdq-int4-flux.1-canny-dev",
            "mit-han-lab/svdq-int4-flux.1-depth-dev",
            "mit-han-lab/svdq-int4-flux.1-fill-dev",
        ]
        prefix = os.path.join(folder_paths.models_dir, "diffusion_models")
        if os.path.exists(prefix):
            local_base_model_folders = sorted(
                [
                    folder
                    for folder in os.listdir(prefix)
                    if not folder.startswith(".") and os.path.isdir(os.path.join(prefix, folder))
                ]
            )
            base_model_paths = local_base_model_folders + base_model_paths
        
        return {
            "required": {
                "model": ("MODEL", {"tooltip": "The diffusion model the LoRA will be applied to."}),
                "lora_name": (lora_name_list, {"tooltip": "The name of the LoRA."}),
                "lora_format": (
                    ["auto", "comfyui", "diffusers", "svdquant", "xlab"],
                    {"tooltip": "The format of the LoRA."},
                ),
                "base_model_name": (
                    base_model_paths,
                    {
                        "tooltip": "If the lora format is SVDQuant, this field has no use. Otherwise, the base model's state dictionary is required for converting the LoRA weights to SVDQuant."
                    },
                ),
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
                "converted_lora_sulfix" :("STRING", {"multiline": False, "default": "-converted"}),
                "save_converted_lora": ("BOOLEAN", {"default": False,"tooltip": "Save the converted LoRA next to the original file with 'converted_lora_sulfix' text suffix."}),
            }
        }

    RETURN_TYPES = ("MODEL",)
    OUTPUT_TOOLTIPS = ("The modified diffusion model.",)
    FUNCTION = "load_lora"
    TITLE = "SVDQuant FLUX.1 LoRA Loader"

    CATEGORY = "SVDQuant"
    DESCRIPTION = (
        "LoRAs are used to modify the diffusion model, "
        "altering the way in which latents are denoised such as applying styles. "
        "Currently, only one LoRA node can be applied."
    )

    def load_lora(self, model, lora_name: str, lora_format: str, base_model_name: str, lora_strength: float, save_converted_lora: bool, converted_lora_sulfix: str):
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
                if lora_format == "auto":
                    lora_format = detect_format(lora_path)
                if lora_format != "svdquant":
                    if lora_format == "comfyui":
                        input_lora = comfyui2diffusers(lora_path)
                    elif lora_format == "xlab":
                        input_lora = xlab2diffusers(lora_path)
                    elif lora_format == "diffusers":
                        input_lora = lora_path
                    else:
                        raise ValueError(f"Invalid LoRA format {lora_format}.")
                    prefix = os.path.join(folder_paths.models_dir, "diffusion_models")
                    base_model_path = os.path.join(prefix, base_model_name, "transformer_blocks.safetensors")
                    if not os.path.exists(base_model_path):
                        # download from huggingface
                        base_model_path = os.path.join(base_model_name, "transformer_blocks.safetensors")
                    state_dict = convert_to_nunchaku_flux_lowrank_dict(base_model_path, input_lora)
                    #the converted lora will be deleted on next comfyui startup as part of "temp" folder cleanup
                    with tempfile.NamedTemporaryFile(suffix=".safetensors", dir=folder_paths.temp_directory, delete=False) as tmp_file: 
                        save_file(state_dict, tmp_file.name)
                        model.model.diffusion_model.model.update_lora_params(tmp_file.name)
                    
                    if save_converted_lora:
                        converted_lora_path = lora_path.replace(".safetensors", converted_lora_sulfix + ".safetensors")
                        save_file(state_dict, converted_lora_path)
                        lora_name = os.path.basename(converted_lora_path)
                else:
                    model.model.diffusion_model.model.update_lora_params(lora_path)
                model.model.diffusion_model.model.set_lora_strength(lora_strength)
            self.cur_lora_name = lora_name

        return (model,)
