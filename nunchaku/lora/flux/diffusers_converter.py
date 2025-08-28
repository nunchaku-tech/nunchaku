"""
This module implements the functions to convert FLUX LoRA weights from various formats
to the Diffusers format, which will later be converted to Nunchaku format.
"""

import argparse
import logging
import os

import torch
from diffusers.loaders import FluxLoraLoaderMixin
from diffusers.utils.state_dict_utils import convert_unet_state_dict_to_peft
from safetensors.torch import save_file

from ...utils import load_state_dict_in_safetensors

# Get log level from environment variable (default to INFO)
log_level = os.getenv("LOG_LEVEL", "INFO").upper()

# Configure logging
logging.basicConfig(level=getattr(logging, log_level, logging.INFO), format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def handle_kohya_lora(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """
    Convert Kohya LoRA format keys to Diffusers format.

    Parameters
    ----------
    state_dict : dict[str, torch.Tensor]
        LoRA weights, possibly in Kohya format.

    Returns
    -------
    dict[str, torch.Tensor]
        LoRA weights in Diffusers format.
    """
    # first check if the state_dict is in the kohya format
    # like: https://civitai.com/models/1118358?modelVersionId=1256866
    if any([not k.startswith("lora_transformer_") for k in state_dict.keys()]):
        return state_dict
    else:
        new_state_dict = {}
        for k, v in state_dict.items():
            new_k = k.replace("lora_transformer_", "transformer.")

            new_k = new_k.replace("norm_out_", "norm_out.")

            new_k = new_k.replace("time_text_embed_", "time_text_embed.")
            new_k = new_k.replace("guidance_embedder_", "guidance_embedder.")
            new_k = new_k.replace("text_embedder_", "text_embedder.")
            new_k = new_k.replace("timestep_embedder_", "timestep_embedder.")

            new_k = new_k.replace("single_transformer_blocks_", "single_transformer_blocks.")
            new_k = new_k.replace("_attn_", ".attn.")
            new_k = new_k.replace("_norm_linear.", ".norm.linear.")
            new_k = new_k.replace("_proj_mlp.", ".proj_mlp.")
            new_k = new_k.replace("_proj_out.", ".proj_out.")

            new_k = new_k.replace("transformer_blocks_", "transformer_blocks.")
            new_k = new_k.replace("to_out_0.", "to_out.0.")
            new_k = new_k.replace("_ff_context_net_0_proj.", ".ff_context.net.0.proj.")
            new_k = new_k.replace("_ff_context_net_2.", ".ff_context.net.2.")
            new_k = new_k.replace("_ff_net_0_proj.", ".ff.net.0.proj.")
            new_k = new_k.replace("_ff_net_2.", ".ff.net.2.")
            new_k = new_k.replace("_norm1_context_linear.", ".norm1_context.linear.")
            new_k = new_k.replace("_norm1_linear.", ".norm1.linear.")

            new_k = new_k.replace(".lora_down.", ".lora_A.")
            new_k = new_k.replace(".lora_up.", ".lora_B.")

            new_state_dict[new_k] = v
        return new_state_dict


def convert_peft_to_comfyui(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """
    Convert PEFT format (base_model.model.*) to ComfyUI format (lora_unet_*).

    Mapping rules:
    - base_model.model.double_blocks.X.img_attn.proj → lora_unet_double_blocks_X_img_attn_proj
    - base_model.model.single_blocks.X.linear1 → lora_unet_single_blocks_X_linear1
    - base_model.model.final_layer.linear → lora_unet_final_layer_linear
    - lora_A/lora_B → lora_down/lora_up

    Parameters
    ----------
    state_dict : dict[str, torch.Tensor]
        LoRA weights in PEFT format

    Returns
    -------
    dict[str, torch.Tensor]
        LoRA weights in ComfyUI format
    """
    converted_dict = {}

    for key, value in state_dict.items():
        new_key = key

        if key.startswith("base_model.model."):
            # Remove base_model.model. prefix
            new_key = key.replace("base_model.model.", "")

            # Convert to ComfyUI format with underscores
            # Handle double_blocks
            if "double_blocks" in new_key:
                # Replace dots with underscores within the block structure
                # e.g., double_blocks.0.img_attn.proj → double_blocks_0_img_attn_proj
                new_key = new_key.replace("double_blocks.", "lora_unet_double_blocks_")
                # Replace remaining dots with underscores
                new_key = new_key.replace(".", "_")

            # Handle single_blocks
            elif "single_blocks" in new_key:
                new_key = new_key.replace("single_blocks.", "lora_unet_single_blocks_")
                # Special handling for modulation.lin → modulation_lin
                new_key = new_key.replace("modulation.lin", "modulation_lin")
                # Replace remaining dots with underscores
                new_key = new_key.replace(".", "_")

            # Handle final_layer
            elif "final_layer" in new_key:
                new_key = new_key.replace("final_layer.linear", "lora_unet_final_layer_linear")
                # Replace remaining dots with underscores
                new_key = new_key.replace(".", "_")

            else:
                # For any other keys, add lora_unet_ prefix and replace dots
                new_key = "lora_unet_" + new_key.replace(".", "_")

        # Convert lora_A/lora_B to lora_down/lora_up
        new_key = new_key.replace("_lora_A_weight", ".lora_down.weight")
        new_key = new_key.replace("_lora_B_weight", ".lora_up.weight")

        converted_dict[new_key] = value

        if key != new_key:
            logger.debug(f"Converted: {key} → {new_key}")

    return converted_dict


def to_diffusers(input_lora: str | dict[str, torch.Tensor], output_path: str | None = None) -> dict[str, torch.Tensor]:
    """
    Convert LoRA weights to Diffusers format, which will later be converted to Nunchaku format.

    Parameters
    ----------
    input_lora : str or dict[str, torch.Tensor]
        Path to a safetensors file or a LoRA weight dictionary.
    output_path : str, optional
        If given, save the converted weights to this path.

    Returns
    -------
    dict[str, torch.Tensor]
        LoRA weights in Diffusers format.
    """
    if isinstance(input_lora, str):
        tensors = load_state_dict_in_safetensors(input_lora, device="cpu")
    else:
        tensors = {k: v for k, v in input_lora.items()}

    tensors = handle_kohya_lora(tensors)

    # Convert FP8 tensors to BF16
    for k, v in tensors.items():
        if v.dtype not in [torch.float64, torch.float32, torch.bfloat16, torch.float16]:
            tensors[k] = v.to(torch.bfloat16)

    # Apply Kontext-specific key conversion for both PEFT format and ComfyUI format
    # This handles LoRAs with base_model.model.* prefix or lora_unet_* prefix (including final_layer_linear)
    if any(k.startswith("base_model.model.") for k in tensors.keys()):
        logger.info("Converting PEFT format to ComfyUI format")
        return convert_peft_to_comfyui(tensors)

    # Handle LoRAs that only have final_layer_linear without adaLN_modulation
    # This is a workaround for incomplete final layer LoRAs
    final_keys = [k for k in tensors.keys() if "final_layer" in k]
    has_linear = any("final_layer_linear" in k for k in final_keys)
    has_adaln = any("final_layer_adaLN_modulation" in k for k in final_keys)

    if has_linear and not has_adaln:
        for key in list(tensors.keys()):
            if "final_layer_linear" in key:
                adaln_key = key.replace("final_layer_linear", "final_layer_adaLN_modulation_1")
                if adaln_key not in tensors:
                    tensors[adaln_key] = torch.zeros_like(tensors[key])

    new_tensors, alphas = FluxLoraLoaderMixin.lora_state_dict(tensors, return_alphas=True)
    new_tensors = convert_unet_state_dict_to_peft(new_tensors)

    if alphas is not None and len(alphas) > 0:
        for k, v in alphas.items():
            key_A = k.replace(".alpha", ".lora_A.weight")
            key_B = k.replace(".alpha", ".lora_B.weight")
            assert key_A in new_tensors, f"Key {key_A} not found in new tensors."
            assert key_B in new_tensors, f"Key {key_B} not found in new tensors."
            rank = new_tensors[key_A].shape[0]
            assert new_tensors[key_B].shape[1] == rank, f"Rank mismatch for {key_B}."
            new_tensors[key_A] = new_tensors[key_A] * v / rank

    if output_path is not None:
        output_dir = os.path.dirname(os.path.abspath(output_path))
        os.makedirs(output_dir, exist_ok=True)
        save_file(new_tensors, output_path)

    return new_tensors


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input-path", type=str, required=True, help="path to the comfyui lora safetensors file")
    parser.add_argument(
        "-o", "--output-path", type=str, required=True, help="path to the output diffusers safetensors file"
    )
    args = parser.parse_args()
    to_diffusers(args.input_path, args.output_path)
