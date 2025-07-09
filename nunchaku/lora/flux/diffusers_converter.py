"""
Diffusers LoRA converter for Flux models.

This module provides functionality to convert LoRA weights from various formats
(including Kohya format) to Diffusers format that can be used with the Diffusers
library. It handles format detection, key renaming, and tensor conversion.

The converter supports:
- Kohya LoRA format conversion
- FP8 to BF16 tensor conversion
- Alpha scaling integration
- Safetensors file I/O
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
    Convert Kohya LoRA format to standard Diffusers format.

    This function detects if the input state dict is in Kohya format and converts
    it to standard Diffusers format by renaming keys and adjusting the structure.
    Kohya format uses different naming conventions for transformer blocks and
    layer components.

    Parameters
    ----------
    state_dict : dict[str, torch.Tensor]
        Dictionary containing LoRA weights in potentially Kohya format.

    Returns
    -------
    dict[str, torch.Tensor]
        Dictionary with LoRA weights converted to standard Diffusers format.

    Notes
    -----
    The function performs the following key transformations:
    - `lora_transformer_` → `transformer.`
    - `norm_out_` → `norm_out.`
    - `time_text_embed_` → `time_text_embed.`
    - `single_transformer_blocks_` → `single_transformer_blocks.`
    - `transformer_blocks_` → `transformer_blocks.`
    - `lora_down` → `lora_A`
    - `lora_up` → `lora_B`
    
    And many other component-specific transformations.

    Examples
    --------
    >>> # Convert Kohya format state dict
    >>> kohya_dict = {"lora_transformer_norm_out_weight": torch.randn(10, 20)}
    >>> converted = handle_kohya_lora(kohya_dict)
    >>> print(list(converted.keys()))
    ['transformer.norm_out.weight']
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


def to_diffusers(input_lora: str | dict[str, torch.Tensor], output_path: str | None = None) -> dict[str, torch.Tensor]:
    """
    Convert LoRA weights to Diffusers format.

    This function takes LoRA weights in various formats and converts them to
    Diffusers format. It handles Kohya format conversion, FP8 to BF16 tensor
    conversion, PEFT state dict conversion, and alpha scaling integration.

    Parameters
    ----------
    input_lora : str or dict[str, torch.Tensor]
        Either a path to a safetensors file containing LoRA weights, or a dictionary
        of LoRA weight tensors.
    output_path : str, optional
        Path to save the converted LoRA weights as a safetensors file. If None,
        the weights are not saved to disk (default: None).

    Returns
    -------
    dict[str, torch.Tensor]
        Dictionary containing the LoRA weights in Diffusers format.

    Notes
    -----
    The conversion process includes:
    1. Loading tensors from file or using provided dictionary
    2. Handling Kohya format conversion if detected
    3. Converting FP8 tensors to BF16 for compatibility
    4. Processing through FluxLoraLoaderMixin for proper format
    5. Converting to PEFT format using Diffusers utilities
    6. Applying alpha scaling if present in the original weights

    Examples
    --------
    >>> # Convert from file path
    >>> diffusers_weights = to_diffusers("path/to/lora.safetensors")
    
    >>> # Convert from weight dictionary
    >>> weights = {"transformer.layer.lora_A.weight": torch.randn(10, 20)}
    >>> diffusers_weights = to_diffusers(weights, "output.safetensors")

    Raises
    ------
    AssertionError
        If expected keys are not found in the state dict after conversion.
    """
    if isinstance(input_lora, str):
        tensors = load_state_dict_in_safetensors(input_lora, device="cpu")
    else:
        tensors = {k: v for k, v in input_lora.items()}

    tensors = handle_kohya_lora(tensors)

    ### convert the FP8 tensors to BF16
    for k, v in tensors.items():
        if v.dtype not in [torch.float64, torch.float32, torch.bfloat16, torch.float16]:
            tensors[k] = v.to(torch.bfloat16)

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
