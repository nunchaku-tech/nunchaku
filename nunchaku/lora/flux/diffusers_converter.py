import argparse
import os
import warnings

import torch
from diffusers.loaders import FluxLoraLoaderMixin
from safetensors.torch import save_file

from .utils import load_state_dict_in_safetensors


def to_diffusers(input_lora: str | dict[str, torch.Tensor], output_path: str | None = None) -> dict[str, torch.Tensor]:
    if isinstance(input_lora, str):
        tensors = load_state_dict_in_safetensors(input_lora, device="cpu")
    else:
        tensors = {k: v for k, v in input_lora.items()}

    ### convert the FP8 tensors to BF16
    for k, v in tensors.items():
        if v.dtype not in [torch.float64, torch.float32, torch.bfloat16, torch.float16]:
            tensors[k] = v.to(torch.bfloat16)

    new_tensors, alphas = FluxLoraLoaderMixin.lora_state_dict(tensors, return_alphas=True)

    if alphas is not None and len(alphas) > 0:
        warnings.warn("Alpha values are not used in the conversion to diffusers format.")

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
