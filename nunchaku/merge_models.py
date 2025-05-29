import argparse
import json
import os
from pathlib import Path

import torch
from huggingface_hub import constants, hf_hub_download
from safetensors.torch import save_file

from .utils import load_state_dict_in_safetensors


def merge_models_into_a_single_file(
    pretrained_model_name_or_path: str | os.PathLike[str], **kwargs
) -> tuple[dict[str, torch.Tensor], dict[str, str]]:
    subfolder = kwargs.get("subfolder", None)

    if isinstance(pretrained_model_name_or_path, str):
        pretrained_model_name_or_path = Path(pretrained_model_name_or_path)
    if pretrained_model_name_or_path.exists():
        dirpath = pretrained_model_name_or_path if subfolder is None else pretrained_model_name_or_path / subfolder
        unquantized_part_path = dirpath / "unquantized_layers.safetensors"
        transformer_block_path = dirpath / "transformer_blocks.safetensors"
        config_path = dirpath / "config.json"
        comfy_config_path = dirpath / "comfy_config.json"
    else:
        download_kwargs = {
            "subfolder": subfolder,
            "repo_type": "model",
            "revision": kwargs.get("revision", None),
            "cache_dir": kwargs.get("cache_dir", None),
            "local_dir": kwargs.get("local_dir", None),
            "user_agent": kwargs.get("user_agent", None),
            "force_download": kwargs.get("force_download", False),
            "proxies": kwargs.get("proxies", None),
            "etag_timeout": kwargs.get("etag_timeout", constants.DEFAULT_ETAG_TIMEOUT),
            "token": kwargs.get("token", None),
            "local_files_only": kwargs.get("local_files_only", None),
            "headers": kwargs.get("headers", None),
            "endpoint": kwargs.get("endpoint", None),
            "resume_download": kwargs.get("resume_download", None),
            "force_filename": kwargs.get("force_filename", None),
            "local_dir_use_symlinks": kwargs.get("local_dir_use_symlinks", "auto"),
        }
        unquantized_part_path = hf_hub_download(
            repo_id=str(pretrained_model_name_or_path), filename="unquantized_layers.safetensors", **download_kwargs
        )
        transformer_block_path = hf_hub_download(
            repo_id=str(pretrained_model_name_or_path), filename="transformer_blocks.safetensors", **download_kwargs
        )
        config_path = hf_hub_download(
            repo_id=str(pretrained_model_name_or_path), filename="config.json", **download_kwargs
        )
        comfy_config_path = hf_hub_download(
            repo_id=str(pretrained_model_name_or_path), filename="comfy_config.json", **download_kwargs
        )

    unquantized_part_sd = load_state_dict_in_safetensors(unquantized_part_path)
    transformer_block_sd = load_state_dict_in_safetensors(transformer_block_path)
    state_dict = unquantized_part_sd
    state_dict.update(transformer_block_sd)

    return state_dict, {
        "config": json.dumps(Path(config_path).read_text()),
        "comfy_config": Path(comfy_config_path).read_text(),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i",
        "--input-path",
        type=Path,
        required=True,
        help="Path to model directory. It can also be a huggingface repo.",
    )
    parser.add_argument("-o", "--output-path", type=Path, required=True, help="Path to output path")
    args = parser.parse_args()
    state_dict, metadata = merge_models_into_a_single_file(args.input_path)
    output_path = Path(args.output_path)
    dirpath = output_path.parent
    dirpath.mkdir(parents=True, exist_ok=True)
    save_file(state_dict, output_path, metadata=metadata)
