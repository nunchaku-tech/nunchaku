import argparse
import json
import os

from huggingface_hub import constants, hf_hub_download

from .utils import load_state_dict_in_safetensors
import argparse
from safetensors.torch import save_file


def merge_models_into_a_single_file(pretrained_model_name_or_path: str, **kwargs) -> dict:
    subfolder = kwargs.get("subfolder", None)
    if os.path.exists(pretrained_model_name_or_path):
        dirname = (
            pretrained_model_name_or_path
            if subfolder is None
            else os.path.join(pretrained_model_name_or_path, subfolder)
        )
        unquantized_part_path = os.path.join(dirname, "unquantized_layers.safetensors")
        transformer_block_path = os.path.join(dirname, "transformer_blocks.safetensors")
        config_path = os.path.join(dirname, "config.json")
        comfy_config_path = os.path.join(dirname, "comfy_config.json")
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
            repo_id=pretrained_model_name_or_path, filename="unquantized_layers.safetensors", **download_kwargs
        )
        transformer_block_path = hf_hub_download(
            repo_id=pretrained_model_name_or_path, filename="transformer_blocks.safetensors", **download_kwargs
        )
        config_path = hf_hub_download(repo_id=pretrained_model_name_or_path, filename="config.json", **download_kwargs)
        comfy_config_path = hf_hub_download(
            repo_id=pretrained_model_name_or_path, filename="comfy_config.json", **download_kwargs
        )

    unquantized_part_sd = load_state_dict_in_safetensors(unquantized_part_path)
    transformer_block_sd = load_state_dict_in_safetensors(transformer_block_path)
    with open(config_path, "r") as f:
        config = json.load(f)
    with open(comfy_config_path, "r") as f:
        comfy_config = json.load(f)
    model_sd = unquantized_part_sd
    model_sd.update(transformer_block_sd)
    state_dict = {}
    state_dict["model"] = model_sd
    state_dict["config"] = config
    state_dict["comfy_config"] = comfy_config
    return state_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i",
        "--input-path",
        type=str,
        required=True,
        help="Path to model directory. It can also be a huggingface repo.",
    )
    parser.add_argument("-o", "--output-path", type=str, required=True, help="Path to output path")
    args = parser.parse_args()
    state_dict = merge_models_into_a_single_file(args.input_path)
    dirpath = os.path.dirname(os.path.abspath(args.output_path))
    os.makedirs(dirpath, exist_ok=True)
    save_file(state_dict, args.output_path)
