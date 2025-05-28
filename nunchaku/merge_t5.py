import argparse
import json
import os
from os import PathLike

from huggingface_hub import constants, hf_hub_download
from safetensors.torch import save_file

from .utils import load_state_dict_in_safetensors
from pathlib import Path


def merge_config_into_model(pretrained_model_name_or_path: str | PathLike[str], **kwargs) -> dict:
    subfolder = kwargs.get("subfolder", None)

    if isinstance(pretrained_model_name_or_path, str):
        pretrained_model_name_or_path = Path(pretrained_model_name_or_path)
    if pretrained_model_name_or_path.exists():
        dirpath = pretrained_model_name_or_path if subfolder is None else pretrained_model_name_or_path / subfolder
        model_path = dirpath / "awq-int4-flux.1-t5xxl.safetensors"
        config_path = dirpath / "config.json"
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
        model_path = hf_hub_download(
            repo_id=str(pretrained_model_name_or_path), filename="awq-int4-flux.1-t5xxl.safetensors", **download_kwargs
        )
        config_path = hf_hub_download(repo_id=pretrained_model_name_or_path, filename="config.json", **download_kwargs)
        model_path = Path(model_path)
        config_path = Path(config_path)

    model_sd = load_state_dict_in_safetensors(model_path)
    with open(config_path, "r") as f:
        config = json.load(f)
    state_dict = {}
    state_dict["model"] = model_sd
    state_dict["config"] = config
    return state_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i",
        "--input-path",
        type=Path,
        default="mit-han-lab/nunchaku-t5",
        help="Path to model directory. It can also be a huggingface repo.",
    )
    parser.add_argument("-o", "--output-path", type=Path, required=True, help="Path to output path")
    args = parser.parse_args()
    state_dict = merge_config_into_model(args.input_path)
    output_path = Path(args.output_path)
    dirpath = output_path.parent
    dirpath.mkdir(parents=True, exist_ok=True)
    save_file(state_dict, output_path)
