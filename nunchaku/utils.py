import hashlib
import os
import warnings
from pathlib import Path

import safetensors
import torch
from huggingface_hub import hf_hub_download


def sha256sum(filepath: str | os.PathLike[str]) -> str:
    sha256 = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def fetch_or_download(path: str | Path, repo_type: str = "model") -> Path:
    path = Path(path)

    if path.exists():
        return path

    parts = path.parts
    if len(parts) < 3:
        raise ValueError(f"Path '{path}' is too short to extract repo_id and subfolder")

    repo_id = "/".join(parts[:2])
    sub_path = Path(*parts[2:])
    filename = sub_path.name
    subfolder = sub_path.parent if sub_path.parent != Path(".") else None

    path = hf_hub_download(repo_id=repo_id, filename=filename, subfolder=subfolder, repo_type=repo_type)
    return Path(path)


def ceil_divide(x: int, divisor: int) -> int:
    """Ceiling division.

    Args:
        x (`int`):
            dividend.
        divisor (`int`):
            divisor.

    Returns:
        `int`:
            ceiling division result.
    """
    return (x + divisor - 1) // divisor


def load_state_dict_in_safetensors(
    path: str | os.PathLike[str],
    device: str | torch.device = "cpu",
    filter_prefix: str = "",
    return_metadata: bool = False,
) -> dict[str, torch.Tensor] | tuple[dict[str, torch.Tensor], dict[str, str]]:
    state_dict = {}
    with safetensors.safe_open(fetch_or_download(path), framework="pt", device=device) as f:
        metadata = f.metadata()
        for k in f.keys():
            if filter_prefix and not k.startswith(filter_prefix):
                continue
            state_dict[k.removeprefix(filter_prefix)] = f.get_tensor(k)
    if return_metadata:
        return state_dict, metadata
    else:
        return state_dict


def filter_state_dict(state_dict: dict[str, torch.Tensor], filter_prefix: str = "") -> dict[str, torch.Tensor]:
    """Filter state dict.

    Args:
        state_dict (`dict`):
            state dict.
        filter_prefix (`str`):
            filter prefix.

    Returns:
        `dict`:
            filtered state dict.
    """
    return {k.removeprefix(filter_prefix): v for k, v in state_dict.items() if k.startswith(filter_prefix)}


def get_precision(
    precision: str = "auto",
    device: str | torch.device = "cuda",
    pretrained_model_name_or_path: str | os.PathLike[str] | None = None,
) -> str:
    assert precision in ("auto", "int4", "fp4")
    if precision == "auto":
        if isinstance(device, str):
            device = torch.device(device)
        capability = torch.cuda.get_device_capability(0 if device.index is None else device.index)
        sm = f"{capability[0]}{capability[1]}"
        precision = "fp4" if sm == "120" else "int4"
    if pretrained_model_name_or_path is not None:
        if precision == "int4":
            if "fp4" in str(pretrained_model_name_or_path):
                warnings.warn("The model may be quantized to fp4, but you are loading it with int4 precision.")
        elif precision == "fp4":
            if "int4" in str(pretrained_model_name_or_path):
                warnings.warn("The model may be quantized to int4, but you are loading it with fp4 precision.")
    return precision


def is_turing(device: str | torch.device = "cuda") -> bool:
    """Check if the current GPU is a Turing GPU.

    Returns:
        `bool`:
            True if the current GPU is a Turing GPU, False otherwise.
    """
    if isinstance(device, str):
        device = torch.device(device)
    device_id = 0 if device.index is None else device.index
    capability = torch.cuda.get_device_capability(device_id)
    sm = f"{capability[0]}{capability[1]}"
    return sm == "75"


def get_gpu_memory(device: str | torch.device = "cuda", unit: str = "GiB") -> int:
    """Get the GPU memory of the current device.

    Args:
        device (`str` | `torch.device`, optional, defaults to `"cuda"`):
            device.

    Returns:
        `int`:
            GPU memory in bytes.
    """
    if isinstance(device, str):
        device = torch.device(device)
    assert unit in ("GiB", "MiB", "B")
    memory = torch.cuda.get_device_properties(device).total_memory
    if unit == "GiB":
        return memory // (1024**3)
    elif unit == "MiB":
        return memory // (1024**2)
    else:
        return memory


def check_hardware_compatibility(quantization_config: dict, device: str | torch.device = "cuda"):
    if isinstance(device, str):
        device = torch.device(device)
    capability = torch.cuda.get_device_capability(0 if device.index is None else device.index)
    sm = f"{capability[0]}{capability[1]}"
    if sm == "120":  # you can only use the fp4 models
        if quantization_config["weight"]["dtype"] != "fp4_e2m1_all":
            raise ValueError('Please use "fp4" quantization for Blackwell GPUs. ')
    elif sm in ["75", "80", "86", "89"]:
        if quantization_config["weight"]["dtype"] != "int4":
            raise ValueError('Please use "int4" quantization for Turing, Ampere and Ada GPUs. ')
    else:
        raise ValueError(
            f"Unsupported GPU architecture {sm} due to the lack of 4-bit tensorcores. "
            "Please use a Turing, Ampere, Ada or Blackwell GPU for this quantization configuration."
        )
