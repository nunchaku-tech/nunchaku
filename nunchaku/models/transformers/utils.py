"""
Utility functions and mixins for Nunchaku transformer model loading and tensor manipulation.

This module provides:
- Logging configuration for Nunchaku transformers.
- The `NunchakuModelLoaderMixin` class for standardized model loading from safetensors or legacy folders.
- The `pad_tensor` utility for padding tensors to multiples of a given size.

These utilities are intended for internal use by Nunchaku's transformer backends.
"""

import json
import logging
import os
from pathlib import Path
from typing import Any, Optional

import torch
from diffusers import __version__
from huggingface_hub import constants, hf_hub_download
from torch import nn

from nunchaku.utils import ceil_divide, load_state_dict_in_safetensors

# Get log level from environment variable (default to INFO)
log_level = os.getenv("LOG_LEVEL", "INFO").upper()

# Configure logging
logging.basicConfig(level=getattr(logging, log_level, logging.INFO), format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class NunchakuModelLoaderMixin:
    """
    Mixin class providing standardized model loading utilities for Nunchaku transformer models.

    This mixin offers methods to build models from safetensors files or legacy folder structures,
    handling configuration, state dict loading, and device placement.

    Methods
    -------
    _build_model(pretrained_model_name_or_path, **kwargs)
        Build a model from a safetensors file, returning the model, state dict, and metadata.

    _build_model_legacy(pretrained_model_name_or_path, **kwargs)
        Build a model from a legacy folder structure, returning the model and paths to weight files.
    """

    @classmethod
    def _build_model(
        cls, pretrained_model_name_or_path: str | os.PathLike[str], **kwargs
    ) -> tuple[nn.Module, dict[str, torch.Tensor], dict[str, str]]:
        """
        Build a transformer model from a safetensors file.

        Parameters
        ----------
        pretrained_model_name_or_path : str or os.PathLike
            Path to the safetensors file containing the model weights and metadata.
        **kwargs
            Additional keyword arguments. Recognized: 'torch_dtype'.

        Returns
        -------
        tuple
            (transformer, state_dict, metadata)
            - transformer : nn.Module
                The instantiated transformer model (on 'meta' device).
            - state_dict : dict[str, torch.Tensor]
                The loaded state dictionary from safetensors.
            - metadata : dict[str, str]
                Metadata dictionary from safetensors file.
        """
        if isinstance(pretrained_model_name_or_path, str):
            pretrained_model_name_or_path = Path(pretrained_model_name_or_path)
        state_dict, metadata = load_state_dict_in_safetensors(pretrained_model_name_or_path, return_metadata=True)

        # Load the config file
        config = json.loads(metadata["config"])

        with torch.device("meta"):
            transformer = cls.from_config(config).to(kwargs.get("torch_dtype", torch.bfloat16))

        return transformer, state_dict, metadata

    @classmethod
    def _build_model_legacy(
        cls, pretrained_model_name_or_path: str | os.PathLike, **kwargs
    ) -> tuple[nn.Module, str, str]:
        """
        Build a transformer model from a legacy folder structure.

        .. warning::
            This method is deprecated and will be removed in v0.4.
            Please migrate to safetensors-based model loading.

        Parameters
        ----------
        pretrained_model_name_or_path : str or os.PathLike
            Path to the folder containing model weights.
        **kwargs
            Additional keyword arguments for HuggingFace Hub download and config loading.

        Returns
        -------
        tuple
            (transformer, unquantized_part_path, transformer_block_path)
            - transformer : nn.Module
                The instantiated transformer model (on 'meta' device).
            - unquantized_part_path : str
                Path to the unquantized layers safetensors file.
            - transformer_block_path : str
                Path to the transformer blocks safetensors file.
        """
        logger.warning(
            "Loading models from a folder will be deprecated in v0.4. "
            "Please download the latest safetensors model, or use one of the following tools to "
            "merge your model into a single file: the CLI utility `python -m nunchaku.merge_safetensors` "
            "or the ComfyUI workflow `merge_safetensors.json`."
        )
        subfolder = kwargs.get("subfolder", None)
        if os.path.exists(pretrained_model_name_or_path):
            dirname = (
                pretrained_model_name_or_path
                if subfolder is None
                else os.path.join(pretrained_model_name_or_path, subfolder)
            )
            unquantized_part_path = os.path.join(dirname, "unquantized_layers.safetensors")
            transformer_block_path = os.path.join(dirname, "transformer_blocks.safetensors")
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

        cache_dir = kwargs.pop("cache_dir", None)
        force_download = kwargs.pop("force_download", False)
        proxies = kwargs.pop("proxies", None)
        local_files_only = kwargs.pop("local_files_only", None)
        token = kwargs.pop("token", None)
        revision = kwargs.pop("revision", None)
        config, _, _ = cls.load_config(
            pretrained_model_name_or_path,
            subfolder=subfolder,
            cache_dir=cache_dir,
            return_unused_kwargs=True,
            return_commit_hash=True,
            force_download=force_download,
            proxies=proxies,
            local_files_only=local_files_only,
            token=token,
            revision=revision,
            user_agent={"diffusers": __version__, "file_type": "model", "framework": "pytorch"},
            **kwargs,
        )

        with torch.device("meta"):
            transformer = cls.from_config(config).to(kwargs.get("torch_dtype", torch.bfloat16))
        return transformer, unquantized_part_path, transformer_block_path


def pad_tensor(tensor: Optional[torch.Tensor], multiples: int, dim: int, fill: Any = 0) -> torch.Tensor | None:
    """
    Pad a tensor along a given dimension to the next multiple of a specified value.

    This is useful for ensuring tensor shapes are compatible with hardware or model requirements.

    Parameters
    ----------
    tensor : torch.Tensor or None
        The input tensor to pad. If None, returns None.
    multiples : int
        The multiple to pad to. If <= 1, no padding is applied.
    dim : int
        The dimension along which to pad.
    fill : Any, optional
        The value to use for padding (default: 0).

    Returns
    -------
    torch.Tensor or None
        The padded tensor, or None if input was None.
    """
    if multiples <= 1:
        return tensor
    if tensor is None:
        return None
    shape = list(tensor.shape)
    if shape[dim] % multiples == 0:
        return tensor
    shape[dim] = ceil_divide(shape[dim], multiples) * multiples
    result = torch.empty(shape, dtype=tensor.dtype, device=tensor.device)
    result.fill_(fill)
    result[[slice(0, extent) for extent in tensor.shape]] = tensor
    return result
