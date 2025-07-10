"""
Nunchaku T5 Encoder
===================

This module implements the Nunchaku T5 Encoder, an extension of HuggingFace's T5EncoderModel
with support for memory-efficient low-bit inference.

Overview
--------
The NunchakuT5EncoderModel class enables loading T5 encoder weights from safetensors files,
automatically replacing supported linear layers with quantized :class:`~nunchaku.models.text_encoders.linear.W4Linear`
modules for improved performance and memory efficiency.

Example
-------
.. code-block:: python

    from nunchaku import NunchakuT5EncoderModel
    model = NunchakuT5EncoderModel.from_pretrained(
        "mit-han-lab/nunchaku-t5/awq-int4-flux.1-t5xxl.safetensors"
    )

"""

import json
import logging
import os
from pathlib import Path

import torch
from accelerate import init_empty_weights
from torch import nn
from transformers import T5Config, T5EncoderModel

from ...utils import load_state_dict_in_safetensors
from .linear import W4Linear

# Get log level from environment variable (default to INFO)
log_level = os.getenv("LOG_LEVEL", "INFO").upper()

# Configure logging
logging.basicConfig(level=getattr(logging, log_level, logging.INFO), format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class NunchakuT5EncoderModel(T5EncoderModel):
    """
    Nunchaku T5 Encoder Model

    Extends :class:`transformers.T5EncoderModel` to support quantized weights and
    memory-efficient inference using :class:`~nunchaku.models.text_encoders.linear.W4Linear`.

    This class provides a convenient interface for loading T5 encoder weights from
    safetensors files, automatically replacing supported linear layers with quantized
    modules for improved speed and reduced memory usage.

    Example
    -------
    .. code-block:: python

        model = NunchakuT5EncoderModel.from_pretrained(
            "mit-han-lab/nunchaku-t5/awq-int4-flux.1-t5xxl.safetensors"
        )
    """

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str | os.PathLike[str], **kwargs):
        """
        Load a :class:`NunchakuT5EncoderModel` from a safetensors file.

        This method loads the model configuration and weights from a safetensors file,
        initializes the model on the 'meta' device (no memory allocation for weights),
        and replaces supported linear layers with quantized :class:`~nunchaku.models.text_encoders.linear.W4Linear` modules.

        Parameters
        ----------
        pretrained_model_name_or_path : str or os.PathLike
            Path to the safetensors file containing the model weights and metadata.
        torch_dtype : torch.dtype, optional
            Data type for model initialization (default: ``torch.bfloat16``).
            Set to ``torch.float16`` for Turing GPUs.
        device : str or torch.device, optional
            Device to load the model onto (default: ``"cuda"``).
            If the model is loaded on CPU, it will be automatically moved to GPU.

        Returns
        -------
        NunchakuT5EncoderModel
            The loaded and quantized T5 encoder model.

        Example
        -------
        .. code-block:: python

            model = NunchakuT5EncoderModel.from_pretrained(
                "mit-han-lab/nunchaku-t5/awq-int4-flux.1-t5xxl.safetensors"
            )
        """
        pretrained_model_name_or_path = Path(pretrained_model_name_or_path)
        state_dict, metadata = load_state_dict_in_safetensors(pretrained_model_name_or_path, return_metadata=True)

        # Load the config file from metadata
        config = json.loads(metadata["config"])
        config = T5Config(**config)

        # Initialize model on 'meta' device (no memory allocation for weights)
        with init_empty_weights():
            t5_encoder = T5EncoderModel(config).to(kwargs.get("torch_dtype", torch.bfloat16))

        t5_encoder.eval()

        # Load the model weights from the safetensors file and quantize supported linear layers
        named_modules = {}
        for name, module in t5_encoder.named_modules():
            assert isinstance(name, str)
            if isinstance(module, nn.Linear):
                if f"{name}.qweight" in state_dict:
                    logger.debug(f"Switching {name} to W4Linear")
                    qmodule = W4Linear.from_linear(module, group_size=128, init_only=True)
                    # modeling_t5.py: T5DenseGatedActDense needs dtype of weight
                    qmodule.weight = torch.empty([1], dtype=module.weight.dtype, device=module.weight.device)

                    parent_name, child_name = name.rsplit(".", 1)
                    setattr(named_modules[parent_name], child_name, qmodule)
            else:
                named_modules[name] = module

        device = kwargs.get("device", "cuda")
        if isinstance(device, str):
            device = torch.device(device)
        t5_encoder.to_empty(device=device)
        t5_encoder.load_state_dict(state_dict, strict=True)

        return t5_encoder
