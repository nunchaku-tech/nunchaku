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

logger = logging.getLogger(__name__)


class NunchakuT5EncoderModel(T5EncoderModel):
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str | os.PathLike[str], **kwargs):
        pretrained_model_name_or_path = Path(pretrained_model_name_or_path)
        state_dict = load_state_dict_in_safetensors(pretrained_model_name_or_path)

        # Load the config file
        metadata = state_dict.pop("__metadata__", {})
        config = json.loads(metadata["config"])
        config = T5Config(**config)

        # Initialize model on 'meta' device (no memory allocation for weights)
        with init_empty_weights():
            t5_encoder = T5EncoderModel(config).to(kwargs.get("torch_dtype", torch.bfloat16))

        t5_encoder.eval()

        # Load the model weights from the safetensors file
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
