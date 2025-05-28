import os

import torch
from accelerate import init_empty_weights
from huggingface_hub import constants, hf_hub_download
from safetensors.torch import load_file
from torch import nn
from transformers import PretrainedConfig, T5Config, T5EncoderModel

from .linear import W4Linear


class NunchakuT5EncoderModel(T5EncoderModel):
    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str | os.PathLike,
        config: PretrainedConfig | str | os.PathLike | None = None,
        cache_dir: str | os.PathLike | None = None,
        force_download: bool = False,
        local_files_only: bool = False,
        token: str | bool | None = None,
        revision: str = "main",
        **kwargs,
    ):
        subfolder = kwargs.get("subfolder", None)
        if os.path.exists(pretrained_model_name_or_path):
            dirname = (
                pretrained_model_name_or_path
                if subfolder is None
                else os.path.join(pretrained_model_name_or_path, subfolder)
            )
            qmodel_path = os.path.join(dirname, "svdq-t5.safetensors")
            config_path = os.path.join(dirname, "config.json")
        else:
            shared_kwargs = {
                "repo_id": pretrained_model_name_or_path,
                "subfolder": subfolder,
                "repo_type": "model",
                "revision": revision,
                "library_name": kwargs.get("library_name"),
                "library_version": kwargs.get("library_version"),
                "cache_dir": cache_dir,
                "local_dir": kwargs.get("local_dir"),
                "user_agent": kwargs.get("user_agent"),
                "force_download": force_download,
                "proxies": kwargs.get("proxies"),
                "etag_timeout": kwargs.get("etag_timeout", constants.DEFAULT_ETAG_TIMEOUT),
                "token": token,
                "local_files_only": local_files_only,
                "headers": kwargs.get("headers"),
                "endpoint": kwargs.get("endpoint"),
                "resume_download": kwargs.get("resume_download"),
                "force_filename": kwargs.get("force_filename"),
                "local_dir_use_symlinks": kwargs.get("local_dir_use_symlinks", "auto"),
            }
            qmodel_path = hf_hub_download(filename="svdq-t5.safetensors", **shared_kwargs)
            config_path = hf_hub_download(filename="config.json", **shared_kwargs)

        # Load the config file
        config = T5Config.from_json_file(config_path)
        # Initialize model on 'meta' device (no memory allocation for weights)
        with init_empty_weights():
            t5_encoder = T5EncoderModel(config).to(kwargs.get("torch_dtype", torch.bfloat16))

        t5_encoder.eval()
        # Load the model weights from the safetensors file
        state_dict = load_file(qmodel_path)

        named_modules = {}
        for name, module in t5_encoder.named_modules():
            assert isinstance(name, str)
            if isinstance(module, nn.Linear):
                if f"{name}.qweight" in state_dict:
                    print(f"Switching {name} to W4Linear")
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
