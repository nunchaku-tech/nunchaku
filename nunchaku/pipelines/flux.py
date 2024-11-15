import os

import torch
from diffusers import FluxPipeline
from huggingface_hub import hf_hub_download

from ..models.flux import inject_pipeline, load_quantized_model


def quantize_t5(pipe: FluxPipeline, qencoder_path: str):
    assert os.path.exists(qencoder_path), f"qencoder_path {qencoder_path} does not exist"
    from deepcompressor.backend.tinychat.linear import W4Linear

    named_modules = {}
    qencoder_state_dict = torch.load(qencoder_path, map_location="cpu")
    for name, module in pipe.text_encoder_2.named_modules():
        assert isinstance(name, str)
        if isinstance(module, torch.nn.Linear):
            suffix = [".q", ".k", ".v", ".o", ".wi_0"]
            if f"{name}.qweight" in qencoder_state_dict and name.endswith(tuple(suffix)):
                print(f"Switching {name} to W4Linear")
                qmodule = W4Linear.from_linear(module, group_size=128, init_only=True)
                qmodule.qweight.data.copy_(qencoder_state_dict[f"{name}.qweight"])
                if qmodule.bias is not None:
                    qmodule.bias.data.copy_(qencoder_state_dict[f"{name}.bias"])
                qmodule.scales.data.copy_(qencoder_state_dict[f"{name}.scales"])
                qmodule.scaled_zeros.data.copy_(qencoder_state_dict[f"{name}.scaled_zeros"])

                # modeling_t5.py: T5DenseGatedActDense needs dtype of weight
                qmodule.weight = torch.empty([1], dtype=module.weight.dtype, device=module.weight.device)

                parent_name, child_name = name.rsplit(".", 1)
                setattr(named_modules[parent_name], child_name, qmodule)
        else:
            named_modules[name] = module


def from_pretrained(pretrained_model_name_or_path: str | os.PathLike, **kwargs) -> FluxPipeline:
    qmodel_device = kwargs.pop("qmodel_device", "cuda:0")
    qmodel_device = torch.device(qmodel_device)
    if qmodel_device.type != "cuda":
        raise ValueError(f"qmodel_device = {qmodel_device} is not a CUDA device")

    qmodel_path = kwargs.pop("qmodel_path")
    qencoder_path = kwargs.pop("qencoder_path", None)

    if not os.path.exists(qmodel_path):
        hf_repo_id = os.path.dirname(qmodel_path)
        filename = os.path.basename(qmodel_path)
        qmodel_path = hf_hub_download(repo_id=hf_repo_id, filename=filename)

    pipeline = FluxPipeline.from_pretrained(pretrained_model_name_or_path, **kwargs)
    m = load_quantized_model(qmodel_path, 0 if qmodel_device.index is None else qmodel_device.index)
    inject_pipeline(pipeline, m)

    if qencoder_path is not None:
        assert isinstance(qencoder_path, str)
        if not os.path.exists(qencoder_path):
            hf_repo_id = os.path.dirname(qencoder_path)
            filename = os.path.basename(qencoder_path)
            qencoder_path = hf_hub_download(repo_id=hf_repo_id, filename=filename)
        quantize_t5(pipeline, qencoder_path)

    return pipeline
