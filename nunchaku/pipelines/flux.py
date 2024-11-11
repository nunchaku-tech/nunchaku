import os

import torch
from diffusers import FluxPipeline
from huggingface_hub import hf_hub_download

from ..models.flux import inject_pipeline, load_quantized_model


def from_pretrained(
    pretrained_model_name_or_path: str | os.PathLike, **kwargs
) -> FluxPipeline:
    qmodel_device = kwargs.pop("qmodel_device", "cuda:0")
    qmodel_device = torch.device(qmodel_device)
    if qmodel_device.type != "cuda":
        raise ValueError(f"qmodel_device = {qmodel_device} is not a CUDA device")

    qmodel_path = kwargs.pop("qmodel_path")

    if not os.path.exists(qmodel_path):
        hf_repo_id = os.path.dirname(qmodel_path)
        filename = os.path.basename(qmodel_path)
        qmodel_path = hf_hub_download(repo_id=hf_repo_id, filename=filename)

    pipeline = FluxPipeline.from_pretrained(pretrained_model_name_or_path, **kwargs)
    m = load_quantized_model(
        qmodel_path, 0 if qmodel_device.index is None else qmodel_device.index
    )
    inject_pipeline(pipeline, m)

    # if qencoder_path is not None:
    #     assert isinstance(qencoder_path, str)
    #     if not os.path.exists(qencoder_path):
    #         hf_repo_id = os.path.dirname(qencoder_path)
    #         filename = os.path.basename(qencoder_path)
    #         qencoder_path = hf_hub_download(repo_id=hf_repo_id, filename=filename)
    #     quantize_t5(pipeline, qencoder_path)

    return pipeline
