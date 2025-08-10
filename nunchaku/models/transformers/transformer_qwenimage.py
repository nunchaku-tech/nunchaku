import os
from pathlib import Path

import torch
from diffusers.models.transformers.transformer_qwenimage import QwenImageTransformer2DModel, QwenImageTransformerBlock
from huggingface_hub import utils

from ...utils import get_precision
from ..attention import NunchakuFeedForward
from ..linear import AWQW4A16Linear
from .utils import NunchakuModelLoaderMixin


class NunchakuQwenDoubleStreamAttention:
    pass


class NunchakuQwenImageTransformerBlock(QwenImageTransformerBlock):
    def __init__(self, other: QwenImageTransformerBlock, scale_shift: float = 0.0, **kwargs):
        super(QwenImageTransformerBlock, self).__init__()

        self.dim = other.dim
        self.img_mod = other.img_mod
        self.img_mod[1] = AWQW4A16Linear(self.img_mod[1])
        self.img_norm1 = self.img_norm1
        self.attn = NunchakuQwenDoubleStreamAttention(other.attn)
        self.img_norm2 = self.img_norm2
        self.img_mlp = NunchakuFeedForward(self.img_mlp)

        # Text processing modules
        self.txt_mod = self.txt_mod
        self.txt_mod[1] = AWQW4A16Linear(self.txt_mod[1])
        self.txt_norm1 = self.txt_norm1
        # Text doesn't need separate attention - it's handled by img_attn joint computation
        self.txt_norm2 = self.txt_norm2
        self.txt_mlp = NunchakuFeedForward(self.txt_mlp)


class NunchakuQwenImageTransformer2DModel(QwenImageTransformer2DModel, NunchakuModelLoaderMixin):

    def _patch_model(self, **kwargs):
        for i, block in enumerate(self.transformer_blocks):
            self.transformer_blocks[i] = NunchakuQwenImageTransformerBlock(block, scale_shift=0, **kwargs)
        return self

    @classmethod
    @utils.validate_hf_hub_args
    def from_pretrained(cls, pretrained_model_name_or_path: str | os.PathLike[str], **kwargs):
        device = kwargs.get("device", "cpu")
        offload = kwargs.get("offload", False)

        if offload:
            raise NotImplementedError("Offload is not supported for FluxTransformer2DModelV2")

        torch_dtype = kwargs.get("torch_dtype", torch.bfloat16)

        if isinstance(pretrained_model_name_or_path, str):
            pretrained_model_name_or_path = Path(pretrained_model_name_or_path)

        assert pretrained_model_name_or_path.is_file() or pretrained_model_name_or_path.name.endswith(
            (".safetensors", ".sft")
        ), "Only safetensors are supported"
        transformer, model_state_dict, metadata = cls._build_model(pretrained_model_name_or_path, **kwargs)
        transformer = transformer.to(torch_dtype)

        precision = get_precision()
        if precision == "fp4":
            precision = "nvfp4"
        transformer._patch_model(precision=precision)

        transformer = transformer.to_empty(device=device)
        transformer.load_state_dict(model_state_dict)

        return transformer
