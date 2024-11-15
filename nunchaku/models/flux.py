import os
import types

import diffusers
import torch
from diffusers import FluxPipeline, FluxTransformer2DModel
from huggingface_hub import hf_hub_download
from packaging.version import Version
from torch import nn

from .._C import QuantizedFluxModel

SVD_RANK = 32


class NunchakuFluxModel(nn.Module):
    def __init__(self, m: QuantizedFluxModel):
        super().__init__()
        self.m = m
        self.dtype = torch.bfloat16

    def forward(
        self,
        /,
        hidden_states: torch.Tensor,
        temb: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        image_rotary_emb: torch.Tensor,
        joint_attention_kwargs=None,
        controlnet_block_samples=None,
        controlnet_single_block_samples=None,
    ):
        batch_size = hidden_states.shape[0]
        txt_tokens = encoder_hidden_states.shape[1]
        img_tokens = hidden_states.shape[1]

        original_dtype = hidden_states.dtype

        hidden_states = hidden_states.to(self.dtype)
        encoder_hidden_states = encoder_hidden_states.to(self.dtype)
        temb = temb.to(self.dtype)

        assert image_rotary_emb.ndim == 6
        assert image_rotary_emb.shape[0] == 1
        assert image_rotary_emb.shape[1] == 1
        assert image_rotary_emb.shape[2] == batch_size * (txt_tokens + img_tokens)
        # [bs, tokens, head_dim / 2, 1, 2] (sincos)
        image_rotary_emb = image_rotary_emb.reshape(
            [batch_size, txt_tokens + img_tokens, *image_rotary_emb.shape[3:]]
        )
        rotary_emb_txt = image_rotary_emb[:, :txt_tokens, ...]  # .to(self.dtype)
        rotary_emb_img = image_rotary_emb[:, txt_tokens:, ...]  # .to(self.dtype)
        rotary_emb_single = image_rotary_emb  # .to(self.dtype)

        hidden_states = self.m.forward(
            hidden_states,
            encoder_hidden_states,
            temb,
            rotary_emb_img,
            rotary_emb_txt,
            rotary_emb_single,
            controlnet_block_samples,
            controlnet_single_block_samples,
        )

        hidden_states = hidden_states.to(original_dtype)

        encoder_hidden_states = hidden_states[:, :txt_tokens, ...]
        hidden_states = hidden_states[:, txt_tokens:, ...]

        return encoder_hidden_states, hidden_states


## copied from diffusers 0.30.3
def rope(pos: torch.Tensor, dim: int, theta: int) -> torch.Tensor:
    assert dim % 2 == 0, "The dimension must be even."

    scale = torch.arange(0, dim, 2, dtype=torch.float64, device=pos.device) / dim
    omega = 1.0 / (theta**scale)

    batch_size, seq_length = pos.shape
    out = torch.einsum("...n,d->...nd", pos, omega)

    USE_SINCOS = True
    if USE_SINCOS:
        cos_out = torch.cos(out)
        sin_out = torch.sin(out)
        stacked_out = torch.stack([sin_out, cos_out], dim=-1)
        out = stacked_out.view(batch_size, -1, dim // 2, 1, 2)
    else:
        out = out.view(batch_size, -1, dim // 2, 1, 1)

    # stacked_out = torch.stack([cos_out, -sin_out, sin_out, cos_out], dim=-1)
    # out = stacked_out.view(batch_size, -1, dim // 2, 2, 2)
    return out.float()


class EmbedND(torch.nn.Module):
    def __init__(self, dim: int, theta: int, axes_dim: list[int]):
        super().__init__()
        self.dim = dim
        self.theta = theta
        self.axes_dim = axes_dim

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        if Version(diffusers.__version__) >= Version("0.31.0"):
            ids = ids[None, ...]
        n_axes = ids.shape[-1]
        emb = torch.cat(
            [rope(ids[..., i], self.axes_dim[i], self.theta) for i in range(n_axes)],
            dim=-3,
        )
        return emb.unsqueeze(1)


def load_quantized_model(path: str, device: str | torch.device) -> QuantizedFluxModel:
    device = torch.device(device)
    assert device.type == "cuda"

    m = QuantizedFluxModel()
    m.disableMemoryAutoRelease()
    m.init(True, 0 if device.index is None else device.index)
    m.load(path)
    return m


def inject_pipeline(pipe: FluxPipeline, m: QuantizedFluxModel) -> FluxPipeline:
    net: FluxTransformer2DModel = pipe.transformer
    net.pos_embed = EmbedND(dim=net.inner_dim, theta=10000, axes_dim=[16, 56, 56])

    net.transformer_blocks = torch.nn.ModuleList([NunchakuFluxModel(m)])
    net.single_transformer_blocks = torch.nn.ModuleList([])

    def update_params(self: FluxTransformer2DModel, path: str):
        if not os.path.exists(path):
            hf_repo_id = os.path.dirname(path)
            filename = os.path.basename(path)
            path = hf_hub_download(repo_id=hf_repo_id, filename=filename)
        block = self.transformer_blocks[0]
        assert isinstance(block, NunchakuFluxModel)
        block.m.load(path, True)

    def set_lora_scale(self: FluxTransformer2DModel, scale: float):
        block = self.transformer_blocks[0]
        assert isinstance(block, NunchakuFluxModel)
        block.m.setLoraScale(SVD_RANK, scale)

    net.nunchaku_update_params = types.MethodType(update_params, net)
    net.nunchaku_set_lora_scale = types.MethodType(set_lora_scale, net)

    return pipe


def inject_transformer(
    transformer_model: FluxTransformer2DModel, m: QuantizedFluxModel
) -> None:
    """注入自定义transformer模型

    Args:
        transformer_model: 原始transformer模型
        custom_model: 要注入的自定义模型
    """
    # 注入位置编码
    transformer_model.pos_embed = EmbedND(
        dim=transformer_model.inner_dim, theta=10000, axes_dim=[16, 56, 56]
    )

    # 替换transformer块
    transformer_model.transformer_blocks = torch.nn.ModuleList([NunchakuFluxModel(m)])
    transformer_model.single_transformer_blocks = torch.nn.ModuleList([])

    def update_params(self: FluxTransformer2DModel, path: str):
        if not os.path.exists(path):
            hf_repo_id = os.path.dirname(path)
            filename = os.path.basename(path)
            path = hf_hub_download(repo_id=hf_repo_id, filename=filename)
        block = self.transformer_blocks[0]
        assert isinstance(block, NunchakuFluxModel)
        block.m.load(path, True)

    def set_lora_scale(self: FluxTransformer2DModel, scale: float):
        block = self.transformer_blocks[0]
        assert isinstance(block, NunchakuFluxModel)
        block.m.setLoraScale(SVD_RANK, scale)

    transformer_model.nunchaku_update_params = types.MethodType(
        update_params, transformer_model
    )
    transformer_model.nunchaku_set_lora_scale = types.MethodType(
        set_lora_scale, transformer_model
    )

    return transformer_model
