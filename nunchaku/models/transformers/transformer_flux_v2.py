import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import torch
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.models.transformers.transformer_flux import (
    FluxAttention,
    FluxSingleTransformerBlock,
    FluxTransformer2DModel,
    FluxTransformerBlock,
)
from huggingface_hub import utils
from torch import nn
from torch.nn import GELU

from ...ops.fused import fused_gelu_mlp
from ...utils import get_precision
from ..attention import NunchakuFeedForward
from ..attention_processor import NunchakuFA2Processor, NunchakuFP16AttnProcessor
from ..embeddings import NunchakuFluxPosEmbed, pack_rotemb
from ..linear import SVDQW4A4Linear
from ..normalization import NunchakuAdaLayerNormZero, NunchakuAdaLayerNormZeroSingle
from ..utils import fuse_linears
from .utils import NunchakuModelLoaderMixin, pad_tensor


class NunchakuFluxAttention(nn.Module):
    def __init__(self, flux_attention: FluxAttention, processor: str = "nunchaku-fp16", **kwargs):
        super(NunchakuFluxAttention, self).__init__()

        self.head_dim = flux_attention.head_dim
        self.inner_dim = flux_attention.inner_dim
        self.query_dim = flux_attention.query_dim
        self.use_bias = flux_attention.use_bias
        self.dropout = flux_attention.dropout
        self.out_dim = flux_attention.out_dim
        self.context_pre_only = flux_attention.context_pre_only
        self.pre_only = flux_attention.pre_only
        self.heads = flux_attention.heads
        self.added_kv_proj_dim = flux_attention.added_kv_proj_dim
        self.added_proj_bias = flux_attention.added_proj_bias

        self.norm_q = flux_attention.norm_q
        self.norm_k = flux_attention.norm_k

        # fuse the qkv
        with torch.device("meta"):
            to_qkv = fuse_linears([flux_attention.to_q, flux_attention.to_k, flux_attention.to_v])
        self.to_qkv = SVDQW4A4Linear.from_linear(to_qkv, **kwargs)

        if not self.pre_only:
            self.to_out = flux_attention.to_out
            self.to_out[0] = SVDQW4A4Linear.from_linear(self.to_out[0], **kwargs)

        if self.added_kv_proj_dim is not None:
            self.norm_added_q = flux_attention.norm_added_q
            self.norm_added_k = flux_attention.norm_added_k

            # fuse the add_qkv
            with torch.device("meta"):
                add_qkv_proj = fuse_linears(
                    [flux_attention.add_q_proj, flux_attention.add_k_proj, flux_attention.add_v_proj]
                )
            self.add_qkv_proj = SVDQW4A4Linear.from_linear(add_qkv_proj, **kwargs)
            self.to_add_out = SVDQW4A4Linear.from_linear(flux_attention.to_add_out, **kwargs)

        self.processor = None
        self.set_processor(processor)

    def set_processor(self, processor: str):
        if processor == "flashattn2":
            self.processor = NunchakuFA2Processor()
        elif processor == "nunchaku-fp16":
            self.processor = NunchakuFP16AttnProcessor()
        else:
            raise ValueError(f"Processor {processor} is not supported")

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        image_rotary_emb: Tuple[torch.Tensor, torch.Tensor] | torch.Tensor = None,
        **kwargs,
    ):
        return self.processor(
            attn=self,
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            attention_mask=attention_mask,
            image_rotary_emb=image_rotary_emb,
        )


class NunchakuFluxTransformerBlock(FluxTransformerBlock):

    def __init__(self, block: FluxTransformerBlock, scale_shift: float = 1, **kwargs):
        super(FluxTransformerBlock, self).__init__()
        self.scale_shift = scale_shift

        # The scale_shift=1 from AdaLayerNormZero has already been fused into the linear weights,
        # so we set scale_shift=0 here to avoid applying it again.
        self.norm1 = NunchakuAdaLayerNormZero(block.norm1, scale_shift=scale_shift)
        self.norm1_context = NunchakuAdaLayerNormZero(block.norm1_context, scale_shift=scale_shift)

        self.attn = NunchakuFluxAttention(block.attn, **kwargs)
        self.norm2 = block.norm2
        self.norm2_context = block.norm2_context
        self.ff = NunchakuFeedForward(block.ff, **kwargs)
        self.ff_context = NunchakuFeedForward(block.ff_context, **kwargs)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb: torch.Tensor,
        image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
    ):
        if joint_attention_kwargs is not None and len(joint_attention_kwargs) > 0:
            raise NotImplementedError("joint_attention_kwargs is not supported")

        norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm1(hidden_states, emb=temb)

        norm_encoder_hidden_states, c_gate_msa, c_shift_mlp, c_scale_mlp, c_gate_mlp = self.norm1_context(
            encoder_hidden_states, emb=temb
        )

        joint_attention_kwargs = joint_attention_kwargs or {}

        # Attention.
        attention_outputs = self.attn(
            hidden_states=norm_hidden_states,
            encoder_hidden_states=norm_encoder_hidden_states,
            image_rotary_emb=image_rotary_emb,
            **joint_attention_kwargs,
        )

        if len(attention_outputs) == 2:
            attn_output, context_attn_output = attention_outputs
        elif len(attention_outputs) == 3:
            attn_output, context_attn_output, ip_attn_output = attention_outputs

        # Process attention outputs for the `hidden_states`.
        attn_output = gate_msa.unsqueeze(1) * attn_output
        hidden_states = hidden_states + attn_output

        norm_hidden_states = self.norm2(hidden_states)
        norm_hidden_states = norm_hidden_states * scale_mlp[:, None] + shift_mlp[:, None]

        ff_output = self.ff(norm_hidden_states)
        ff_output = gate_mlp.unsqueeze(1) * ff_output

        hidden_states = hidden_states + ff_output
        if len(attention_outputs) == 3:
            hidden_states = hidden_states + ip_attn_output

        # Process attention outputs for the `encoder_hidden_states`.
        context_attn_output = c_gate_msa.unsqueeze(1) * context_attn_output
        encoder_hidden_states = encoder_hidden_states + context_attn_output

        norm_encoder_hidden_states = self.norm2_context(encoder_hidden_states)
        norm_encoder_hidden_states = norm_encoder_hidden_states * c_scale_mlp[:, None] + c_shift_mlp[:, None]

        context_ff_output = self.ff_context(norm_encoder_hidden_states)
        encoder_hidden_states = encoder_hidden_states + c_gate_mlp.unsqueeze(1) * context_ff_output
        if encoder_hidden_states.dtype == torch.float16:
            encoder_hidden_states = encoder_hidden_states.clip(-65504, 65504)

        return encoder_hidden_states, hidden_states


class NunchakuFluxSingleTransformerBlock(FluxSingleTransformerBlock):
    def __init__(self, block: FluxSingleTransformerBlock, scale_shift: float = 1, **kwargs):
        super(FluxSingleTransformerBlock, self).__init__()
        self.mlp_hidden_dim = block.mlp_hidden_dim
        self.norm = block.norm
        self.norm = NunchakuAdaLayerNormZeroSingle(block.norm, scale_shift=scale_shift)

        self.mlp_fc1 = SVDQW4A4Linear.from_linear(block.proj_mlp, **kwargs)
        self.act_mlp = block.act_mlp
        self.mlp_fc2 = SVDQW4A4Linear.from_linear(block.proj_out, in_features=self.mlp_hidden_dim, **kwargs)
        # for int4, we shift the activation of mlp_fc2 to make it unsigned
        self.mlp_fc2.act_unsigned = self.mlp_fc2.precision != "nvfp4"

        self.attn = NunchakuFluxAttention(block.attn, **kwargs)
        self.attn.to_out = SVDQW4A4Linear.from_linear(block.proj_out, in_features=self.mlp_fc1.in_features, **kwargs)

    def forward(
        self,
        hidden_states: torch.Tensor,
        temb: torch.Tensor,
        image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
    ) -> torch.Tensor:

        residual = hidden_states
        norm_hidden_states, gate = self.norm(hidden_states, emb=temb)

        # Feedforward
        if isinstance(self.act_mlp, GELU):
            # use fused gelu mlp
            mlp_hidden_states = fused_gelu_mlp(norm_hidden_states, self.mlp_fc1, self.mlp_fc2)
        else:
            # fallback to original gelu mlp
            mlp_hidden_states = self.mlp_fc1(norm_hidden_states)
            mlp_hidden_states = self.act_mlp(mlp_hidden_states)
            mlp_hidden_states = self.mlp_fc2(mlp_hidden_states)

        # Attention
        joint_attention_kwargs = joint_attention_kwargs or {}
        attn_output = self.attn(
            hidden_states=norm_hidden_states, image_rotary_emb=image_rotary_emb, **joint_attention_kwargs
        )

        hidden_states = attn_output + mlp_hidden_states
        gate = gate.unsqueeze(1)
        hidden_states = gate * hidden_states
        hidden_states = residual + hidden_states
        if hidden_states.dtype == torch.float16:
            hidden_states = hidden_states.clip(-65504, 65504)

        return hidden_states


class NunchakuFluxTransformer2DModelV2(FluxTransformer2DModel, NunchakuModelLoaderMixin):

    def _patch_model(self, **kwargs):
        self.pos_embed = NunchakuFluxPosEmbed(dim=self.inner_dim, theta=10000, axes_dim=self.pos_embed.axes_dim)
        for i, block in enumerate(self.transformer_blocks):
            self.transformer_blocks[i] = NunchakuFluxTransformerBlock(block, scale_shift=0, **kwargs)
        for i, block in enumerate(self.single_transformer_blocks):
            self.single_transformer_blocks[i] = NunchakuFluxSingleTransformerBlock(block, scale_shift=0, **kwargs)
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
        converted_state_dict = convert_flux_state_dict(model_state_dict)

        state_dict = transformer.state_dict()

        for k in state_dict.keys():
            if k not in converted_state_dict:
                assert ".wtscale" in k or ".wcscales" in k
                converted_state_dict[k] = torch.ones_like(state_dict[k])
            else:
                assert state_dict[k].dtype == converted_state_dict[k].dtype

        transformer.load_state_dict(converted_state_dict)

        return transformer

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor = None,
        pooled_projections: torch.Tensor = None,
        timestep: torch.LongTensor = None,
        img_ids: torch.Tensor = None,
        txt_ids: torch.Tensor = None,
        guidance: torch.Tensor = None,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        controlnet_block_samples=None,
        controlnet_single_block_samples=None,
        return_dict: bool = True,
        controlnet_blocks_repeat: bool = False,
    ) -> Union[torch.Tensor, Transformer2DModelOutput]:
        """
        The [`FluxTransformer2DModel`] forward method.

        Args:
            hidden_states (`torch.Tensor` of shape `(batch_size, image_sequence_length, in_channels)`):
                Input `hidden_states`.
            encoder_hidden_states (`torch.Tensor` of shape `(batch_size, text_sequence_length, joint_attention_dim)`):
                Conditional embeddings (embeddings computed from the input conditions such as prompts) to use.
            pooled_projections (`torch.Tensor` of shape `(batch_size, projection_dim)`): Embeddings projected
                from the embeddings of input conditions.
            timestep ( `torch.LongTensor`):
                Used to indicate denoising step.
            block_controlnet_hidden_states: (`list` of `torch.Tensor`):
                A list of tensors that if specified are added to the residuals of transformer blocks.
            joint_attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
                `self.processor` in
                [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~models.transformer_2d.Transformer2DModelOutput`] instead of a plain
                tuple.

        Returns:
            If `return_dict` is True, an [`~models.transformer_2d.Transformer2DModelOutput`] is returned, otherwise a
            `tuple` where the first element is the sample tensor.
        """
        hidden_states = self.x_embedder(hidden_states)

        timestep = timestep.to(hidden_states.dtype) * 1000
        if guidance is not None:
            guidance = guidance.to(hidden_states.dtype) * 1000

        temb = (
            self.time_text_embed(timestep, pooled_projections)
            if guidance is None
            else self.time_text_embed(timestep, guidance, pooled_projections)
        )
        encoder_hidden_states = self.context_embedder(encoder_hidden_states)

        if txt_ids.ndim == 3:
            txt_ids = txt_ids[0]
        if img_ids.ndim == 3:
            img_ids = img_ids[0]

        ids = torch.cat((txt_ids, img_ids), dim=0)
        image_rotary_emb = self.pos_embed(ids)

        if joint_attention_kwargs is not None and "ip_adapter_image_embeds" in joint_attention_kwargs:
            ip_adapter_image_embeds = joint_attention_kwargs.pop("ip_adapter_image_embeds")
            ip_hidden_states = self.encoder_hid_proj(ip_adapter_image_embeds)
            joint_attention_kwargs.update({"ip_hidden_states": ip_hidden_states})

        txt_tokens = encoder_hidden_states.shape[1]
        img_tokens = hidden_states.shape[1]

        assert image_rotary_emb.ndim == 6
        assert image_rotary_emb.shape[0] == 1
        assert image_rotary_emb.shape[1] == 1
        assert image_rotary_emb.shape[2] == 1 * (txt_tokens + img_tokens)
        # [1, tokens, head_dim / 2, 1, 2] (sincos)
        image_rotary_emb = image_rotary_emb.reshape([1, txt_tokens + img_tokens, *image_rotary_emb.shape[3:]])
        rotary_emb_txt = image_rotary_emb[:, :txt_tokens, ...]  # .to(self.dtype)
        rotary_emb_img = image_rotary_emb[:, txt_tokens:, ...]  # .to(self.dtype)
        rotary_emb_single = image_rotary_emb

        rotary_emb_txt = pack_rotemb(pad_tensor(rotary_emb_txt, 256, 1))
        rotary_emb_img = pack_rotemb(pad_tensor(rotary_emb_img, 256, 1))
        rotary_emb_single = pack_rotemb(pad_tensor(rotary_emb_single, 256, 1))

        for index_block, block in enumerate(self.transformer_blocks):
            encoder_hidden_states, hidden_states = block(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                temb=temb,
                image_rotary_emb=(rotary_emb_img, rotary_emb_txt),
                joint_attention_kwargs=joint_attention_kwargs,
            )

            # controlnet residual
            if controlnet_block_samples is not None:
                raise NotImplementedError("Controlnet is not supported for FluxTransformer2DModelV2 for now")

        hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)
        for index_block, block in enumerate(self.single_transformer_blocks):
            hidden_states = block(
                hidden_states=hidden_states,
                temb=temb,
                image_rotary_emb=rotary_emb_single,
                joint_attention_kwargs=joint_attention_kwargs,
            )

            # controlnet residual
            if controlnet_single_block_samples is not None:
                raise NotImplementedError("Controlnet is not supported for FluxTransformer2DModelV2 for now")

        hidden_states = hidden_states[:, txt_tokens:]
        hidden_states = self.norm_out(hidden_states, temb)
        output = self.proj_out(hidden_states)

        if not return_dict:
            return (output,)

        return Transformer2DModelOutput(sample=output)


def convert_flux_state_dict(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    new_state_dict = {}
    for k, v in state_dict.items():
        if "single_transformer_blocks." in k:
            if ".qkv_proj." in k:
                new_k = k.replace(".qkv_proj.", ".attn.to_qkv.")
            elif ".out_proj." in k:
                new_k = k.replace(".out_proj.", ".attn.to_out.")
            elif ".norm_q." in k or ".norm_k." in k:
                new_k = k.replace(".norm_k.", ".attn.norm_k.")
                new_k = new_k.replace(".norm_q.", ".attn.norm_q.")
            else:
                new_k = k
            new_k = new_k.replace(".lora_down", ".proj_down")
            new_k = new_k.replace(".lora_up", ".proj_up")
            if ".smooth_orig" in k:
                new_k = new_k.replace(".smooth_orig", ".smooth_factor_orig")
            elif ".smooth" in k:
                new_k = new_k.replace(".smooth", ".smooth_factor")
            new_state_dict[new_k] = v
        elif "transformer_blocks." in k:
            if ".mlp_context_fc1" in k:
                new_k = k.replace(".mlp_context_fc1.", ".ff_context.net.0.proj.")
            elif ".mlp_context_fc2" in k:
                new_k = k.replace(".mlp_context_fc2.", ".ff_context.net.2.")
            elif ".mlp_fc1" in k:
                new_k = k.replace(".mlp_fc1.", ".ff.net.0.proj.")
            elif ".mlp_fc2" in k:
                new_k = k.replace(".mlp_fc2.", ".ff.net.2.")
            elif ".qkv_proj_context." in k:
                new_k = k.replace(".qkv_proj_context.", ".attn.add_qkv_proj.")
            elif ".qkv_proj." in k:
                new_k = k.replace(".qkv_proj.", ".attn.to_qkv.")
            elif ".norm_q." in k or ".norm_k." in k:
                new_k = k.replace(".norm_k.", ".attn.norm_k.")
                new_k = new_k.replace(".norm_q.", ".attn.norm_q.")
            elif ".norm_added_q." in k or ".norm_added_k." in k:
                new_k = k.replace(".norm_added_k.", ".attn.norm_added_k.")
                new_k = new_k.replace(".norm_added_q.", ".attn.norm_added_q.")
            elif ".out_proj." in k:
                new_k = k.replace(".out_proj.", ".attn.to_out.0.")
            elif ".out_proj_context." in k:
                new_k = k.replace(".out_proj_context.", ".attn.to_add_out.")
            else:
                new_k = k
            new_k = new_k.replace(".lora_down", ".proj_down")
            new_k = new_k.replace(".lora_up", ".proj_up")
            if ".smooth_orig" in k:
                new_k = new_k.replace(".smooth_orig", ".smooth_factor_orig")
            elif ".smooth" in k:
                new_k = new_k.replace(".smooth", ".smooth_factor")
            new_state_dict[new_k] = v
        else:
            new_state_dict[k] = v

    return new_state_dict
