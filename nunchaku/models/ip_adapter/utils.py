"""
IP-Adapter utility functions and classes for FluxTransformer2DModel.

This module provides the core implementation for integrating IP-Adapter
conditioning into Flux-based transformer models, including block modification,
weight loading, and image embedding support.
"""

import cv2
import torch
import torch.nn.functional as F
from diffusers import FluxTransformer2DModel
from huggingface_hub import hf_hub_download
from safetensors import safe_open
from torch import nn

from nunchaku.caching.utils import FluxCachedTransformerBlocks, check_and_apply_cache
from nunchaku.models.transformers.utils import pad_tensor

num_transformer_blocks = 19  # FIXME
num_single_transformer_blocks = 38  # FIXME


class IPA_TransformerBlocks(FluxCachedTransformerBlocks):
    """
    Transformer block wrapper for IP-Adapter integration.

    This class extends FluxCachedTransformerBlocks to enable per-layer
    IP-Adapter conditioning, efficient caching, and flexible output control.

    Parameters
    ----------
    transformer : nn.Module, optional
        The base transformer module to wrap.
    ip_adapter_scale : float, default=1.0
        Scaling factor for the IP-Adapter output.
    return_hidden_states_first : bool, default=True
        If True, return hidden states before encoder states.
    return_hidden_states_only : bool, default=False
        If True, return only hidden states.
    verbose : bool, default=False
        If True, print verbose debug information.
    device : str or torch.device
        Device to use for computation.

    Attributes
    ----------
    ip_adapter_scale : float
        Scaling factor for IP-Adapter output.
    image_embeds : torch.Tensor or None
        Image embeddings for IP-Adapter conditioning.
    """

    def __init__(
        self,
        *,
        transformer: nn.Module = None,
        ip_adapter_scale: float = 1.0,
        return_hidden_states_first: bool = True,
        return_hidden_states_only: bool = False,
        verbose: bool = False,
        device: str | torch.device,
    ):
        super().__init__(
            transformer=transformer,
            use_double_fb_cache=False,
            residual_diff_threshold_multi=-1,
            residual_diff_threshold_single=-1,
            return_hidden_states_first=return_hidden_states_first,
            return_hidden_states_only=return_hidden_states_only,
            verbose=verbose,
        )
        self.ip_adapter_scale = ip_adapter_scale
        self.image_embeds = None

    def forward(
        self,
        hidden_states: torch.Tensor,
        temb: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        image_rotary_emb: torch.Tensor,
        id_embeddings=None,
        id_weight=None,
        joint_attention_kwargs=None,
        controlnet_block_samples=None,
        controlnet_single_block_samples=None,
        skip_first_layer=False,
    ):
        """
        Forward pass with IP-Adapter conditioning.

        Parameters
        ----------
        hidden_states : torch.Tensor
            Input hidden states.
        temb : torch.Tensor
            Temporal embedding tensor.
        encoder_hidden_states : torch.Tensor
            Encoder hidden states.
        image_rotary_emb : torch.Tensor
            Rotary embedding for image tokens.
        id_embeddings : optional
            Not used.
        id_weight : optional
            Not used.
        joint_attention_kwargs : dict, optional
            Additional attention arguments, may include 'ip_hidden_states'.
        controlnet_block_samples : list, optional
            ControlNet block samples for multi-blocks.
        controlnet_single_block_samples : list, optional
            ControlNet block samples for single blocks.
        skip_first_layer : bool, default=False
            If True, skip the first transformer block.

        Returns
        -------
        tuple or torch.Tensor
            Final hidden states and encoder states, or only hidden states if
            `return_hidden_states_only` is True.
        """
        batch_size = hidden_states.shape[0]
        txt_tokens = encoder_hidden_states.shape[1]
        img_tokens = hidden_states.shape[1]

        original_dtype = hidden_states.dtype
        original_device = hidden_states.device

        hidden_states = hidden_states.to(self.dtype).to(original_device)
        encoder_hidden_states = encoder_hidden_states.to(self.dtype).to(original_device)
        temb = temb.to(self.dtype).to(original_device)
        image_rotary_emb = image_rotary_emb.to(original_device)

        if controlnet_block_samples is not None:
            controlnet_block_samples = (
                torch.stack(controlnet_block_samples).to(original_device) if len(controlnet_block_samples) > 0 else None
            )
        if controlnet_single_block_samples is not None:
            controlnet_single_block_samples = (
                torch.stack(controlnet_single_block_samples).to(original_device)
                if len(controlnet_single_block_samples) > 0
                else None
            )

        assert image_rotary_emb.ndim == 6
        assert image_rotary_emb.shape[0] == 1
        assert image_rotary_emb.shape[1] == 1
        # [1, tokens, head_dim/2, 1, 2] (sincos)
        total_tokens = txt_tokens + img_tokens
        assert image_rotary_emb.shape[2] == 1 * total_tokens

        image_rotary_emb = image_rotary_emb.reshape([1, txt_tokens + img_tokens, *image_rotary_emb.shape[3:]])
        rotary_emb_txt = image_rotary_emb[:, :txt_tokens, ...]
        rotary_emb_img = image_rotary_emb[:, txt_tokens:, ...]
        rotary_emb_single = image_rotary_emb

        rotary_emb_txt = self.pack_rotemb(pad_tensor(rotary_emb_txt, 256, 1))
        rotary_emb_img = self.pack_rotemb(pad_tensor(rotary_emb_img, 256, 1))
        rotary_emb_single = self.pack_rotemb(pad_tensor(rotary_emb_single, 256, 1))

        if joint_attention_kwargs is not None and "ip_hidden_states" in joint_attention_kwargs:
            ip_hidden_states = joint_attention_kwargs.pop("ip_hidden_states")
        elif self.image_embeds is not None:
            ip_hidden_states = self.image_embeds

        remaining_kwargs = {
            "temb": temb,
            "rotary_emb_img": rotary_emb_img,
            "rotary_emb_txt": rotary_emb_txt,
            "rotary_emb_single": rotary_emb_single,
            "controlnet_block_samples": controlnet_block_samples,
            "controlnet_single_block_samples": controlnet_single_block_samples,
            "txt_tokens": txt_tokens,
            "ip_hidden_states": ip_hidden_states if ip_hidden_states is not None else None,
        }

        torch._dynamo.graph_break()

        if (self.residual_diff_threshold_multi <= 0.0) or (batch_size > 1):
            updated_h, updated_enc, _, _ = self.call_IPA_multi_transformer_blocks(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                skip_block=False,
                **remaining_kwargs,
            )

            remaining_kwargs.pop("ip_hidden_states", None)
            cat_hidden_states = torch.cat([updated_enc, updated_h], dim=1)

            updated_cat, _ = self.call_remaining_single_transformer_blocks(
                hidden_states=cat_hidden_states, encoder_hidden_states=None, start_idx=0, **remaining_kwargs
            )
            # torch._dynamo.graph_break()

            final_enc = updated_cat[:, :txt_tokens, ...]
            final_h = updated_cat[:, txt_tokens:, ...]

            final_h = final_h.to(original_dtype).to(original_device)
            final_enc = final_enc.to(original_dtype).to(original_device)

            if self.return_hidden_states_only:
                return final_h
            if self.return_hidden_states_first:
                return final_h, final_enc
            return final_enc, final_h

        original_hidden_states = hidden_states
        first_hidden_states, first_encoder_hidden_states, _, _ = self.call_IPA_multi_transformer_blocks(
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            first_block=True,
            skip_block=False,
            **remaining_kwargs,
        )

        hidden_states = first_hidden_states
        encoder_hidden_states = first_encoder_hidden_states
        first_hidden_states_residual_multi = hidden_states - original_hidden_states
        del original_hidden_states

        call_remaining_fn = self.call_IPA_multi_transformer_blocks

        torch._dynamo.graph_break()
        updated_h, updated_enc, threshold = check_and_apply_cache(
            first_residual=first_hidden_states_residual_multi,
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            threshold=self.residual_diff_threshold_multi,
            parallelized=False,
            mode="multi",
            verbose=self.verbose,
            call_remaining_fn=call_remaining_fn,
            remaining_kwargs=remaining_kwargs,
        )
        self.residual_diff_threshold_multi = threshold

        # Single layer
        remaining_kwargs.pop("ip_hidden_states", None)

        cat_hidden_states = torch.cat([updated_enc, updated_h], dim=1)
        original_cat = cat_hidden_states
        if not self.use_double_fb_cache:
            ##NO FBCache
            updated_cat, _ = self.call_remaining_single_transformer_blocks(
                hidden_states=cat_hidden_states, encoder_hidden_states=None, start_idx=0, **remaining_kwargs
            )
        else:
            # USE FBCache
            cat_hidden_states = self.m.forward_single_layer(0, cat_hidden_states, temb, rotary_emb_single)

            first_hidden_states_residual_single = cat_hidden_states - original_cat
            del original_cat

            call_remaining_fn_single = self.call_remaining_single_transformer_blocks

            updated_cat, _, threshold = check_and_apply_cache(
                first_residual=first_hidden_states_residual_single,
                hidden_states=cat_hidden_states,
                encoder_hidden_states=None,
                threshold=self.residual_diff_threshold_single,
                parallelized=False,
                mode="single",
                verbose=self.verbose,
                call_remaining_fn=call_remaining_fn_single,
                remaining_kwargs=remaining_kwargs,
            )
            self.residual_diff_threshold_single = threshold

        # torch._dynamo.graph_break()

        final_enc = updated_cat[:, :txt_tokens, ...]
        final_h = updated_cat[:, txt_tokens:, ...]

        final_h = final_h.to(original_dtype).to(original_device)
        final_enc = final_enc.to(original_dtype).to(original_device)

        if self.return_hidden_states_only:
            return final_h
        if self.return_hidden_states_first:
            return final_h, final_enc
        return final_enc, final_h

    def call_IPA_multi_transformer_blocks(
        self,
        hidden_states: torch.Tensor,
        temb: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        rotary_emb_img: torch.Tensor,
        rotary_emb_txt: torch.Tensor,
        rotary_emb_single: torch.Tensor,
        controlnet_block_samples=None,
        controlnet_single_block_samples=None,
        skip_first_layer=False,
        txt_tokens=None,
        ip_hidden_states=None,
        first_block: bool = False,
        skip_block: bool = True,
    ):
        """
        Apply IP-Adapter conditioning to multiple transformer blocks.

        Parameters
        ----------
        hidden_states : torch.Tensor
            Input hidden states.
        temb : torch.Tensor
            Temporal embedding tensor.
        encoder_hidden_states : torch.Tensor
            Encoder hidden states.
        rotary_emb_img : torch.Tensor
            Rotary embedding for image tokens.
        rotary_emb_txt : torch.Tensor
            Rotary embedding for text tokens.
        rotary_emb_single : torch.Tensor
            Rotary embedding for single block.
        controlnet_block_samples : list, optional
            ControlNet block samples for multi-blocks.
        controlnet_single_block_samples : list, optional
            ControlNet block samples for single blocks.
        skip_first_layer : bool, default=False
            If True, skip the first transformer block.
        txt_tokens : int, optional
            Number of text tokens.
        ip_hidden_states : torch.Tensor, optional
            Image prompt hidden states.
        first_block : bool, default=False
            If True, only process the first block.
        skip_block : bool, default=True
            If True, skip the first block.

        Returns
        -------
        tuple
            (hidden_states, encoder_hidden_states, hidden_states_residual, encoder_hidden_states_residual)
        """
        if first_block and skip_block:
            raise ValueError("`first_block` and `skip_block` cannot both be True.")

        start_idx = 1 if skip_block else 0
        end_idx = 1 if first_block else num_transformer_blocks

        original_hidden_states = hidden_states.clone()
        original_encoder_hidden_states = encoder_hidden_states.clone()
        ip_hidden_states[0] = ip_hidden_states[0].to(self.dtype).to(self.device)

        for idx in range(start_idx, end_idx):
            k_img = self.ip_k_projs[idx](ip_hidden_states[0])
            v_img = self.ip_v_projs[idx](ip_hidden_states[0])

            hidden_states, encoder_hidden_states, ip_query = self.m.forward_layer_ip_adapter(
                idx,
                hidden_states,
                encoder_hidden_states,
                temb,
                rotary_emb_img,
                rotary_emb_txt,
                controlnet_block_samples,
                controlnet_single_block_samples,
            )

            ip_query = ip_query.contiguous().to(self.dtype)
            ip_query = ip_query.view(1, -1, 24, 128).transpose(1, 2)

            k_img = k_img.view(1, -1, 24, 128).transpose(1, 2)
            v_img = v_img.view(1, -1, 24, 128).transpose(1, 2)

            real_ip_attn_output = F.scaled_dot_product_attention(
                ip_query, k_img, v_img, attn_mask=None, dropout_p=0.0, is_causal=False
            )
            real_ip_attn_output = real_ip_attn_output.transpose(1, 2).reshape(1, -1, 24 * 128)

            hidden_states = hidden_states + self.ip_adapter_scale * real_ip_attn_output

        hidden_states = hidden_states.contiguous()
        encoder_hidden_states = encoder_hidden_states.contiguous()
        hs_res = hidden_states - original_hidden_states
        enc_res = encoder_hidden_states - original_encoder_hidden_states

        return hidden_states, encoder_hidden_states, hs_res, enc_res

    def load_ip_adapter_weights_per_layer(
        self,
        repo_id: str,
        filename: str = "ip_adapter.safetensors",
        prefix: str = "double_blocks.",
        joint_attention_dim: int = 4096,
        inner_dim: int = 3072,
    ):
        """
        Load per-layer IP-Adapter weights from a HuggingFace Hub repository.

        Parameters
        ----------
        repo_id : str
            HuggingFace Hub repository ID.
        filename : str, default="ip_adapter.safetensors"
            Name of the safetensors file.
        prefix : str, default="double_blocks."
            Prefix for block keys in the file.
        joint_attention_dim : int, default=4096
            Input dimension for joint attention.
        inner_dim : int, default=3072
            Output dimension for projections.

        Returns
        -------
        None
        """
        path = hf_hub_download(repo_id=repo_id, filename=filename)
        raw_cpu = {}
        with safe_open(path, framework="pt", device="cpu") as f:
            for key in f.keys():
                if key.startswith(prefix):
                    raw_cpu[key] = f.get_tensor(key)

        raw = {k: v.to(self.device) for k, v in raw_cpu.items()}
        layer_ids = sorted({int(k.split(".")[1]) for k in raw.keys()})
        layers = []
        for i in layer_ids:
            base = f"double_blocks.{i}.processor.ip_adapter_double_stream"
            layers.append(
                {
                    "k_weight": raw[f"{base}_k_proj.weight"],
                    "k_bias": raw[f"{base}_k_proj.bias"],
                    "v_weight": raw[f"{base}_v_proj.weight"],
                    "v_bias": raw[f"{base}_v_proj.bias"],
                }
            )

        cross_dim = joint_attention_dim
        hidden_dim = inner_dim
        self.ip_k_projs = nn.ModuleList()
        self.ip_v_projs = nn.ModuleList()

        for layer in layers:
            k_proj = nn.Linear(cross_dim, hidden_dim, bias=True, device=self.device, dtype=self.dtype)
            v_proj = nn.Linear(cross_dim, hidden_dim, bias=True, device=self.device, dtype=self.dtype)

            k_proj.weight.data.copy_(layer["k_weight"])
            k_proj.bias.data.copy_(layer["k_bias"])
            v_proj.weight.data.copy_(layer["v_weight"])
            v_proj.bias.data.copy_(layer["v_bias"])

            self.ip_k_projs.append(k_proj)
            self.ip_v_projs.append(v_proj)

    def set_ip_hidden_states(self, image_embeds, negative_image_embeds=None):
        """
        Set the image embeddings for IP-Adapter conditioning.

        Parameters
        ----------
        image_embeds : torch.Tensor
            Image embeddings to use.
        negative_image_embeds : optional
            Not used.

        Returns
        -------
        None
        """
        self.image_embeds = image_embeds


def resize_numpy_image_long(image, resize_long_edge=768):
    """
    Resize a numpy image so its longest edge matches a target size.

    Parameters
    ----------
    image : np.ndarray
        Input image as a numpy array.
    resize_long_edge : int, default=768
        Target size for the longest edge.

    Returns
    -------
    np.ndarray
        Resized image.
    """
    h, w = image.shape[:2]
    if max(h, w) <= resize_long_edge:
        return image
    k = resize_long_edge / max(h, w)
    h = int(h * k)
    w = int(w * k)
    image = cv2.resize(image, (w, h), interpolation=cv2.INTER_LANCZOS4)
    return image


def undo_all_mods_on_transformer(transformer: FluxTransformer2DModel):
    """
    Restore a FluxTransformer2DModel to its original, unmodified state.

    This function undoes any modifications made for IP-Adapter integration,
    restoring the original forward method and transformer blocks.

    Parameters
    ----------
    transformer : FluxTransformer2DModel
        The transformer model to restore.

    Returns
    -------
    FluxTransformer2DModel
        The restored transformer model.
    """
    if hasattr(transformer, "_original_forward"):
        transformer.forward = transformer._original_forward
        del transformer._original_forward
    if hasattr(transformer, "_original_blocks"):
        transformer.transformer_blocks = transformer._original_blocks
        del transformer._original_blocks
    return transformer
