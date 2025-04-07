import os

import diffusers
import torch
from diffusers import FluxTransformer2DModel
from diffusers.configuration_utils import register_to_config
from huggingface_hub import utils
from packaging.version import Version
from torch import nn
import matplotlib.pyplot as plt

from nunchaku.utils import fetch_or_download
from .utils import NunchakuModelLoaderMixin, pad_tensor
from ..._C import QuantizedFluxModel, utils as cutils
from ...utils import load_state_dict_in_safetensors
from .DFB_cache import *

SVD_RANK = 32
num_transformer_blocks = 19
num_single_transformer_blocks = 38


class NunchakuFluxTransformerBlocks(nn.Module):
    def __init__(
        self,
        m,
        device: str | torch.device,
        residual_diff_threshold_multi: float = 0.06,
        residual_diff_threshold_single: float = 0.06,
        adaptive_th: float = 0.90,
        USE_FBCache : bool = False,
        return_hidden_states_first: bool = False,
        return_hidden_states_only: bool = False,
        VERBOSE_Threshold: bool = True, 
    ):
        super(NunchakuFluxTransformerBlocks, self).__init__()
        self.m = m
        self.dtype = torch.bfloat16
        self.device = device
        self.residual_diff_threshold_multi = residual_diff_threshold_multi
        self.residual_diff_threshold_single = residual_diff_threshold_single
        self.return_hidden_states_first = return_hidden_states_first
        self.return_hidden_states_only = return_hidden_states_only
        self.adaptive_th = adaptive_th
        self.USE_FBCache = USE_FBCache
        
        self.VERBOSE_Threshold = VERBOSE_Threshold
        self.threshold_multi_history = []
        self.residual_diff_threshold_multi_history = []
        self.threshold_single_history = []
        self.residual_diff_threshold_single_history = []

    
    def call_remaining_multi_transformer_blocks(
        self,
        start_idx: int,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb: torch.Tensor,
        rotary_emb_img: torch.Tensor,
        rotary_emb_txt: torch.Tensor,
    ):

        original_hidden_states = hidden_states.clone()
        original_encoder_hidden_states = encoder_hidden_states.clone()

        for idx in range(start_idx, num_transformer_blocks):
            updated_h, updated_enc = self.m.forward_layer(
                idx,
                hidden_states,
                encoder_hidden_states,
                temb,
                rotary_emb_img,
                rotary_emb_txt,
            )
            hidden_states = updated_h
            encoder_hidden_states = updated_enc

        hs_res = hidden_states - original_hidden_states
        enc_res = encoder_hidden_states - original_encoder_hidden_states
        return hidden_states, encoder_hidden_states, hs_res, enc_res

    def call_remaining_single_transformer_blocks(
        self,
        start_idx: int,
        cat_hidden_states: torch.Tensor,
        temb: torch.Tensor,
        rotary_emb_single: torch.Tensor,
    ):

        original_cat = cat_hidden_states.clone()

        for idx in range(start_idx, num_single_transformer_blocks):
            cat_hidden_states = self.m.forward_single_layer(
                idx,
                cat_hidden_states,
                temb,
                rotary_emb_single,
            )

        cat_res = cat_hidden_states - original_cat
        return cat_hidden_states, cat_res
    

    def forward(
        self,
        hidden_states: torch.Tensor,          # [batch, img_tokens, dim]
        temb: torch.Tensor,                   # [batch, dim]
        encoder_hidden_states: torch.Tensor,  # [batch, txt_tokens, dim]
        image_rotary_emb: torch.Tensor,       # shape [1, 1, batch*(txt+img), head_dim/2, 1, 2]
        joint_attention_kwargs=None,
    ):
        batch_size = hidden_states.shape[0]
        txt_tokens = encoder_hidden_states.shape[1]
        img_tokens = hidden_states.shape[1]

        original_dtype = hidden_states.dtype
        original_device = hidden_states.device

        # Move to appropriate dtype/device
        hidden_states = hidden_states.to(self.dtype).to(self.device)
        encoder_hidden_states = encoder_hidden_states.to(self.dtype).to(self.device)
        temb = temb.to(self.dtype).to(self.device)
        image_rotary_emb = image_rotary_emb.to(self.device)

        assert image_rotary_emb.ndim == 6
        assert image_rotary_emb.shape[0] == 1
        assert image_rotary_emb.shape[1] == 1
        #assert image_rotary_emb.shape[2] == batch_size * (txt_tokens + img_tokens)
        # [bs, tokens, head_dim / 2, 1, 2] (sincos)
        image_rotary_emb = image_rotary_emb.reshape([1, txt_tokens + img_tokens, *image_rotary_emb.shape[3:]])
        image_rotary_emb = image_rotary_emb.expand(batch_size, -1, -1, -1, -1)
        image_rotary_emb = image_rotary_emb.contiguous()
 
        
        rotary_emb_txt = image_rotary_emb[:, :txt_tokens, ...].contiguous()  # .to(self.dtype)
        rotary_emb_img = image_rotary_emb[:, txt_tokens:, ...].contiguous()  # .to(self.dtype)
        rotary_emb_single = image_rotary_emb.contiguous()  # .to(self.dtype)

        rotary_emb_txt = pad_tensor(rotary_emb_txt, 256, 1).contiguous()
        rotary_emb_img = pad_tensor(rotary_emb_img, 256, 1).contiguous()
        rotary_emb_single = pad_tensor(rotary_emb_single, 256, 1).contiguous()
        
        
        if  self.USE_FBCache==False:
            ####TODO
            hidden_states = self.m.forward(
            hidden_states, encoder_hidden_states, temb, rotary_emb_img, rotary_emb_txt, rotary_emb_single
            )

            hidden_states = hidden_states.to(original_dtype).to(original_device)

            encoder_hidden_states = hidden_states[:, :txt_tokens, ...]
            hidden_states = hidden_states[:, txt_tokens:, ...]

            return encoder_hidden_states, hidden_states
        
        
        context = get_current_cache_context()
    
        if (self.residual_diff_threshold_multi <= 0.0) and (self.residual_diff_threshold_single <= 0.0):
            for idx in range(num_transformer_blocks):
                updated_h, updated_enc = self.m.forward_layer(
                    idx,
                    hidden_states,
                    encoder_hidden_states,
                    temb,
                    rotary_emb_img,
                    rotary_emb_txt,
                )
                hidden_states = updated_h
                encoder_hidden_states = updated_enc

            cat_hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)
            for idx in range(num_single_transformer_blocks):
                cat_hidden_states = self.m.forward_single_layer(
                    idx,
                    cat_hidden_states,
                    temb,
                    rotary_emb_single,
                )
        

            final_encoder_hidden_states = cat_hidden_states[:, :txt_tokens, ...]
            final_hidden_states = cat_hidden_states[:, txt_tokens:, ...]

        else:
            original_hs = hidden_states
            first_updated_h, first_updated_enc = self.m.forward_layer(
                0,
                hidden_states,
                encoder_hidden_states,
                temb,
                rotary_emb_img,
                rotary_emb_txt,
            )
            hidden_states = first_updated_h
            encoder_hidden_states = first_updated_enc

            first_residual_multi = hidden_states - original_hs
            del original_hs

            parallelized = False
            can_use_cache_multi = False
            if self.residual_diff_threshold_multi > 0:
                can_use_cache_multi,threshold_multi = get_can_use_cache_multi(
                    first_residual_multi,
                    threshold=self.residual_diff_threshold_multi,
                    parallelized=parallelized,
                )
            if self.VERBOSE_Threshold:
                self.threshold_multi_history.append(threshold_multi)
                self.residual_diff_threshold_multi_history.append(self.residual_diff_threshold_multi)

            torch._dynamo.graph_break()

            if can_use_cache_multi:
                self.residual_diff_threshold_multi = self.residual_diff_threshold_multi*self.adaptive_th
                #del first_residual_multi
                hidden_states, encoder_hidden_states = apply_prev_hidden_states_residual_multi(
                    hidden_states, encoder_hidden_states
                )
            else:
                self.residual_diff_threshold_multi = threshold_multi
                #set_buffer("first_hidden_states_residual_multi", first_residual_multi)
                context.first_hidden_states_residual_multi = first_residual_multi#.clone()
                #del first_residual_multi

                hidden_states, encoder_hidden_states, hs_res_multi, enc_res_multi = self.call_remaining_multi_transformer_blocks(
                    start_idx=1,
                    hidden_states=hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    temb=temb,
                    rotary_emb_img=rotary_emb_img,
                    rotary_emb_txt=rotary_emb_txt,
                )

                #set_buffer("hidden_states_residual_multi", hs_res_multi)
                #set_buffer("encoder_hidden_states_residual_multi", enc_res_multi)
                context.hidden_states_residual_multi = hs_res_multi#.clone()
                context.encoder_hidden_states_residual_multi = enc_res_multi#.clone()

            torch._dynamo.graph_break()


            cat_hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)

            original_cat = cat_hidden_states
            
            
            first_single_cat = self.m.forward_single_layer(
                0,
                cat_hidden_states,
                temb,
                rotary_emb_single,
            )
            cat_hidden_states = first_single_cat

            first_cat_hidden_states_residual_single = cat_hidden_states - original_cat
            del original_cat

            can_use_cache_singles = False
            if self.residual_diff_threshold_single > 0:
                can_use_cache_singles, threshold_single = get_can_use_cache_single(
                    first_cat_hidden_states_residual_single,
                    threshold=self.residual_diff_threshold_single,
                    parallelized=parallelized,
                )

            torch._dynamo.graph_break()
            if self.VERBOSE_Threshold:
                self.threshold_single_history.append(threshold_single)
                self.residual_diff_threshold_single_history.append(self.residual_diff_threshold_single)

            if can_use_cache_singles:
                self.residual_diff_threshold_single = self.residual_diff_threshold_single * self.adaptive_th
                #del first_cat_hidden_states_residual_single
                cat_hidden_states = apply_prev_cat_hidden_states_residual_single(cat_hidden_states)
            else:
                self.residual_diff_threshold_single = threshold_single
                #set_buffer("first_cat_hidden_states_residual_single", first_cat_hidden_states_residual_single)
                context.first_cat_hidden_states_residual_single = first_cat_hidden_states_residual_single#.clone()
                
                #del first_cat_hidden_states_residual_single

                cat_hidden_states, cat_res_single = self.call_remaining_single_transformer_blocks(
                    start_idx=1,
                    cat_hidden_states=cat_hidden_states,
                    temb=temb,
                    rotary_emb_single=rotary_emb_single,
                )
                #set_buffer("cat_hidden_states_residual_single", cat_res_single)
                context.cat_hidden_states_residual_single = cat_res_single.clone()

            torch._dynamo.graph_break()

            final_encoder_hidden_states = cat_hidden_states[:, :txt_tokens, ...]
            final_hidden_states = cat_hidden_states[:, txt_tokens:, ...]

        final_encoder_hidden_states = final_encoder_hidden_states.to(original_dtype).to(original_device)
        final_hidden_states = final_hidden_states.to(original_dtype).to(original_device)


        if self.return_hidden_states_only:
            return final_hidden_states
        else:
            if self.return_hidden_states_first:
                return (final_hidden_states, final_encoder_hidden_states)
            else:
                return (final_encoder_hidden_states, final_hidden_states)

    def plot_thresholds(self, save_path: str = None):
            fig, axs = plt.subplots(1, 2, figsize=(12, 5))

            axs[0].plot(self.threshold_multi_history, label="threshold_multi")
            axs[0].plot(self.residual_diff_threshold_multi_history, label="residual_diff_threshold_multi")
            axs[0].set_title("Multi Threshold Tracking")
            axs[0].set_xlabel("Iteration")
            axs[0].set_ylabel("Threshold Value")
            axs[0].legend()

            axs[1].plot(self.threshold_single_history, label="threshold_single")
            axs[1].plot(self.residual_diff_threshold_single_history, label="residual_diff_threshold_single")
            axs[1].set_title("Single Threshold Tracking")
            axs[1].set_xlabel("Iteration")
            axs[1].set_ylabel("Threshold Value")
            axs[1].legend()

            plt.tight_layout()

            if save_path:
                safe_save_fig(fig, save_path)
                print(f" * Threshold save: {save_path}")
            else:
                plt.show()

            plt.close(fig)

    def reset_threshold_logs(self):
        self.threshold_multi_history.clear()
        self.residual_diff_threshold_multi_history.clear()
        self.threshold_single_history.clear()
        self.residual_diff_threshold_single_history.clear()


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

    return out.float()


class EmbedND(nn.Module):
    def __init__(self, dim: int, theta: int, axes_dim: list[int]):
        super(EmbedND, self).__init__()
        self.dim = dim
        self.theta = theta
        self.axes_dim = axes_dim

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        if Version(diffusers.__version__) >= Version("0.31.0"):
            ids = ids[None, ...]
        n_axes = ids.shape[-1]
        emb = torch.cat([rope(ids[..., i], self.axes_dim[i], self.theta) for i in range(n_axes)], dim=-3)
        return emb.unsqueeze(1)


def load_quantized_module(
    path: str, device: str | torch.device = "cuda", use_fp4: bool = False, offload: bool = False
) -> QuantizedFluxModel:
    device = torch.device(device)
    assert device.type == "cuda"
    m = QuantizedFluxModel()
    cutils.disable_memory_auto_release()
    m.init(use_fp4, offload, True, 0 if device.index is None else device.index)
    m.load(path)
    return m


class NunchakuFluxTransformer2dModel(FluxTransformer2DModel, NunchakuModelLoaderMixin):
    @register_to_config
    def __init__(
        self,
        patch_size: int = 1,
        in_channels: int = 64,
        out_channels: int | None = None,
        num_layers: int = 19,
        num_single_layers: int = 38,
        attention_head_dim: int = 128,
        num_attention_heads: int = 24,
        joint_attention_dim: int = 4096,
        pooled_projection_dim: int = 768,
        guidance_embeds: bool = False,
        axes_dims_rope: tuple[int] = (16, 56, 56),
    ):
        super(NunchakuFluxTransformer2dModel, self).__init__(
            patch_size=patch_size,
            in_channels=in_channels,
            out_channels=out_channels,
            num_layers=0,
            num_single_layers=0,
            attention_head_dim=attention_head_dim,
            num_attention_heads=num_attention_heads,
            joint_attention_dim=joint_attention_dim,
            pooled_projection_dim=pooled_projection_dim,
            guidance_embeds=guidance_embeds,
            axes_dims_rope=axes_dims_rope,
        )
        self.unquantized_loras = {}
        self.unquantized_state_dict = None
        self.batch_size = 1
        self.VERBOSE_Threshold = True
        self.adaptive_th = 0.99

    @classmethod
    @utils.validate_hf_hub_args
    def from_pretrained(cls, pretrained_model_name_or_path: str | os.PathLike, **kwargs):
        device = kwargs.get("device", "cuda")
        precision = kwargs.get("precision", "int4")
        offload = kwargs.get("offload", False)
        assert precision in ["int4", "fp4"]
        transformer, transformer_block_path = cls._build_model(pretrained_model_name_or_path, **kwargs)
        m = load_quantized_module(transformer_block_path, device=device, use_fp4=precision == "fp4", offload=offload)
        transformer.inject_quantized_module(m, device)
        return transformer,m

    def update_unquantized_lora_params(self, strength: float = 1):
        new_state_dict = {}
        for k in self.unquantized_state_dict.keys():
            v = self.unquantized_state_dict[k]
            if k.replace(".weight", ".lora_B.weight") in self.unquantized_loras:
                new_state_dict[k] = v + strength * (
                    self.unquantized_loras[k.replace(".weight", ".lora_B.weight")]
                    @ self.unquantized_loras[k.replace(".weight", ".lora_A.weight")]
                )
            else:
                new_state_dict[k] = v
        self.load_state_dict(new_state_dict, strict=True)

    def update_lora_params(self, path: str):
        state_dict = load_state_dict_in_safetensors(path)

        unquantized_loras = {}
        for k in state_dict.keys():
            if "transformer_blocks" not in k:
                unquantized_loras[k] = state_dict[k]

        self.unquantized_loras = unquantized_loras
        if len(unquantized_loras) > 0:
            if self.unquantized_state_dict is None:
                unquantized_state_dict = self.state_dict()
                self.unquantized_state_dict = {k: v.cpu() for k, v in unquantized_state_dict.items()}
            self.update_unquantized_lora_params(1)

        path = fetch_or_download(path)
        block = self.transformer_blocks[0]
        assert isinstance(block, NunchakuFluxTransformerBlocks)
        block.m.load(path, True)

    def set_lora_strength(self, strength: float = 1):
        block = self.transformer_blocks[0]
        assert isinstance(block, NunchakuFluxTransformerBlocks)
        block.m.setLoraScale(SVD_RANK, strength)
        if len(self.unquantized_loras) > 0:
            self.update_unquantized_lora_params(strength)

    def inject_quantized_module(self, m: QuantizedFluxModel, device: str | torch.device = "cuda"):
        self.pos_embed = EmbedND(dim=self.inner_dim, theta=10000, axes_dim=[16, 56, 56])

        ### Compatible with the original forward method
        self.transformer_blocks = nn.ModuleList([NunchakuFluxTransformerBlocks(m, device)])
        self.single_transformer_blocks = nn.ModuleList([])

        return self
    
    def set_residual_diff_threshold(self, threshold_multi: float,threshold_single: float, adaptive_th: float, USE_FBCahe : bool):
        for block in self.transformer_blocks:
            if isinstance(block, NunchakuFluxTransformerBlocks):
                block.residual_diff_threshold_multi = threshold_multi
                block.residual_diff_threshold_single = threshold_single
                block.adaptive_th = adaptive_th
                block.USE_FBCache = USE_FBCahe
    

    def get_residual_diff_threshold(self) -> float:
        for block in self.transformer_blocks:
            if isinstance(block, NunchakuFluxTransformerBlocks):
                return block.residual_diff_threshold

        return 0.0
    
def safe_save_fig(fig, save_path: str):

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    _, ext = os.path.splitext(save_path)
    
    fd, temp_path = tempfile.mkstemp(suffix=ext, dir=os.path.dirname(save_path))
    os.close(fd)  
    try:
        fig.savefig(temp_path)
        os.replace(temp_path, save_path) 
        print(f"Figure saved successfully at: {save_path}")
    except Exception as e:
        print(f"Error saving figure at {save_path}: {e}")
        if os.path.exists(temp_path):
            os.remove(temp_path)