import sys
import os
import logging
from typing import Any, Dict, Optional, Union

import diffusers
import torch
from diffusers import FluxTransformer2DModel
from diffusers.configuration_utils import register_to_config
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from huggingface_hub import utils
from packaging.version import Version
from safetensors.torch import load_file
from torch import nn

from ..._C import QuantizedOminiFluxModel
from ..._C import utils as cutils
from ...lora.flux.nunchaku_converter import fuse_vectors, to_nunchaku
from ...lora.flux.utils import is_nunchaku_format
from ...utils import get_precision, load_state_dict_in_safetensors
from .utils import NunchakuModelLoaderMixin, pad_tensor

SVD_RANK = 32

# Get log level from environment variable (default to INFO)
log_level = os.getenv("LOG_LEVEL", "INFO").upper()
print(f"Log level: {log_level}")
# Configure logging
logging.basicConfig(level=getattr(logging, log_level, logging.INFO), format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Initialize C++ logging explicitly
# from ..._C import _C
# Force C++ to use the same log level as Python
print(f"Setting C++ log level to {log_level}")
cutils.set_log_level(log_level.lower())  # spdlog uses lowercase level names

# Force stderr to flush immediately to ensure logs appear
sys.stderr.flush()

# Explicitly print with syscalls to ensure visibility
os.write(2, f"C++ logging initialized with level {log_level}\n".encode())

cutils.set_log_level(log_level)


class NunchakuOminiFluxTransformerBlocks(nn.Module):
    """A PyTorch nn.Module wrapper for the Nunchaku C++ OminiFluxModel.

    This class handles the interface between the Python-based Diffusers pipeline
    and the underlying C++ implementation of the OminiFlux transformer blocks.
    It prepares inputs, calls the C++ forward pass, and processes outputs.
    """

    def __init__(self, m: QuantizedOminiFluxModel, device: str | torch.device):
        super(NunchakuOminiFluxTransformerBlocks, self).__init__()
        self.m = m  # The C++ QuantizedOminiFluxModel instance
        self.dtype = torch.bfloat16 if m.isBF16() else torch.float16
        self.device = device

    @staticmethod
    def pack_rotemb(rotemb: torch.Tensor) -> torch.Tensor:
        """Packs rotary embeddings into the format expected by the C++ kernel.

        The C++ kernel expects a specific memory layout for rotary embeddings
        to optimize performance. This function reshapes and permutes the input
        `rotemb` tensor to match that layout.

        Args:
            rotemb: The rotary embedding tensor, expected to be of shape
                    (B, M, D // 2, 1, 2) and dtype float32, where D is head dimension.

        Returns:
            The packed rotary embedding tensor of shape (B, M, D).
        """
        assert rotemb.dtype == torch.float32
        B = rotemb.shape[0]
        M = rotemb.shape[1]
        D = rotemb.shape[2] * 2
        logger.debug(f"pack_rotemb input shape: {rotemb.shape}")
        logger.debug(f"B={B}, M={M}, D={D}")
        assert rotemb.shape == (B, M, D // 2, 1, 2)
        assert M % 16 == 0, f"M ({M}) must be divisible by 16"
        assert D % 8 == 0, f"D ({D}) must be divisible by 8"

        # Log intermediate shapes
        rotemb = rotemb.reshape(B, M // 16, 16, D // 8, 8)
        logger.debug(f"After first reshape: {rotemb.shape}")

        rotemb = rotemb.permute(0, 1, 3, 2, 4)
        logger.debug(f"After permute: {rotemb.shape}")

        rotemb = rotemb.reshape(*rotemb.shape[0:3], 2, 8, 4, 2)
        logger.debug(f"After second reshape: {rotemb.shape}")

        rotemb = rotemb.permute(0, 1, 2, 4, 5, 3, 6)
        logger.debug(f"After second permute: {rotemb.shape}")

        rotemb = rotemb.contiguous()
        rotemb = rotemb.view(B, M, D)
        logger.debug(f"Final output shape: {rotemb.shape}")
        return rotemb

    def forward(
        self,
        hidden_states: torch.Tensor,
        cond_hidden_states: torch.Tensor,
        temb: torch.Tensor,
        cond_temb: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        image_rotary_emb: torch.Tensor,
        cond_rotary_emb: torch.Tensor,
        id_embeddings=None,
        id_weight=None,
        joint_attention_kwargs=None,
        controlnet_block_samples=None,
        controlnet_single_block_samples=None,
        skip_first_layer=False,
    ):
        """Performs the forward pass through the entire OminiFlux model.

        This method prepares all inputs (converting to appropriate dtypes and devices,
        packing rotary embeddings, handling ControlNet inputs) and then calls the
        forward method of the C++ `QuantizedOminiFluxModel`.

        Args:
            hidden_states: Image latents.
            cond_hidden_states: Conditional latents.
            temb: Time embeddings.
            cond_temb: Conditional time embeddings.
            encoder_hidden_states: Text encoder hidden states.
            image_rotary_emb: Rotary embeddings for image and text combined.
            cond_rotary_emb: Rotary embeddings for conditional inputs.
            id_embeddings: Optional ID embeddings.
            id_weight: Weight for ID embeddings.
            joint_attention_kwargs: Optional kwargs for joint attention.
            controlnet_block_samples: ControlNet features for joint blocks.
            controlnet_single_block_samples: ControlNet features for single blocks.
            skip_first_layer: If True, skips the first layer in the C++ model.

        Returns:
            A tuple containing (encoder_hidden_states, hidden_states) after processing,
            both converted back to the original dtype and device.
        """
        # batch_size = hidden_states.shape[0]
        txt_tokens = encoder_hidden_states.shape[1]
        img_tokens = hidden_states.shape[1]
        cond_tokens = cond_hidden_states.shape[1]

        self.id_embeddings = id_embeddings
        self.id_weight = id_weight
        self.pulid_ca_idx = 0
        if self.id_embeddings is not None:
            self.set_residual_callback()

        original_dtype = hidden_states.dtype
        original_device = hidden_states.device

        hidden_states = hidden_states.to(self.dtype).to(self.device)
        cond_hidden_states = cond_hidden_states.to(self.dtype).to(self.device)
        encoder_hidden_states = encoder_hidden_states.to(self.dtype).to(self.device)
        temb = temb.to(self.dtype).to(self.device)
        cond_temb = cond_temb.to(self.dtype).to(self.device)
        image_rotary_emb = image_rotary_emb.to(self.device)
        cond_rotary_emb = cond_rotary_emb.to(self.device)

        if controlnet_block_samples is not None:
            if isinstance(controlnet_block_samples, list) and len(controlnet_block_samples) > 0:
                controlnet_block_samples = torch.stack(controlnet_block_samples).to(self.device)
            elif isinstance(controlnet_block_samples, list):
                controlnet_block_samples = None

        if controlnet_single_block_samples is not None:
            if isinstance(controlnet_single_block_samples, list) and len(controlnet_single_block_samples) > 0:
                controlnet_single_block_samples = torch.stack(controlnet_single_block_samples).to(self.device)
            elif isinstance(controlnet_single_block_samples, list):
                controlnet_single_block_samples = None

        assert image_rotary_emb.ndim == 6
        assert image_rotary_emb.shape[0] == 1
        assert image_rotary_emb.shape[1] == 1
        assert image_rotary_emb.shape[2] == 1 * (txt_tokens + img_tokens)
        # [1, tokens, head_dim / 2, 1, 2] (sincos)
        image_rotary_emb = image_rotary_emb.reshape([1, txt_tokens + img_tokens, *image_rotary_emb.shape[3:]])
        rotary_emb_txt = image_rotary_emb[:, :txt_tokens, ...]  # .to(self.dtype)
        rotary_emb_img = image_rotary_emb[:, txt_tokens:, ...]  # .to(self.dtype)
        rotary_emb_single = image_rotary_emb  # .to(self.dtype)
        cond_rotary_emb = cond_rotary_emb.reshape([1, cond_tokens, *cond_rotary_emb.shape[3:]])

        rotary_emb_txt = self.pack_rotemb(pad_tensor(rotary_emb_txt, 256, 1))
        rotary_emb_img = self.pack_rotemb(pad_tensor(rotary_emb_img, 256, 1))
        rotary_emb_single = self.pack_rotemb(pad_tensor(rotary_emb_single, 256, 1))
        rotary_emb_cond = self.pack_rotemb(pad_tensor(cond_rotary_emb, 256, 1))

        hidden_states = self.m.forward(
            hidden_states,
            cond_hidden_states,
            encoder_hidden_states,
            temb,
            cond_temb,
            rotary_emb_img,
            rotary_emb_txt,
            rotary_emb_single,
            rotary_emb_cond,
            controlnet_block_samples,
            controlnet_single_block_samples,
            skip_first_layer,
        )

        if self.id_embeddings is not None:
            self.reset_residual_callback()

        hidden_states = hidden_states.to(original_dtype).to(original_device)
        encoder_hidden_states = hidden_states[:, :txt_tokens, ...]
        hidden_states = hidden_states[:, txt_tokens:, ...]
        # cond_hidden_states = cond_hidden_states[:, txt_tokens:txt_tokens+img_tokens, ...]

        return encoder_hidden_states, hidden_states

    def forward_layer_at(
        self,
        idx: int,
        hidden_states: torch.Tensor,
        cond_hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb: torch.Tensor,
        cond_temb: torch.Tensor,
        image_rotary_emb: torch.Tensor,
        cond_rotary_emb: torch.Tensor,
        joint_attention_kwargs=None,
        controlnet_block_samples=None,
        controlnet_single_block_samples=None,
    ):
        """
        Performs a forward pass through a single specified layer of the OminiFlux model.

        Similar to the main `forward` method, but only executes a single layer
        identified by `idx`. This is useful for debugging, analysis, or layer-wise
        interventions.

        Args:
            idx: The index of the transformer layer to execute.
            hidden_states: Image latents.
            cond_hidden_states: Conditional latents.
            encoder_hidden_states: Text encoder hidden states.
            temb: Time embeddings.
            cond_temb: Conditional time embeddings.
            image_rotary_emb: Rotary embeddings for image and text combined.
            cond_rotary_emb: Rotary embeddings for conditional inputs.
            joint_attention_kwargs: Optional kwargs for joint attention.
            controlnet_block_samples: ControlNet features for joint blocks.
            controlnet_single_block_samples: ControlNet features for single blocks.

        Returns:
            A tuple (hidden_states, encoder_hidden_states, cond_hidden_states)
            after processing by the specified layer, converted back to original dtype/device.
            Note the order matches the C++ backend's return order for `forward_layer`.
        """
        # batch_size = hidden_states.shape[0]
        txt_tokens = encoder_hidden_states.shape[1]
        img_tokens = hidden_states.shape[1]
        cond_tokens = cond_hidden_states.shape[1]

        original_dtype = hidden_states.dtype
        original_device = hidden_states.device

        hidden_states = hidden_states.to(self.dtype).to(self.device)
        encoder_hidden_states = encoder_hidden_states.to(self.dtype).to(self.device)
        cond_hidden_states = cond_hidden_states.to(self.dtype).to(self.device)
        temb = temb.to(self.dtype).to(self.device)
        cond_temb = cond_temb.to(self.dtype).to(self.device)
        image_rotary_emb = image_rotary_emb.to(self.device)
        cond_rotary_emb = cond_rotary_emb.to(self.device)

        if controlnet_block_samples is not None:
            if isinstance(controlnet_block_samples, list) and len(controlnet_block_samples) > 0:
                controlnet_block_samples = torch.stack(controlnet_block_samples).to(self.device)
            elif isinstance(controlnet_block_samples, list):
                controlnet_block_samples = None
        if controlnet_single_block_samples is not None:
            if isinstance(controlnet_single_block_samples, list) and len(controlnet_single_block_samples) > 0:
                controlnet_single_block_samples = torch.stack(controlnet_single_block_samples).to(self.device)
            elif isinstance(controlnet_single_block_samples, list):
                controlnet_single_block_samples = None

        assert image_rotary_emb.ndim == 6
        assert image_rotary_emb.shape[0] == 1
        assert image_rotary_emb.shape[1] == 1
        assert image_rotary_emb.shape[2] == 1 * (txt_tokens + img_tokens)
        # [1, tokens, head_dim / 2, 1, 2] (sincos)
        image_rotary_emb = image_rotary_emb.reshape([1, txt_tokens + img_tokens, *image_rotary_emb.shape[3:]])
        rotary_emb_txt = image_rotary_emb[:, :txt_tokens, ...]  # .to(self.dtype)
        rotary_emb_img = image_rotary_emb[:, txt_tokens:, ...]  # .to(self.dtype)

        cond_rotary_emb = cond_rotary_emb.reshape([1, cond_tokens, *cond_rotary_emb.shape[3:]])

        rotary_emb_txt = self.pack_rotemb(pad_tensor(rotary_emb_txt, 256, 1))
        rotary_emb_img = self.pack_rotemb(pad_tensor(rotary_emb_img, 256, 1))
        rotary_emb_cond = self.pack_rotemb(pad_tensor(cond_rotary_emb, 256, 1))

        hidden_states, encoder_hidden_states, cond_hidden_states = self.m.forward_layer(
            idx,
            hidden_states,
            cond_hidden_states,
            encoder_hidden_states,
            temb,
            cond_temb,
            rotary_emb_img,
            rotary_emb_txt,
            rotary_emb_cond,
            controlnet_block_samples,
            controlnet_single_block_samples,
        )

        hidden_states = hidden_states.to(original_dtype).to(original_device)
        cond_hidden_states = cond_hidden_states.to(original_dtype).to(original_device)
        encoder_hidden_states = encoder_hidden_states.to(original_dtype).to(original_device)
        return hidden_states, encoder_hidden_states, cond_hidden_states

    def set_residual_callback(self):
        """Sets a residual callback function for the C++ model.

        This allows Python code to inject or modify residuals at specific points
        within the C++ model's execution, typically used for techniques like PuLID.
        The callback receives a tensor from C++, processes it, and returns a tensor
        that C++ will add as a residual.
        `self.pulid_ca` and `self.pulid_ca_idx` are assumed to be set externally
        if this callback is used for PuLID-like functionality.
        """
        id_embeddings = self.id_embeddings
        pulid_ca = self.pulid_ca
        pulid_ca_idx = [self.pulid_ca_idx]
        id_weight = self.id_weight

        def callback(hidden_states):
            ip = id_weight * pulid_ca[pulid_ca_idx[0]](id_embeddings, hidden_states.to("cuda"))
            pulid_ca_idx[0] += 1
            return ip

        self.callback_holder = callback
        self.m.set_residual_callback(callback)

    def reset_residual_callback(self):
        """Clears any previously set residual callback in the C++ model."""
        self.callback_holder = None
        self.m.set_residual_callback(None)

    def __del__(self):
        """Ensures the C++ model resources are released when the Python object is deleted."""
        self.m.reset()

    def norm1(
        self,
        hidden_states: torch.Tensor,
        emb: torch.Tensor,
        idx: int = 0,  # Index of the transformer block
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Calls the `norm_one_forward` method of the C++ model for a specific block.

        This is typically used to get the output of the first AdaLayerNormZero layer
        within a OminiJointTransformerBlock. This can be useful for caching or other
        optimizations (e.g., TeaCache).

        Args:
            hidden_states: The input hidden states to the norm layer.
            emb: The conditioning embedding for the norm layer.
            idx: The index of the transformer block whose norm1 is to be called.

        Returns:
            A tuple containing: (norm_x, gate_msa, shift_mlp, scale_mlp, gate_mlp),
            which are the outputs of the OminiAdaLayerNormZero module.
        """
        return self.m.norm_one_forward(idx, hidden_states, emb)


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

    return out.float()


class EmbedND(nn.Module):
    def __init__(self, dim: int, theta: int, axes_dim: list[int]):
        super(EmbedND, self).__init__()
        self.dim = dim  # Dimensionality of the embedding for each axis
        self.theta = theta  # Base for the sinusoidal frequency calculation
        self.axes_dim = axes_dim  # List of embedding dimensions for each axis in `ids`

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        """Computes N-dimensional rotary positional embeddings.

        Args:
            ids: A tensor of shape (..., N_axes) containing integer coordinates for N axes.

        Returns:
            A tensor of shape (1, 1, sum(axes_dim), 1, 2) containing the
            concatenated rotary embeddings for all axes.
        """
        if Version(diffusers.__version__) >= Version("0.31.0"):
            ids = ids[None, ...]
        n_axes = ids.shape[-1]
        emb = torch.cat([rope(ids[..., i], self.axes_dim[i], self.theta) for i in range(n_axes)], dim=-3)
        return emb.unsqueeze(1)


def load_quantized_module(
    path: str, device: str | torch.device = "cuda", use_fp4: bool = False, offload: bool = False, bf16: bool = True
) -> QuantizedOminiFluxModel:
    """Loads a quantized OminiFluxModel from a saved Nunchaku checkpoint.

    Args:
        path: Path to the Nunchaku checkpoint file.
        device: The CUDA device to load the model onto.
        use_fp4: Whether the model was quantized with FP4 (as opposed to INT4).
        offload: Whether to enable layer offloading in the C++ model.
        bf16: Whether to use bfloat16 (True) or float16 (False) for the model's dtype.

    Returns:
        An instance of the C++ `QuantizedOminiFluxModel`.
    """
    device = torch.device(device)
    assert device.type == "cuda"
    m = QuantizedOminiFluxModel()
    cutils.disable_memory_auto_release()
    m.init(use_fp4, offload, bf16, 0 if device.index is None else device.index)
    m.load(path)
    return m


class NunchakuOminiFluxTransformer2dModel(FluxTransformer2DModel, NunchakuModelLoaderMixin):
    @register_to_config
    def __init__(
        self,
        patch_size: int = 1,  # Patch size for image tokenization (relevant for original DiT, less so here)
        in_channels: int = 64,  # Number of input channels to the first layer (x_embedder)
        out_channels: int | None = None,  # Number of output channels from the last layer (proj_out)
        num_layers: int = 19,  # Number of joint transformer blocks
        num_single_layers: int = 38,  # Number of single transformer blocks
        attention_head_dim: int = 128,  # Dimension of each attention head
        num_attention_heads: int = 24,  # Number of attention heads
        joint_attention_dim: int = 4096,  # Not directly used by OminiFlux, legacy from DiT
        pooled_projection_dim: int = 768,  # Dimension of pooled text projections
        guidance_embeds: bool = False,  # Whether the model uses explicit guidance embeddings
        axes_dims_rope: tuple[int] = (16, 56, 56),  # Dimensions for N-D RoPE (e.g., text, height, width)
    ):
        super(NunchakuOminiFluxTransformer2dModel, self).__init__(
            patch_size=patch_size,
            in_channels=in_channels,
            out_channels=out_channels,
            num_layers=num_layers,
            num_single_layers=num_single_layers,
            attention_head_dim=attention_head_dim,
            num_attention_heads=num_attention_heads,
            joint_attention_dim=joint_attention_dim,
            pooled_projection_dim=pooled_projection_dim,
            guidance_embeds=guidance_embeds,
            axes_dims_rope=axes_dims_rope,
        )
        # these state_dicts are used for supporting lora
        self._unquantized_part_sd: dict[str, torch.Tensor] = {}
        self._unquantized_part_loras: dict[str, torch.Tensor] = {}
        self._quantized_part_sd: dict[str, torch.Tensor] = {}
        self._quantized_part_vectors: dict[str, torch.Tensor] = {}
        self._original_in_channels = in_channels

        # Comfyui LoRA related
        self.comfy_lora_meta_list = []
        self.comfy_lora_sd_list = []

    @classmethod
    @utils.validate_hf_hub_args
    def from_pretrained(cls, pretrained_model_name_or_path: str | os.PathLike, **kwargs):
        device = kwargs.get("device", "cuda")
        if isinstance(device, str):
            device = torch.device(device)
        offload = kwargs.get("offload", False)
        torch_dtype = kwargs.get("torch_dtype", torch.bfloat16)
        precision = get_precision(kwargs.get("precision", "auto"), device, pretrained_model_name_or_path)
        transformer, unquantized_part_path, transformer_block_path = cls._build_model(
            pretrained_model_name_or_path, **kwargs
        )

        # get the default LoRA branch and all the vectors
        quantized_part_sd = load_file(transformer_block_path)
        new_quantized_part_sd = {}
        for k, v in quantized_part_sd.items():
            if v.ndim == 1:
                new_quantized_part_sd[k] = v
            elif "qweight" in k:
                # only the shape information of this tensor is needed
                new_quantized_part_sd[k] = v.to("meta")
            elif "lora" in k:
                new_quantized_part_sd[k] = v
        transformer._quantized_part_sd = new_quantized_part_sd
        m = load_quantized_module(
            transformer_block_path,
            device=device,
            use_fp4=precision == "fp4",
            offload=offload,
            bf16=torch_dtype == torch.bfloat16,
        )
        transformer.inject_quantized_module(m, device)
        transformer.to_empty(device=device)

        unquantized_part_sd = load_file(unquantized_part_path)
        transformer.load_state_dict(unquantized_part_sd, strict=False)
        transformer._unquantized_part_sd = unquantized_part_sd

        return transformer

    def inject_quantized_module(self, m: QuantizedOminiFluxModel, device: str | torch.device = "cuda"):
        """
        Injects the C++ `QuantizedOminiFluxModel` into this PyTorch module.

        This method sets up the necessary connections and replaces parts of the
        standard Diffusers FluxTransformer2DModel with the Nunchaku-accelerated
        components.

        Args:
            m: The initialized C++ `QuantizedOminiFluxModel` instance.
            device: The device the model is on.
        """
        print("Injecting quantized module")
        self.pos_embed = EmbedND(dim=self.inner_dim, theta=10000, axes_dim=[16, 56, 56])

        ### Compatible with the original forward method
        # The entire C++ model is wrapped in a single NunchakuOminiFluxTransformerBlocks module.
        # The original `transformer_blocks` and `single_transformer_blocks` lists from Diffusers
        # are effectively replaced by this single wrapper.
        self.transformer_blocks = nn.ModuleList([NunchakuOminiFluxTransformerBlocks(m, device)])
        self.single_transformer_blocks = nn.ModuleList([])

        return self

    def set_attention_impl(self, impl: str):
        """Sets the attention implementation in the C++ model.

        Args:
            impl: A string identifying the attention implementation, e.g.,
                  "flashattn2" or "nunchaku-fp16".
        """
        block = self.transformer_blocks[0]
        assert isinstance(block, NunchakuOminiFluxTransformerBlocks)
        block.m.setAttentionImpl(impl)

    ### LoRA Related Functions

    def _expand_module(self, module_name: str, new_shape: tuple[int, int]):
        """Expands a linear module (nn.Linear) to a new shape if LoRA weights require it.

        If a LoRA layer has dimensions larger than the original weight matrix,
        this function creates a new `nn.Linear` module with the expanded shape,
        copies the original weights, and updates the stored state dict.

        Args:
            module_name: The fully qualified name of the nn.Linear module to expand.
            new_shape: The target shape (out_features, in_features) for the module.
        """
        module = self.get_submodule(module_name)
        assert isinstance(module, nn.Linear)
        weight_shape = module.weight.shape
        logger.info("Expand the shape of module {} from {} to {}".format(module_name, tuple(weight_shape), new_shape))
        assert new_shape[0] >= weight_shape[0] and new_shape[1] >= weight_shape[1]
        new_module = nn.Linear(
            new_shape[1],
            new_shape[0],
            bias=module.bias is not None,
            device=module.weight.device,
            dtype=module.weight.dtype,
        )
        new_module.weight.data.zero_()
        new_module.weight.data[: weight_shape[0], : weight_shape[1]] = module.weight.data
        self._unquantized_part_sd[f"{module_name}.weight"] = new_module.weight.data.clone()
        if new_module.bias is not None:
            new_module.bias.data.zero_()
            new_module.bias.data[: weight_shape[0]] = module.bias.data
            self._unquantized_part_sd[f"{module_name}.bias"] = new_module.bias.data.clone()
        parent_name = ".".join(module_name.split(".")[:-1])
        parent_module = self.get_submodule(parent_name)
        parent_module.add_module(module_name.split(".")[-1], new_module)

        if module_name == "x_embedder":
            new_value = int(new_module.weight.data.shape[1])
            old_value = getattr(self.config, "in_channels")
            if new_value != old_value:
                logger.info(f"Update in_channels from {old_value} to {new_value}")
                setattr(self.config, "in_channels", new_value)

    def _update_unquantized_part_lora_params(self, strength: float = 1):
        """Applies loaded LoRA parameters to the unquantized parts of the model.

        This method first checks if any unquantized linear layers need to be expanded
        to accommodate the LoRA dimensions. Then, it iterates through the original
        unquantized weights, adds the LoRA delta (scaled by `strength`), and updates
        the model's state dict.

        Args:
            strength: The scaling factor for the LoRA weights.
        """
        # check if we need to expand the linear layers
        device = next(self.parameters()).device
        for k, v in self._unquantized_part_loras.items():
            if "lora_A" in k:
                lora_a = v
                lora_b = self._unquantized_part_loras[k.replace(".lora_A.", ".lora_B.")]
                diff_shape = (lora_b.shape[0], lora_a.shape[1])
                weight_shape = self._unquantized_part_sd[k.replace(".lora_A.", ".")].shape
                if diff_shape[0] > weight_shape[0] or diff_shape[1] > weight_shape[1]:
                    module_name = ".".join(k.split(".")[:-2])
                    self._expand_module(module_name, diff_shape)
            elif v.ndim == 1:
                diff_shape = v.shape
                weight_shape = self._unquantized_part_sd[k].shape
                if diff_shape[0] > weight_shape[0]:
                    assert diff_shape[0] >= weight_shape[0]
                    module_name = ".".join(k.split(".")[:-1])
                    module = self.get_submodule(module_name)
                    weight_shape = module.weight.shape
                    diff_shape = (diff_shape[0], weight_shape[1])
                    self._expand_module(module_name, diff_shape)
        new_state_dict = {}
        for k in self._unquantized_part_sd.keys():
            v = self._unquantized_part_sd[k]
            v = v.to(device)
            self._unquantized_part_sd[k] = v

            if v.ndim == 1 and k in self._unquantized_part_loras:
                diff = strength * self._unquantized_part_loras[k]
                if diff.shape[0] < v.shape[0]:
                    diff = torch.cat(
                        [diff, torch.zeros(v.shape[0] - diff.shape[0], device=device, dtype=v.dtype)], dim=0
                    )
                new_state_dict[k] = v + diff
            elif v.ndim == 2 and k.replace(".weight", ".lora_B.weight") in self._unquantized_part_loras:
                lora_a = self._unquantized_part_loras[k.replace(".weight", ".lora_A.weight")]
                lora_b = self._unquantized_part_loras[k.replace(".weight", ".lora_B.weight")]

                if lora_a.shape[1] < v.shape[1]:
                    lora_a = torch.cat(
                        [
                            lora_a,
                            torch.zeros(lora_a.shape[0], v.shape[1] - lora_a.shape[1], device=device, dtype=v.dtype),
                        ],
                        dim=1,
                    )
                if lora_b.shape[0] < v.shape[0]:
                    lora_b = torch.cat(
                        [
                            lora_b,
                            torch.zeros(v.shape[0] - lora_b.shape[0], lora_b.shape[1], device=device, dtype=v.dtype),
                        ],
                        dim=0,
                    )

                diff = strength * (lora_b @ lora_a)
                new_state_dict[k] = v + diff
            else:
                new_state_dict[k] = v
        self.load_state_dict(new_state_dict, strict=True)

    def update_lora_params(self, path_or_state_dict: str | dict[str, torch.Tensor]):
        """Loads and applies LoRA parameters to both quantized and unquantized parts.

        This is the main entry point for applying LoRA to the model.
        It handles:
        1. Loading LoRA weights from a file or dictionary.
        2. Converting to Nunchaku format if necessary.
        3. Separating LoRA weights for unquantized and quantized parts.
        4. Applying unquantized LoRAs via `_update_unquantized_part_lora_params`.
        5. Fusing LoRA vectors with base quantized vectors for the C++ model.
        6. Loading the updated quantized weights/vectors into the C++ model.

        Args:
            path_or_state_dict: Path to a LoRA safetensors file or a state dictionary
                                containing LoRA weights.
        """
        if isinstance(path_or_state_dict, dict):
            state_dict = {
                k: v for k, v in path_or_state_dict.items()
            }  # copy a new one to avoid modifying the original one
        else:
            state_dict = load_state_dict_in_safetensors(path_or_state_dict)

        if not is_nunchaku_format(state_dict):
            state_dict = to_nunchaku(state_dict, base_sd=self._quantized_part_sd)

        unquantized_part_loras = {}
        for k, v in list(state_dict.items()):
            device = next(self.parameters()).device
            if "transformer_blocks" not in k:
                unquantized_part_loras[k] = state_dict.pop(k).to(device)

        if len(self._unquantized_part_loras) > 0 or len(unquantized_part_loras) > 0:
            self._unquantized_part_loras = unquantized_part_loras

            self._unquantized_part_sd = {k: v for k, v in self._unquantized_part_sd.items() if "pulid_ca" not in k}
            self._update_unquantized_part_lora_params(1)

        quantized_part_vectors = {}
        for k, v in list(state_dict.items()):
            if v.ndim == 1:
                quantized_part_vectors[k] = state_dict.pop(k)
        if len(self._quantized_part_vectors) > 0 or len(quantized_part_vectors) > 0:
            self._quantized_part_vectors = quantized_part_vectors
            updated_vectors = fuse_vectors(quantized_part_vectors, self._quantized_part_sd, 1)
            state_dict.update(updated_vectors)

        # Get the vectors from the quantized part

        block = self.transformer_blocks[0]
        assert isinstance(block, NunchakuOminiFluxTransformerBlocks)

        block.m.loadDict(state_dict, True)

    # This function can only be used with a single LoRA.
    # For multiple LoRAs, please fuse the lora scale into the weights.
    def set_lora_strength(self, strength: float = 1):
        """Sets the strength of the currently loaded LoRA layers.

        This method adjusts the scaling factor applied to the LoRA weights.
        It affects both the unquantized parts (by re-applying with new strength)
        and the quantized parts (by telling the C++ model the new scale or
        re-fusing vectors).

        Note: This function is primarily designed for a single active LoRA.
        For multiple LoRAs, it's generally better to fuse them with their respective
        strengths into the base weights.

        Args:
            strength: The new LoRA strength/scaling factor.
        """
        block = self.transformer_blocks[0]
        assert isinstance(block, NunchakuOminiFluxTransformerBlocks)
        block.m.setLoraScale(SVD_RANK, strength)
        if len(self._unquantized_part_loras) > 0:
            self._update_unquantized_part_lora_params(strength)
        if len(self._quantized_part_vectors) > 0:
            vector_dict = fuse_vectors(self._quantized_part_vectors, self._quantized_part_sd, strength)
            block.m.loadDict(vector_dict, True)

    def reset_x_embedder(self):
        """Resets the `x_embedder` (initial convolution) to its original `in_channels`.

        If LoRA loading expanded `in_channels` of `x_embedder`, this function
        reverts it to the original channel count defined during model initialization.
        This is important when removing or changing LoRAs that modified input channels.
        """
        # if change the model in channels, we need to update the x_embedder
        if self._original_in_channels != self.config.in_channels:
            assert self._original_in_channels < self.config.in_channels
            old_module = self.x_embedder
            new_module = nn.Linear(
                in_features=self._original_in_channels,
                out_features=old_module.out_features,
                bias=old_module.bias is not None,
                device=old_module.weight.device,
                dtype=old_module.weight.dtype,
            )
            new_module.weight.data.copy_(old_module.weight.data[: new_module.out_features, : new_module.in_features])
            self._unquantized_part_sd["x_embedder.weight"] = new_module.weight.data.clone()
            if new_module.bias is not None:
                new_module.bias.data.zero_()
                new_module.bias.data.copy_(old_module.bias.data[: new_module.out_features])
                self._unquantized_part_sd["x_embedder.bias"] = new_module.bias.data.clone()
            self.x_embedder = new_module
            setattr(self.config, "in_channels", self._original_in_channels)

    def reset_lora(self):
        """Resets all applied LoRA modifications.

        This involves:
        1. Removing LoRA effects from unquantized parameters by reapplying with zero strength (effectively).
        2. Reloading the original (non-LoRA) quantized weights/vectors into the C++ model.
        3. Resetting the `x_embedder` if its `in_channels` was changed by LoRA.
        """
        unquantized_part_loras = {}
        if len(self._unquantized_part_loras) > 0 or len(unquantized_part_loras) > 0:
            self._unquantized_part_loras = unquantized_part_loras
            self._update_unquantized_part_lora_params(1)
        state_dict = {k: v for k, v in self._quantized_part_sd.items() if "lora" in k}
        quantized_part_vectors = {}
        if len(self._quantized_part_vectors) > 0 or len(quantized_part_vectors) > 0:
            self._quantized_part_vectors = quantized_part_vectors
            updated_vectors = fuse_vectors(quantized_part_vectors, self._quantized_part_sd, 1)
            state_dict.update(updated_vectors)
        self.transformer_blocks[0].m.loadDict(state_dict, True)
        self.reset_x_embedder()

    def forward(
        self,
        hidden_states: torch.Tensor,
        cond_hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor = None,
        pooled_projections: torch.Tensor = None,
        timestep: torch.LongTensor = None,
        cond_ids: torch.Tensor = None,
        img_ids: torch.Tensor = None,
        txt_ids: torch.Tensor = None,
        guidance: torch.Tensor = None,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        controlnet_block_samples=None,
        controlnet_single_block_samples=None,
        return_dict: bool = True,
        controlnet_blocks_repeat: bool = False,
    ) -> Union[torch.FloatTensor, Transformer2DModelOutput]:
        """
        Main forward pass of the NunchakuOminiFluxTransformer2dModel.

        This method orchestrates the entire generation process for the transformer part.
        It takes various inputs (hidden states, text embeddings, time steps, etc.),
        prepares them, and passes them to the Nunchaku-accelerated transformer blocks.

        Args:
            hidden_states: The image latents, shape `(B, C, H, W)`.
            cond_hidden_states: The conditional latents, shape `(B, C, H_cond, W_cond)`.
            encoder_hidden_states: Text encoder outputs, shape `(B, S_txt, D_txt)`.
            pooled_projections: Pooled text embeddings, shape `(B, D_pool)`.
            timestep: Current denoising timestep.
            cond_ids: Positional IDs for conditional latents.
            img_ids: Positional IDs for image latents.
            txt_ids: Positional IDs for text latents.
            guidance: Guidance scale tensor (if `guidance_embeds` is True).
            joint_attention_kwargs: Kwargs for joint attention (e.g., IP-Adapter).
            controlnet_block_samples: ControlNet features for joint blocks.
            controlnet_single_block_samples: ControlNet features for single blocks.
            return_dict: Whether to return a `Transformer2DModelOutput` or a tuple.
            controlnet_blocks_repeat: Not used in this version.

        Returns:
            If `return_dict` is True, a `Transformer2DModelOutput` with the denoised sample.
            Otherwise, a tuple containing the denoised sample.
        """
        # Initial convolution to project input latents to inner dimension
        hidden_states = self.x_embedder(hidden_states)
        cond_hidden_states = self.x_embedder(cond_hidden_states)

        timestep = timestep.to(hidden_states.dtype) * 1000
        if guidance is not None:
            guidance = guidance.to(hidden_states.dtype) * 1000
        else:
            guidance = None

        temb = (
            self.time_text_embed(timestep, pooled_projections)
            if guidance is None
            else self.time_text_embed(timestep, guidance, pooled_projections)
        )
        cond_temb = (
            self.time_text_embed(torch.zeros_like(timestep), pooled_projections)
            if guidance is None
            else self.time_text_embed(torch.zeros_like(timestep), guidance, pooled_projections)
        )

        encoder_hidden_states = self.context_embedder(encoder_hidden_states)

        if txt_ids.ndim == 3:
            logger.warning(
                "Passing `txt_ids` 3d torch.Tensor is deprecated."
                "Please remove the batch dimension and pass it as a 2d torch Tensor"
            )
            txt_ids = txt_ids[0]
        if img_ids.ndim == 3:
            logger.warning(
                "Passing `img_ids` 3d torch.Tensor is deprecated."
                "Please remove the batch dimension and pass it as a 2d torch Tensor"
            )
            img_ids = img_ids[0]

        # print(txt_ids.shape, img_ids.shape)
        # print(cond_ids.shape)
        ids = torch.cat((txt_ids, img_ids), dim=0)
        image_rotary_emb = self.pos_embed(ids)
        cond_rotary_emb = self.pos_embed(cond_ids)

        # print(image_rotary_emb.shape, cond_rotary_emb.shape)

        if joint_attention_kwargs is not None and "ip_adapter_image_embeds" in joint_attention_kwargs:
            ip_adapter_image_embeds = joint_attention_kwargs.pop("ip_adapter_image_embeds")
            ip_hidden_states = self.encoder_hid_proj(ip_adapter_image_embeds)
            joint_attention_kwargs.update({"ip_hidden_states": ip_hidden_states})

        nunchaku_block = self.transformer_blocks[0]
        encoder_hidden_states, hidden_states = nunchaku_block(
            hidden_states=hidden_states,
            cond_hidden_states=cond_hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            temb=temb,
            cond_temb=cond_temb,
            image_rotary_emb=image_rotary_emb,
            cond_rotary_emb=cond_rotary_emb,
            joint_attention_kwargs=joint_attention_kwargs,
            controlnet_block_samples=controlnet_block_samples,
            controlnet_single_block_samples=controlnet_single_block_samples,
        )
        # hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)
        # hidden_states = hidden_states[:, encoder_hidden_states.shape[1] :, ...]
        hidden_states = self.norm_out(hidden_states, temb)

        output = self.proj_out(hidden_states)

        if not return_dict:
            return (output,)

        return Transformer2DModelOutput(sample=output)
