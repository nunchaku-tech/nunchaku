import math

import cv2
import insightface
import numpy as np
import torch
import torch.nn.functional as F
from diffusers import FluxTransformer2DModel
from facexlib.parsing import init_parsing_model
from facexlib.utils.face_restoration_helper import FaceRestoreHelper
from huggingface_hub import hf_hub_download
from insightface.app import FaceAnalysis
from safetensors import safe_open
from torch import nn
from torchvision.transforms import InterpolationMode
from torchvision.transforms.functional import normalize, resize
from torchvision.utils import make_grid

from nunchaku.caching.utils import FluxCachedTransformerBlocks, check_and_apply_cache
from nunchaku.models.pulid.encoders_transformer import IDFormer
from nunchaku.models.pulid.eva_clip import create_model_and_transforms
from nunchaku.models.pulid.eva_clip.constants import OPENAI_DATASET_MEAN, OPENAI_DATASET_STD
from nunchaku.models.transformers.utils import pad_tensor

num_transformer_blocks = 19  # FIXME
num_single_transformer_blocks = 38  # FIXME


class IPA_TransformerBlocks(FluxCachedTransformerBlocks):
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
        self.image_embeds = image_embeds


def resize_numpy_image_long(image, resize_long_edge=768):
    h, w = image.shape[:2]
    if max(h, w) <= resize_long_edge:
        return image
    k = resize_long_edge / max(h, w)
    h = int(h * k)
    w = int(w * k)
    image = cv2.resize(image, (w, h), interpolation=cv2.INTER_LANCZOS4)
    return image


# from basicsr
def img2tensor(imgs, bgr2rgb=True, float32=True):
    """Numpy array to tensor.

    Args:
        imgs (list[ndarray] | ndarray): Input images.
        bgr2rgb (bool): Whether to change bgr to rgb.
        float32 (bool): Whether to change to float32.

    Returns:
        list[tensor] | tensor: Tensor images. If returned results only have
            one element, just return tensor.
    """

    def _totensor(img, bgr2rgb, float32):
        if img.shape[2] == 3 and bgr2rgb:
            if img.dtype == "float64":
                img = img.astype("float32")
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = torch.from_numpy(img.transpose(2, 0, 1))
        if float32:
            img = img.float()
        return img

    if isinstance(imgs, list):
        return [_totensor(img, bgr2rgb, float32) for img in imgs]
    return _totensor(imgs, bgr2rgb, float32)


def tensor2img(tensor, rgb2bgr=True, out_type=np.uint8, min_max=(0, 1)):
    """Convert torch Tensors into image numpy arrays.

    After clamping to [min, max], values will be normalized to [0, 1].

    Args:
        tensor (Tensor or list[Tensor]): Accept shapes:
            1) 4D mini-batch Tensor of shape (B x 3/1 x H x W);
            2) 3D Tensor of shape (3/1 x H x W);
            3) 2D Tensor of shape (H x W).
            Tensor channel should be in RGB order.
        rgb2bgr (bool): Whether to change rgb to bgr.
        out_type (numpy type): output types. If ``np.uint8``, transform outputs
            to uint8 type with range [0, 255]; otherwise, float type with
            range [0, 1]. Default: ``np.uint8``.
        min_max (tuple[int]): min and max values for clamp.

    Returns:
        (Tensor or list): 3D ndarray of shape (H x W x C) OR 2D ndarray of
        shape (H x W). The channel order is BGR.
    """
    if not (torch.is_tensor(tensor) or (isinstance(tensor, list) and all(torch.is_tensor(t) for t in tensor))):
        raise TypeError(f"tensor or list of tensors expected, got {type(tensor)}")

    if torch.is_tensor(tensor):
        tensor = [tensor]
    result = []
    for _tensor in tensor:
        _tensor = _tensor.squeeze(0).float().detach().cpu().clamp_(*min_max)
        _tensor = (_tensor - min_max[0]) / (min_max[1] - min_max[0])

        n_dim = _tensor.dim()
        if n_dim == 4:
            img_np = make_grid(_tensor, nrow=int(math.sqrt(_tensor.size(0))), normalize=False).numpy()
            img_np = img_np.transpose(1, 2, 0)
            if rgb2bgr:
                img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        elif n_dim == 3:
            img_np = _tensor.numpy()
            img_np = img_np.transpose(1, 2, 0)
            if img_np.shape[2] == 1:  # gray image
                img_np = np.squeeze(img_np, axis=2)
            else:
                if rgb2bgr:
                    img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        elif n_dim == 2:
            img_np = _tensor.numpy()
        else:
            raise TypeError(f"Only support 4D, 3D or 2D tensor. But received with dimension: {n_dim}")
        if out_type == np.uint8:
            # Unlike MATLAB, numpy.unit8() WILL NOT round by default.
            img_np = (img_np * 255.0).round()
        img_np = img_np.astype(out_type)
        result.append(img_np)
    if len(result) == 1:
        result = result[0]
    return result


def to_gray(img):
    x = 0.299 * img[:, 0:1] + 0.587 * img[:, 1:2] + 0.114 * img[:, 2:3]
    x = x.repeat(1, 3, 1, 1)
    return x


def get_puild_embed(image, device, cal_uncond=False, weight_dtype=torch.bfloat16, onnx_provider="gpu"):
    face_helper = FaceRestoreHelper(
        upscale_factor=1,
        face_size=512,
        crop_ratio=(1, 1),
        det_model="retinaface_resnet50",
        save_ext="png",
        device=device,
    )
    face_helper.face_parse = None
    face_helper.face_parse = init_parsing_model(model_name="bisenet", device=device)

    providers = (
        ["CPUExecutionProvider"] if onnx_provider == "cpu" else ["CUDAExecutionProvider", "CPUExecutionProvider"]
    )

    app = FaceAnalysis(name="antelopev2", root=".", providers=providers)
    app.prepare(ctx_id=0, det_size=(640, 640))
    handler_ante = insightface.model_zoo.get_model("models/antelopev2/glintr100.onnx", providers=providers)
    handler_ante.prepare(ctx_id=0)

    from safetensors.torch import load_file

    ckpt_path = hf_hub_download("guozinan/PuLID", "pulid_flux_v0.9.0.safetensors", local_dir="models")

    full_sd = load_file(ckpt_path)

    encoder_sd = {k.split("pulid_encoder.")[1]: v for k, v in full_sd.items() if k.startswith("pulid_encoder.")}

    pulid_encoder = IDFormer().to(device, weight_dtype)
    pulid_encoder.load_state_dict(encoder_sd, strict=True)
    pulid_encoder.eval()

    model, _, _ = create_model_and_transforms("EVA02-CLIP-L-14-336", "eva_clip", force_custom_clip=True)
    model = model.visual
    clip_vision_model = model.to(device, dtype=weight_dtype)

    face_helper.clean_all()
    debug_img_list = []
    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    # get antelopev2 embedding
    face_info = app.get(image_bgr)
    if len(face_info) > 0:
        face_info = sorted(face_info, key=lambda x: (x["bbox"][2] - x["bbox"][0]) * (x["bbox"][3] - x["bbox"][1]))[
            -1
        ]  # only use the maximum face
        id_ante_embedding = face_info["embedding"]
        debug_img_list.append(
            image[
                int(face_info["bbox"][1]) : int(face_info["bbox"][3]),
                int(face_info["bbox"][0]) : int(face_info["bbox"][2]),
            ]
        )
    else:
        id_ante_embedding = None

    # using facexlib to detect and align face
    face_helper.read_image(image_bgr)
    face_helper.get_face_landmarks_5(only_center_face=True)
    face_helper.align_warp_face()
    if len(face_helper.cropped_faces) == 0:
        raise RuntimeError("facexlib align face fail")
    align_face = face_helper.cropped_faces[0]
    # incase insightface didn't detect face
    if id_ante_embedding is None:
        print("fail to detect face using insightface, extract embedding on align face")
        id_ante_embedding = handler_ante.get_feat(align_face)

    id_ante_embedding = torch.from_numpy(id_ante_embedding).to(device, weight_dtype)
    if id_ante_embedding.ndim == 1:
        id_ante_embedding = id_ante_embedding.unsqueeze(0)

    # parsing
    input = img2tensor(align_face, bgr2rgb=True).unsqueeze(0) / 255.0
    input = input.to(device)
    parsing_out = face_helper.face_parse(normalize(input, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]))[0]
    parsing_out = parsing_out.argmax(dim=1, keepdim=True)
    bg_label = [0, 16, 18, 7, 8, 9, 14, 15]
    bg = sum(parsing_out == i for i in bg_label).bool()
    white_image = torch.ones_like(input)
    # only keep the face features
    face_features_image = torch.where(bg, white_image, to_gray(input))
    debug_img_list.append(tensor2img(face_features_image, rgb2bgr=False))

    eva_transform_mean = getattr(clip_vision_model, "image_mean", OPENAI_DATASET_MEAN)
    eva_transform_std = getattr(clip_vision_model, "image_std", OPENAI_DATASET_STD)
    if not isinstance(eva_transform_mean, (list, tuple)):
        eva_transform_mean = (eva_transform_mean,) * 3
    if not isinstance(eva_transform_std, (list, tuple)):
        eva_transform_std = (eva_transform_std,) * 3
    eva_transform_mean = eva_transform_mean
    eva_transform_std = eva_transform_std

    # transform img before sending to eva-clip-vit
    face_features_image = resize(face_features_image, clip_vision_model.image_size, InterpolationMode.BICUBIC)
    face_features_image = normalize(face_features_image, eva_transform_mean, eva_transform_std)
    id_cond_vit, id_vit_hidden = clip_vision_model(
        face_features_image.to(weight_dtype), return_all_features=False, return_hidden=True, shuffle=False
    )
    id_cond_vit_norm = torch.norm(id_cond_vit, 2, 1, True)
    id_cond_vit = torch.div(id_cond_vit, id_cond_vit_norm)

    id_cond = torch.cat([id_ante_embedding, id_cond_vit], dim=-1)

    id_embedding = pulid_encoder(id_cond, id_vit_hidden)

    if not cal_uncond:
        return id_embedding, None

    id_uncond = torch.zeros_like(id_cond)
    id_vit_hidden_uncond = []
    for layer_idx in range(0, len(id_vit_hidden)):
        id_vit_hidden_uncond.append(torch.zeros_like(id_vit_hidden[layer_idx]))
    uncond_id_embedding = pulid_encoder(id_uncond, id_vit_hidden_uncond)

    return id_embedding, uncond_id_embedding


def undo_all_mods_on_transformer(transformer: FluxTransformer2DModel):
    if hasattr(transformer, "_original_forward"):
        transformer.forward = transformer._original_forward
        del transformer._original_forward
    if hasattr(transformer, "_original_blocks"):
        transformer.transformer_blocks = transformer._original_blocks
        del transformer._original_blocks
    return transformer
