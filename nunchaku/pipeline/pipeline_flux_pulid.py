"""
This module provides the PuLID FluxPipeline for personalized image generation with identity preservation.
It integrates face analysis, alignment, and embedding extraction using InsightFace and FaceXLib, and injects
identity embeddings into a Flux transformer pipeline.
"""

# Adapted from https://github.com/ToTheBeginning/PuLID/blob/main/pulid/pipeline.py
import gc
import logging
import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

import cv2
import insightface
import numpy as np
import torch
from diffusers import FluxPipeline
from diffusers.image_processor import PipelineImageInput
from diffusers.pipelines.flux.pipeline_flux import calculate_shift, retrieve_timesteps
from diffusers.pipelines.flux.pipeline_output import FluxPipelineOutput
from facexlib.parsing import init_parsing_model
from facexlib.utils.face_restoration_helper import FaceRestoreHelper
from huggingface_hub import snapshot_download
from huggingface_hub.constants import HUGGINGFACE_HUB_CACHE
from insightface.app import FaceAnalysis
from torch import nn
from torchvision.transforms import InterpolationMode
from torchvision.transforms.functional import normalize, resize

from ..models.pulid.encoders_transformer import IDFormer, PerceiverAttentionCA
from ..models.pulid.eva_clip import create_model_and_transforms
from ..models.pulid.eva_clip.constants import OPENAI_DATASET_MEAN, OPENAI_DATASET_STD
from ..models.pulid.utils import img2tensor, resize_numpy_image_long, tensor2img
from ..models.transformers import NunchakuFluxTransformer2dModel
from ..utils import load_state_dict_in_safetensors, sha256sum

# Get log level from environment variable (default to INFO)
log_level = os.getenv("LOG_LEVEL", "INFO").upper()

# Configure logging
logging.basicConfig(level=getattr(logging, log_level, logging.INFO), format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def check_antelopev2_dir(antelopev2_dirpath: str | os.PathLike[str]) -> bool:
    """
    Check if the given directory contains all required AntelopeV2 ONNX model files with correct SHA256 hashes.

    Parameters
    ----------
    antelopev2_dirpath : str or os.PathLike
        Path to the directory containing AntelopeV2 ONNX models.

    Returns
    -------
    bool
        True if all required files exist and have correct hashes, False otherwise.
    """
    antelopev2_dirpath = Path(antelopev2_dirpath)
    required_files = {
        "1k3d68.onnx": "df5c06b8a0c12e422b2ed8947b8869faa4105387f199c477af038aa01f9a45cc",
        "2d106det.onnx": "f001b856447c413801ef5c42091ed0cd516fcd21f2d6b79635b1e733a7109dbf",
        "genderage.onnx": "4fde69b1c810857b88c64a335084f1c3fe8f01246c9a191b48c7bb756d6652fb",
        "glintr100.onnx": "4ab1d6435d639628a6f3e5008dd4f929edf4c4124b1a7169e1048f9fef534cdf",
        "scrfd_10g_bnkps.onnx": "5838f7fe053675b1c7a08b633df49e7af5495cee0493c7dcf6697200b85b5b91",
    }

    if not antelopev2_dirpath.is_dir():
        logger.debug(f"Directory does not exist: {antelopev2_dirpath}")
        return False

    for filename, expected_hash in required_files.items():
        filepath = antelopev2_dirpath / filename
        if not filepath.exists():
            logger.debug(f"Missing file: {filename}")
            return False
        if expected_hash != "<SKIP_HASH>" and not sha256sum(filepath) == expected_hash:
            logger.debug(f"Hash mismatch for: {filename}")
            return False
    return True


class PuLIDPipeline(nn.Module):
    """
    PyTorch module for extracting identity embeddings using PuLID, InsightFace, and EVA-CLIP.

    This class handles face detection, alignment, parsing, and embedding extraction for use in personalized
    diffusion pipelines.

    Parameters
    ----------
    dit : NunchakuFluxTransformer2dModel
        The transformer model to inject PuLID attention modules into.
    device : str or torch.device
        Device to run the pipeline on.
    weight_dtype : str or torch.dtype, optional
        Data type for model weights (default: torch.bfloat16).
    onnx_provider : str, optional
        ONNX runtime provider, "gpu" or "cpu" (default: "gpu").
    pulid_path : str or os.PathLike, optional
        Path to PuLID weights in safetensors format.
    eva_clip_path : str or os.PathLike, optional
        Path to EVA-CLIP weights.
    insightface_dirpath : str or os.PathLike or None, optional
        Path to InsightFace models directory.
    facexlib_dirpath : str or os.PathLike or None, optional
        Path to FaceXLib models directory.

    Attributes
    ----------
    pulid_encoder : IDFormer
        The IDFormer encoder for identity embedding.
    pulid_ca : nn.ModuleList
        List of PerceiverAttentionCA modules injected into the transformer.
    face_helper : FaceRestoreHelper
        Helper for face alignment and parsing.
    clip_vision_model : nn.Module
        EVA-CLIP visual backbone.
    eva_transform_mean : tuple
        Mean for image normalization.
    eva_transform_std : tuple
        Std for image normalization.
    app : FaceAnalysis
        InsightFace face analysis application.
    handler_ante : insightface.model_zoo.model_zoo.Model
        InsightFace embedding model.
    debug_img_list : list
        List of debug images (for visualization).
    """

    def __init__(
        self,
        dit: NunchakuFluxTransformer2dModel,
        device: str | torch.device,
        weight_dtype: str | torch.dtype = torch.bfloat16,
        onnx_provider: str = "gpu",
        pulid_path: str | os.PathLike[str] = "guozinan/PuLID/pulid_flux_v0.9.1.safetensors",
        eva_clip_path: str | os.PathLike[str] = "QuanSun/EVA-CLIP/EVA02_CLIP_L_336_psz14_s6B.pt",
        insightface_dirpath: str | os.PathLike[str] | None = None,
        facexlib_dirpath: str | os.PathLike[str] | None = None,
    ):
        super().__init__()
        self.device = device
        self.weight_dtype = weight_dtype
        double_interval = 2
        single_interval = 4

        # init encoder
        self.pulid_encoder = IDFormer().to(self.device, self.weight_dtype)

        num_ca = 19 // double_interval + 38 // single_interval
        if 19 % double_interval != 0:
            num_ca += 1
        if 38 % single_interval != 0:
            num_ca += 1
        self.pulid_ca = nn.ModuleList(
            [PerceiverAttentionCA().to(self.device, self.weight_dtype) for _ in range(num_ca)]
        )

        dit.transformer_blocks[0].pulid_ca = self.pulid_ca

        # preprocessors
        # face align and parsing

        if facexlib_dirpath is None:
            facexlib_dirpath = Path(HUGGINGFACE_HUB_CACHE) / "facexlib"
        facexlib_dirpath = Path(facexlib_dirpath)

        self.face_helper = FaceRestoreHelper(
            upscale_factor=1,
            face_size=512,
            crop_ratio=(1, 1),
            det_model="retinaface_resnet50",
            save_ext="png",
            device=self.device,
            model_rootpath=str(facexlib_dirpath),
        )
        self.face_helper.face_parse = None
        self.face_helper.face_parse = init_parsing_model(
            model_name="bisenet", device=self.device, model_rootpath=str(facexlib_dirpath)
        )

        # clip-vit backbone
        model, _, _ = create_model_and_transforms(
            "EVA02-CLIP-L-14-336", "eva_clip", force_custom_clip=True, pretrained_path=eva_clip_path
        )
        model = model.visual
        self.clip_vision_model = model.to(self.device, dtype=self.weight_dtype)
        eva_transform_mean = getattr(self.clip_vision_model, "image_mean", OPENAI_DATASET_MEAN)
        eva_transform_std = getattr(self.clip_vision_model, "image_std", OPENAI_DATASET_STD)
        if not isinstance(eva_transform_mean, (list, tuple)):
            eva_transform_mean = (eva_transform_mean,) * 3
        if not isinstance(eva_transform_std, (list, tuple)):
            eva_transform_std = (eva_transform_std,) * 3
        self.eva_transform_mean = eva_transform_mean
        self.eva_transform_std = eva_transform_std

        # antelopev2
        if insightface_dirpath is None:
            insightface_dirpath = Path(HUGGINGFACE_HUB_CACHE) / "insightface"
        insightface_dirpath = Path(insightface_dirpath)

        if insightface_dirpath is not None:
            antelopev2_dirpath = insightface_dirpath / "models" / "antelopev2"
        else:
            antelopev2_dirpath = None

        if antelopev2_dirpath is None or not check_antelopev2_dir(antelopev2_dirpath):
            snapshot_download("DIAMONIK7777/antelopev2", local_dir=antelopev2_dirpath)
        providers = (
            ["CPUExecutionProvider"] if onnx_provider == "cpu" else ["CUDAExecutionProvider", "CPUExecutionProvider"]
        )
        self.app = FaceAnalysis(name="antelopev2", root=insightface_dirpath, providers=providers)
        self.app.prepare(ctx_id=0, det_size=(640, 640))
        self.handler_ante = insightface.model_zoo.get_model(
            str(antelopev2_dirpath / "glintr100.onnx"), providers=providers
        )
        self.handler_ante.prepare(ctx_id=0)

        # pulid model
        state_dict = load_state_dict_in_safetensors(pulid_path)
        module_state_dict = {}

        for k, v in state_dict.items():
            module = k.split(".")[0]
            module_state_dict.setdefault(module, {})
            new_k = k[len(module) + 1 :]
            module_state_dict[module][new_k] = v

        for module in module_state_dict:
            logging.debug(f"loading from {module}")
            getattr(self, module).load_state_dict(module_state_dict[module], strict=True)

        del state_dict
        del module_state_dict

        gc.collect()
        torch.cuda.empty_cache()

        # other configs
        self.debug_img_list = []

    def to_gray(self, img):
        """
        Convert an image tensor to grayscale (3 channels).

        Parameters
        ----------
        img : torch.Tensor
            Image tensor of shape (B, 3, H, W).

        Returns
        -------
        torch.Tensor
            Grayscale image tensor of shape (B, 3, H, W).
        """
        x = 0.299 * img[:, 0:1] + 0.587 * img[:, 1:2] + 0.114 * img[:, 2:3]
        x = x.repeat(1, 3, 1, 1)
        return x

    @torch.no_grad()
    def get_id_embedding(self, image, cal_uncond=False):
        """
        Extract identity embedding from an RGB image.

        Parameters
        ----------
        image : np.ndarray
            Input RGB image as a numpy array, range [0, 255].
        cal_uncond : bool, optional
            If True, also compute unconditional embedding (default: False).

        Returns
        -------
        id_embedding : torch.Tensor
            Identity embedding tensor.
        uncond_id_embedding : torch.Tensor or None
            Unconditional embedding tensor if cal_uncond is True, else None.
        """
        self.face_helper.clean_all()
        self.debug_img_list = []
        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        # get antelopev2 embedding
        face_info = self.app.get(image_bgr)
        if len(face_info) > 0:
            face_info = sorted(face_info, key=lambda x: (x["bbox"][2] - x["bbox"][0]) * (x["bbox"][3] - x["bbox"][1]))[
                -1
            ]  # only use the maximum face
            id_ante_embedding = face_info["embedding"]
            self.debug_img_list.append(
                image[
                    int(face_info["bbox"][1]) : int(face_info["bbox"][3]),
                    int(face_info["bbox"][0]) : int(face_info["bbox"][2]),
                ]
            )
        else:
            id_ante_embedding = None

        # using facexlib to detect and align face
        self.face_helper.read_image(image_bgr)
        self.face_helper.get_face_landmarks_5(only_center_face=True)
        self.face_helper.align_warp_face()
        if len(self.face_helper.cropped_faces) == 0:
            raise RuntimeError("facexlib align face fail")
        align_face = self.face_helper.cropped_faces[0]
        # incase insightface didn't detect face
        if id_ante_embedding is None:
            print("fail to detect face using insightface, extract embedding on align face")
            id_ante_embedding = self.handler_ante.get_feat(align_face)

        id_ante_embedding = torch.from_numpy(id_ante_embedding).to(self.device, self.weight_dtype)
        if id_ante_embedding.ndim == 1:
            id_ante_embedding = id_ante_embedding.unsqueeze(0)

        # parsing
        input = img2tensor(align_face, bgr2rgb=True).unsqueeze(0) / 255.0
        input = input.to(self.device)
        parsing_out = self.face_helper.face_parse(normalize(input, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]))[0]
        parsing_out = parsing_out.argmax(dim=1, keepdim=True)
        bg_label = [0, 16, 18, 7, 8, 9, 14, 15]
        bg = sum(parsing_out == i for i in bg_label).bool()
        white_image = torch.ones_like(input)
        # only keep the face features
        face_features_image = torch.where(bg, white_image, self.to_gray(input))
        self.debug_img_list.append(tensor2img(face_features_image, rgb2bgr=False))

        # transform img before sending to eva-clip-vit
        face_features_image = resize(face_features_image, self.clip_vision_model.image_size, InterpolationMode.BICUBIC)
        face_features_image = normalize(face_features_image, self.eva_transform_mean, self.eva_transform_std)
        id_cond_vit, id_vit_hidden = self.clip_vision_model(
            face_features_image.to(self.weight_dtype), return_all_features=False, return_hidden=True, shuffle=False
        )
        id_cond_vit_norm = torch.norm(id_cond_vit, 2, 1, True)
        id_cond_vit = torch.div(id_cond_vit, id_cond_vit_norm)

        id_cond = torch.cat([id_ante_embedding, id_cond_vit], dim=-1)

        id_embedding = self.pulid_encoder(id_cond, id_vit_hidden)

        if not cal_uncond:
            return id_embedding, None

        id_uncond = torch.zeros_like(id_cond)
        id_vit_hidden_uncond = []
        for layer_idx in range(0, len(id_vit_hidden)):
            id_vit_hidden_uncond.append(torch.zeros_like(id_vit_hidden[layer_idx]))
        uncond_id_embedding = self.pulid_encoder(id_uncond, id_vit_hidden_uncond)

        return id_embedding, uncond_id_embedding


class PuLIDFluxPipeline(FluxPipeline):
    """
    FluxPipeline with PuLID identity embedding support.

    This pipeline extends the standard FluxPipeline to support personalized image generation using
    identity embeddings extracted from a reference image. It injects the PuLID identity encoder into
    the transformer and supports all standard FluxPipeline features.

    Parameters
    ----------
    scheduler : SchedulerMixin
        Scheduler for diffusion process.
    vae : AutoencoderKL
        Variational autoencoder for image encoding/decoding.
    text_encoder : PreTrainedModel
        Text encoder for prompt embeddings.
    tokenizer : PreTrainedTokenizer
        Tokenizer for text encoder.
    text_encoder_2 : PreTrainedModel
        Second text encoder (optional).
    tokenizer_2 : PreTrainedTokenizer
        Second tokenizer (optional).
    transformer : NunchakuFluxTransformer2dModel
        Transformer model for denoising.
    image_encoder : nn.Module, optional
        Image encoder for IP-Adapter (default: None).
    feature_extractor : nn.Module, optional
        Feature extractor for images (default: None).
    pulid_device : str, optional
        Device for PuLID pipeline (default: "cuda").
    weight_dtype : torch.dtype, optional
        Data type for model weights (default: torch.bfloat16).
    onnx_provider : str, optional
        ONNX runtime provider (default: "gpu").
    """

    def __init__(
        self,
        scheduler,
        vae,
        text_encoder,
        tokenizer,
        text_encoder_2,
        tokenizer_2,
        transformer,
        image_encoder=None,
        feature_extractor=None,
        pulid_device="cuda",
        weight_dtype=torch.bfloat16,
        onnx_provider="gpu",
    ):
        super().__init__(
            scheduler=scheduler,
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            text_encoder_2=text_encoder_2,
            tokenizer_2=tokenizer_2,
            transformer=transformer,
            image_encoder=image_encoder,
            feature_extractor=feature_extractor,
        )

        # Save custom parameters
        self.pulid_device = torch.device(pulid_device)
        self.weight_dtype = weight_dtype
        self.onnx_provider = onnx_provider

        # Init PuLID pipeline (injects ID encoder into transformer)
        self.pulid_model = PuLIDPipeline(
            dit=self.transformer,  # directly mutate transformer with pulid_ca
            device=self.pulid_device,
            weight_dtype=self.weight_dtype,
            onnx_provider=self.onnx_provider,
        )

    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        prompt_2: Optional[Union[str, List[str]]] = None,
        negative_prompt: Union[str, List[str]] = None,
        negative_prompt_2: Optional[Union[str, List[str]]] = None,
        true_cfg_scale: float = 1.0,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 28,
        sigmas: Optional[List[float]] = None,
        guidance_scale: float = 3.5,
        num_images_per_prompt: Optional[int] = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        ip_adapter_image: Optional[PipelineImageInput] = None,
        id_image=None,
        id_weight=1.0,
        start_step=0,
        ip_adapter_image_embeds: Optional[List[torch.Tensor]] = None,
        negative_ip_adapter_image: Optional[PipelineImageInput] = None,
        negative_ip_adapter_image_embeds: Optional[List[torch.Tensor]] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        max_sequence_length: int = 512,
    ):
        r"""
        Function invoked when calling the pipeline for generation.

        See the parent class :class:`diffusers.FluxPipeline` for full documentation.

        Parameters
        ----------
        prompt : str or List[str], optional
            The prompt(s) to guide image generation.
        prompt_2 : str or List[str], optional
            Second prompt(s) for dual-encoder pipelines.
        negative_prompt : str or List[str], optional
            Negative prompt(s) to avoid in generation.
        negative_prompt_2 : str or List[str], optional
            Second negative prompt(s) for dual-encoder pipelines.
        true_cfg_scale : float, optional
            True classifier-free guidance scale.
        height : int, optional
            Output image height.
        width : int, optional
            Output image width.
        num_inference_steps : int, optional
            Number of denoising steps.
        sigmas : List[float], optional
            Custom sigmas for the scheduler.
        guidance_scale : float, optional
            Classifier-free guidance scale.
        num_images_per_prompt : int, optional
            Number of images per prompt.
        generator : torch.Generator or List[torch.Generator], optional
            Random generator(s) for reproducibility.
        latents : torch.FloatTensor, optional
            Pre-generated latents.
        prompt_embeds : torch.FloatTensor, optional
            Pre-generated prompt embeddings.
        pooled_prompt_embeds : torch.FloatTensor, optional
            Pre-generated pooled prompt embeddings.
        ip_adapter_image : PipelineImageInput, optional
            Image input for IP-Adapter.
        id_image : PIL.Image.Image or np.ndarray, optional
            Reference image for identity embedding.
        id_weight : float, optional
            Weight for identity embedding.
        start_step : int, optional
            Step to start from (for advanced use).
        ip_adapter_image_embeds : List[torch.Tensor], optional
            Precomputed IP-Adapter image embeddings.
        negative_ip_adapter_image : PipelineImageInput, optional
            Negative image input for IP-Adapter.
        negative_ip_adapter_image_embeds : List[torch.Tensor], optional
            Precomputed negative IP-Adapter image embeddings.
        negative_prompt_embeds : torch.FloatTensor, optional
            Precomputed negative prompt embeddings.
        negative_pooled_prompt_embeds : torch.FloatTensor, optional
            Precomputed negative pooled prompt embeddings.
        output_type : str, optional
            Output format ("pil" or "np").
        return_dict : bool, optional
            Whether to return a dict or tuple.
        joint_attention_kwargs : dict, optional
            Additional kwargs for joint attention.
        callback_on_step_end : Callable, optional
            Callback at the end of each denoising step.
        callback_on_step_end_tensor_inputs : List[str], optional
            List of tensor names for callback.
        max_sequence_length : int, optional
            Maximum sequence length for prompts.

        Returns
        -------
        FluxPipelineOutput or tuple
            Output images and additional info.
        """
        height = height or self.default_sample_size * self.vae_scale_factor
        width = width or self.default_sample_size * self.vae_scale_factor

        if id_image is not None:
            # pil_image = Image.open(id_image)
            pil_image = id_image.convert("RGB")
            numpy_image = np.array(pil_image)
            id_image = resize_numpy_image_long(numpy_image, 1024)
            id_embeddings, uncond_id_embeddings = self.pulid_model.get_id_embedding(id_image)
        else:
            id_embeddings = None
            uncond_id_embeddings = None

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt,
            prompt_2,
            height,
            width,
            negative_prompt=negative_prompt,
            negative_prompt_2=negative_prompt_2,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
            callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
            max_sequence_length=max_sequence_length,
        )

        self._guidance_scale = guidance_scale
        self._joint_attention_kwargs = joint_attention_kwargs
        self._current_timestep = None
        self._interrupt = False

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device

        lora_scale = self.joint_attention_kwargs.get("scale", None) if self.joint_attention_kwargs is not None else None
        has_neg_prompt = negative_prompt is not None or (
            negative_prompt_embeds is not None and negative_pooled_prompt_embeds is not None
        )
        do_true_cfg = true_cfg_scale > 1 and has_neg_prompt
        (
            prompt_embeds,
            pooled_prompt_embeds,
            text_ids,
        ) = self.encode_prompt(
            prompt=prompt,
            prompt_2=prompt_2,
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            device=device,
            num_images_per_prompt=num_images_per_prompt,
            max_sequence_length=max_sequence_length,
            lora_scale=lora_scale,
        )
        if do_true_cfg:
            (
                negative_prompt_embeds,
                negative_pooled_prompt_embeds,
                _,
            ) = self.encode_prompt(
                prompt=negative_prompt,
                prompt_2=negative_prompt_2,
                prompt_embeds=negative_prompt_embeds,
                pooled_prompt_embeds=negative_pooled_prompt_embeds,
                device=device,
                num_images_per_prompt=num_images_per_prompt,
                max_sequence_length=max_sequence_length,
                lora_scale=lora_scale,
            )

        # 4. Prepare latent variables
        num_channels_latents = self.transformer.config.in_channels // 4
        latents, latent_image_ids = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )

        # 5. Prepare timesteps
        sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps) if sigmas is None else sigmas
        image_seq_len = latents.shape[1]
        mu = calculate_shift(
            image_seq_len,
            self.scheduler.config.get("base_image_seq_len", 256),
            self.scheduler.config.get("max_image_seq_len", 4096),
            self.scheduler.config.get("base_shift", 0.5),
            self.scheduler.config.get("max_shift", 1.15),
        )
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler,
            num_inference_steps,
            device,
            sigmas=sigmas,
            mu=mu,
        )
        num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)
        self._num_timesteps = len(timesteps)

        # handle guidance
        if self.transformer.config.guidance_embeds:
            guidance = torch.full([1], guidance_scale, device=device, dtype=torch.float32)
            guidance = guidance.expand(latents.shape[0])
        else:
            guidance = None

        if (ip_adapter_image is not None or ip_adapter_image_embeds is not None) and (
            negative_ip_adapter_image is None and negative_ip_adapter_image_embeds is None
        ):
            negative_ip_adapter_image = np.zeros((width, height, 3), dtype=np.uint8)
            negative_ip_adapter_image = [negative_ip_adapter_image] * self.transformer.encoder_hid_proj.num_ip_adapters

        elif (ip_adapter_image is None and ip_adapter_image_embeds is None) and (
            negative_ip_adapter_image is not None or negative_ip_adapter_image_embeds is not None
        ):
            ip_adapter_image = np.zeros((width, height, 3), dtype=np.uint8)
            ip_adapter_image = [ip_adapter_image] * self.transformer.encoder_hid_proj.num_ip_adapters

        if self.joint_attention_kwargs is None:
            self._joint_attention_kwargs = {}

        image_embeds = None
        negative_image_embeds = None
        if ip_adapter_image is not None or ip_adapter_image_embeds is not None:
            image_embeds = self.prepare_ip_adapter_image_embeds(
                ip_adapter_image,
                ip_adapter_image_embeds,
                device,
                batch_size * num_images_per_prompt,
            )
        if negative_ip_adapter_image is not None or negative_ip_adapter_image_embeds is not None:
            negative_image_embeds = self.prepare_ip_adapter_image_embeds(
                negative_ip_adapter_image,
                negative_ip_adapter_image_embeds,
                device,
                batch_size * num_images_per_prompt,
            )

        # 6. Denoising loop
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue

                self._current_timestep = t
                if image_embeds is not None:
                    self._joint_attention_kwargs["ip_adapter_image_embeds"] = image_embeds
                # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
                timestep = t.expand(latents.shape[0]).to(latents.dtype)
                noise_pred = self.transformer(
                    hidden_states=latents,
                    id_embeddings=id_embeddings,
                    id_weight=id_weight,
                    timestep=timestep / 1000,
                    guidance=guidance,
                    pooled_projections=pooled_prompt_embeds,
                    encoder_hidden_states=prompt_embeds,
                    txt_ids=text_ids,
                    img_ids=latent_image_ids,
                    joint_attention_kwargs=self.joint_attention_kwargs,
                    return_dict=False,
                )[0]

                if do_true_cfg:
                    if negative_image_embeds is not None:
                        self._joint_attention_kwargs["ip_adapter_image_embeds"] = negative_image_embeds
                    neg_noise_pred = self.transformer(
                        hidden_states=latents,
                        id_embeddings=id_embeddings,
                        id_weight=id_weight,
                        timestep=timestep / 1000,
                        guidance=guidance,
                        pooled_projections=negative_pooled_prompt_embeds,
                        encoder_hidden_states=negative_prompt_embeds,
                        txt_ids=text_ids,
                        img_ids=latent_image_ids,
                        joint_attention_kwargs=self.joint_attention_kwargs,
                        return_dict=False,
                    )[0]
                    noise_pred = neg_noise_pred + true_cfg_scale * (noise_pred - neg_noise_pred)

                # compute the previous noisy sample x_t -> x_t-1
                latents_dtype = latents.dtype
                latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

                if latents.dtype != latents_dtype:
                    if torch.backends.mps.is_available():
                        # some platforms (eg. apple mps) misbehave due to a pytorch bug: https://github.com/pytorch/pytorch/pull/99272
                        latents = latents.to(latents_dtype)

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                    latents = callback_outputs.pop("latents", latents)
                    prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()

        self._current_timestep = None

        if output_type == "latent":
            image = latents
        else:
            latents = self._unpack_latents(latents, height, width, self.vae_scale_factor)
            latents = (latents / self.vae.config.scaling_factor) + self.vae.config.shift_factor
            image = self.vae.decode(latents, return_dict=False)[0]
            image = self.image_processor.postprocess(image, output_type=output_type)

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (image,)

        return FluxPipelineOutput(images=image)
