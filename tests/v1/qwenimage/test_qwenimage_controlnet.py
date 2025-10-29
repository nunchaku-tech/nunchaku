import gc
import os
from pathlib import Path

import diffusers
import packaging.version
import pytest
import torch
from diffusers.utils import load_image

from nunchaku.models.transformers.transformer_qwenimage import NunchakuQwenImageTransformer2DModel
from nunchaku.utils import get_gpu_memory, get_precision, is_turing

from ...utils import already_generate, compute_lpips
from ..utils import run_pipeline

try:
    from diffusers import QwenImageControlNetModel, QwenImageControlNetPipeline
except ImportError:
    QwenImageControlNetModel = None
    QwenImageControlNetPipeline = None

# Skip the test if diffusers<0.36
pytestmark = pytest.mark.skipif(
    packaging.version.parse(diffusers.__version__) <= packaging.version.parse("0.35.1"),
    reason="QwenImageControlNetPipeline requires diffusers>=0.36",
)


precision = get_precision()
torch_dtype = torch.float16 if is_turing() else torch.bfloat16
dtype_str = "fp16" if torch_dtype == torch.float16 else "bf16"


class Case:

    def __init__(self, num_inference_steps: int, rank: int, expected_lpips: dict[str, float]):
        self.model_name = "qwen-image-controlnet-union"
        self.num_inference_steps = num_inference_steps
        self.rank = rank
        self.expected_lpips = expected_lpips


@pytest.mark.parametrize(
    "case",
    [
        pytest.param(
            Case(
                num_inference_steps=20,
                rank=32,
                expected_lpips={"int4-bf16": 0.13, "fp4-bf16": 0.11},
            ),
            id="qwen-image-controlnet-union-r32",
        ),
        pytest.param(
            Case(
                num_inference_steps=20,
                rank=128,
                expected_lpips={"int4-bf16": 0.12, "fp4-bf16": 0.1},
            ),
            id="qwen-image-controlnet-union-r128",
        ),
    ],
)
def test_qwenimage_controlnet(case: Case):
    batch_size = 1
    true_cfg_scale = 4.0
    rank = case.rank
    expected_lpips = case.expected_lpips[f"{precision}-{dtype_str}"]
    model_name = case.model_name
    num_inference_steps = case.num_inference_steps
    forward_kwargs = {
        "num_inference_steps": num_inference_steps,
        "true_cfg_scale": true_cfg_scale,
        "controlnet_conditioning_scale": 1.0,
    }

    ref_root = os.environ.get("NUNCHAKU_TEST_CACHE_ROOT", os.path.join("test_results", "ref"))
    folder_name = f"t{num_inference_steps}g{true_cfg_scale}"
    save_dir_16bit = Path(ref_root) / model_name / dtype_str / folder_name

    repo_id = "Qwen/Qwen-Image"

    dataset = [
        {
            "prompt": "Aesthetics art, traditional asian pagoda, elaborate golden accents, sky blue and white color palette, swirling cloud pattern, digital illustration, east asian architecture, ornamental rooftop, intricate detailing on building, cultural representation.",
            "negative_prompt": " ",
            "filename": "canny",
            "control_image": load_image(
                "https://huggingface.co/InstantX/Qwen-Image-ControlNet-Union/resolve/main/conds/canny.png"
            ).convert("RGB"),
        },
        {
            "prompt": "A swanky, minimalist living room with a huge floor-to-ceiling window letting in loads of natural light. A beige couch with white cushions sits on a wooden floor, with a matching coffee table in front. The walls are a soft, warm beige, decorated with two framed botanical prints. A potted plant chills in the corner near the window. Sunlight pours through the leaves outside, casting cool shadows on the floor.",
            "negative_prompt": " ",
            "filename": "depth",
            "control_image": load_image(
                "https://huggingface.co/InstantX/Qwen-Image-ControlNet-Union/resolve/main/conds/depth.png"
            ).convert("RGB"),
        },
        {
            "prompt": "Photograph of a young man with light brown hair and a beard, wearing a beige flat cap, black leather jacket, gray shirt, brown pants, and white sneakers. He's sitting on a concrete ledge in front of a large circular window, with a cityscape reflected in the glass. The wall is cream-colored, and the sky is clear blue. His shadow is cast on the wall.",
            "negative_prompt": " ",
            "filename": "pose",
            "control_image": load_image(
                "https://huggingface.co/InstantX/Qwen-Image-ControlNet-Union/resolve/main/conds/pose.png"
            ).convert("RGB"),
        },
    ]
    for item in dataset:
        item["width"] = item["control_image"].size[0]
        item["height"] = item["control_image"].size[1]

    if not already_generate(save_dir_16bit, len(dataset)):
        controlnet = QwenImageControlNetModel.from_pretrained(
            "InstantX/Qwen-Image-ControlNet-Union", torch_dtype=torch_dtype
        )
        pipe = QwenImageControlNetPipeline.from_pretrained(repo_id, controlnet=controlnet, torch_dtype=torch_dtype)
        pipe.enable_sequential_cpu_offload()
        run_pipeline(
            dataset=dataset, batch_size=1, pipeline=pipe, save_dir=save_dir_16bit, forward_kwargs=forward_kwargs
        )
        del pipe
        gc.collect()
        torch.cuda.empty_cache()

    save_dir_nunchaku = (
        Path("test_results")
        / "nunchaku"
        / model_name
        / f"{precision}_r{rank}-{dtype_str}"
        / f"{folder_name}-bs{batch_size}"
    )

    model_path = f"nunchaku-tech/nunchaku-qwen-image/svdq-{precision}_r{rank}-qwen-image.safetensors"
    transformer = NunchakuQwenImageTransformer2DModel.from_pretrained(model_path, torch_dtype=torch_dtype)
    controlnet = QwenImageControlNetModel.from_pretrained(
        "InstantX/Qwen-Image-ControlNet-Union", torch_dtype=torch_dtype
    )
    pipe = QwenImageControlNetPipeline.from_pretrained(
        repo_id, transformer=transformer, controlnet=controlnet, torch_dtype=torch_dtype
    )

    if get_gpu_memory() > 18:
        pipe.enable_model_cpu_offload()
    else:
        transformer.set_offload(True, use_pin_memory=True, num_blocks_on_gpu=20)
        pipe._exclude_from_cpu_offload.append("transformer")
        pipe.enable_sequential_cpu_offload()

    run_pipeline(
        dataset=dataset, batch_size=batch_size, pipeline=pipe, save_dir=save_dir_nunchaku, forward_kwargs=forward_kwargs
    )
    del transformer
    del pipe
    gc.collect()
    torch.cuda.empty_cache()

    lpips = compute_lpips(save_dir_16bit, save_dir_nunchaku, batch_size=1)
    print(f"lpips: {lpips}")
    assert lpips < expected_lpips * 1.10
