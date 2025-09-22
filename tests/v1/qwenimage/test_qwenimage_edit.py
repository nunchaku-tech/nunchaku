import gc
import os
from pathlib import Path

import pytest
import torch
from diffusers import QwenImageEditPipeline
from diffusers.utils import load_image

from nunchaku.models.transformers.transformer_qwenimage import NunchakuQwenImageTransformer2DModel
from nunchaku.utils import get_gpu_memory, get_precision, is_turing

from ...utils import already_generate, compute_lpips
from ..utils import run_pipeline

precision = get_precision()
torch_dtype = torch.float16 if is_turing() else torch.bfloat16
dtype_str = "fp16" if torch_dtype == torch.float16 else "bf16"


class TestCase:

    def __init__(self, num_inference_steps: int, rank: int, expected_lpips: dict[str, float]):
        self.model_name = "qwen-image-edit"
        self.num_inference_steps = num_inference_steps
        self.rank = rank
        self.expected_lpips = expected_lpips


@pytest.mark.parametrize(
    "case",
    [
        pytest.param(
            TestCase(
                num_inference_steps=20,
                rank=32,
                expected_lpips={"int4": 0.35, "fp4": 0.3},
            ),
            id="qwen-image-edit-r32",
        ),
        pytest.param(
            TestCase(
                num_inference_steps=20,
                rank=128,
                expected_lpips={"int4": 0.32, "fp4": 0.3},
            ),
            id="qwen-image-edit-r128",
        ),
    ],
)
def test_qwenimage_edit(case: TestCase):
    batch_size = 1
    true_cfg_scale = 4.0
    rank = case.rank
    expected_lpips = case.expected_lpips[precision]
    model_name = case.model_name
    num_inference_steps = case.num_inference_steps

    ref_root = os.environ.get("NUNCHAKU_TEST_CACHE_ROOT", os.path.join("test_results", "ref"))
    folder_name = f"t{num_inference_steps}g{true_cfg_scale}"
    save_dir_16bit = Path(ref_root) / model_name / dtype_str / folder_name

    repo_id = "Qwen/Qwen-Image-Edit"
    dataset = [
        {
            "prompt": "change the text to read '双截棍 Qwen Image Edit is here'",
            "negative_prompt": " ",
            "filename": "neon_sign",
            "image": load_image(
                "https://huggingface.co/datasets/nunchaku-tech/test-data/resolve/main/inputs/neon_sign.png"
            ).convert("RGB"),
        },
        {
            "prompt": "Remove all UI text elements from the image. Keep the feeling that the characters and scene are in water. Also, remove the green UI elements at the bottom.",
            "negative_prompt": " ",
            "filename": "comfy_poster",
            "image": load_image(
                "https://huggingface.co/datasets/nunchaku-tech/test-data/resolve/main/inputs/comfy_poster.png"
            ).convert("RGB"),
        },
    ]

    if not already_generate(save_dir_16bit, len(dataset)):
        pipe = QwenImageEditPipeline.from_pretrained(repo_id, torch_dtype=torch_dtype)
        pipe.enable_sequential_cpu_offload()
        run_pipeline(
            dataset=dataset,
            batch_size=1,
            pipeline=pipe,
            save_dir=save_dir_16bit,
            forward_kwargs={"num_inference_steps": num_inference_steps, "true_cfg_scale": true_cfg_scale},
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

    model_path = f"nunchaku-tech/nunchaku-qwen-image-edit/svdq-{get_precision()}_r{rank}-qwen-image-edit.safetensors"
    transformer = NunchakuQwenImageTransformer2DModel.from_pretrained(model_path, torch_dtype=torch_dtype)

    pipe = QwenImageEditPipeline.from_pretrained(repo_id, transformer=transformer, torch_dtype=torch_dtype)

    if get_gpu_memory() > 18:
        pipe.enable_model_cpu_offload()
    else:
        transformer.set_offload(True, use_pin_memory=True, num_blocks_on_gpu=20)
        pipe._exclude_from_cpu_offload.append("transformer")
        pipe.enable_sequential_cpu_offload()

    run_pipeline(
        dataset=dataset,
        batch_size=batch_size,
        pipeline=pipe,
        save_dir=save_dir_nunchaku,
        forward_kwargs={"num_inference_steps": num_inference_steps, "true_cfg_scale": true_cfg_scale},
    )
    del transformer
    del pipe
    gc.collect()
    torch.cuda.empty_cache()

    lpips = compute_lpips(save_dir_16bit, save_dir_nunchaku)
    print(f"lpips: {lpips}")
    assert lpips < expected_lpips * 1.10
