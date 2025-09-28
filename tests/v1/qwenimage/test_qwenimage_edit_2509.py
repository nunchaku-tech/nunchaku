import gc
import os
from pathlib import Path

import pytest
import torch
from diffusers import QwenImageEditPlusPipeline
from diffusers.utils import load_image

from nunchaku.models.transformers.transformer_qwenimage import NunchakuQwenImageTransformer2DModel
from nunchaku.utils import get_gpu_memory, get_precision, is_turing

from ...utils import already_generate, compute_lpips
from ..utils import run_pipeline

precision = get_precision()
torch_dtype = torch.float16 if is_turing() else torch.bfloat16
dtype_str = "fp16" if torch_dtype == torch.float16 else "bf16"


class Case:

    def __init__(self, num_inference_steps: int, rank: int, expected_lpips: dict[str, float]):
        self.model_name = "qwen-image-edit-2509"
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
                expected_lpips={"int4-bf16": 0.17, "fp4-bf16": 0.17},
            ),
            id="qwen-image-edit-2509-r32",
        ),
        pytest.param(
            Case(
                num_inference_steps=20,
                rank=128,
                expected_lpips={"int4-bf16": 0.21, "fp4-bf16": 0.13},
            ),
            id="qwen-image-edit-2509-r128",
        ),
    ],
)
def test_qwenimage_edit(case: Case):
    batch_size = 1
    true_cfg_scale = 4.0
    rank = case.rank
    expected_lpips = case.expected_lpips[f"{precision}-{dtype_str}"]
    model_name = case.model_name
    num_inference_steps = case.num_inference_steps

    ref_root = os.environ.get("NUNCHAKU_TEST_CACHE_ROOT", os.path.join("test_results", "ref"))
    folder_name = f"t{num_inference_steps}g{true_cfg_scale}"
    save_dir_16bit = Path(ref_root) / model_name / dtype_str / folder_name

    repo_id = "Qwen/Qwen-Image-Edit-2509"
    dataset = [
        {
            "prompt": "make the cat floating in the air and holding a sign that reads 'this is fun' written with a blue crayon",
            "negative_prompt": " ",
            "filename": "cat_sitting.png",
            "image": load_image(
                "https://huggingface.co/datasets/nunchaku-tech/test-data/resolve/main/inputs/cat_sitting.jpg"
            ).convert("RGB"),
        },
        {
            "prompt": "turn the style of the photo to vintage comic book",
            "negative_prompt": " ",
            "filename": "pie",
            "image": load_image(
                "https://huggingface.co/datasets/nunchaku-tech/test-data/resolve/main/inputs/pie.png"
            ).convert("RGB"),
        },
    ]

    if not already_generate(save_dir_16bit, len(dataset)):
        pipe = QwenImageEditPlusPipeline.from_pretrained(repo_id, torch_dtype=torch_dtype)
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

    model_path = (
        f"nunchaku-tech/nunchaku-qwen-image-edit-2509/svdq-{get_precision()}_r{rank}-qwen-image-edit-2509.safetensors"
    )
    transformer = NunchakuQwenImageTransformer2DModel.from_pretrained(model_path, torch_dtype=torch_dtype)

    pipe = QwenImageEditPlusPipeline.from_pretrained(repo_id, transformer=transformer, torch_dtype=torch_dtype)

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

    lpips = compute_lpips(save_dir_16bit, save_dir_nunchaku, batch_size=1)
    print(f"lpips: {lpips}")
    assert lpips < expected_lpips * 1.10
