import gc
import math
import os
from pathlib import Path

import pytest
import torch
from diffusers import FlowMatchEulerDiscreteScheduler, QwenImageEditPipeline
from diffusers.utils import load_image

from nunchaku.models.transformers.transformer_qwenimage import NunchakuQwenImageTransformer2DModel
from nunchaku.utils import get_gpu_memory, get_precision, is_turing

from ...utils import already_generate, compute_lpips
from ..utils import run_pipeline

precision = get_precision()
torch_dtype = torch.float16 if is_turing() else torch.bfloat16
dtype_str = "fp16" if torch_dtype == torch.float16 else "bf16"

model_paths = {
    "qwen-image-edit-lightningv1.0-4steps": "nunchaku-tech/nunchaku-qwen-image-edit/svdq-{precision}_r{rank}-qwen-image-edit-lightningv1.0-4steps.safetensors",
    "qwen-image-edit-lightningv1.0-8steps": "nunchaku-tech/nunchaku-qwen-image-edit/svdq-{precision}_r{rank}-qwen-image-edit-lightningv1.0-8steps.safetensors",
}
lora_paths = {
    "qwen-image-edit-lightningv1.0-4steps": (
        "lightx2v/Qwen-Image-Lightning",
        "Qwen-Image-Edit-Lightning-4steps-V1.0-bf16.safetensors",
    ),
    "qwen-image-edit-lightningv1.0-8steps": (
        "lightx2v/Qwen-Image-Lightning",
        "Qwen-Image-Edit-Lightning-8steps-V1.0-bf16.safetensors",
    ),
}


class TestCase:

    def __init__(self, model_name: str, num_inference_steps: int, rank: int, expected_lpips: dict[str, float]):
        self.model_name = model_name
        self.num_inference_steps = num_inference_steps
        self.rank = rank
        self.expected_lpips = expected_lpips


@pytest.mark.parametrize(
    "case",
    [
        pytest.param(
            TestCase(
                model_name="qwen-image-edit-lightningv1.0-4steps",
                num_inference_steps=4,
                rank=32,
                expected_lpips={"int4-bf16": 0.1, "fp4-bf16": 0.4},
            ),
            id="qwen-image-edit-lightningv1.0-4steps-r32",
        ),
        pytest.param(
            TestCase(
                model_name="qwen-image-edit-lightningv1.0-4steps",
                num_inference_steps=4,
                rank=128,
                expected_lpips={"int4-bf16": 0.1, "fp4-bf16": 0.4},
            ),
            id="qwen-image-edit-lightningv1.0-4steps-r128",
        ),
        pytest.param(
            TestCase(
                model_name="qwen-image-edit-lightningv1.0-8steps",
                num_inference_steps=8,
                rank=32,
                expected_lpips={"int4-bf16": 0.1, "fp4-bf16": 0.4},
            ),
            id="qwen-image-edit-lightningv1.0-8steps-r32",
        ),
        pytest.param(
            TestCase(
                model_name="qwen-image-edit-lightningv1.0-8steps",
                num_inference_steps=8,
                rank=128,
                expected_lpips={"int4-bf16": 0.1, "fp4-bf16": 0.4},
            ),
            id="qwen-image-edit-lightningv1.0-8steps-r128",
        ),
    ],
)
def test_qwenimage_edit_lightning(case: TestCase):
    batch_size = 1
    true_cfg_scale = 1.0
    rank = case.rank
    expected_lpips = case.expected_lpips[f"{precision}-{dtype_str}"]
    model_name = case.model_name
    num_inference_steps = case.num_inference_steps

    ref_root = os.environ.get("NUNCHAKU_TEST_CACHE_ROOT", os.path.join("test_results", "ref"))
    folder_name = f"t{num_inference_steps}g{true_cfg_scale}"
    save_dir_16bit = Path(ref_root) / model_name / dtype_str / folder_name

    repo_id = "Qwen/Qwen-Image-Edit"

    # From https://github.com/ModelTC/Qwen-Image-Lightning/blob/342260e8f5468d2f24d084ce04f55e101007118b/generate_with_diffusers.py#L82C9-L97C10
    scheduler_config = {
        "base_image_seq_len": 256,
        "base_shift": math.log(3),  # We use shift=3 in distillation
        "invert_sigmas": False,
        "max_image_seq_len": 8192,
        "max_shift": math.log(3),  # We use shift=3 in distillation
        "num_train_timesteps": 1000,
        "shift": 1.0,
        "shift_terminal": None,  # set shift_terminal to None
        "stochastic_sampling": False,
        "time_shift_type": "exponential",
        "use_beta_sigmas": False,
        "use_dynamic_shifting": True,
        "use_exponential_sigmas": False,
        "use_karras_sigmas": False,
    }
    scheduler = FlowMatchEulerDiscreteScheduler.from_config(scheduler_config)

    dataset = [
        {
            "prompt": "change the text to read '双截棍 Qwen Image Edit is here'",
            "filename": "neon_sign",
            "image": load_image(
                "https://huggingface.co/datasets/nunchaku-tech/test-data/resolve/main/inputs/neon_sign.png"
            ).convert("RGB"),
        },
        {
            "prompt": "Remove all UI text elements from the image. Keep the feeling that the characters and scene are in water. Also, remove the green UI elements at the bottom.",
            "filename": "comfy_poster",
            "image": load_image(
                "https://huggingface.co/datasets/nunchaku-tech/test-data/resolve/main/inputs/comfy_poster.png"
            ).convert("RGB"),
        },
    ]

    if not already_generate(save_dir_16bit, len(dataset)):
        pipe = QwenImageEditPipeline.from_pretrained(repo_id, scheduler=scheduler, torch_dtype=torch_dtype)
        pipe.load_lora_weights(lora_paths[model_name][0], weight_name=lora_paths[model_name][1])
        pipe.fuse_lora()
        pipe.unload_lora_weights()
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

    pipe = QwenImageEditPipeline.from_pretrained(
        repo_id, transformer=transformer, scheduler=scheduler, torch_dtype=torch_dtype
    )

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
