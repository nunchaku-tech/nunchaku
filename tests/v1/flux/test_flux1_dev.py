import gc
import os
from pathlib import Path

import pytest
import torch
from diffusers import FluxPipeline

from nunchaku import NunchakuFluxTransformer2DModelV2
from nunchaku.utils import get_gpu_memory, get_precision, is_turing

from ...utils import already_generate, compute_lpips
from ..utils import run_pipeline

precision = get_precision()
torch_dtype = torch.float16 if is_turing() else torch.bfloat16
dtype_str = "fp16" if torch_dtype == torch.float16 else "bf16"


class Case:
    def __init__(
        self,
        rank: int = 32,
        batch_size: int = 1,
        width: int = 1024,
        height: int = 1024,
        num_inference_steps: int = 20,
        attention_impl: str = "flashattn2",
        expected_lpips: dict[str, float] = {},
        model_name: str = "flux.1-dev",
        repo_id: str = "black-forest-labs/FLUX.1-dev",
    ):
        self.rank = rank
        self.batch_size = batch_size
        self.width = width
        self.height = height
        self.num_inference_steps = num_inference_steps
        self.attention_impl = attention_impl
        self.expected_lpips = expected_lpips
        self.model_name = model_name
        self.repo_id = repo_id

        self.model_path = f"nunchaku-tech/nunchaku-flux.1-dev/svdq-{precision}_r{rank}-flux.1-dev.safetensors"

        ref_root = os.environ.get("NUNCHAKU_TEST_CACHE_ROOT", os.path.join("test_results", "ref"))
        folder_name = f"w{width}h{height}t{num_inference_steps}"

        self.save_dir_16bit = Path(ref_root) / model_name / dtype_str / folder_name
        self.save_dir_nunchaku = (
            Path("test_results")
            / "nunchaku"
            / model_name
            / f"{precision}_r{rank}-{dtype_str}"
            / f"{folder_name}-bs{batch_size}"
        )

        self.forward_kwargs = {
            "width": width,
            "height": height,
            "num_inference_steps": num_inference_steps,
            "guidance_scale": 3.5,
        }


@pytest.mark.parametrize(
    "case", [pytest.param(Case(expected_lpips={"int4-bf16": 0.17, "fp4-bf16": 0.19}), id="flux.1-dev-r32")]
)
def test_flux_dev(case: Case):
    batch_size = case.batch_size
    expected_lpips = case.expected_lpips
    rank = case.rank
    repo_id = case.repo_id

    dataset = [
        {
            "prompt": "Plain light background, man to the side, light, happy, eye contact, black man aged 25  50, stylish confident man, suit, great straight hair, ",
            "filename": "man",
        },
        {
            "prompt": "3d rendering of isometric cupcake logo, pastel colors, octane rendering, unreal egine ",
            "filename": "cupcake_logo",
        },
        {
            "prompt": "character design and sketch, evil, female, drow elf, sorcerer, sharp facial features, large iris, dark blue and indigo colors, long and ornate cape, rainbowcolored gems and jewelry, leather armor, jeweled dagger, dark purple long hair, gothic ",
            "filename": "character_design",
        },
        # {
        #     "prompt": "a hauntingly sparse drivein theater with a single red car and a single audio post. ",
        #     "filename": "drivein_theater",
        # },
    ]

    if not already_generate(case.save_dir_16bit, len(dataset)):
        pipeline = FluxPipeline.from_pretrained(case.repo_id, torch_dtype=torch_dtype)
        if get_gpu_memory() > 25:
            pipeline.enable_model_cpu_offload()
        else:
            pipeline.enable_sequential_cpu_offload()
        run_pipeline(
            dataset=dataset,
            batch_size=case.batch_size,
            pipeline=pipeline,
            save_dir=case.save_dir_16bit,
            forward_kwargs=case.forward_kwargs,
        )

    transformer = NunchakuFluxTransformer2DModelV2.from_pretrained(
        f"nunchaku-tech/nunchaku-flux.1-dev/svdq-{precision}_r{rank}-flux.1-dev.safetensors",
        torch_dtype=torch_dtype,
    )

    pipe = FluxPipeline.from_pretrained(repo_id, transformer=transformer, torch_dtype=torch_dtype)

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
        save_dir=case.save_dir_nunchaku,
        forward_kwargs=case.forward_kwargs,
    )
    del transformer
    del pipe
    gc.collect()
    torch.cuda.empty_cache()

    lpips = compute_lpips(case.save_dir_16bit, case.save_dir_nunchaku)
    print(f"lpips: {lpips}")
    assert lpips < expected_lpips[f"{precision}-{dtype_str}"] * 1.10
