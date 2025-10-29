import gc
import os
from pathlib import Path

import pytest
import torch
from diffusers import FluxFillPipeline
from diffusers.utils import load_image

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
        model_name: str = "flux.1-fill-dev",
        repo_id: str = "black-forest-labs/FLUX.1-Fill-dev",
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

        self.model_path = f"nunchaku-tech/nunchaku-flux.1-fill-dev/svdq-{precision}_r{rank}-flux.1-fill-dev.safetensors"

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

        self.pipeline_cls = FluxFillPipeline
        self.forward_kwargs = {
            "width": width,
            "height": height,
            "num_inference_steps": num_inference_steps,
            "guidance_scale": 30,
        }


@pytest.mark.parametrize(
    "case", [pytest.param(Case(expected_lpips={"int4-bf16": 0.1, "fp4-bf16": 0.1}), id="flux.1-fill-dev-r32")]
)
def test_flux_fill_dev(case: Case):
    batch_size = case.batch_size
    expected_lpips = case.expected_lpips
    repo_id = case.repo_id

    dataset = [
        {
            "prompt": "the insanely extreme muscle car, Big foot wheels, dragster style, flames, 6 wheels ",
            "filename": "1ce4f3b8627ab16e8f09e6e169d8744d32274880",
            "image": load_image(
                "https://huggingface.co/datasets/nunchaku-tech/test-data/resolve/main/inputs/1ce4f3b8627ab16e8f09e6e169d8744d32274880-image.png"
            ).convert("RGB"),
            "mask_image": load_image(
                "https://huggingface.co/datasets/nunchaku-tech/test-data/resolve/main/inputs/1ce4f3b8627ab16e8f09e6e169d8744d32274880-mask.png"
            ).convert("RGB"),
        },
        # {
        #     "prompt": "sunlower, Folk Art ",
        #     "filename": "8c2fef24a984d4c76bebcfa406b7240fd25d7c36",
        #     "image": load_image(
        #         "https://huggingface.co/datasets/nunchaku-tech/test-data/resolve/main/inputs/8c2fef24a984d4c76bebcfa406b7240fd25d7c36-image.png"
        #     ).convert("RGB"),
        #     "mask_image": load_image(
        #         "https://huggingface.co/datasets/nunchaku-tech/test-data/resolve/main/inputs/8c2fef24a984d4c76bebcfa406b7240fd25d7c36-mask.png"
        #     ).convert("RGB"),
        # },
        {
            "prompt": "modern realistic allium flowers, clean straight lines, black and white, a lot of white space to color, coloring book style ",
            "filename": "94f2b6fc3ab734ccdf6e57f72287f0a6df522dc0",
            "image": load_image(
                "https://huggingface.co/datasets/nunchaku-tech/test-data/resolve/main/inputs/94f2b6fc3ab734ccdf6e57f72287f0a6df522dc0-image.png"
            ).convert("RGB"),
            "mask_image": load_image(
                "https://huggingface.co/datasets/nunchaku-tech/test-data/resolve/main/inputs/94f2b6fc3ab734ccdf6e57f72287f0a6df522dc0-mask.png"
            ).convert("RGB"),
        },
        {
            "prompt": " Content Spirit Wraith Coin Medium engraved metallic coin Style symmetrical, detailed design Lighting Reflective natural light Colors purples and grays Composition the beast centered, surrounded by elemental symbols, stats, and abilities Create a Spirit Wraith Elemental Guardian Coin featuring a symmetrical, detailed design of the Spirit Wraith guardian at the center, signifying its affinity for the spirit element. The coin should have reflective natural light with mystical purples and ethereal grays. Encircle the guardian with elemental symbols, stats, and abilities relevant to its spiritbased prowess. ",
            "filename": "d38575d92bfd143930c4e57daa69aad5a4be48a6",
            "image": load_image(
                "https://huggingface.co/datasets/nunchaku-tech/test-data/resolve/main/inputs/d38575d92bfd143930c4e57daa69aad5a4be48a6-image.png"
            ).convert("RGB"),
            "mask_image": load_image(
                "https://huggingface.co/datasets/nunchaku-tech/test-data/resolve/main/inputs/d38575d92bfd143930c4e57daa69aad5a4be48a6-mask.png"
            ).convert("RGB"),
        },
    ]

    if not already_generate(case.save_dir_16bit, len(dataset)):
        pipeline = case.pipeline_cls.from_pretrained(case.repo_id, torch_dtype=torch_dtype)
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

    transformer = NunchakuFluxTransformer2DModelV2.from_pretrained(case.model_path, torch_dtype=torch_dtype)

    pipe = case.pipeline_cls.from_pretrained(repo_id, transformer=transformer, torch_dtype=torch_dtype)

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
