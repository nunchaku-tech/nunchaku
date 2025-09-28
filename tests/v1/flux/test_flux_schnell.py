# import os
# from pathlib import Path

# import torch
# from diffusers import FluxPipeline

# from nunchaku.utils import get_gpu_memory, get_precision, is_turing

# from ...utils import already_generate
# from ..utils import run_pipeline

# precision = get_precision()
# torch_dtype = torch.float16 if is_turing() else torch.bfloat16
# dtype_str = "fp16" if torch_dtype == torch.float16 else "bf16"


# class Case:
#     def __init__(
#         self,
#         rank: int = 32,
#         batch_size: int = 1,
#         width: int = 1024,
#         height: int = 1024,
#         num_inference_steps: int = 4,
#         expected_lpips: dict[str, float] = {},
#         model_name: str = "flux.1-schnell",
#         repo_id: str = "black-forest-labs/FLUX.1-schnell",
#     ):
#         self.rank = rank
#         self.batch_size = batch_size
#         self.width = width
#         self.height = height
#         self.num_inference_steps = num_inference_steps
#         self.expected_lpips = expected_lpips
#         self.model_name = model_name
#         self.repo_id = repo_id

#         self.model_path = f"nunchaku-tech/nunchaku-flux.1-schnell/svdq-{precision}_r{rank}-flux.1-schnell.safetensors"

#         ref_root = os.environ.get("NUNCHAKU_TEST_CACHE_ROOT", os.path.join("test_results", "ref"))
#         folder_name = f"w{width}h{height}t{num_inference_steps}"

#         self.save_dir_16bit = Path(ref_root) / model_name / dtype_str / folder_name
#         self.save_dir_nunchaku = (
#             Path("test_results")
#             / "nunchaku"
#             / model_name
#             / f"{precision}_r{rank}-{dtype_str}"
#             / f"{folder_name}-bs{batch_size}"
#         )


# def test_flux_schnell(case: Case):
#     batch_size = case.batch_size
#     width = case.width
#     height = case.height
#     num_inference_steps = case.num_inference_steps
#     expected_lpips = case.expected_lpips
#     rank = case.rank
#     expected_lpips = case.expected_lpips[f"{precision}-{dtype_str}"]
#     model_name = case.model_name
#     repo_id = case.repo_id

#     dataset = [
#         {
#             "prompt": "Plain light background, man to the side, light, happy, eye contact, black man aged 25  50, stylish confident man, suit, great straight hair, ",
#             "filename": "man",
#         },
#         {
#             "prompt": "3d rendering of isometric cupcake logo, pastel colors, octane rendering, unreal egine ",
#             "filename": "cupcake_logo",
#         },
#         {
#             "prompt": "character design and sketch, evil, female, drow elf, sorcerer, sharp facial features, large iris, dark blue and indigo colors, long and ornate cape, rainbowcolored gems and jewelry, leather armor, jeweled dagger, dark purple long hair, gothic ",
#             "filename": "character_design",
#         },
#         {
#             "prompt": "a hauntingly sparse drivein theater with a single red car and a single audio post. ",
#             "filename": "drivein_theater",
#         },
#     ]

#     if not already_generate(case.save_dir_16bit, len(dataset)):
#         pipeline = FluxPipeline.from_pretrained(case.repo_id, torch_dtype=torch_dtype)
#         if get_gpu_memory() > 25:
#             pipeline = pipeline.enable_model_cpu_offload()
#         else:
#             pipeline.enable_sequential_cpu_offload()
#         run_pipeline(
#             dataset=dataset,
#             batch_size=case.batch_size,
#             pipeline=pipeline,
#             save_dir=case.save_dir_16bit,
#             forward_kwargs={
#                 "height": case.height,
#                 "width": case.width,
#                 "num_inference_steps": case.num_inference_steps,
#             },
#         )
