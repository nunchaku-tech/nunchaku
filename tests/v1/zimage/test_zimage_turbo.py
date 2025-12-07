import gc
import os
from pathlib import Path

import pytest
import torch
from diffusers import ZImagePipeline

from nunchaku import NunchakuZImageTransformer2DModel
from nunchaku.utils import get_precision, is_turing

from ...utils import already_generate, compute_lpips
from ..utils import run_pipeline

precision = get_precision()
torch_dtype = torch.float16 if is_turing() else torch.bfloat16
dtype_str = "fp16" if torch_dtype == torch.float16 else "bf16"


@pytest.mark.parametrize(
    "rank,expected_lpips",
    [
        (32, {"int4-bf16": 0.37, "fp4-bf16": 0.33}),
        (128, {"int4-bf16": 0.34, "fp4-bf16": 0.3}),
    ],
)
def test_zimage_turbo(rank: int, expected_lpips: dict[str, float]):
    model_name = "z-image-turbo"
    batch_size = 1
    width = 1024
    height = 1024
    num_inference_steps = 9
    guidance_scale = 0.0

    ref_root = os.environ.get("NUNCHAKU_TEST_CACHE_ROOT", os.path.join("test_results", "ref"))
    folder_name = f"w{width}h{height}t{num_inference_steps}g{guidance_scale}"
    save_dir_16bit = Path(ref_root) / model_name / dtype_str / folder_name

    repo_id = "Tongyi-MAI/Z-Image-Turbo"
    dataset = [
        {
            "prompt": "Table Mountain, South Africa, covered in clouds on a hot, bright summers day. Use a Sony alpha 1 to capture a lot of details. use a 100mm lense. Use aperture F 1.2 to make the mountain standout. Photo taken from Blouberg Beach ",
            "negative_prompt": " ",
            "filename": "landscape",
        },
        {
            "prompt": "A futuristic tibetan god wearing ornate robes embroidered with an infinitely complex gold mandala, very old man, white beard, character concept  full body,  a weathered magical Gate with glowing runes carved into a granite cliff face, stairs lined with cherry blossom trees and jacaranda trees the entrance of goddess, ornate, beautiful, weapons, lush, nature, low angle, Protoctist style Zeng Chuanxing, widescreen, anamorphic 2 39, gold , intricate detail, hyper realistic, low angle  Symmetrical, epic scale  Cinematic, Color Grading, F 2. 8, 8K, Ultra  HD, AMOLED, Ray Tracing Global Illumination, spiritual vibes, Transparent, Translucent, Iridescent, Ray Tracing Reflections, Harris Shutter, De  Noise, VFX, SFX, anamorphic 2 39 ",
            "negative_prompt": " ",
            "filename": "art",
        },
        {
            "prompt": "年轻的中国女子，身着红色汉服，绣工细密。妆容精致无瑕，额间点着红色花钿。发髻高盘而华丽，簪着金色凤凰头饰、红花与串珠。右手持一柄圆形折扇，扇面绘有仕女、树木与鸟。左手微抬，掌上方悬着一盏霓虹闪电形灯（⚡️），散发明亮的黄色光辉。背景是柔和灯光下的户外夜景，层叠的宝塔（西安大雁塔）成剪影状隐现，远处彩光朦胧。",
            "negative_prompt": " ",
            "filename": "portrait_chinese_prompt",
        },
    ]

    if not already_generate(save_dir_16bit, len(dataset)):
        pipe = ZImagePipeline.from_pretrained(repo_id, torch_dtype=torch_dtype).to("cuda")
        run_pipeline(
            dataset=dataset,
            batch_size=1,
            pipeline=pipe,
            save_dir=save_dir_16bit,
            forward_kwargs={
                "width": width,
                "height": height,
                "num_inference_steps": num_inference_steps,
                "guidance_scale": guidance_scale,
            },
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
    path = (f"nunchaku-tech/nunchaku-z-image-turbo/svdq-{precision}_r{rank}-z-image-turbo.safetensors",)
    transformer = NunchakuZImageTransformer2DModel.from_pretrained(path, torch_dtype=torch_dtype)

    pipe = ZImagePipeline.from_pretrained(repo_id, transformer=transformer, torch_dtype=torch_dtype).to("cuda")

    run_pipeline(
        dataset=dataset,
        batch_size=batch_size,
        pipeline=pipe,
        save_dir=save_dir_nunchaku,
        forward_kwargs={
            "width": width,
            "height": height,
            "num_inference_steps": num_inference_steps,
            "guidance_scale": guidance_scale,
        },
    )
    del transformer
    del pipe
    gc.collect()
    torch.cuda.empty_cache()

    lpips = compute_lpips(save_dir_16bit, save_dir_nunchaku)
    print(f"lpips: {lpips}")
    assert lpips < expected_lpips[f"{precision}-{dtype_str}"] * 1.10
