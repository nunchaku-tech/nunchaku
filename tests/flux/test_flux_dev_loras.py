import pytest
import torch
from diffusers import FluxPipeline

from nunchaku import NunchakuFluxTransformer2dModel
from nunchaku.utils import get_precision, is_turing

from .utils import run_test


@pytest.mark.skipif(is_turing(), reason="Skip tests due to using Turing GPUs")
@pytest.mark.parametrize(
    "num_inference_steps,lora_name,lora_strength,cpu_offload,expected_lpips",
    [
        (25, "realism", 0.9, True, 0.136 if get_precision() == "int4" else 0.112),
        # (25, "ghibsky", 1, False, 0.186),
        # (28, "anime", 1, False, 0.284),
        (24, "sketch", 1, True, 0.291 if get_precision() == "int4" else 0.182),
        # (28, "yarn", 1, False, 0.211),
        # (25, "haunted_linework", 1, True, 0.317),
    ],
)
def test_flux_dev_loras(num_inference_steps, lora_name, lora_strength, cpu_offload, expected_lpips):
    run_test(
        precision=get_precision(),
        model_name="flux.1-dev",
        dataset_name=lora_name,
        height=1024,
        width=1024,
        num_inference_steps=num_inference_steps,
        guidance_scale=3.5,
        use_qencoder=False,
        attention_impl="nunchaku-fp16",
        cpu_offload=cpu_offload,
        lora_names=lora_name,
        lora_strengths=lora_strength,
        cache_threshold=0,
        expected_lpips=expected_lpips,
    )


# lora composition & large rank loras
@pytest.mark.skipif(is_turing(), reason="Skip tests due to using Turing GPUs")
def test_flux_dev_turbo8_ghibsky_1024x1024():
    run_test(
        precision=get_precision(),
        model_name="flux.1-dev",
        dataset_name="haunted_linework",
        height=1024,
        width=1024,
        num_inference_steps=8,
        guidance_scale=3.5,
        use_qencoder=False,
        cpu_offload=True,
        lora_names=["realism", "ghibsky", "anime", "sketch", "yarn", "haunted_linework", "turbo8"],
        lora_strengths=[0, 1, 0, 0, 0, 0, 1],
        cache_threshold=0,
        expected_lpips=0.310 if get_precision() == "int4" else 0.168,
    )


def test_kohya_lora():
    precision = get_precision()  # auto-detect your precision is 'int4' or 'fp4' based on your GPU
    transformer = NunchakuFluxTransformer2dModel.from_pretrained(
        f"mit-han-lab/nunchaku-flux.1-dev/svdq-{precision}_r32-flux.1-dev.safetensors"
    )
    pipeline = FluxPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-dev", transformer=transformer, torch_dtype=torch.bfloat16
    ).to("cuda")

    transformer.update_lora_params("mit-han-lab/nunchaku-test-models/hand_drawn_game.safetensors")
    transformer.set_lora_strength(1)

    prompt = (
        "masterful impressionism oil painting titled 'the violinist', the composition follows the rule of thirds, "
        "placing the violinist centrally in the frame. the subject is a young woman with fair skin and light blonde "
        "hair is styled in a long, flowing hairstyle with natural waves. she is dressed in an opulent, "
        "luxurious silver silk gown with a high waist and intricate gold detailing along the bodice. "
        "the gown's texture is smooth and reflective. she holds a violin under her chin, "
        "her right hand poised to play, and her left hand supporting the neck of the instrument. "
        "she wears a delicate gold necklace with small, sparkling gemstones that catch the light. "
        "her beautiful eyes focused on the viewer. the background features an elegantly furnished room "
        "with classical late 19th century decor. to the left, there is a large, ornate portrait of "
        "a man in a dark suit, set in a gilded frame. below this, a wooden desk with a closed book. "
        "to the right, a red upholstered chair with a wooden frame is partially visible. "
        "the room is bathed in natural light streaming through a window with red curtains, "
        "creating a warm, inviting atmosphere. the lighting highlights the violinist, "
        "casting soft shadows that enhance the depth and realism of the scene, highly aesthetic, "
        "harmonious colors, impressioniststrokes, "
        "<lora:style-impressionist_strokes-flux-by_daalis:1.0> <lora:image_upgrade-flux-by_zeronwo7829:1.0>"
    )

    image = pipeline(prompt, num_inference_steps=20, guidance_scale=3.5).images[0]
    image.save(f"flux.1-dev-{precision}-1.png")
