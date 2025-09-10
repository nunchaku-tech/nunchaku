import torch
from diffusers import QwenImageEditPipeline
from diffusers.utils import load_image

from nunchaku import NunchakuQwenImageTransformer2DModel
from nunchaku.utils import get_gpu_memory, get_precision

rank = 128  # you can also use rank=128 model to improve the quality

# Load the model
transformer = NunchakuQwenImageTransformer2DModel.from_pretrained(
    f"nunchaku-tech/nunchaku-qwen-image-edit/svdq-{get_precision()}_r{rank}-qwen-image-edit.safetensors"
)

pipeline = QwenImageEditPipeline.from_pretrained(
    "Qwen/Qwen-Image-Edit", transformer=transformer, torch_dtype=torch.bfloat16
)

if get_gpu_memory() > 18:
    pipeline.enable_model_cpu_offload()
else:
    # use per-layer offloading for low VRAM. This only requires 3-4GB of VRAM.
    transformer.set_offload(
        True, use_pin_memory=False, num_blocks_on_gpu=1
    )  # increase num_blocks_on_gpu if you have more VRAM
    pipeline._exclude_from_cpu_offload.append("transformer")
    pipeline.enable_sequential_cpu_offload()

image = load_image(
    "https://qwen-qwen-image-edit.hf.space/gradio_api/file=/tmp/gradio/d02be0b3422c33fc0ad3c64445959f17d3d61286c2d7dba985df3cd53d484b77/neon_sign.png"
).convert("RGB")
prompt = "change the text to read '双截棍 Qwen Image Edit is here'"
inputs = {
    "image": image,
    "prompt": prompt,
    "true_cfg_scale": 4.0,
    "negative_prompt": " ",
    "num_inference_steps": 50,
}

output = pipeline(**inputs)
output_image = output.images[0]
output_image.save(f"qwen-image-edit-r{rank}.png")
