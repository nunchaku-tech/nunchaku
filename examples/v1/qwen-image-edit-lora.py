import torch
from diffusers import QwenImageEditPipeline
from diffusers.utils import load_image
from huggingface_hub import hf_hub_download

from nunchaku import NunchakuQwenImageTransformer2DModel
from nunchaku.lora.flux.v1.lora_flux_v2 import update_lora_params_v2
from nunchaku.utils import get_gpu_memory, get_precision

rank = 128  # you can also use rank=128 model to improve the quality

# Load the model
transformer = NunchakuQwenImageTransformer2DModel.from_pretrained(
    f"nunchaku-tech/nunchaku-qwen-image-edit/svdq-{get_precision()}_r{rank}-qwen-image-edit.safetensors"
)

pipeline = QwenImageEditPipeline.from_pretrained(
    "Qwen/Qwen-Image-Edit", transformer=transformer, torch_dtype=torch.bfloat16
)

lora_configs = {
    "lightning": {
        "path": "lightx2v/Qwen-Image-Lightning",
        "weight_name": "Qwen-Image-Lightning-4steps-V2.0-bf16.safetensors",
        "prompt": "GHIBSKY style painting, sign saying 'Flux Ghibsky'",
        "strength": 1.0,
        "inference_step": 4,
    },
    "slider": {
        "path": "ostris/qwen_image_detail_slider",
        "weight_name": "qwen_image_detail_slider.safetensors",
        "prompt": "a horse is a DJ at a night club, fish eye lens, smoke machine, lazer lights, holding a martini",
        "strength": 1.0,
        "inference_step": 50,
    },
}

selected_style = "lightning"  # Change this to test different styles
config = lora_configs[selected_style]

lora_path = hf_hub_download(repo_id=config["path"], filename=config.get("weight_name", "lora.safetensors"))
print(f"   Downloaded to: {lora_path}")

update_lora_params_v2(transformer, lora_path, strength=config["strength"])

if get_gpu_memory() > 18:
    pipeline.enable_model_cpu_offload()
else:
    # use per-layer offloading for low VRAM. This only requires 3-4GB of VRAM.
    transformer.set_offload(
        True, use_pin_memory=False, num_blocks_on_gpu=1
    )  # increase num_blocks_on_gpu if you have more VRAM
    pipeline._exclude_from_cpu_offload.append("transformer")
    pipeline.enable_sequential_cpu_offload()


image = load_image("https://huggingface.co/datasets/nunchaku-tech/test-data/resolve/main/inputs/neon_sign.png")
image = image.convert("RGB")

prompt = config["prompt"]
inference_step = config["inference_step"]

inputs = {
    "image": image,
    "prompt": prompt,
    "true_cfg_scale": 4.0,
    "negative_prompt": " ",
    "num_inference_steps": inference_step,
}

output = pipeline(**inputs)
output_image = output.images[0]
output_image.save(f"qwen-image-edit-r{rank}.png")
