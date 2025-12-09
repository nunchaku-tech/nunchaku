import torch
from diffusers import QwenImagePipeline
from huggingface_hub import hf_hub_download

from nunchaku.lora.flux.v1.lora_flux_v2 import update_lora_params_v2
from nunchaku.models.transformers.transformer_qwenimage import NunchakuQwenImageTransformer2DModel
from nunchaku.utils import get_gpu_memory, get_precision

model_name = "Qwen/Qwen-Image"
rank = 32  # you can also use rank=128 model to improve the quality

# Load the model
transformer = NunchakuQwenImageTransformer2DModel.from_pretrained(
    f"nunchaku-tech/nunchaku-qwen-image/svdq-{get_precision()}_r{rank}-qwen-image.safetensors"
)

lora_configs = {
    "lightning": {
        "path": "lightx2v/Qwen-Image-Lightning",
        "weight_name": "Qwen-Image-Lightning-4steps-V2.0-bf16.safetensors",
        "prompt": "GHIBSKY style painting, sign saying 'Flux Ghibsky'",
        "negative_prompt": " ",
        "strength": 1.0,
        "inference_step": 4,
    },
    "slider": {
        "path": "ostris/qwen_image_detail_slider",
        "weight_name": "qwen_image_detail_slider.safetensors",
        "prompt": "a horse is a DJ at a night club, fish eye lens, smoke machine, lazer lights, holding a martini",
        "negative_prompt": " ",
        "strength": 1.0,
        "inference_step": 50,
    },
    "Qwen-Image-Studio-Realism": {
        "path": "prithivMLmods/Qwen-Image-Studio-Realism",
        "weight_name": "qwen-studio-realism.safetensors",
        "prompt": "Studio Realism, a close-up shot of a young womans face features a neutral expression. The womans eyes are a deep brown, her hair is a dark brown, and her eyebrows are a lighter shade of brown. She is wearing a black and yellow t-shirt, with a cream collar around her neck. Her ear is adorned with a gold earring, adding a touch of shine to her face. The backdrop is a stark white, and the womans hair is pulled back in a ponytail.",
        "negative_prompt": " ",
        "strength": 1.0,
        "inference_step": 20,
    },
    "qwen-image-realism-lora": {
        "path": "flymy-ai/qwen-image-realism-lora",
        "weight_name": "flymy_realism.safetensors",
        "prompt": """Super Realism portrait of a teenager woman of African descent, serene calmness, arms crossed, illuminated by dramatic studio lighting, sunlit park in the background, adorned with delicate jewelry, three-quarter view, sun-kissed skin with natural imperfections, loose shoulder-length curls, slightly squinting eyes, environmental street portrait with text "FLYMY AI" on t-shirt.""",
        "negative_prompt": " ",
        "strength": 1.0,
        "inference_step": 20,
    },
    "qwen-image-modern-anime-lora": {
        "path": "alfredplpl/qwen-image-modern-anime-lora",
        "weight_name": "lora.safetensors",
        "prompt": "Japanese modern anime style, an upper body shot of an woman standing on the rainy street. She watches the sky.",
        "negative_prompt": "photo, cg, 3d",
        "strength": 1.0,
        "inference_step": 20,
    },
}

selected_style = "qwen-image-modern-anime-lora"  # Change this to test different styles
# selected_style = "lightning"
config = lora_configs[selected_style]

lora_path = hf_hub_download(repo_id=config["path"], filename=config.get("weight_name", "lora.safetensors"))
print(f"   Downloaded to: {lora_path}")

update_lora_params_v2(transformer, lora_path, strength=config["strength"])

# currently, you need to use this pipeline to offload the model to CPU
pipe = QwenImagePipeline.from_pretrained("Qwen/Qwen-Image", transformer=transformer, torch_dtype=torch.bfloat16)

if get_gpu_memory() > 18:
    pipe.enable_model_cpu_offload()
else:
    # use per-layer offloading for low VRAM. This only requires 3-4GB of VRAM.
    transformer.set_offload(True)
    pipe._exclude_from_cpu_offload.append("transformer")
    pipe.enable_sequential_cpu_offload()

positive_magic = {
    "en": "Ultra HD, 4K, cinematic composition.",  # for english prompt,
    "zh": "超清，4K，电影级构图",  # for chinese prompt,
}

# Generate image
prompt = config["prompt"]
inference_step = config["inference_step"]

negative_prompt = config["negative_prompt"]  # using an empty string if you do not have specific concept to remove

image = pipe(
    prompt=prompt,  # + positive_magic["en"],
    negative_prompt=negative_prompt,
    width=1024,
    height=1024,
    num_inference_steps=inference_step,
    true_cfg_scale=2.5,
).images[0]

image.save(f"qwen-image-r{rank}.png")
