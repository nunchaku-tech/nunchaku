import torch
from diffusers import QwenImagePipeline

from nunchaku.models.transformers.transformer_qwenimage import NunchakuQwenImageTransformer2DModel
from nunchaku.utils import get_gpu_memory, get_precision
from nunchaku.lora.flux.v1.lora_flux_v2 import update_lora_params_v2
from huggingface_hub import hf_hub_download
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
        "strength": 1.0,
    },
}
selected_style = "lightning"  # Change this to test different styles
config = lora_configs[selected_style]

lora_path = hf_hub_download(repo_id=config["path"], filename=config.get("weight_name", "lora.safetensors"))
print(f"   Downloaded to: {lora_path}")

#update_lora_params_v2(transformer, lora_path, strength=config["strength"])

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
prompt = """Bookstore window display. A sign displays “New Arrivals This Week”. Below, a shelf tag with the text “Best-Selling Novels Here”. To the side, a colorful poster advertises “Author Meet And Greet on Saturday” with a central portrait of the author. There are four books on the bookshelf, namely “The light between worlds” “When stars are scattered” “The slient patient” “The night circus”"""
negative_prompt = " "  # using an empty string if you do not have specific concept to remove

image = pipe(
    prompt=prompt + positive_magic["en"],
    negative_prompt=negative_prompt,
    width=1664,
    height=928,
    num_inference_steps=4,
    true_cfg_scale=4.0,
).images[0]

image.save(f"qwen-image-r{rank}.png")
