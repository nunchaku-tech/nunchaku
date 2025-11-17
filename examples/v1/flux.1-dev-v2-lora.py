import torch
from diffusers import FluxPipeline
from huggingface_hub import hf_hub_download

from nunchaku.lora.flux.v1.lora_flux_v2 import update_lora_params_v2
from nunchaku.models.transformers.transformer_flux_v2 import NunchakuFluxTransformer2DModelV2
from nunchaku.utils import get_precision

precision = get_precision()

lora_configs = {
    "ghibli": {
        "path": "aleksa-codes/flux-ghibsky-illustration",
        "weight_name": "lora.safetensors",
        "prompt": "GHIBSKY style painting, sign saying 'Flux Ghibsky'",
        "strength": 1.0,
    },
    "anime": {
        "path": "alvdansen/sonny-anime-fixed",
        "weight_name": "araminta_k_sonnyanime_fluxd_fixed.safetensors",
        "prompt": "a cute creature, nm22 style",
        "strength": 1.0,
    },
}

# Select LoRA style
selected_style = "anime"  # Change this to test different styles
config = lora_configs[selected_style]

# Load V2 transformer with quantization
print(f"\nLoading Flux V2 model with {precision} precision...")
transformer = NunchakuFluxTransformer2DModelV2.from_pretrained(
    f"nunchaku-tech/nunchaku-flux.1-dev/svdq-{precision}_r32-flux.1-dev.safetensors",
    device="cuda",
    torch_dtype=torch.bfloat16,
)

lora_path = hf_hub_download(repo_id=config["path"], filename=config.get("weight_name", "lora.safetensors"))
print(f"   Downloaded to: {lora_path}")

update_lora_params_v2(transformer, lora_path, strength=config["strength"])

print("\nCreating Flux pipeline...")
pipe = FluxPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-dev", transformer=transformer, torch_dtype=torch.bfloat16
).to("cuda")

# Generate image with LoRA
prompt = config["prompt"]
print(f"\nGenerating image with prompt: {prompt}")

image = pipe(
    prompt,
    height=1024,
    width=1024,
    guidance_scale=3.5,
    num_inference_steps=25,
    generator=torch.Generator("cuda").manual_seed(0),
).images[0]

# Save image
output_path = f"output_{selected_style}_v2_runtime.png"
image.save(output_path)
print(f"Image saved to {output_path}")
