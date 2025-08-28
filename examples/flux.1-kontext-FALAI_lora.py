import torch
from diffusers import FluxKontextPipeline
from diffusers.utils import load_image

from nunchaku import NunchakuFluxTransformer2dModel
from nunchaku.utils import get_precision

transformer = NunchakuFluxTransformer2dModel.from_pretrained(
    f"nunchaku-tech/nunchaku-flux.1-kontext-dev/svdq-{get_precision()}_r32-flux.1-kontext-dev.safetensors"
)

pipeline = FluxKontextPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-Kontext-dev", transformer=transformer, torch_dtype=torch.bfloat16
).to("cuda")

image = load_image(
    "https://huggingface.co/datasets/nunchaku-tech/test-data/resolve/main/ComfyUI-nunchaku/inputs/monalisa.jpg"
).convert("RGB")

### LoRA Related Code ###
transformer.update_lora_params(
    "nunchaku-tech/nunchaku-test-models/relight-kontext-lora-single-caption_comfy.safetensors"
    # "linoyts/relight-kontext-lora-single-caption/relight-kontext-lora-single-caption.safetensors"
)  # Path to your LoRA safetensors, can also be a remote HuggingFace path
transformer.set_lora_strength(1)  # Your LoRA strength here
### End of LoRA Related Code ###

prompt = "neon light, city"
image = pipeline(image=image, prompt=prompt, generator=torch.Generator().manual_seed(23), guidance_scale=2.5).images[0]
image.save("flux-kontext-dev.png")
