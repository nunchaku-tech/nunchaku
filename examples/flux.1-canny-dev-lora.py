import torch
from controlnet_aux import CannyDetector
from diffusers import FluxControlPipeline
from diffusers.utils import load_image

from nunchaku import NunchakuFluxTransformer2dModel
from nunchaku.utils import get_precision

precision = get_precision()  # auto-detect your precision is 'int4' or 'fp4' based on your GPU
transformer = NunchakuFluxTransformer2dModel.from_pretrained(
    f"mit-han-lab/nunchaku-flux.1-canny-dev/svdq-{precision}_r32-flux.1-canny-dev.safetensors"
)
pipe = FluxControlPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-dev", transformer=transformer, torch_dtype=torch.bfloat16
).to("cuda")

### LoRA Related Code ###
transformer.update_lora_params(
    "black-forest-labs/FLUX.1-Canny-dev-lora/flux1-canny-dev-lora.safetensors"
)  # Path to your LoRA safetensors, can also be a remote HuggingFace path
transformer.set_lora_strength(0.85)  # Your LoRA strength here
### End of LoRA Related Code ###

prompt = (
    "A robot made of exotic candies and chocolates of different kinds. "
    "The background is filled with confetti and celebratory gifts."
)
control_image = load_image("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/robot.png")

processor = CannyDetector()
control_image = processor(
    control_image, low_threshold=50, high_threshold=200, detect_resolution=1024, image_resolution=1024
)

image = pipe(
    prompt=prompt, control_image=control_image, height=1024, width=1024, num_inference_steps=50, guidance_scale=30.0
).images[0]
image.save(f"flux.1-canny-dev-lora-{precision}.png")
