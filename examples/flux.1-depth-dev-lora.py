import torch
from diffusers import FluxControlPipeline
from diffusers.utils import load_image
from image_gen_aux import DepthPreprocessor

from nunchaku import NunchakuFluxTransformer2dModel
from nunchaku.utils import get_precision

precision = get_precision()  # auto-detect your precision is 'int4' or 'fp4' based on your GPU
transformer = NunchakuFluxTransformer2dModel.from_pretrained(
    f"nunchaku-tech/nunchaku-flux.1-depth-dev/svdq-{precision}_r32-flux.1-depth-dev.safetensors"
)
pipe = FluxControlPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-dev", transformer=transformer, torch_dtype=torch.bfloat16
).to("cuda")

### LoRA Related Code ###
transformer.update_lora_params(
    "black-forest-labs/FLUX.1-Depth-dev-lora/flux1-depth-dev-lora.safetensors"
)  # Path to your LoRA safetensors, can also be a remote HuggingFace path
transformer.set_lora_strength(0.85)  # Your LoRA strength here
### End of LoRA Related Code ###

control_image = load_image("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/robot.png")

processor = DepthPreprocessor.from_pretrained("LiheYoung/depth-anything-large-hf")
control_image = processor(control_image)[0].convert("RGB")

image = pipe(
    prompt="A robot made of exotic candies and chocolates of different kinds. The background is filled with confetti and celebratory gifts.",  # noqa: E501
    control_image=control_image,
    height=1024,
    width=1024,
    num_inference_steps=30,
    guidance_scale=10.0,
    generator=torch.Generator().manual_seed(42),
).images[0]
image.save(f"flux.1-depth-dev-lora-{precision}.png")
