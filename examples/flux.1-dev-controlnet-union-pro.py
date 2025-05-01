import torch
from diffusers import FluxControlNetModel, FluxControlNetPipeline
from diffusers.models import FluxMultiControlNetModel
from diffusers.utils import load_image

from nunchaku import NunchakuFluxTransformer2dModel
from nunchaku.utils import get_gpu_memory, get_precision

base_model = "black-forest-labs/FLUX.1-dev"
controlnet_model_union = "Shakker-Labs/FLUX.1-dev-ControlNet-Union-Pro"

controlnet_union = FluxControlNetModel.from_pretrained(controlnet_model_union, torch_dtype=torch.bfloat16)
controlnet = FluxMultiControlNetModel([controlnet_union])  # we always recommend loading via FluxMultiControlNetModel

precision = get_precision()
need_offload = get_gpu_memory() < 36
transformer = NunchakuFluxTransformer2dModel.from_pretrained(
    f"mit-han-lab/svdq-{precision}-flux.1-dev", torch_dtype=torch.bfloat16, offload=need_offload
)
transformer.set_attention_impl("nunchaku-fp16")

pipeline = FluxControlNetPipeline.from_pretrained(
    base_model, transformer=transformer, controlnet=controlnet, torch_dtype=torch.bfloat16
)

if need_offload:
    pipeline.enable_sequential_cpu_offload()
else:
    pipeline = pipeline.to("cuda")

prompt = "A anime style girl with messy beach waves."
control_image_depth = load_image(
    "https://huggingface.co/Shakker-Labs/FLUX.1-dev-ControlNet-Union-Pro/resolve/main/assets/depth.jpg"
)
control_mode_depth = 2

control_image_canny = load_image(
    "https://huggingface.co/Shakker-Labs/FLUX.1-dev-ControlNet-Union-Pro/resolve/main/assets/canny.jpg"
)
control_mode_canny = 0

width, height = control_image_depth.size

image = pipeline(
    prompt,
    control_image=[control_image_depth, control_image_canny],
    control_mode=[control_mode_depth, control_mode_canny],
    width=width,
    height=height,
    controlnet_conditioning_scale=[0.3, 0.1],
    num_inference_steps=28,
    guidance_scale=3.5,
    generator=torch.manual_seed(233),
).images[0]


image.save(f"flux.1-dev-controlnet-union-pro-{precision}.png")
