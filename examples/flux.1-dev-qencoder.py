import torch
from diffusers import FluxPipeline

from nunchaku import NunchakuFluxTransformer2dModel, NunchakuT5EncoderModel
from nunchaku.utils import get_precision

precision = get_precision()  # auto-detect your precision is 'int4' or 'fp4' based on your GPU
transformer = NunchakuFluxTransformer2dModel.from_pretrained(f"mit-han-lab/svdq-{precision}-flux.1-dev")
text_encoder_2 = NunchakuT5EncoderModel.from_pretrained("mit-han-lab/nunchaku-t5/awq-int4-flux.1-t5xxl.safetensors")
pipeline = FluxPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-dev",
    text_encoder_2=text_encoder_2,
    transformer=transformer,
    torch_dtype=torch.bfloat16,
).to("cuda")
image = pipeline("A cat holding a sign that says hello world", num_inference_steps=50, guidance_scale=3.5).images[0]
image.save(f"flux.1-dev-{precision}.png")
