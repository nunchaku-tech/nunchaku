import torch
from diffusers import FluxPipeline

from nunchaku import NunchakuFluxTransformer2dModel

transformer,m = NunchakuFluxTransformer2dModel.from_pretrained("mit-han-lab/svdq-int4-flux.1-dev")
pipeline = FluxPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-dev", transformer=transformer, torch_dtype=torch.bfloat16
).to("cuda")

prompts = ["A cat holding a sign that says hello world",
           "A Welsh Corgi holding a sign that says hello world"]

#prompts = "A Welsh Corgi holding a sign that says hello world"
           
image = pipeline(prompts, num_inference_steps=50, guidance_scale=3.5).images[0]
image.save("flux.1-dev.png")
