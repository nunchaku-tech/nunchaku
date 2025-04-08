import torch
from diffusers import FluxPipeline

from nunchaku import NunchakuFluxTransformer2dModel

transformer,m = NunchakuFluxTransformer2dModel.from_pretrained("mit-han-lab/svdq-int4-flux.1-dev")
pipeline = FluxPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-dev", transformer=transformer, torch_dtype=torch.bfloat16
).to("cuda")

prompts = ["A very cute giant panda is watching a soccer game while drinking beer. Very realistic photo.",
           "A Welsh Corgi holding a sign that says hello world. Very realistic photo."]

#prompts = "A Welsh Corgi holding a sign that says hello world"
           
image = pipeline(prompts, num_inference_steps=50, guidance_scale=3.5).images

image[0].save("results/flux.1-dev.png")
image[1].save("results/flux.2-dev.png")

