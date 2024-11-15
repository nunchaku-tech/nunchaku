import torch

from nunchaku.pipelines import flux as nunchaku_flux

pipeline = nunchaku_flux.from_pretrained(
    "black-forest-labs/FLUX.1-schnell",
    torch_dtype=torch.bfloat16,
    qmodel_path="mit-han-lab/svdquant-models/svdq-int4-flux.1-schnell.safetensors",  # download from Huggingface
).to("cuda")
image = pipeline("A cat holding a sign that says hello world", num_inference_steps=4, guidance_scale=0).images[0]
image.save("example.png")
