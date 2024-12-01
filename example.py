import torch
import time

from nunchaku.pipelines import flux as nunchaku_flux

# Initialize the pipeline
pipeline = nunchaku_flux.from_pretrained(
    "black-forest-labs/FLUX.1-schnell",
    cache_dir="model-cache",
    torch_dtype=torch.bfloat16,
    qmodel_path="mit-han-lab/svdquant-models/svdq-int4-flux.1-schnell.safetensors",  # download from Huggingface
).to("cuda")

# List of prompts
prompts = [
    "A cat holding a sign that says hello world",
    "A dog playing with a ball in the park",
    "A beautiful sunset over the mountains",
    "A futuristic cityscape at night",
    "A group of people having a picnic in the park"
]

# Generate images and calculate average time per image
total_time = 0
for i, prompt in enumerate(prompts):
    start_time = time.time()
    image = pipeline(prompt, num_inference_steps=4, guidance_scale=0).images[0]
    end_time = time.time()
    image.save(f"example_{i}.png")
    total_time += end_time - start_time

average_time_per_image = total_time / len(prompts)
print(f"Average time per image: {average_time_per_image:.2f} seconds")
