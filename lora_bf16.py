import torch
from diffusers.pipelines import FluxPipeline

model_id = "black-forest-labs/FLUX.1-dev"

pipe = FluxPipeline.from_pretrained(model_id, torch_dtype=torch.bfloat16)

import ipdb

ipdb.set_trace()

pipe.load_lora_weights("./loras/skin_texture.safetensors", adapter_name="default")
pipe.set_adapters(["default"], adapter_weights=[10])

# with open("flux.txt", "w") as f:
#     f.write(str(pipe.transformer) + "\n")

pipe.enable_model_cpu_offload()

prompt = "Hyperrealistic art of  <lora:skin texture style v5:1> a closeup of a man's hand holding a colt pistol in his hand, perfect skin, detailed skin pore, realism style, perfect image, perfect body, perfect anatomy, sharp image, detailed image, high quality photography, skin texture style, solo, long sleeves, holding, weapon, holding weapon, gun, blue background, holding gun, handgun, m1911, photorealistic, hand focus, Extremely high-resolution details, photographic, realism pushed to extreme, fine texture, incredibly lifelike"
image = pipe(
    prompt=prompt,
    guidance_scale=3.5,
    height=1024,
    width=1024,
    num_inference_steps=20,
    max_sequence_length=512,
    generator=torch.Generator().manual_seed(42),
).images[0]
image.save("bf16_0.png")
