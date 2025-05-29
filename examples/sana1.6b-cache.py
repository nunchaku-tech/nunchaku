import torch
from diffusers import SanaPipeline

from nunchaku import NunchakuSanaTransformer2DModel
from nunchaku.caching.diffusers_adapters import apply_cache_on_pipe

transformer = NunchakuSanaTransformer2DModel.from_pretrained(
    "mit-han-lab/nunchaku-sana/svdq-int4_r32-sana1.6b.safetensors"
)
pipe = SanaPipeline.from_pretrained(
    "Efficient-Large-Model/Sana_1600M_1024px_BF16_diffusers",
    transformer=transformer,
    variant="bf16",
    torch_dtype=torch.bfloat16,
).to("cuda")
pipe.vae.to(torch.bfloat16)
pipe.text_encoder.to(torch.bfloat16)

apply_cache_on_pipe(pipe, residual_diff_threshold=0.25)

# WarmUp

prompt = "A cute 🐼 eating 🎋, ink drawing style"
image = pipe(
    prompt=prompt,
    height=1024,
    width=1024,
    guidance_scale=4.5,
    num_inference_steps=20,
    generator=torch.Generator().manual_seed(42),
).images[0]

image.save("sana1.6b-int4.png")
