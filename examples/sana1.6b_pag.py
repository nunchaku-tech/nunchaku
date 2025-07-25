import torch
from diffusers import SanaPAGPipeline

from nunchaku import NunchakuSanaTransformer2DModel

transformer = NunchakuSanaTransformer2DModel.from_pretrained(
    "nunchaku-tech/nunchaku-sana/svdq-int4_r32-sana1.6b.safetensors", pag_layers=8
)
pipe = SanaPAGPipeline.from_pretrained(
    "Efficient-Large-Model/Sana_1600M_1024px_BF16_diffusers",
    transformer=transformer,
    variant="bf16",
    torch_dtype=torch.bfloat16,
    pag_applied_layers="transformer_blocks.8",
).to("cuda")
pipe._set_pag_attn_processor = lambda *args, **kwargs: None

pipe.text_encoder.to(torch.bfloat16)
pipe.vae.to(torch.bfloat16)

image = pipe(
    prompt="A cute 🐼 eating 🎋, ink drawing style",
    height=1024,
    width=1024,
    guidance_scale=5.0,
    pag_scale=2.0,
    num_inference_steps=20,
).images[0]
image.save("sana1.6b_pag-int4.png")
