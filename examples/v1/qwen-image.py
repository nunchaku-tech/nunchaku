import torch

from nunchaku.models.transformers.transformer_qwenimage import NunchakuQwenImageTransformer2DModel
from nunchaku.pipeline.pipeline_qwenimage import NunchakuQwenImagePipeline

model_name = "Qwen/Qwen-Image"

# Load the model
transformer = NunchakuQwenImageTransformer2DModel.from_pretrained(
    "Lmxyy/nunchaku-qwen-image/svdq-fp4_r32-qwen-image.safetensors"
)

# currently, you need to use this pipeline to offload the model to CPU
pipe = NunchakuQwenImagePipeline.from_pretrained("Qwen/Qwen-Image", transformer=transformer, torch_dtype=torch.bfloat16)

positive_magic = {
    "en": "Ultra HD, 4K, cinematic composition.",  # for english prompt,
    "zh": "超清，4K，电影级构图",  # for chinese prompt,
}

# Generate image
prompt = """A coffee shop entrance features a chalkboard sign reading "Qwen Coffee 😊 $2 per cup," with a neon light beside it displaying "通义千问". Next to it hangs a poster showing a beautiful Chinese woman, and beneath the poster is written "π≈3.1415926-53589793-23846264-33832795-02384197". Ultra HD, 4K, cinematic composition"""
negative_prompt = " "  # using an empty string if you do not have specific concept to remove

image = pipe(
    prompt=prompt + positive_magic["en"],
    negative_prompt=negative_prompt,
    width=1328,
    height=1328,
    num_inference_steps=50,
    true_cfg_scale=4.0,
    generator=torch.Generator().manual_seed(2333),
).images[0]

image.save("qwen-image.png")
