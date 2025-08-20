import torch

from nunchaku.models.transformers.transformer_qwenimage import NunchakuQwenImageTransformer2DModel
from nunchaku.pipeline.pipeline_qwenimage import NunchakuQwenImagePipeline
from nunchaku.utils import get_precision

model_name = "Qwen/Qwen-Image"

# Load the model
transformer = NunchakuQwenImageTransformer2DModel.from_pretrained(
    f"nunchaku-tech/nunchaku-qwen-image/svdq-{get_precision()}_r32-qwen-image.safetensors"
)  # you can also use r128 model to improve the quality

# currently, you need to use this pipeline to offload the model to CPU
pipe = NunchakuQwenImagePipeline.from_pretrained("Qwen/Qwen-Image", transformer=transformer, torch_dtype=torch.bfloat16)

positive_magic = {
    "en": "Ultra HD, 4K, cinematic composition.",  # for english prompt,
    "zh": "超清，4K，电影级构图",  # for chinese prompt,
}

# Generate image
prompt = """Bookstore window display. A sign displays “New Arrivals This Week”. Below, a shelf tag with the text “Best-Selling Novels Here”. To the side, a colorful poster advertises “Author Meet And Greet on Saturday” with a central portrait of the author. There are four books on the bookshelf, namely “The light between worlds” “When stars are scattered” “The slient patient” “The night circus”"""
negative_prompt = " "  # using an empty string if you do not have specific concept to remove

image = pipe(
    prompt=prompt + positive_magic["en"],
    negative_prompt=negative_prompt,
    width=1664,
    height=928,
    num_inference_steps=50,
    true_cfg_scale=4.0,
).images[0]

image.save("qwen-image-r32.png")
