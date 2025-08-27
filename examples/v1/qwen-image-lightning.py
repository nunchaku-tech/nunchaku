from diffusers import FlowMatchEulerDiscreteScheduler
import torch
import math
from nunchaku.models.transformers.transformer_qwenimage import NunchakuQwenImageTransformer2DModel
from nunchaku.pipeline.pipeline_qwenimage import NunchakuQwenImagePipeline
from nunchaku.utils import get_precision

# From https://github.com/ModelTC/Qwen-Image-Lightning/blob/342260e8f5468d2f24d084ce04f55e101007118b/generate_with_diffusers.py#L82C9-L97C10
scheduler_config = {
    "base_image_seq_len": 256,
    "base_shift": math.log(3),  # We use shift=3 in distillation
    "invert_sigmas": False,
    "max_image_seq_len": 8192,
    "max_shift": math.log(3),  # We use shift=3 in distillation
    "num_train_timesteps": 1000,
    "shift": 1.0,
    "shift_terminal": None,  # set shift_terminal to None
    "stochastic_sampling": False,
    "time_shift_type": "exponential",
    "use_beta_sigmas": False,
    "use_dynamic_shifting": True,
    "use_exponential_sigmas": False,
    "use_karras_sigmas": False,
}
scheduler = FlowMatchEulerDiscreteScheduler.from_config(scheduler_config)

# Load the model
transformer = NunchakuQwenImageTransformer2DModel.from_pretrained(
    f"Lmxyy/nunchaku-qwen-image/svdq-{get_precision()}_r128-qwen-image-lightningv1.0-4steps.safetensors"
)  # you can also use the r128 or 8-step model to improve the quality
pipe = NunchakuQwenImagePipeline.from_pretrained(
    "Qwen/Qwen-Image", transformer=transformer, scheduler=scheduler, torch_dtype=torch.bfloat16
)

prompt = """Bookstore window display. A sign displays “New Arrivals This Week”. Below, a shelf tag with the text “Best-Selling Novels Here”. To the side, a colorful poster advertises “Author Meet And Greet on Saturday” with a central portrait of the author. There are four books on the bookshelf, namely “The light between worlds” “When stars are scattered” “The slient patient” “The night circus”"""
negative_prompt = " "
image = pipe(
    prompt=prompt,
    negative_prompt=negative_prompt,
    width=1024,
    height=1024,
    num_inference_steps=4,  # change the num_inference_steps to 8 if you're using the 8-step model
    true_cfg_scale=1.0,
).images[0]
image.save("qwen-image-lightning.png")
