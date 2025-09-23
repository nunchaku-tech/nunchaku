import math

import torch
from diffusers import FlowMatchEulerDiscreteScheduler, QwenImagePipeline

from nunchaku.models.transformers.transformer_qwenimage import NunchakuQwenImageTransformer2DModel
from nunchaku.utils import get_gpu_memory, get_precision

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

num_inference_steps = 4  # you can also use the 8-step model to improve the quality
rank = 32  # you can also use the rank=128 model to improve the quality
model_paths = {
    4: f"nunchaku-tech/nunchaku-qwen-image/svdq-{get_precision()}_r{rank}-qwen-image-lightningv1.0-4steps.safetensors",
    8: f"nunchaku-tech/nunchaku-qwen-image/svdq-{get_precision()}_r{rank}-qwen-image-lightningv1.1-8steps.safetensors",
}

# Load the model
transformer = NunchakuQwenImageTransformer2DModel.from_pretrained(model_paths[num_inference_steps])
pipe = QwenImagePipeline.from_pretrained(
    "Qwen/Qwen-Image", transformer=transformer, scheduler=scheduler, torch_dtype=torch.bfloat16
)

if get_gpu_memory() > 18:
    pipe.enable_model_cpu_offload()
else:
    # use per-layer offloading for low VRAM. This only requires 3-4GB of VRAM.
    transformer.set_offload(
        True, use_pin_memory=False, num_blocks_on_gpu=1
    )  # increase num_blocks_on_gpu if you have more VRAM
    pipe._exclude_from_cpu_offload.append("transformer")
    pipe.enable_sequential_cpu_offload()

prompt = """Bookstore window display. A sign displays “New Arrivals This Week”. Below, a shelf tag with the text “Best-Selling Novels Here”. To the side, a colorful poster advertises “Author Meet And Greet on Saturday” with a central portrait of the author. There are four books on the bookshelf, namely “The light between worlds” “When stars are scattered” “The slient patient” “The night circus”"""
image = pipe(
    prompt=prompt,
    width=1024,
    height=1024,
    num_inference_steps=num_inference_steps,
    true_cfg_scale=1.0,
).images[0]
image.save(f"qwen-image-lightning_r{rank}.png")
