import torch
from diffusers.pipelines.z_image.pipeline_z_image import ZImagePipeline

from nunchaku import NunchakuZImageTransformer2DModel
from nunchaku.utils import get_precision, is_turing

if __name__ == "__main__":
    precision = get_precision()  # auto-detect your precision is 'int4' or 'fp4' based on your GPU
    rank = 128  # Use 32 for faster sampling; 256 (INT4 only) for best quality
    dtype = torch.float16 if is_turing() else torch.bfloat16  # Use float16 when Turing (20- series) GPU is used.
    transformer = NunchakuZImageTransformer2DModel.from_pretrained(
        f"nunchaku-tech/nunchaku-z-image-turbo/svdq-{precision}_r{rank}-z-image-turbo.safetensors", torch_dtype=dtype
    )

    pipe = ZImagePipeline.from_pretrained(
        "Tongyi-MAI/Z-Image-Turbo", transformer=transformer, torch_dtype=dtype, low_cpu_mem_usage=False
    )
    pipe.enable_sequential_cpu_offload() # enable sequential CPU offload for low vram
    # pipe = pipe.to("cuda") # or else comment the line above and uncomment this line to put all components to GPU

    prompt = "a young military male cooking in the kitchen for therapy"

    image = pipe(
        prompt=prompt,
        height=1024,
        width=1024,
        num_inference_steps=8,  # This actually results in 8 DiT forwards
        guidance_scale=0.0,  # Guidance should be 0 for the Turbo models
        generator=torch.Generator().manual_seed(12345),
    ).images[0]

    image.save(f"z-image-turbo-{precision}_r{rank}_{str(dtype)}.png")
