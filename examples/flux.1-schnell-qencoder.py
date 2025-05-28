import torch
from diffusers import FluxPipeline

from nunchaku import NunchakuFluxTransformer2dModel
from nunchaku.models.text_encoders.t5_encoder import NunchakuT5EncoderModel
from nunchaku.utils import get_precision


def main():
    pipeline_init_kwargs = {}
    text_encoder_2 = NunchakuT5EncoderModel.from_pretrained("mit-han-lab/svdq-flux.1-t5")
    pipeline_init_kwargs["text_encoder_2"] = text_encoder_2
    precision = get_precision()  # auto-detect your precision is 'int4' or 'fp4' based on your GPU
    transformer = NunchakuFluxTransformer2dModel.from_pretrained(f"mit-han-lab/svdq-{precision}-flux.1-schnell")
    pipeline = FluxPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-schnell", transformer=transformer, torch_dtype=torch.bfloat16, **pipeline_init_kwargs
    ).to("cuda")
    image = pipeline(
        "A cat holding a sign that says hello world", width=1024, height=1024, num_inference_steps=4, guidance_scale=0
    ).images[0]
    image.save(f"flux.1-schnell-qencoder-{precision}.png")


if __name__ == "__main__":
    main()
