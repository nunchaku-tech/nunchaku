import argparse

import torch
from flux_pix2pix_pipeline import FluxPix2pixTurboPipeline


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_path", type=str, help="Path to the input image")
    parser.add_argument("-o", "--output-path", type=str, help="Path to save the output image", default="output.png")
    parser.add_argument("-t", "--type", type=str, help="Input type", default="sketch", choices=["sketch", "canny"])
    parser.add_argument(
        "-m", "--model", type=str, default="pretrained/converted/sketch.safetensors", help="Path to the model"
    )
    parser.add_argument("-p", "--prompt", type=str, help="Prompt to use for the model", default="a cat")
    parser.add_argument("-a", "--alpha", type=float, default=0.4, help="Alpha value for LoRA")
    args = parser.parse_args()
    return args


def main():
    args = get_args()
    pipeline = FluxPix2pixTurboPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-schnell", torch_dtype=torch.bfloat16
    ).to("cuda")
    pipeline.load_model(args.model, alpha=args.alpha)
    img = pipeline(image=args.input_path, image_type=args.type, alpha=args.alpha, prompt=args.prompt).images[0]
    img.save(args.output_path)


if __name__ == "__main__":
    main()
