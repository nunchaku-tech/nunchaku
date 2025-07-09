"""
Command-line interface for converting LoRA weights to Nunchaku format.

This module provides a command-line interface for converting LoRA weights from
various formats to Nunchaku format. It handles format detection, automatic
naming, and provides options for specifying quantization paths and output
locations.

The script supports:
- Automatic format detection and conversion
- Flexible output path specification
- Support for different quantization backends (int4/fp4)
- Integration with the Nunchaku conversion pipeline
"""

import argparse
import os

from .nunchaku_converter import to_nunchaku
from .utils import is_nunchaku_format

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--quant-path",
        type=str,
        help="path to the quantized model safetensor file",
        default="mit-han-lab/svdq-int4-flux.1-dev/transformer_blocks.safetensors",
    )
    parser.add_argument("--lora-path", type=str, required=True, help="path to LoRA weights safetensor file")
    parser.add_argument("--output-root", type=str, default="", help="root to the output safetensor file")
    parser.add_argument("--lora-name", type=str, default=None, help="name of the LoRA weights")
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["bfloat16", "float16"],
        help="data type of the converted weights",
    )
    args = parser.parse_args()

    if is_nunchaku_format(args.lora_path):
        print("Already in nunchaku format, no conversion needed.")
        exit(0)

    if not args.output_root:
        # output to the parent directory of the lora safetensors file
        args.output_root = os.path.dirname(args.lora_path)
    if args.lora_name is None:
        base_name = os.path.basename(args.lora_path)
        lora_name = base_name.rsplit(".", 1)[0]
        precision = "fp4" if "fp4" in args.quant_path else "int4"
        lora_name = f"svdq-{precision}-{lora_name}"
        print(f"LoRA name not provided, using {lora_name} as the LoRA name")
    else:
        lora_name = args.lora_name
    assert lora_name, "LoRA name must be provided."

    to_nunchaku(
        args.lora_path,
        args.quant_path,
        dtype=args.dtype,
        output_path=os.path.join(args.output_root, f"{lora_name}.safetensors"),
    )
