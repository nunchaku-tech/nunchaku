"""
CLI tool to convert LoRA weights to Nunchaku format.

**Usage:**

.. code-block:: bash

    python -m nunchaku.lora.flux.convert \\
        --lora-path composed_lora.safetensors \\
        --quant-path mit-han-lab/svdq-int4-flux.1-dev/transformer_blocks.safetensors \\
        --output-root ./converted \\
        --dtype bfloat16

**Arguments:**

- ``--lora-path``: Path to the LoRA weights safetensor file (required)
- ``--quant-path``: Path to the quantized model safetensor file (default: ``mit-han-lab/svdq-int4-flux.1-dev/transformer_blocks.safetensors``)
- ``--output-root``: Root directory for the output safetensor file (default: parent directory of the lora file)
- ``--lora-name``: Name of the LoRA weights (optional, auto-generated if not provided)
- ``--dtype``: Data type of the converted weights, either ``bfloat16`` or ``float16`` (default: ``bfloat16``)

**Main Function**

:func:`nunchaku.lora.flux.nunchaku_converter.to_nunchaku`
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
        help="Path to the quantized model safetensors file.",
        default="mit-han-lab/svdq-int4-flux.1-dev/transformer_blocks.safetensors",
    )
    parser.add_argument("--lora-path", type=str, required=True, help="Path to LoRA weights safetensors file.")
    parser.add_argument("--output-root", type=str, default="", help="Root directory for output safetensors file.")
    parser.add_argument("--lora-name", type=str, default=None, help="Name for the output LoRA weights.")
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["bfloat16", "float16"],
        help="Data type of the converted weights.",
    )
    args = parser.parse_args()

    if is_nunchaku_format(args.lora_path):
        print("Already in Nunchaku format, no conversion needed.")
        exit(0)

    if not args.output_root:
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
