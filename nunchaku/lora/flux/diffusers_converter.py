import argparse
import os
import warnings

import torch
from diffusers.loaders import FluxLoraLoaderMixin
from diffusers.utils.state_dict_utils import convert_unet_state_dict_to_peft
from safetensors.torch import save_file

from .utils import load_state_dict_in_safetensors


def handle_skin_like_lora(input_lora: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """
    Special check if the LoRA is like https://civitai.com/models/580857?modelVersionId=1081450, which mixes various formats.
    Check whether the LoRA contains keys with all the following prefixes:
    - "diffusion_model.double_blocks."
    - "lora_unet_double_blocks_"
    - "lora_unet_single_blocks_"
    - "transformer.single_transformer_blocks."
    - "transformer.transformer_blocks."
    """
    prefixes = [
        "diffusion_model.double_blocks.",
        "lora_unet_double_blocks_",
        "lora_unet_single_blocks_",
        "transformer.single_transformer_blocks.",
        "transformer.transformer_blocks.",
    ]
    found = {prefix: False for prefix in prefixes}
    for k in input_lora.keys():
        for prefix in prefixes:
            if k.startswith(prefix):
                found[prefix] = True

    if all(found.values()):
        new_tensors = {}

        for k, v in input_lora.items():
            # process the comfyui-style LoRA tensors
            if k.startswith(("transformer.transformer_blocks.", "transformer.single_transformer_blocks.")):
                new_tensors[k] = v
                continue
            if k.startswith("diffusion_model.double_blocks."):
                new_k = k.replace("diffusion_model.double_blocks.", "transformer.transformer_blocks.")
                new_k = new_k.replace(".lora_down.", ".lora_A.")
                new_k = new_k.replace(".lora_up.", ".lora_B.")
                new_k = new_k.replace(".img_attn.", ".attn.")
                new_k = new_k.replace(".txt_attn.", ".attn.")

                if ".proj." in new_k:
                    if ".img_attn." in k:
                        new_k = new_k.replace(".proj.", ".to_out.0.")
                    else:
                        assert ".txt_attn." in k
                        new_k = new_k.replace(".proj.", ".to_add_out.")
                    assert new_k not in new_tensors
                    new_tensors[new_k] = v
                else:
                    assert ".qkv." in new_k
                    if ".lora_A." in new_k:
                        for p in ["q", "k", "v"]:
                            if ".img_attn." in k:
                                assert new_k.replace(".qkv.", f".to_{p}.") not in new_tensors
                                new_tensors[new_k.replace(".qkv.", f".to_{p}.")] = v.clone()
                            else:
                                assert ".txt_attn." in k
                                assert new_k.replace(".qkv.", f".add_{p}_proj.") not in new_tensors
                                new_tensors[new_k.replace(".qkv.", f".add_{p}_proj.")] = v.clone()
                    else:
                        assert ".lora_B." in new_k
                        for i, p in enumerate(["q", "k", "v"]):
                            assert v.shape[0] % 3 == 0
                            chunk_size = v.shape[0] // 3
                            if ".img_attn." in k:
                                assert new_k.replace(".qkv.", f".to_{p}.") not in new_tensors
                                new_tensors[new_k.replace(".qkv.", f".to_{p}.")] = v[
                                    i * chunk_size : (i + 1) * chunk_size
                                ]
                            else:
                                assert new_k.replace(".qkv.", f".add_{p}_proj.") not in new_tensors
                                assert ".txt_attn." in k
                                new_tensors[new_k.replace(".qkv.", f".add_{p}_proj.")] = v[
                                    i * chunk_size : (i + 1) * chunk_size
                                ]
                continue

            if "alpha" in k:
                continue
            new_k = k.replace("lora_down", "lora_A").replace("lora_up", "lora_B")
            if "lora_unet_double_blocks_" in k:
                new_k = new_k.replace("lora_unet_double_blocks_", "transformer.transformer_blocks.")
                if "qkv" in new_k:
                    for i, p in enumerate(["q", "k", "v"]):
                        if "lora_A" in new_k:
                            # Copy the tensor
                            new_k = new_k.replace("_img_attn_qkv", f".attn.to_{p}")
                            new_k = new_k.replace("_txt_attn_qkv", f".attn.add_{p}_proj")
                            rank = v.shape[0]
                            alpha = input_lora[k.replace("lora_down.weight", "alpha")]
                            assert new_k not in new_tensors
                            new_tensors[new_k] = v.clone() * alpha / rank
                        else:
                            assert "lora_B" in new_k
                            assert v.shape[0] % 3 == 0
                            chunk_size = v.shape[0] // 3
                            new_k = new_k.replace("_img_attn_qkv", f".attn.to_{p}")
                            new_k = new_k.replace("_txt_attn_qkv", f".attn.add_{p}_proj")
                            assert new_k not in new_tensors
                            new_tensors[new_k] = v[i * chunk_size : (i + 1) * chunk_size]
                else:
                    new_k = new_k.replace("_img_attn_proj", ".attn.to_out.0")
                    new_k = new_k.replace("_img_mlp_0", ".ff.net.0.proj")
                    new_k = new_k.replace("_img_mlp_2", ".ff.net.2")
                    new_k = new_k.replace("_img_mod_lin", ".norm1.linear")
                    new_k = new_k.replace("_txt_attn_proj", ".attn.to_add_out")
                    new_k = new_k.replace("_txt_mlp_0", ".ff_context.net.0.proj")
                    new_k = new_k.replace("_txt_mlp_2", ".ff_context.net.2")
                    new_k = new_k.replace("_txt_mod_lin", ".norm1_context.linear")
                    assert new_k not in new_tensors
                    new_tensors[new_k] = v
            else:
                assert "lora_unet_single_blocks_" in k
                new_k = new_k.replace("lora_unet_single_blocks_", "transformer.single_transformer_blocks.")
                if "linear1" in k:
                    start = 0
                    for i, p in enumerate(["q", "k", "v", "i"]):
                        if "lora_A" in new_k:
                            if p == "i":
                                new_k1 = new_k.replace("_linear1", ".proj_mlp")
                            else:
                                new_k1 = new_k.replace("_linear1", f".attn.to_{p}")
                            rank = v.shape[0]
                            alpha = input_lora[k.replace("lora_down.weight", "alpha")]
                            assert new_k1 not in new_tensors
                            new_tensors[new_k1] = v.clone() * alpha / rank
                        else:
                            if p == "i":
                                new_k1 = new_k.replace("_linear1", ".proj_mlp")
                            else:
                                new_k1 = new_k.replace("_linear1", f".attn.to_{p}")
                            chunk_size = 12288 if p == "i" else 3072
                            assert new_k1 not in new_tensors
                            new_tensors[new_k1] = v[start : start + chunk_size]
                            start += chunk_size
                else:
                    new_k = new_k.replace("_linear2", ".proj_out")
                    new_k = new_k.replace("_modulation_lin", ".norm.linear")
                    if "lora_down" in k:
                        rank = v.shape[0]
                        alpha = input_lora[k.replace("lora_down.weight", "alpha")]
                        v = v * alpha / rank
                    assert new_k not in new_tensors
                    new_tensors[new_k] = v

        return new_tensors
    else:
        return input_lora


def to_diffusers(input_lora: str | dict[str, torch.Tensor], output_path: str | None = None) -> dict[str, torch.Tensor]:
    if isinstance(input_lora, str):
        tensors = load_state_dict_in_safetensors(input_lora, device="cpu")
    else:
        tensors = {k: v for k, v in input_lora.items()}

    tensors = handle_skin_like_lora(tensors)
    with open("modified_skin.txt", "w") as f:
        # for k, v in tensors.items():
        #     f.write(f"{k}: {v.shape}\n")
        for k in sorted(tensors.keys()):
            f.write(f"{k}: {tensors[k].shape}\n")

    from safetensors.torch import save_file

    save_file(tensors, "loras/new_skin.safetensors")

    exit(0)

    ### convert the FP8 tensors to BF16
    for k, v in tensors.items():
        if v.dtype not in [torch.float64, torch.float32, torch.bfloat16, torch.float16]:
            tensors[k] = v.to(torch.bfloat16)

    new_tensors, alphas = FluxLoraLoaderMixin.lora_state_dict(tensors, return_alphas=True)
    new_tensors = convert_unet_state_dict_to_peft(new_tensors)

    if alphas is not None and len(alphas) > 0:
        warnings.warn("Alpha values are not used in the conversion to diffusers format.")

    if output_path is not None:
        output_dir = os.path.dirname(os.path.abspath(output_path))
        os.makedirs(output_dir, exist_ok=True)
        save_file(new_tensors, output_path)

    return new_tensors


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input-path", type=str, required=True, help="path to the comfyui lora safetensors file")
    parser.add_argument(
        "-o", "--output-path", type=str, required=True, help="path to the output diffusers safetensors file"
    )
    args = parser.parse_args()
    to_diffusers(args.input_path, args.output_path)
