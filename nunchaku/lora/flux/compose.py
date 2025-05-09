import argparse
import os

import torch
from safetensors.torch import save_file

from .diffusers_converter import to_diffusers
from .utils import is_nunchaku_format


def compose_lora(
    loras: list[tuple[str | dict[str, torch.Tensor], float]], output_path: str | None = None
) -> dict[str, torch.Tensor]:
    composed = {}
    for lora, strength in loras:
        assert not is_nunchaku_format(lora)
        lora = to_diffusers(lora)
        for k, v in list(lora.items()):
            if v.ndim == 1:
                previous_tensor = composed.get(k, None)
                if previous_tensor is None:
                    if "norm_q" in k or "norm_k" in k or "norm_added_q" in k or "norm_added_k" in k:
                        composed[k] = v
                    else:
                        composed[k] = v * strength
                else:
                    assert not ("norm_q" in k or "norm_k" in k or "norm_added_q" in k or "norm_added_k" in k)
                    composed[k] = previous_tensor + v * strength
            else:
                assert v.ndim == 2
                if ".to_q." in k or ".add_q_proj." in k:  # qkv must all exist
                    if "lora_B" in k:
                        continue

                    q_a = v
                    k_a = lora[k.replace(".to_q.", ".to_k.").replace(".add_q_proj.", ".add_k_proj.")]
                    v_a = lora[k.replace(".to_q.", ".to_v.").replace(".add_q_proj.", ".add_v_proj.")]

                    q_b = lora[k.replace("lora_A", "lora_B")]
                    k_b = lora[
                        k.replace("lora_A", "lora_B")
                        .replace(".to_q.", ".to_k.")
                        .replace(".add_q_proj.", ".add_k_proj.")
                    ]
                    v_b = lora[
                        k.replace("lora_A", "lora_B")
                        .replace(".to_q.", ".to_v.")
                        .replace(".add_q_proj.", ".add_v_proj.")
                    ]

                    assert q_a.shape[0] == k_a.shape[0] == v_a.shape[0]
                    assert q_b.shape[1] == k_b.shape[1] == v_b.shape[1]

                    if torch.isclose(q_a, k_a).all() and torch.isclose(q_a, v_a).all():
                        lora_a = q_a
                        lora_b = torch.cat((q_b, k_b, v_b), dim=0)
                    else:
                        lora_a_group = (q_a, k_a, v_a)
                        new_shape_a = [sum([_.shape[0] for _ in lora_a_group]), q_a.shape[1]]
                        lora_a = torch.zeros(new_shape_a, dtype=q_a.dtype, device=q_a.device)
                        start_dim = 0
                        for tensor in lora_a_group:
                            lora_a[start_dim : start_dim + tensor.shape[0]] = tensor
                            start_dim += tensor.shape[0]

                        lora_b_group = (q_b, k_b, v_b)
                        new_shape_b = [sum([_.shape[0] for _ in lora_b_group]), sum([_.shape[1] for _ in lora_b_group])]
                        lora_b = torch.zeros(new_shape_b, dtype=q_b.dtype, device=q_b.device)
                        start_dims = (0, 0)
                        for tensor in lora_b_group:
                            end_dims = (start_dims[0] + tensor.shape[0], start_dims[1] + tensor.shape[1])
                            lora_b[start_dims[0] : end_dims[0], start_dims[1] : end_dims[1]] = tensor
                            start_dims = end_dims

                    lora_a = lora_a * strength

                    new_k_a = k.replace(".to_q.", ".to_qkv.").replace(".add_q_proj.", ".add_qkv_proj.")
                    new_k_b = new_k_a.replace("lora_A", "lora_B")

                    for kk, vv, dim in ((new_k_a, lora_a, 0), (new_k_b, lora_b, 1)):
                        previous_lora = composed.get(kk, None)
                        composed[kk] = vv if previous_lora is None else torch.cat([previous_lora, vv], dim=dim)

                elif ".to_k." in k or ".to_v." in k or ".add_k_proj." in k or ".add_v_proj." in k:
                    continue
                else:
                    if "lora_A" in k:
                        v = v * strength

                    previous_lora = composed.get(k, None)
                    if previous_lora is None:
                        composed[k] = v
                    else:
                        if "lora_A" in k:
                            if previous_lora.shape[1] != v.shape[1]:  # flux.1-tools LoRA compatibility
                                assert "x_embedder" in k
                                expanded_size = max(previous_lora.shape[1], v.shape[1])
                                if expanded_size > previous_lora.shape[1]:
                                    expanded_previous_lora = torch.zeros(
                                        (previous_lora.shape[0], expanded_size),
                                        device=previous_lora.device,
                                        dtype=previous_lora.dtype,
                                    )
                                    expanded_previous_lora[:, : previous_lora.shape[1]] = previous_lora
                                else:
                                    expanded_previous_lora = previous_lora
                                if expanded_size > v.shape[1]:
                                    expanded_v = torch.zeros(
                                        (v.shape[0], expanded_size), device=v.device, dtype=v.dtype
                                    )
                                    expanded_v[:, : v.shape[1]] = v
                                else:
                                    expanded_v = v
                                composed[k] = torch.cat([expanded_previous_lora, expanded_v], dim=0)
                            else:
                                composed[k] = torch.cat([previous_lora, v], dim=0)
                        else:
                            composed[k] = torch.cat([previous_lora, v], dim=1)

                    composed[k] = (
                        v if previous_lora is None else torch.cat([previous_lora, v], dim=0 if "lora_A" in k else 1)
                    )
    if output_path is not None:
        output_dir = os.path.dirname(os.path.abspath(output_path))
        os.makedirs(output_dir, exist_ok=True)
        save_file(composed, output_path)
    return composed


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i", "--input-paths", type=str, nargs="*", required=True, help="paths to the lora safetensors files"
    )
    parser.add_argument("-s", "--strengths", type=float, nargs="*", required=True, help="strengths for each lora")
    parser.add_argument("-o", "--output-path", type=str, required=True, help="path to the output safetensors file")
    args = parser.parse_args()
    assert len(args.input_paths) == len(args.strengths)
    compose_lora(list(zip(args.input_paths, args.strengths)), args.output_path)
