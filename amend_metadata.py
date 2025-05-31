import json
from pathlib import Path

import yaml
from safetensors.torch import save_file
from tqdm import tqdm

from nunchaku.utils import load_state_dict_in_safetensors


def load_yaml(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as file:
        data = yaml.safe_load(file)
    return data


if __name__ == "__main__":
    # data = load_yaml("nunchaku_models.yaml")
    # for model in tqdm(data["diffusion_models"]):
    #     for precision in ["int4", "fp4"]:
    #         repo_id = model["repo_id"]
    #         filename = model["filename"].format(precision=precision)
    #         sd, metadata = load_state_dict_in_safetensors(Path(repo_id) / filename, return_metadata=True)
    #         metadata["model_class"] = "NunchakuFluxTransformer2dModel"
    #         quantization_config = {
    #             "method": "svdquant",
    #             "weight": {
    #                 "dtype": "fp4_e2m1_all" if precision == "fp4" else "int4",
    #                 "scale_dtype": [None, "fp8_e4m3_nan"] if precision == "fp4" else None,
    #                 "group_size": 16 if precision == "fp4" else 64,
    #             },
    #             "activation": {
    #                 "dtype": "fp4_e2m1_all" if precision == "fp4" else "int4",
    #                 "scale_dtype": "fp8_e4m3_nan" if precision == "fp4" else None,
    #                 "group_size": 16 if precision == "fp4" else 64,
    #             },
    #         }
    #         metadata["quantization_config"] = json.dumps(quantization_config)
    #         output_dir = Path("nunchaku-models") / Path(repo_id).name
    #         output_dir.mkdir(parents=True, exist_ok=True)
    #         save_file(sd, output_dir / filename, metadata=metadata)
    # sd, metadata = load_state_dict_in_safetensors(
    #     "mit-han-lab/nunchaku-t5/awq-int4-flux.1-t5xxl.safetensors", return_metadata=True
    # )
    # metadata["model_class"] = "NunchakuT5EncoderModel"
    # quantization_config = {"method": "awq", "weight": {"dtype": "int4", "scale_dtype": None, "group_size": 128}}
    # output_dir = Path("nunchaku-models") / "nunchaku-t5"
    # output_dir.mkdir(parents=True, exist_ok=True)
    # save_file(sd, output_dir / "awq-int4-flux.1-t5xxl.safetensors", metadata=metadata)
    sd, metadata = load_state_dict_in_safetensors(
        "mit-han-lab/nunchaku-sana/svdq-int4_r32-sana1.6b.safetensors", return_metadata=True
    )
    metadata["model_class"] = "NunchakuSanaTransformer2DModel"
    precision = "int4"
    quantization_config = {
        "method": "svdquant",
        "weight": {
            "dtype": "fp4_e2m1_all" if precision == "fp4" else "int4",
            "scale_dtype": [None, "fp8_e4m3_nan"] if precision == "fp4" else None,
            "group_size": 16 if precision == "fp4" else 64,
        },
        "activation": {
            "dtype": "fp4_e2m1_all" if precision == "fp4" else "int4",
            "scale_dtype": "fp8_e4m3_nan" if precision == "fp4" else None,
            "group_size": 16 if precision == "fp4" else 64,
        },
    }
    output_dir = Path("nunchaku-models") / "nunchaku-sana"
    output_dir.mkdir(parents=True, exist_ok=True)
    save_file(sd, output_dir / "svdq-int4_r32-sana1.6b.safetensors", metadata=metadata)
