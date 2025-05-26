import torch
from safetensors.torch import save_file

from nunchaku.utils import load_state_dict_in_safetensors

if __name__ == "__main__":
    sd = load_state_dict_in_safetensors("loras/removalV2.safetensors")
    new_sd = {}
    for k, v in sd.items():
        if ".single_transformer_blocks." in k:
            new_sd[k] = v
        else:
            new_sd[k] = torch.zeros_like(v)
    save_file(new_sd, "loras/removalV2-single.safetensors")
