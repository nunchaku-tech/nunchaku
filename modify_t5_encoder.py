from nunchaku.utils import load_state_dict_in_safetensors

if __name__ == "__main__":
    sd = load_state_dict_in_safetensors(
        "mit-han-lab/nunchaku-t5/awq-int4-flux.1-t5xxl.safetensors", return_metadata=True
    )
    print(sd["__metadata__"])
