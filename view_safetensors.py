import os
import sys

from safetensors import safe_open

if __name__ == "__main__":
    input_path = sys.argv[1]
    output_path = sys.argv[2]
    tensors = {}
    with safe_open(input_path, framework="pt", device="cpu") as f:
        for k in f.keys():
            tensors[k] = f.get_tensor(k)
    keys = sorted(tensors.keys())
    output_dir = os.path.dirname(os.path.abspath(output_path))
    os.makedirs(output_dir, exist_ok=True)
    with open(output_path, "w") as f:
        for k in sorted(keys):
            v = tensors[k]
            if v.ndim == 1:
                print(k, v.abs().sum())
            f.write(f"{k}: {v.shape if v.ndim > 0 else v}\n")
