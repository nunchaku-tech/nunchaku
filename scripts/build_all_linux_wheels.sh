#!/bin/bash
# Define the versions for Python, Torch, and CUDA
python_versions=("3.10" "3.11" "3.12" "3.13")
torch_versions=("2.5" "2.6")
cuda_versions=("12.4")

# Loop through all combinations of Python, Torch, and CUDA versions
for python_version in "${python_versions[@]}"; do
  for torch_version in "${torch_versions[@]}"; do
    # Skip building for Python 3.13 and PyTorch 2.5
    if [[ "$python_version" == "3.13" && "$torch_version" == "2.5" ]]; then
      echo "Skipping Python 3.13 with PyTorch 2.5"
      continue
    fi
    for cuda_version in "${cuda_versions[@]}"; do
      bash scripts/build_linux_wheel.sh "$python_version" "$torch_version" "$cuda_version"
    done
  done
done

bash scripts/build_linux_wheel_torch2.7_cu128.sh "3.10" "2.7" "12.8"
bash scripts/build_linux_wheel_torch2.7_cu128.sh "3.11" "2.7" "12.8"
bash scripts/build_linux_wheel_torch2.7_cu128.sh "3.12" "2.7" "12.8"
bash scripts/build_linux_wheel_torch2.7_cu128.sh "3.13" "2.7" "12.8"

bash scripts/build_linux_wheel_cu128.sh "3.10" "2.8" "12.8"
bash scripts/build_linux_wheel_cu128.sh "3.11" "2.8" "12.8"
bash scripts/build_linux_wheel_cu128.sh "3.12" "2.8" "12.8"
bash scripts/build_linux_wheel_cu128.sh "3.13" "2.8" "12.8"
