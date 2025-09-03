#!/bin/bash
# Define the versions for Python, Torch, and CUDA
NUNCHAKU_VERSION=$1
python_versions=("3.10" "3.11" "3.12" "3.13")
torch_versions=("2.5" "2.6")
cuda_versions=("12.4" "12.8")

# Loop through all combinations of Python, Torch, and CUDA versions
for python_version in "${python_versions[@]}"; do
  for torch_version in "${torch_versions[@]}"; do
    # Skip building for Python 3.13 and PyTorch 2.5
    if [[ "$python_version" == "3.13" && "$torch_version" == "2.5" ]]; then
      echo "Skipping Python 3.13 with PyTorch 2.5"
      continue
    fi
    for cuda_version in "${cuda_versions[@]}"; do
      bash scripts/build_docker.sh "$python_version" "$torch_version" "$cuda_version" "$NUNCHAKU_VERSION"
    done
  done
done

for python_version in "${python_versions[@]}"; do
  for cuda_version in "${cuda_versions[@]}"; do
    bash scripts/build_docker_torch27.sh "$python_version" "2.7" "$cuda_version" "$NUNCHAKU_VERSION"
    bash scripts/build_docker_torch28.sh "$python_version" "2.8" "$cuda_version" "$NUNCHAKU_VERSION"
  done
done
