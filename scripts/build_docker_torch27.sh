#!/bin/bash
PYTHON_VERSION=$1
TORCH_VERSION=$2
CUDA_VERSION=$3
NUNCHAKU_VERSION=$4
TORCHVISION_VERSION=""
TORCHAUDIO_VERSION=""

if [ "$CUDA_VERSION" == "12.8" ]; then
  CUDA_IMAGE="12.8.1-devel-ubuntu24.04"
  echo "CUDA_VERSION is 12.8, setting CUDA_IMAGE to $CUDA_IMAGE"
elif [ "$CUDA_VERSION" == "12.4" ]; then
  CUDA_IMAGE="12.4.1-devel-ubuntu22.04"
  echo "CUDA_VERSION is 12.4, setting CUDA_IMAGE to $CUDA_IMAGE"
else
  echo "CUDA_VERSION is not 12.8 or 12.4. Exit."
  exit 2
fi

docker build -f docker/Dockerfile.torch27 --no-cache \
--build-arg PYTHON_VERSION=${PYTHON_VERSION} \
--build-arg CUDA_SHORT_VERSION=${CUDA_VERSION//.} \
--build-arg CUDA_IMAGE=${CUDA_IMAGE} \
--build-arg TORCH_VERSION=${TORCH_VERSION} \
--build-arg TORCHVISION_VERSION=${TORCHVISION_VERSION} \
--build-arg TORCHAUDIO_VERSION=${TORCHAUDIO_VERSION} \
-t lmxyy/nunchaku:${NUNCHAKU_VERSION}-py${PYTHON_VERSION}-torch${TORCH_VERSION}-cuda${CUDA_VERSION} .

docker push lmxyy/nunchaku:${NUNCHAKU_VERSION}-py${PYTHON_VERSION}-torch${TORCH_VERSION}-cuda${CUDA_VERSION}
docker rmi lmxyy/nunchaku:${NUNCHAKU_VERSION}-py${PYTHON_VERSION}-torch${TORCH_VERSION}-cuda${CUDA_VERSION}
