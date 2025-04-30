#!/bin/bash
PYTHON_VERSION=$1
TORCH_VERSION=$2
CUDA_VERSION=$3
NUNCHAKU_VERSION=$4

# Check if TORCH_VERSION is 2.5 or 2.6 and set the corresponding versions for TORCHVISION and TORCHAUDIO
if [ "$TORCH_VERSION" == "2.5" ]; then
  TORCHVISION_VERSION="0.20"
  TORCHAUDIO_VERSION="2.5"
  echo "TORCH_VERSION is 2.5, setting TORCHVISION_VERSION to $TORCHVISION_VERSION and TORCHAUDIO_VERSION to $TORCHAUDIO_VERSION"
elif [ "$TORCH_VERSION" == "2.6" ]; then
  TORCHVISION_VERSION="0.21"
  TORCHAUDIO_VERSION="2.6"
  echo "TORCH_VERSION is 2.6, setting TORCHVISION_VERSION to $TORCHVISION_VERSION and TORCHAUDIO_VERSION to $TORCHAUDIO_VERSION"
else
  echo "TORCH_VERSION is not 2.5 or 2.6. Exit."
  exit 2
fi

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

docker build -f docker/Dockerfile --no-cache \
--build-arg PYTHON_VERSION=${PYTHON_VERSION} \
--build-arg CUDA_SHORT_VERSION=${CUDA_VERSION//.} \
--build-arg CUDA_IMAGE=${CUDA_IMAGE} \
--build-arg TORCH_VERSION=${TORCH_VERSION} \
--build-arg TORCHVISION_VERSION=${TORCHVISION_VERSION} \
--build-arg TORCHAUDIO_VERSION=${TORCHAUDIO_VERSION} \
-t lmxyy/nunchaku:${NUNCHAKU_VERSION}-py${PYTHON_VERSION}-torch${TORCH_VERSION}-cuda${CUDA_VERSION} .

docker push lmxyy/nunchaku:${NUNCHAKU_VERSION}-py${PYTHON_VERSION}-torch${TORCH_VERSION}-cuda${CUDA_VERSION}
docker rmi lmxyy/nunchaku:${NUNCHAKU_VERSION}-py${PYTHON_VERSION}-torch${TORCH_VERSION}-cuda${CUDA_VERSION}
