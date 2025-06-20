# Windows Setup Guide

## Environment Setup

### 1. Install Cuda

Download and install the latest CUDA Toolkit from the official [NVIDIA CUDA Downloads](https://developer.nvidia.com/cuda-downloads?target_os=Windows&target_arch=x86_64&target_version=Server2022&target_type=exe_local). After installation, verify the installation:

```bat
nvcc --version
```

### 2. Install Visual Studio C++ Build Tools

Download from the official [Visual Studio Build Tools page](https://visualstudio.microsoft.com/visual-cpp-build-tools/). During installation, select the following workloads:

- **Desktop development with C++**
- **C++ tools for Linux development**

#### 3. Install Git

Download Git from [https://git-scm.com/downloads/win](https://git-scm.com/downloads/win) and follow the installation steps.

### 4. (Optional) Install Conda

Conda helps manage Python environments. You can install either Anaconda or Miniconda from the [official site](https://www.anaconda.com/download/success).

### 5. (Optional) Install ComfyUI

You may have some various ways to install ComfyUI. For example, I used ComfyUI CLI. Once Python is installed, you can install ComfyUI via the CLI:

```bat
pip install comfy-cli
comfy install
```

To launch ComfyUI:

```bat
comfy launch
```

## Installing Nunchaku

### Step 1: Identify Your Python Environment

To ensure correct installation, you need to find the Python interpreter used by ComfyUI. Launch ComfyUI and look for this line in the log:

```text
** Python executable: G:\ComfyuI\python\python.exe
```

Then verify the Python version and installed PyTorch version:

```bat
"G:\ComfyuI\python\python.exe" --version
"G:\ComfyuI\python\python.exe" -m pip show torch
```

### Step 2: Install PyTorch (≥2.5) if you haven’t

Install PyTorch appropriate for your setup

- **For most users**:

  ```bat
  "G:\ComfyuI\python\python.exe" -m pip install torch==2.6 torchvision==0.21 torchaudio==2.6
  ```

- **For RTX 50-series GPUs** (requires PyTorch ≥2.7 with CUDA 12.8):

  ```bat
  "G:\ComfyuI\python\python.exe" -m pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128
  ```

### Step 3: Install Nunchaku

#### Option 1: Use NunchakuWheelInstaller Node in ComfyUI

With [ComfyUI-nunchaku v0.3.2+](https://github.com/mit-han-lab/ComfyUI-nunchaku), you can install Nunchaku using the provided [workflow](https://github.com/mit-han-lab/ComfyUI-nunchaku/blob/main/example_workflows/install_wheel.json) directly in ComfyUI. This automates installation once ComfyUI-nunchaku and its dependencies are set up.

![install_wheel.png](https://huggingface.co/mit-han-lab/nunchaku-artifacts/resolve/main/ComfyUI-nunchaku/assets/install_wheel.png)

#### Option 2: Manually Install Prebuilt Wheels

You can install Nunchaku wheels from one of the following:

- [Hugging Face](https://huggingface.co/mit-han-lab/nunchaku/tree/main)
- [ModelScope](https://modelscope.cn/models/Lmxyy1999/nunchaku)
- [GitHub Releases](https://github.com/mit-han-lab/nunchaku/releases)

Example (for Python 3.11 + PyTorch 2.7):

```bat
"G:\ComfyUI\python\python.exe" -m pip install https://github.com/mit-han-lab/nunchaku/releases/download/v0.3.1/nunchaku-0.3.1+torch2.7-cp311-cp311-linux_x86_64.whl
```

To verify the installation:

```bat
"G:\ComfyuI\python\python.exe" -c "import nunchaku"
```

You can also run a test (requires a Hugging Face token for downloading the models):

```bat
"G:\ComfyuI\python\python.exe" -m huggingface-cli login
"G:\ComfyuI\python\python.exe" -m nunchaku.test
```

#### Option 3: Build Nunchaku from Source

Please use CMD instead of PowerShell for building.

- Step 1: Install Build Tools

  ```bat
  "G:\ComfyuI\python\python.exe" -m pip install ninja setuptools wheel build
  ```

- Step 2: Clone the Repository

  ```bat
  git clone https://github.com/mit-han-lab/nunchaku.git
  cd nunchaku
  git submodule init
  git submodule update
  ```

- Step 3: Set Up Visual Studio Environment

  Locate the `VsDevCmd.bat` script on your system. Example path:

  ```text
  C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\Tools\VsDevCmd.bat
  ```

  Then run:

  ```bat
  "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\Tools\VsDevCmd.bat" -startdir=none -arch=x64 -host_arch=x64
  set DISTUTILS_USE_SDK=1
  ```

- Step 4: Build Nunchaku

  ```bat
  "G:\ComfyuI\python\python.exe" setup.py develop
  ```

  Verify with:

  ```bat
  "G:\ComfyuI\python\python.exe" -c "import nunchaku"
  ```

  You can also run a test (requires a Hugging Face token for downloading the models):

  ```bat
  "G:\ComfyuI\python\python.exe" -m huggingface-cli login
  "G:\ComfyuI\python\python.exe" -m nunchaku.test
  ```

- (Optional) Step 5: Building wheel for Portable Python

  If building directly with portable Python fails, you can first build the wheel in a working Conda environment, then install the `.whl` file using your portable Python:

  ```bat
  set NUNCHAKU_INSTALL_MODE=ALL
  "G:\ComfyuI\python\python.exe" python -m build --wheel --no-isolation
  ```

## Use Nunchaku in ComfyUI

### 1. Install the Plugin

Clone the [ComfyUI-Nunchaku](https://github.com/mit-han-lab/ComfyUI-nunchaku) plugin into the `custom_nodes` folder:

```bat
cd ComfyUI/custom_nodes
git clone https://github.com/mit-han-lab/ComfyUI-nunchaku.git
```

Alternatively, install it using [ComfyUI-Manager](https://github.com/Comfy-Org/ComfyUI-Manager) or `comfy-cli`.

### 2. Download Models

- **Standard FLUX.1-dev Models**

  Start by downloading the standard [FLUX.1-dev text encoders](https://huggingface.co/comfyanonymous/flux_text_encoders/tree/main) and [VAE](https://huggingface.co/black-forest-labs/FLUX.1-dev/blob/main/ae.safetensors). You can also optionally download the original [BF16 FLUX.1-dev](https://huggingface.co/black-forest-labs/FLUX.1-dev/blob/main/flux1-dev.safetensors) model. An example command:

  ```bat
  huggingface-cli download comfyanonymous/flux_text_encoders clip_l.safetensors --local-dir models/text_encoders
  huggingface-cli download comfyanonymous/flux_text_encoders t5xxl_fp16.safetensors --local-dir models/text_encoders
  huggingface-cli download black-forest-labs/FLUX.1-schnell ae.safetensors --local-dir models/vae
  huggingface-cli download black-forest-labs/FLUX.1-dev flux1-dev.safetensors --local-dir models/diffusion_models
  ```

- **Nunchaku 4-bit FLUX.1-dev Models**

  Next, download the Nunchaku 4-bit models to `models/diffusion_models`:

  - For **50-series GPUs**, use the [FP4 model](https://huggingface.co/mit-han-lab/nunchaku-flux.1-dev/blob/main/svdq-fp4_r32-flux.1-dev.safetensors).
  - For **other GPUs**, use the [INT4 model](https://huggingface.co/mit-han-lab/nunchaku-flux.1-dev/blob/main/svdq-int4_r32-flux.1-dev.safetensors).

- **(Optional): Download Sample LoRAs**

  You can test with some sample LoRAs like [FLUX.1-Turbo](https://huggingface.co/alimama-creative/FLUX.1-Turbo-Alpha/blob/main/diffusion_pytorch_model.safetensors) and [Ghibsky](https://huggingface.co/aleksa-codes/flux-ghibsky-illustration/blob/main/lora.safetensors). Place these files in the `models/loras` directory:

  ```bat
  huggingface-cli download alimama-creative/FLUX.1-Turbo-Alpha diffusion_pytorch_model.safetensors --local-dir models/loras
  huggingface-cli download aleksa-codes/flux-ghibsky-illustration lora.safetensors --local-dir models/loras
  ```

### 3. Set Up Workflows

To use the official workflows, download them from the [ComfyUI-nunchaku](https://github.com/mit-han-lab/ComfyUI-nunchaku/tree/main/workflows) and place them in your `ComfyUI/user/default/workflows` directory. The command can be

```bat
# From the root of your ComfyUI folder
cp -r custom_nodes/ComfyUI-nunchaku/example_workflows user/default/workflows/nunchaku_examples
```

You can now launch ComfyUI and try running the example workflows.
