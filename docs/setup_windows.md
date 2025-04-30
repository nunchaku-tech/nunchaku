# Nunchaku Setup Guide (Windows)

# Environment Setup

## 1. Install Cuda

Download and install the latest CUDA Toolkit from the official [NVIDIA CUDA Downloads](https://developer.nvidia.com/cuda-downloads?target_os=Windows&target_arch=x86_64&target_version=Server2022&target_type=exe_local). After installation, verify the installation:

```bash
nvcc --version
```

## 2. Install Visual Studio C++ Build Tools

Download from the official [Visual Studio Build Tools page](https://visualstudio.microsoft.com/visual-cpp-build-tools/). During installation, select the following workloads:

- **Desktop development with C++**
- **C++ tools for Linux development**

### 3. Git

Download Git from [https://git-scm.com/downloads/win](https://git-scm.com/downloads/win) and follow the installation steps.

## 4. (Optional) Install Conda

Conda helps manage Python environments. You can install either Anaconda or Miniconda from the [official site](https://www.anaconda.com/download/success).

## 5. (Optional) Installing ComfyUI

You may have some various ways to install ComfyUI. For example, I used ComfyUI CLI. Once Python is installed, you can install ComfyUI via the CLI:

```shell
pip install comfy-cli
comfy-cli install
```

To launch ComfyUI:

```shell
comfy-cli launch
```

# Installing Nunchaku

## Step 1: Identify Your Python Environment

To ensure correct installation, you need to find the Python interpreter used by ComfyUI. Launch ComfyUI and look for this line in the log:

```bash
** Python executable: G:\ComfyuI\python\python.exe
```

Then verify the Python version and installed PyTorch version:

```bash
"G:\ComfyuI\python\python.exe" --version
"G:\ComfyuI\python\python.exe" -m pip show torch
```

## Step 2: Install PyTorch (≥2.5) if you haven’t

Install PyTorch appropriate for your setup

- **For most users**:

    ```bash
    "G:\ComfyuI\python\python.exe" -m pip install torch==2.6 torchvision==0.21 torchaudio==2.6
    ```

- **For RTX 50-series GPUs** (requires PyTorch ≥2.7 with CUDA 12.8):

    ```bash
    "G:\ComfyuI\python\python.exe" -m pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128
    ```


## Step 3: Install Nunchaku

### Prebuilt Wheels

You can install Nunchaku wheels from one of the following:

- [Hugging Face](https://huggingface.co/mit-han-lab/nunchaku/tree/main)
- [ModelScope](https://modelscope.cn/models/Lmxyy1999/nunchaku)
- [GitHub Releases](https://github.com/mit-han-lab/nunchaku/releases)

Example (for Python 3.10 + PyTorch 2.6):

```bash
"G:\ComfyuI\python\python.exe" -m pip install https://huggingface.co/mit-han-lab/nunchaku/resolve/main/nunchaku-0.2.0+torch2.6-cp310-cp310-win_amd64.whl
```

To verify the installation:

```bash
"G:\ComfyuI\python\python.exe" -c "import nunchaku"
```

You can also run a test (requires a Hugging Face token for downloading the models):

```bash
"G:\ComfyuI\python\python.exe" -m huggingface-cli login
"G:\ComfyuI\python\python.exe" -m nunchaku.test
```

### (Alternative) Build Nunchaku from Source

Please use CMD instead of PowerShell for building.

- Step 1: Install Build Tools

    ```bash
    C:\Users\muyang\miniconda3\envs\comfyui\python.exe
    "G:\ComfyuI\python\python.exe" -m pip install ninja setuptools wheel build
    ```

- Step 2: Clone the Repository

    ```bash
    git clone https://github.com/mit-han-lab/nunchaku.git
    cd nunchaku
    git submodule init
    git submodule update
    ```

- Step 3: Set Up Visual Studio Environment

    Locate the `VsDevCmd.bat` script on your system. Example path:

    ```
    C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\Tools\VsDevCmd.bat
    ```

    Then run:

    ```bash
    "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\Tools\VsDevCmd.bat" -startdir=none -arch=x64 -host_arch=x64
    set DISTUTILS_USE_SDK=1
    ```

- Step 4: Build Nunchaku

    ```bash
    "G:\ComfyuI\python\python.exe" setup.py develop
    ```

    Verify with:

    ```bash
    "G:\ComfyuI\python\python.exe" -c "import nunchaku"
    ```

    You can also run a test (requires a Hugging Face token for downloading the models):

    ```bash
    "G:\ComfyuI\python\python.exe" -m huggingface-cli login
    "G:\ComfyuI\python\python.exe" -m nunchaku.test
    ```

- (Optional) Step 5: Building wheel for Portable Python

    If building directly with portable Python fails, you can first build the wheel in a working Conda environment, then install the `.whl` file using your portable Python:

    ```shell
    set NUNCHAKU_INSTALL_MODE=ALL
    "G:\ComfyuI\python\python.exe" python -m build --wheel --no-isolation
    ```

# Use Nunchaku in ComfyUI

## 1. Install the Plugin

Clone the [ComfyUI-Nunchaku](https://github.com/mit-han-lab/ComfyUI-nunchaku) plugin into the `custom_nodes` folder:

```
cd ComfyUI/custom_nodes
git clone https://github.com/mit-han-lab/ComfyUI-nunchaku.git
```

Alternatively, install using [ComfyUI-Manager](https://github.com/Comfy-Org/ComfyUI-Manager) or `comfy-cli`.

## 2. Download Models

- **Standard FLUX.1-dev Models**

    Start by downloading the standard [FLUX.1-dev text encoders](https://huggingface.co/comfyanonymous/flux_text_encoders/tree/main) and [VAE](https://huggingface.co/black-forest-labs/FLUX.1-dev/blob/main/ae.safetensors). You can also optionally download the original [BF16 FLUX.1-dev](https://huggingface.co/black-forest-labs/FLUX.1-dev/blob/main/flux1-dev.safetensors) model. An example command:

    ```bash
    huggingface-cli download comfyanonymous/flux_text_encoders clip_l.safetensors --local-dir models/text_encoders
    huggingface-cli download comfyanonymous/flux_text_encoders t5xxl_fp16.safetensors --local-dir models/text_encoders
    huggingface-cli download black-forest-labs/FLUX.1-schnell ae.safetensors --local-dir models/vae
    huggingface-cli download black-forest-labs/FLUX.1-dev flux1-dev.safetensors --local-dir models/diffusion_models
    ```

- **SVDQuant 4-bit FLUX.1-dev Models**

    Next, download the SVDQuant 4-bit models:

    - For **50-series GPUs**, use the [FP4 model](https://huggingface.co/mit-han-lab/svdq-fp4-flux.1-dev).
    - For **other GPUs**, use the [INT4 model](https://huggingface.co/mit-han-lab/svdq-int4-flux.1-dev).

    Make sure to place the **entire downloaded folder** into `models/diffusion_models`. For example:

    ```bash
    huggingface-cli download mit-han-lab/svdq-int4-flux.1-dev --local-dir models/diffusion_models/svdq-int4-flux.1-dev
    ```

- **(Optional): Download Sample LoRAs**

    You can test with some sample LoRAs like [FLUX.1-Turbo](https://huggingface.co/alimama-creative/FLUX.1-Turbo-Alpha/blob/main/diffusion_pytorch_model.safetensors) and [Ghibsky](https://huggingface.co/aleksa-codes/flux-ghibsky-illustration/blob/main/lora.safetensors). Place these files in the `models/loras` directory:

    ```bash
    huggingface-cli download alimama-creative/FLUX.1-Turbo-Alpha diffusion_pytorch_model.safetensors --local-dir models/loras
    huggingface-cli download aleksa-codes/flux-ghibsky-illustration lora.safetensors --local-dir models/loras
    ```


## 3. Set Up Workflows

To use the official workflows, download them from the [ComfyUI-nunchaku](https://github.com/mit-han-lab/ComfyUI-nunchaku/tree/main/workflows) and place them in your `ComfyUI/user/default/workflows` directory. The command can be

```bash
# From the root of your ComfyUI folder
cp -r custom_nodes/ComfyUI-nunchaku/workflows user/default/workflows/nunchaku_examples
```

You can now launch ComfyUI and try running the example workflows.

# Troubleshooting

If you encounter issues, refer to our:

- [FAQs](https://github.com/mit-han-lab/nunchaku/discussions/262)
- [GitHub Issues](https://github.com/mit-han-lab/nunchaku/issues)
