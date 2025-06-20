# Installation

We provide step-by-step tutorial videos to help you install and use **Nunchaku on Windows**, available in both [**English**](https://youtu.be/YHAVe-oM7U8?si=cM9zaby_aEHiFXk0) and [**Chinese**](https://www.bilibili.com/video/BV1BTocYjEk5/?share_source=copy_web&vd_source=8926212fef622f25cc95380515ac74ee). You can also follow the corresponding text guide at [`Window Setup Guide`](./setup_windows.md). If you encounter any issues, these resources are a good place to start.

## Option 1: Installing Prebuilt Wheels (Recommended)

### Prerequisites

Ensure that you have [PyTorch ≥ 2.5](https://pytorch.org/) installed. For example, to install **PyTorch 2.7 with CUDA 12.8**, use:

```shell
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

### Installing Nunchaku

Once PyTorch is installed, you can install `nunchaku` from one of the following sources:

- [GitHub Releases](https://github.com/mit-han-lab/nunchaku/releases)

- [Hugging Face](https://huggingface.co/mit-han-lab/nunchaku/tree/main)

- [ModelScope](https://modelscope.cn/models/Lmxyy1999/nunchaku)

```shell
pip install https://github.com/mit-han-lab/nunchaku/releases/download/v0.3.1/nunchaku-0.3.1+torch2.7-cp311-cp311-linux_x86_64.whl
```

#### For ComfyUI Users

If you're using the **ComfyUI portable package**, ensure that `nunchaku` is installed into the Python environment bundled with ComfyUI. You can either:

- Use our **NunchakuWheelInstaller Node** in [ComfyUI-nunchaku](https://github.com/mit-han-lab/ComfyUI-nunchaku), or

- Manually install the wheel using the correct Python path.

##### Option 1: Using NunchakuWheelInstaller

With [ComfyUI-nunchaku v0.3.2+](https://github.com/mit-han-lab/ComfyUI-nunchaku), you can install Nunchaku using the provided [workflow](https://github.com/mit-han-lab/ComfyUI-nunchaku/blob/main/example_workflows/install_wheel.json) directly in ComfyUI. This automates installation once ComfyUI-nunchaku and its dependencies are set up.

![install_wheel.png](https://huggingface.co/mit-han-lab/nunchaku-artifacts/resolve/main/ComfyUI-nunchaku/assets/install_wheel.png)

##### Option 2: Manual Installation

To find the correct Python path:

1. Launch ComfyUI.

1. Check the console log—look for a line like:

   ```text
   ** Python executable: G:\ComfyUI\python\python.exe
   ```

1. Use that executable to install the wheel manually:

   ```bat
   "G:\ComfyUI\python\python.exe" -m pip install <your-wheel-file>.whl
   ```

**Example:** Installing for Python 3.11 and PyTorch 2.7:

```bat
"G:\ComfyUI\python\python.exe" -m pip install https://github.com/mit-han-lab/nunchaku/releases/download/v0.3.1/nunchaku-0.3.1+torch2.7-cp311-cp311-linux_x86_64.whl
```

#### For Blackwell GPUs (50-series)

If you're using a **Blackwell (RTX 50-series)** GPU:

- Use **PyTorch ≥ 2.7** with **CUDA ≥ 12.8**.
- Use **FP4 models** instead of **INT4 models** for best compatibility and performance.

## Option 2: Build from Source

### Requirements

- **CUDA version**:
  - **Linux**: ≥ 12.2
  - **Windows**: ≥ 12.6
  - **Blackwell GPUs**: CUDA ≥ 12.8 required
- **Compiler**:
  - Linux: `gcc/g++ ≥ 11`
  - Windows: Latest **MSVC** via [Visual Studio](https://visualstudio.microsoft.com/visual-cpp-build-tools/)

> Currently supported GPU architectures:
> * `sm_75` (Turing: RTX 2080)
> * `sm_80` (Ampere: A100)
> * `sm_86` (Ampere: RTX 3090, A6000)
> * `sm_89` (Ada: RTX 4090)
> * `sm_120` (Blackwell: RTX 5090)

### Step 1: Set Up Environment

```shell
conda create -n nunchaku python=3.11
conda activate nunchaku

# Install PyTorch
pip install torch torchvision torchaudio

# Install dependencies
pip install ninja wheel diffusers transformers accelerate sentencepiece protobuf huggingface_hub

# Optional: For gradio demos
pip install peft opencv-python gradio spaces
```

For Blackwell users (50-series), install PyTorch ≥ 2.7 with CUDA ≥ 12.8:

```shell
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

### Step 2: Build and Install Nunchaku

**For Linux (if `gcc/g++` is not recent enough):**

```shell
conda install -c conda-forge gxx=11 gcc=11
```

For Windows users, you can download and install the latest [Visual Studio](https://visualstudio.microsoft.com/thank-you-downloading-visual-studio/?sku=Community&channel=Release&version=VS2022&source=VSLandingPage&cid=2030&passive=false) and use its development environment. See [`Window Setup Guide`](./setup_windows.md) for more details.

**Clone and build:**

```shell
git clone https://github.com/mit-han-lab/nunchaku.git
cd nunchaku
git submodule init
git submodule update
python setup.py develop
```

**To build a wheel for distribution:**

```shell
NUNCHAKU_INSTALL_MODE=ALL NUNCHAKU_BUILD_WHEELS=1 python -m build --wheel --no-isolation
```

> **Important:**
> Set `NUNCHAKU_INSTALL_MODE=ALL` to ensure the wheel works on all supported GPU architectures. Otherwise, it may only run on the GPU type used for building.
