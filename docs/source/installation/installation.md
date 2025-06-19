# Installation

We provide tutorial videos to help you install and use Nunchaku on Windows, available in both [**English**](https://youtu.be/YHAVe-oM7U8?si=cM9zaby_aEHiFXk0) and [**Chinese**](https://www.bilibili.com/video/BV1BTocYjEk5/?share_source=copy_web&vd_source=8926212fef622f25cc95380515ac74ee). You can also follow the corresponding step-by-step text guide at [`docs/setup_windows.md`](docs/setup_windows.md). If you run into issues, these resources are a good place to start.

## Wheels

### Prerequisites

Before installation, ensure you have [PyTorch>=2.5](https://pytorch.org/) installed. For example, you can use the following command to install PyTorch 2.6:

```shell
pip install torch==2.6 torchvision==0.21 torchaudio==2.6
```

### Install nunchaku

Once PyTorch is installed, you can directly install `nunchaku` from [Hugging Face](https://huggingface.co/mit-han-lab/nunchaku/tree/main), [ModelScope](https://modelscope.cn/models/Lmxyy1999/nunchaku) or [GitHub release](https://github.com/mit-han-lab/nunchaku/releases). Be sure to select the appropriate wheel for your Python and PyTorch version. For example, for Python 3.11 and PyTorch 2.6:

```shell
pip install https://huggingface.co/mit-han-lab/nunchaku/resolve/main/nunchaku-0.2.0+torch2.6-cp311-cp311-linux_x86_64.whl
```

#### For ComfyUI Users

If you're using the **ComfyUI portable package**, make sure to install `nunchaku` into the correct Python environment bundled with ComfyUI. To find the right Python path, launch ComfyUI and check the log output. You'll see something like this in the first several lines:

```text
** Python executable: G:\ComfyuI\python\python.exe
```

Use that Python executable to install `nunchaku`:

```shell
"G:\ComfyUI\python\python.exe" -m pip install <your-wheel-file>.whl
```

**Example:** Installing for Python 3.11 and PyTorch 2.6:

```shell
"G:\ComfyUI\python\python.exe" -m pip install https://github.com/mit-han-lab/nunchaku/releases/download/v0.2.0/nunchaku-0.2.0+torch2.6-cp311-cp311-linux_x86_64.whl
```

#### For Blackwell GPUs (50-series)

If you're using a Blackwell GPU (e.g., 50-series GPUs), install a wheel with PyTorch 2.7 and higher. Additionally, use **FP4 models** instead of INT4 models."

## Build from Source

**Note**:

- Make sure your CUDA version is **at least 12.2 on Linux** and **at least 12.6 on Windows**. If you're using a Blackwell GPU (e.g., 50-series GPUs), CUDA **12.8 or higher is required**.

- For Windows users, please refer to [this issue](https://github.com/mit-han-lab/nunchaku/issues/6) for the instruction. Please upgrade your MSVC compiler to the latest version.

- We currently support only NVIDIA GPUs with architectures sm_75 (Turing: RTX 2080), sm_86 (Ampere: RTX 3090, A6000), sm_89 (Ada: RTX 4090), and sm_80 (A100). See [this issue](https://github.com/mit-han-lab/nunchaku/issues/1) for more details.

1. Install dependencies:

   ```shell
   conda create -n nunchaku python=3.11
   conda activate nunchaku
   pip install torch torchvision torchaudio
   pip install ninja wheel diffusers transformers accelerate sentencepiece protobuf huggingface_hub

   # For gradio demos
   pip install peft opencv-python gradio spaces
   ```

   To enable NVFP4 on Blackwell GPUs (e.g., 50-series GPUs), please install nightly PyTorch>=2.7 with CUDA>=12.8. The installation command can be:

   ```shell
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
   ```

1. Install `nunchaku` package:
   Make sure you have `gcc/g++>=11`. If you don't, you can install it via Conda on Linux:

   ```shell
   conda install -c conda-forge gxx=11 gcc=11
   ```

   For Windows users, you can download and install the latest [Visual Studio](https://visualstudio.microsoft.com/thank-you-downloading-visual-studio/?sku=Community&channel=Release&version=VS2022&source=VSLandingPage&cid=2030&passive=false).

   Then build the package from source with

   ```shell
   git clone https://github.com/mit-han-lab/nunchaku.git
   cd nunchaku
   git submodule init
   git submodule update
   python setup.py develop
   ```

   If you are building wheels for distribution, use:

   ```shell
   NUNCHAKU_INSTALL_MODE=ALL NUNCHAKU_BUILD_WHEELS=1 python -m build --wheel --no-isolation
   ```

   Make sure to set the environment variable `NUNCHAKU_INSTALL_MODE` to `ALL`. Otherwise, the generated wheels will only work on GPUs with the same architecture as the build machine.
