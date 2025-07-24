.. _installation-installation:

Installation
============

We provide step-by-step tutorial videos to help you install and use **Nunchaku on Windows**,
available in both `English <nunchaku_windows_tutorial_en_>`_ and `Chinese <nunchaku_windows_tutorial_zh_>`_.
You can also follow the corresponding text guide at :doc:`Windows Setup Guide <setup_windows>`.
If you encounter any issues, these resources are a good place to start.

(Recommended) Option 1: Installing Prebuilt Wheels
--------------------------------------------------

Prerequisites
^^^^^^^^^^^^^

Ensure that you have `PyTorch ≥ 2.5 <pytorch_home_>`_ installed. For example, to install **PyTorch 2.7 with CUDA 12.8**, use:

.. code-block:: shell

    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

Installing Nunchaku
^^^^^^^^^^^^^^^^^^^

Once PyTorch is installed, you can install ``nunchaku`` from one of the following sources:

- `GitHub Releases <github_nunchaku_releases_>`_
- `Hugging Face <hf_nunchaku_>`_
- `ModelScope <ms_nunchaku_>`_

.. code-block:: shell

    pip install https://github.com/nunchaku-tech/nunchaku/releases/download/v0.3.1/nunchaku-0.3.1+torch2.7-cp311-cp311-linux_x86_64.whl

Make sure to replace each version number with the correct one for your environment. `cp311` stands for python version 3.11.x. You can find your own python version using  

.. code-block:: shell

    python --version

For ComfyUI Users
^^^^^^^^^^^^^^^^^

If you're using the **ComfyUI portable package**,
ensure that ``nunchaku`` is installed into the Python environment bundled with ComfyUI. You can either:

- Use our :ref:`comfyui_nunchaku:install-wheel-json` workflow, or
- Manually install the wheel using the correct Python path.

Option 1: Using ``install_wheel.json`` Workflow
"""""""""""""""""""""""""""""""""""""""""""""""

With `ComfyUI-nunchaku <github_comfyui-nunchaku_>`_ v0.3.2+,
you can install Nunchaku using the provided
:ref:`comfyui_nunchaku:install-wheel-json` workflow directly in ComfyUI.

.. image:: https://huggingface.co/datasets/nunchaku-tech/cdn/resolve/main/ComfyUI-nunchaku/workflows/install_wheel.png

Option 2: Manual Installation
"""""""""""""""""""""""""""""

To find the correct Python path:

1. Launch ComfyUI.
2. Check the console log—look for a line like:

   .. code-block:: text

       ** Python executable: G:\ComfyUI\python\python.exe

3. Use that executable to install the wheel manually:

   .. code-block:: bat

       "G:\ComfyUI\python\python.exe" -m pip install <your-wheel-file>.whl

   **Example:** Installing for Python 3.11 and PyTorch 2.7:

   .. code-block:: bat

       "G:\ComfyUI\python\python.exe" -m pip install https://github.com/nunchaku-tech/nunchaku/releases/download/v0.3.1/nunchaku-0.3.1+torch2.7-cp311-cp311-linux_x86_64.whl

For Blackwell GPUs (50-series)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If you're using a **Blackwell (RTX 50-series)** GPU:

- Use **PyTorch ≥ 2.7** with **CUDA ≥ 12.8**.
- Use **FP4 models** instead of **INT4 models** for best compatibility and performance.

Option 2: Build from Source
---------------------------

Requirements
^^^^^^^^^^^^

- **CUDA version**:

  - Linux: ≥ 12.2
  - Windows: ≥ 12.6
  - Blackwell GPUs: CUDA ≥ 12.8 required

- **Compiler**:

  - Linux: ``gcc/g++ >= 11``
  - Windows: Latest **MSVC** via `Visual Studio <visual_studio_>`_

.. important::

   Currently supported GPU architectures:

   - ``sm_75`` (Turing: RTX 2080)
   - ``sm_80`` (Ampere: A100)
   - ``sm_86`` (Ampere: RTX 3090, A6000)
   - ``sm_89`` (Ada: RTX 4090)
   - ``sm_120`` (Blackwell: RTX 5090)

Step 1: Set Up Environment
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: shell

    conda create -n nunchaku python=3.11
    conda activate nunchaku

    # Install PyTorch
    pip install torch torchvision torchaudio

    # Install dependencies
    pip install ninja wheel diffusers transformers accelerate sentencepiece protobuf huggingface_hub

    # Optional: For gradio demos
    pip install peft opencv-python gradio spaces

For Blackwell users (50-series), install PyTorch ≥ 2.7 with CUDA ≥ 12.8:

.. code-block:: shell

    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

Step 2: Build and Install Nunchaku
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**For Linux (if** ``gcc/g++`` **is not recent enough):**

.. code-block:: shell

    conda install -c conda-forge gxx=11 gcc=11

For Windows users, download and install the latest `Visual Studio <visual_studio_>`_ and use its development environment. See :doc:`Window Setup Guide <setup_windows>` for more details.

**Clone and build:**

.. code-block:: shell

    git clone https://github.com/nunchaku-tech/nunchaku.git
    cd nunchaku
    git submodule init
    git submodule update
    python setup.py develop

**To build a wheel for distribution:**

.. code-block:: shell

    NUNCHAKU_INSTALL_MODE=ALL NUNCHAKU_BUILD_WHEELS=1 python -m build --wheel --no-isolation

.. important::

   Set ``NUNCHAKU_INSTALL_MODE=ALL`` to ensure the wheel works on all supported GPU architectures. Otherwise, it may only run on the GPU type used for building.
