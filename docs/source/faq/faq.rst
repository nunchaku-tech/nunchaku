Frequently Asked Questions (FAQ)
=================================

❗ Import Error: ``ImportError: cannot import name 'to_diffusers' from 'nunchaku.lora.flux' (...)``
----------------------------------------------------------------------------------------------------------------

This error usually indicates that the nunchaku library was not installed correctly. We've prepared step-by-step installation guides for Windows users:

📺 `English tutorial <https://youtu.be/YHAVe-oM7U8?si=cM9zaby_aEHiFXk0>`_ | 📺 `Chinese tutorial <https://www.bilibili.com/video/BV1BTocYjEk5/?share_source=copy_web&vd_source=8926212fef622f25cc95380515ac74ee>`_ | 📖 `Corresponding Text guide <https://github.com/mit-han-lab/nunchaku/blob/main/docs/setup_windows.md>`_

Please also check the following common causes:

- **You only installed the ComfyUI plugin (ComfyUI-nunchaku) but not the core nunchaku library.** Please follow the `installation instructions in our README <https://github.com/mit-han-lab/nunchaku?tab=readme-ov-file#installation>`_ to install the correct version of the ``nunchaku`` library.

- **You installed nunchaku using** ``pip install nunchaku`` **, but this is the wrong package.**
  The ``nunchaku`` name on PyPI is already taken by an unrelated project. Please uninstall the incorrect package and follow our `installation guide <https://github.com/mit-han-lab/nunchaku?tab=readme-ov-file#installation>`_ to install the correct version.

- **(MOST LIKELY) You installed nunchaku correctly, but into the wrong Python environment.**
  If you're using the ComfyUI portable package, its Python interpreter is very likely not the system default. To identify the correct Python path, launch ComfyUI and check the several initial lines in the log. For example, you will find

  .. code-block:: text

     ** Python executable: G:\ComfyuI\python\python.exe

  To install ``nunchaku`` into this environment, use the following format:

  .. code-block:: shell

     "G:\ComfyUI\python\python.exe" -m pip install <your-wheel-file>.whl

  Example (for Python 3.11 and torch 2.6):

  .. code-block:: shell

     "G:\ComfyUI\python\python.exe" -m pip install https://github.com/mit-han-lab/nunchaku/releases/download/v0.2.0/nunchaku-0.2.0+torch2.6-cp311-cp311-linux_x86_64.whl

- **You have a folder named nunchaku in your working directory.**
  Python may mistakenly load from that local folder instead of the installed library. Also, make sure your plugin folder under ``custom_nodes`` is named ``ComfyUI-nunchaku``, not ``nunchaku``.

❗ Runtime Error: ``Assertion failed: this->shape.dataExtent == other.shape.dataExtent, file ...Tensor.h``
---------------------------------------------------------------------------------------------------------------

This error is typically due to using the wrong model for your GPU.

- If you're using a **Blackwell GPU (e.g., RTX 50-series)**, please use our **FP4** models.
- For all other GPUs, use our **INT4** models.

❗ System crash or blue screen
-------------------------------

We have observed some cases where memory is not properly released after image generation, especially when using ComfyUI. This may lead to system instability or crashes.

We're actively investigating this issue. If you have experience or insights into memory management in ComfyUI, we would appreciate your help!

❗ Out of Memory or Slow Model Loading
---------------------------------------

Try upgrading your CUDA driver and try setting the environment variable ``NUNCHAKU_LOAD_METHOD`` to either ``READ`` or ``READNOPIN``.

❗ Same Seeds Produce Slightly Different Images
------------------------------------------------

This behavior is due to minor precision noise introduced by the GPU's accumulation order. Because modern GPUs execute operations out of order for better performance, small variations in output can occur, even with the same seed.
Enforcing strict accumulation order would reduce this variability but significantly hurt performance, so we do not plan to change this behavior.

❓ PuLID Support
-----------------

PuLID support is currently in development and will be included in the next major release.

~~❗ Assertion Error: ``Assertion failed: a.dtype() == b.dtype(), file ...misc_kernels.cu``~~
-------------------------------------------------------------------------------------------------

**This issue has now been resolved.** ✅

~~At the moment, we **only support the 16-bit version of** `ControlNet-Union-Pro <https://huggingface.co/Shakker-Labs/FLUX.1-dev-ControlNet-Union-Pro>`_. Support for FP8 and other ControlNets is planned for a future release.~~

~~❗ Assertion Error: ``assert image_rotary_emb.shape[2] == batch_size * (txt_tokens + img_tokens)``~~
-------------------------------------------------------------------------------------------------------

**Multi-batch inference is now supported as of** `v0.3.0dev0 <https://github.com/mit-han-lab/nunchaku/releases/tag/v0.3.0dev0>`_. ✅

~~Currently, **batch sizes greater than 1 are not supported** during inference. We will support this in a future major release.~~ 