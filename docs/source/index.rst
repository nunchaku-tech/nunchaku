Nunchaku Documentation
======================
**Nunchaku** is a high-performance inference engine optimized for low-bit diffusion models and LLMs,
as introduced in our paper `SVDQuant <svdquant_paper>`_.
Check out `DeepCompressor <deepcompressor_repo>`_ for the quantization library.

.. toctree::
   :maxdepth: 2
   :caption: Installation

   installation/installation.rst
   installation/setup_windows.rst

.. toctree::
    :maxdepth: 1
    :caption: Usage Tutorials

    usage/basic_usage.rst
    usage/lora.rst
    usage/kontext.rst
    usage/controlnet.rst
    usage/qencoder.rst
    usage/offload.rst
    usage/attention.rst
    usage/fbcache.rst
    usage/pulid.rst

.. toctree::
    :maxdepth: 1
    :caption: Gradio Demos
    :titlesonly:

    Gradio Demos <https://github.com/mit-han-lab/nunchaku/tree/main/app>

.. toctree::
    :maxdepth: 1
    :caption: ComfyUI Support
    :titlesonly:

    ComfyUI-nunchaku Plugin <https://github.com/mit-han-lab/ComfyUI-nunchaku>

.. toctree::
    :maxdepth: 1
    :caption: Customized Model Quantization
    :titlesonly:

    DeepCompressor <https://github.com/mit-han-lab/deepcompressor>

.. toctree::
    :maxdepth: 1
    :caption: Python API Reference

    python_api/nunchaku.rst

.. toctree::
    :maxdepth: 1
    :caption: Frequently Asked Questions (FAQ)

    faq/model.rst
    faq/usage.rst

.. toctree::
    :maxdepth: 2
    :caption: Developer Guidance

    developer/contribution_guide.rst
