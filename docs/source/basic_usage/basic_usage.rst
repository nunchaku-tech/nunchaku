Basic Usage Example
====================

The following is a minimal script for running 4-bit `FLUX.1 <flux_repo_>`_ using Nunchaku.
Nunchaku provides the same API as `🤗 Diffusers <diffusers_repo_>`_, so you can use it in a familiar way.

.. tabs::

   .. tab:: Default (Ampere, Ada, Blackwell, etc.)

      .. literalinclude:: ../../../examples/flux.1-dev.py
         :language: python
         :caption: Minimal example running FLUX.1 with Nunchaku (default)
         :linenos:

   .. tab:: Turing GPUs (e.g., RTX 20 series)

      .. literalinclude:: ../../../examples/flux.1-dev-turing.py
         :language: python
         :caption: Example for Turing GPUs with `torch.float16` and fp16 attention
         :linenos:

.. note::

   If you're using a **Turing GPU (e.g., NVIDIA 20-series)**, set ``torch_dtype=torch.float16`` and use the ``nunchaku-fp16`` attention module instead.
