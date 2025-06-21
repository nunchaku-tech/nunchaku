Basic Usage Tutorial
====================

The following is a minimal script for running 4-bit `FLUX.1 <flux_repo_>`_ using Nunchaku.
Nunchaku provides the same API as `🤗 Diffusers <diffusers_repo_>`_, so you can use it in a familiar way.

.. tabs::

   .. tab:: Default (Ampere, Ada, Blackwell, etc.)

      .. literalinclude:: ../../../examples/flux.1-dev.py
         :language: python
         :caption: Running FLUX.1-dev (`examples/flux.1-dev.py <https://github.com/mit-han-lab/nunchaku/blob/main/examples/flux.1-dev.py>`__)
         :linenos:

   .. tab:: Turing GPUs (e.g., RTX 20 series)

      .. literalinclude:: ../../../examples/flux.1-dev-turing.py
         :language: python
         :caption: Running FLUX.1-dev on Turing GPUs (`examples/flux.1-dev-turing.py <https://github.com/mit-han-lab/nunchaku/blob/main/examples/flux.1-dev-turing.py>`__)
         :linenos:

.. note::

   If you're using a **Turing GPU (e.g., NVIDIA 20-series)**, set ``torch_dtype=torch.float16`` and use the ``nunchaku-fp16`` attention module instead.
