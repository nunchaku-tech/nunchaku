CPU Offload
===========

Nunchaku supports CPU offload to further reduce GPU memory usage without much communication overhead. 
It is also compatible with `Diffusers <diffusers_repo_>`_ offload.

.. literalinclude:: ../../../examples/flux.1-dev-offload.py
   :language: python
   :caption: Running FLUX.1-dev with CPU Offload (`examples/flux.1-dev-offload.py <https://github.com/mit-han-lab/nunchaku/blob/main/examples/flux.1-dev-offload.py>`__)
   :linenos:
   :emphasize-lines: 9, 13, 14

Compared to `basic usage <../basic_usage/basic_usage>`_, the key changes for CPU offload are:

**Enabling Nunchaku Offload** (line 9):
Set ``offload=True`` in the transformer initialization to enable Nunchaku's CPU offload functionality.
This reduces GPU memory usage by offloading model components to CPU when not in use.

**Enabling Diffusers Offload** (line 14):
Call ``pipeline.enable_sequential_cpu_offload()`` to enable Diffusers' sequential CPU offload.
This automatically handles device placement and further reduces GPU memory usage.

.. note::
    Unlike basic usage, you don't need to explicitly call ``.to('cuda')`` on the pipeline,
    as ``pipeline.enable_sequential_cpu_offload()`` automatically manages device placement.
