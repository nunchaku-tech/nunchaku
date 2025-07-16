.. _usage_fbcache:

First-Block Cache
=================

Nunchaku supports `First-Block Cache (FB Cache) <fbcache>`_ for faster long-step denoising. Example usage:

.. literalinclude:: ../../../examples/flux.1-dev-cache.py
   :language: python
   :caption: Running FLUX.1-dev with FB Cache (`examples/flux.1-dev-cache.py <https://github.com/mit-han-lab/nunchaku/blob/main/examples/flux.1-dev-cache.py>`__)
   :linenos:
   :emphasize-lines: 15-17

Enable it with :func:`~nunchaku.caching.diffusers_adapters.flux.apply_cache_on_pipe`:

.. code-block:: python

    apply_cache_on_pipe(pipeline, residual_diff_threshold=0.12)

Adjust ``residual_diff_threshold`` to trade speed for quality - higher values are faster but lower quality.
Recommended value 0.12 gives 2× speedup for 50-step and 1.4× for 30-step denoising.
