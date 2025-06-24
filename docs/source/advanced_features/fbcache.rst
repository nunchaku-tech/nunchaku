First-Block Cache
=================

Nunchaku supports `First-Block Cache (FB Cache) <fbcache>`_ to accelerate long-step denoising. The usage example is as follows:

.. literalinclude:: ../../../examples/flux.1-dev-cache.py
   :language: python
   :caption: Running FLUX.1-dev with FB Cache (`examples/flux.1-dev-cache.py <https://github.com/mit-han-lab/nunchaku/blob/main/examples/flux.1-dev-cache.py>`__)
   :linenos:
   :emphasize-lines: 15-17

You can easily enable it with:

.. code-block:: python

    apply_cache_on_pipe(pipeline, residual_diff_threshold=0.12)

You can tune the ``residual_diff_threshold`` to balance speed and quality: larger values yield faster inference at the cost of some quality. 
A recommended value is 0.12, which provides up to 2× speedup for 50-step denoising and 1.4× speedup for 30-step denoising.
