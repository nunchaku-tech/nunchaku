Usage
=====

Out of memory or slow model loading
-----------------------------------

If you encounter out-of-memory errors or notice that model loading is unusually slow, please try the following steps:

- **Update your CUDA driver** to the latest version, as this can resolve many compatibility and performance issues.
- **Set the environment variable** `NUNCHAKU_LOAD_METHOD` to either `READ` or `READNOPIN`.

.. note::

   **Related issues:** :nunchaku-issue:`249`, :nunchaku-issue:`276`, :nunchaku-issue:`311`

Why do the same seeds produce slightly different images with Nunchaku?
----------------------------------------------------------------------

This behavior is due to minor precision noise introduced by the GPU’s accumulation order.
Because modern GPUs execute operations out of order for better performance, small variations in output can occur, even with the same seed.
Enforcing strict accumulation order would reduce this variability but significantly hurt performance, so we do not plan to change this behavior.

.. note::

   **Related issues:** :nunchaku-issue:`229`, :nunchaku-issue:`294`
