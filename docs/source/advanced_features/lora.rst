Custom LoRAs
============

.. image:: https://huggingface.co/mit-han-lab/nunchaku-artifacts/resolve/main/nunchaku/assets/lora.jpg
   :alt: LoRA integration with Nunchaku

Single LoRA
-----------

`Nunchaku <nunchaku_repo_>`_ seamlessly integrates with off-the-shelf LoRAs without requiring requantization.
You can simply use your LoRA with:

.. literalinclude:: ../../../examples/flux.1-dev-lora.py
   :language: python
   :caption: Running FLUX.1-dev with `Ghibsky <ghibsky_lora_>`_  LoRA (`examples/flux.1-dev-lora.py <https://github.com/mit-han-lab/nunchaku/blob/main/examples/flux.1-dev-lora.py>`__)
   :linenos:
   :emphasize-lines: 16-19

``path_to_your_lora`` can also be a remote HuggingFace path.

Multiple LoRAs
--------------

To compose multiple LoRAs, you can use ``nunchaku.lora.flux.compose.compose_lora`` to compose them. The usage is:

.. literalinclude:: ../../../examples/flux.1-dev-multiple-lora.py
   :language: python
   :caption: Running FLUX.1-dev with `Ghibsky <ghibsky_lora_>`_ and `FLUX-Turbo <turbo_lora_>`_ LoRA (`examples/flux.1-dev-multiple-lora.py <https://github.com/mit-han-lab/nunchaku/blob/main/examples/flux.1-dev-multiple-lora.py>`__)
   :linenos:
   :emphasize-lines: 17-23

You can specify individual strengths for each LoRA in the list.

**For ComfyUI users, you can directly use our LoRA loader. The converted LoRA is deprecated. Please refer to `mit-han-lab/ComfyUI-nunchaku <comfyui_nunchaku_>`_ for more details.**
