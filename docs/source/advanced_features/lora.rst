Customized LoRAs
================

.. image:: https://huggingface.co/mit-han-lab/nunchaku-artifacts/resolve/main/nunchaku/assets/lora.jpg
   :alt: LoRA integration with Nunchaku

Single LoRA
-----------

`Nunchaku <nunchaku_repo_>`_ seamlessly integrates with off-the-shelf LoRAs without requiring requantization.
Instead of fusing the LoRA branch into the main branch, we directly concatenate the LoRA weights to our low-rank branch.
As Nunchaku uses fused kernel, the overhead of a separate low-rank branch is negligible.
Below is an example of running FLUX.1-dev with `Ghibsky <ghibsky_lora_>`_ LoRA.

.. literalinclude:: ../../../examples/flux.1-dev-lora.py
   :language: python
   :caption: Running FLUX.1-dev with `Ghibsky <ghibsky_lora_>`_  LoRA (`examples/flux.1-dev-lora.py <https://github.com/mit-han-lab/nunchaku/blob/main/examples/flux.1-dev-lora.py>`__)
   :linenos:
   :emphasize-lines: 16-19

The LoRA integration in Nunchaku works through two key methods:

**Loading LoRA Parameters** (lines 16-17):
The ``transformer.update_lora_params`` method loads LoRA weights from a safetensors file. It supports:

- **Local file path**: ``"/path/to/your/lora.safetensors"``
- **HuggingFace repository with specific file**: ``"aleksa-codes/flux-ghibsky-illustration/lora.safetensors"``. The system automatically downloads and caches the LoRA file on first access.

**Controlling LoRA Strength** (lines 18-19):
The ``transformer.set_lora_strength`` method sets the LoRA strength parameter, which controls how much influence the LoRA has on the final output. A value of 1.0 applies the full LoRA effect, while lower values (e.g., 0.5) apply a more subtle influence.

Multiple LoRAs
--------------

To load multiple LoRAs simultaneously, Nunchaku provides the ``nunchaku.lora.flux.compose.compose_lora`` function, 
which combines multiple LoRA weights into a single composed LoRA before loading. 
This approach enables efficient multi-LoRA inference without requiring separate loading operations.

The following example demonstrates how to compose and load multiple LoRAs:

.. literalinclude:: ../../../examples/flux.1-dev-multiple-lora.py
   :language: python
   :caption: Running FLUX.1-dev with `Ghibsky <ghibsky_lora_>`_ and `FLUX-Turbo <turbo_lora_>`_ LoRA (`examples/flux.1-dev-multiple-lora.py <https://github.com/mit-han-lab/nunchaku/blob/main/examples/flux.1-dev-multiple-lora.py>`__)
   :linenos:
   :emphasize-lines: 17-23

The ``compose_lora`` function accepts a list of tuples, where each tuple contains:

- **LoRA path**: Either a local file path or HuggingFace repository path with specific file
- **Strength value**: A float value (typically between 0.0 and 1.0) that controls the influence of that specific LoRA

This composition method allows for precise control over individual LoRA strengths while maintaining computational efficiency through a single loading operation.

.. warning::

   When using multiple LoRAs, avoid using ``transformer.set_lora_strength`` as it applies a uniform strength to all LoRAs. Instead, specify individual strength values for each LoRA within the ``compose_lora`` function call for granular control over each LoRA's influence.


**For ComfyUI users, you can directly use our LoRA loader. The converted LoRA is deprecated. Please refer to `mit-han-lab/ComfyUI-nunchaku <comfyui_nunchaku_>`_ for more details.**
