import pytest
import torch

from nunchaku.lora.flux.compose import compose_lora


def _nunchaku_lora():
    # Any key containing ".mlp_fc" marks a LoRA as already converted to Nunchaku format.
    return {
        "transformer_blocks.0.mlp_fc1.lora_down": torch.ones(2, 8),
        "transformer_blocks.0.mlp_fc1.lora_up": torch.ones(8, 2),
    }


def test_single_nunchaku_lora_at_full_strength_is_returned_unchanged():
    lora = _nunchaku_lora()
    assert compose_lora([(lora, 1.0)]) is lora


@pytest.mark.parametrize("strength", [0.0, 0.5, 1.5])
def test_single_nunchaku_lora_at_other_strengths_is_not_returned_at_full_strength(strength):
    # Before the fix, `(strength - 1) < 1e-5` held for every strength below 1, so the LoRA came back
    # unscaled; strengths above 1 reached the Diffusers-format assertion instead.
    with pytest.raises(AssertionError, match="Nunchaku-format"):
        compose_lora([(_nunchaku_lora(), strength)])
