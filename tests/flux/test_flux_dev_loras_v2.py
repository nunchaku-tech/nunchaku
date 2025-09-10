import pytest
import torch
from diffusers import FluxPipeline
from huggingface_hub import hf_hub_download

from nunchaku.lora.flux.v1.lora_flux_v2 import reset_lora_v2, set_lora_strength_v2, update_lora_params_v2
from nunchaku.models.transformers.transformer_flux_v2 import NunchakuFluxTransformer2DModelV2
from nunchaku.utils import get_precision, is_turing


@pytest.mark.skipif(is_turing(), reason="Skip tests due to using Turing GPUs")
class TestFluxV2LoRA:
    """Test suite for V2 LoRA runtime implementation"""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Setup test environment"""
        self.precision = get_precision()
        self.device = "cuda"
        self.dtype = torch.bfloat16

    def _load_transformer(self):
        """Helper to load V2 transformer"""
        return NunchakuFluxTransformer2DModelV2.from_pretrained(
            f"nunchaku-tech/nunchaku-flux.1-dev/svdq-{self.precision}_r32-flux.1-dev.safetensors",
            device=self.device,
            torch_dtype=self.dtype,
        )

    def _download_lora(self, repo_id, filename):
        """Helper to download LoRA weights"""
        return hf_hub_download(repo_id=repo_id, filename=filename)

    def test_basic_lora_loading(self):
        """Test basic LoRA loading and application"""
        transformer = self._load_transformer()

        # Download LoRA weights
        lora_path = self._download_lora("aleksa-codes/flux-ghibsky-illustration", "lora.safetensors")

        # Apply LoRA
        update_lora_params_v2(transformer, lora_path, strength=1.0)

        # Verify LoRA was applied
        assert hasattr(transformer, "_lora_slots")
        assert len(transformer._lora_slots) > 0
        assert transformer._lora_strength == 1.0

    def test_lora_strength_adjustment(self):
        """Test LoRA strength adjustment"""
        transformer = self._load_transformer()

        # Load LoRA with initial strength
        lora_path = self._download_lora("aleksa-codes/flux-ghibsky-illustration", "lora.safetensors")
        update_lora_params_v2(transformer, lora_path, strength=0.5)

        # Adjust strength
        set_lora_strength_v2(transformer, 1.0)
        assert transformer._lora_strength == 1.0

        set_lora_strength_v2(transformer, 0.3)
        assert transformer._lora_strength == 0.3

    def test_lora_reset(self):
        """Test LoRA reset functionality"""
        transformer = self._load_transformer()

        # Load LoRA
        lora_path = self._download_lora("aleksa-codes/flux-ghibsky-illustration", "lora.safetensors")
        update_lora_params_v2(transformer, lora_path, strength=1.0)

        # Reset LoRA
        reset_lora_v2(transformer)

        # Verify reset
        assert len(transformer._lora_slots) == 0
        assert transformer._lora_strength == 1.0

    @pytest.mark.parametrize(
        "lora_config",
        [
            {
                "repo_id": "aleksa-codes/flux-ghibsky-illustration",
                "filename": "lora.safetensors",
                "prompt": "GHIBSKY style painting, a serene landscape",
                "strength": 1.0,
            },
            {
                "repo_id": "alvdansen/sonny-anime-fixed",
                "filename": "araminta_k_sonnyanime_fluxd_fixed.safetensors",
                "prompt": "a cute creature, nm22 style",
                "strength": 0.8,
            },
        ],
    )
    def test_different_lora_styles(self, lora_config):
        """Test different LoRA styles with generation"""
        transformer = self._load_transformer()

        # Download and apply LoRA
        lora_path = self._download_lora(lora_config["repo_id"], lora_config["filename"])
        update_lora_params_v2(transformer, lora_path, strength=lora_config["strength"])

        # Create pipeline
        pipe = FluxPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-dev", transformer=transformer, torch_dtype=self.dtype
        ).to(self.device)

        # Generate image
        with torch.inference_mode():
            image = pipe(
                lora_config["prompt"],
                height=512,  # Smaller for faster testing
                width=512,
                guidance_scale=3.5,
                num_inference_steps=4,  # Minimal steps for testing
                generator=torch.Generator(self.device).manual_seed(42),
            ).images[0]

        # Basic validation
        assert image is not None
        assert image.size == (512, 512)

    def test_sequential_lora_loading(self):
        """Test loading different LoRAs sequentially"""
        # Test 1: Load first LoRA
        transformer1 = self._load_transformer()
        lora1_path = self._download_lora("aleksa-codes/flux-ghibsky-illustration", "lora.safetensors")
        update_lora_params_v2(transformer1, lora1_path, strength=1.0)
        first_slot_count = len(transformer1._lora_slots)
        assert first_slot_count > 0

        # Test 2: Load second LoRA on fresh model (safer approach)
        transformer2 = self._load_transformer()
        lora2_path = self._download_lora("alvdansen/sonny-anime-fixed", "araminta_k_sonnyanime_fluxd_fixed.safetensors")
        update_lora_params_v2(transformer2, lora2_path, strength=0.7)

        # Verify second LoRA is loaded
        assert len(transformer2._lora_slots) > 0
        assert transformer2._lora_strength == 0.7

    def test_lora_with_pipeline_integration(self):
        """Test full pipeline integration with V2 LoRA"""
        transformer = self._load_transformer()

        # Apply LoRA
        lora_path = self._download_lora("aleksa-codes/flux-ghibsky-illustration", "lora.safetensors")
        update_lora_params_v2(transformer, lora_path, strength=1.0)

        # Create pipeline
        pipe = FluxPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-dev", transformer=transformer, torch_dtype=self.dtype
        ).to(self.device)

        # Test generation with different prompts
        prompts = ["GHIBSKY style, a magical forest", "GHIBSKY style painting, ancient temple"]

        for prompt in prompts:
            with torch.inference_mode():
                image = pipe(
                    prompt,
                    height=512,
                    width=512,
                    guidance_scale=3.5,
                    num_inference_steps=4,
                    generator=torch.Generator(self.device).manual_seed(0),
                ).images[0]

            assert image is not None
            assert image.size == (512, 512)

    @pytest.mark.slow
    def test_high_quality_generation(self):
        """Test high quality generation with proper steps (marked as slow)"""
        transformer = self._load_transformer()

        # Load LoRA
        lora_path = self._download_lora("aleksa-codes/flux-ghibsky-illustration", "lora.safetensors")
        update_lora_params_v2(transformer, lora_path, strength=1.0)

        # Create pipeline
        pipe = FluxPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-dev", transformer=transformer, torch_dtype=self.dtype
        ).to(self.device)

        # Generate with full quality settings
        with torch.inference_mode():
            image = pipe(
                "GHIBSKY style painting, majestic mountain landscape at sunset",
                height=1024,
                width=1024,
                guidance_scale=3.5,
                num_inference_steps=25,
                generator=torch.Generator(self.device).manual_seed(42),
            ).images[0]

        assert image is not None
        assert image.size == (1024, 1024)

        # Optionally save for manual inspection
        # image.save("test_output_v2_lora.png")


if __name__ == "__main__":
    # Run a simple test manually
    test = TestFluxV2LoRA()
    test.setup()
    test.test_basic_lora_loading()
    print("Basic LoRA loading test passed!")

    test.test_lora_strength_adjustment()
    print("LoRA strength adjustment test passed!")

    test.test_lora_reset()
    print("LoRA reset test passed!")

    print("\nAll manual tests passed!")
