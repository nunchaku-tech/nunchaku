import gc
import torch
from nunchaku.utils import get_precision
from pathlib import Path
import os
from .utils import already_generate, compute_lpips, offload_pipeline, hash_str_to_int
from diffusers import FluxKontextPipeline
from diffusers.utils import load_image

def test_flux_kontext(expected_lpips: float):
    gc.collect()
    torch.cuda.empty_cache()

    precision = get_precision()

    ref_root = Path(os.environ.get("NUNCHAKU_TEST_CACHE_ROOT", os.path.join("test_results", "ref")))
    results_dir_16_bit = ref_root / "bf16" / "flux.1-kontext-dev" / "kontext"
    results_dir_4_bit = Path("test_results") / precision / "flux.1-kontext-dev" / "kontext"

    os.makedirs(results_dir_16_bit, exist_ok=True)
    os.makedirs(results_dir_4_bit, exist_ok=True)
    
    image = load_image(
    "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/yarn-art-pikachu.png"
).convert("RGB")
    prompts = ["Make Pikachu hold a sign that says 'Nunchaku is awesome', yarn art style, detailed, vibrant colors",
    'Convert the image to ghibli style',
    'help me convert it to manga style',
    'Convert it to a realistic photo',
    ]

    # First, generate results with the 16-bit model
    if not already_generate(results_dir_16_bit, 4):
        pipeline = FluxKontextPipeline.from_pretrained("black-forest-labs/FLUX.1-Kontext-dev", torch_dtype=torch.bfloat16)

        # Possibly offload the model to CPU when GPU memory is scarce
        pipeline = offload_pipeline(pipeline)

        for prompt in prompts:
            seed = hash_str_to_int(prompt)
            result = pipeline(
                image=image,
                prompt=prompt,
                generator=torch.Generator().manual_seed(seed)
            ).images[0]
            result.save(os.path.join(results_dir_16_bit, f"{seed}.png"))

        # Clean up the 16-bit model
        del pipeline.transformer
        del pipeline.text_encoder
        del pipeline.text_encoder_2
        del pipeline.vae
        del pipeline
        del result
        gc.collect()
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

    free, total = torch.cuda.mem_get_info()  # bytes
    print(f"After 16-bit generation: Free: {free/1024**2:.0f} MB  /  Total: {total/1024**2:.0f} MB")

    # Then, generate results with the 4-bit model
    if not already_generate(results_dir_4_bit, 1):
        transformer = NunchakuFluxTransformer2dModel.from_pretrained(
            f"nunchaku-models/nunchaku-flux.1-kontext-dev/svdq-{precision}_r32-flux.1-kontext-dev.safetensors"
        )
        pipeline = FluxKontextPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-dev", transformer=transformer, torch_dtype=torch.bfloat16
        ).to("cuda")
        for prompt in prompts:
            seed = hash_str_to_int(prompt)
            result = pipeline(
                image=image,
                prompt=prompt,
                generator=torch.Generator().manual_seed(seed)
            ).images[0]
            result.save(os.path.join(results_dir_4_bit, f"{seed}.png"))


        # Clean up the 4-bit model
        del pipeline
        del transformer
        gc.collect()
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

    free, total = torch.cuda.mem_get_info()  # bytes
    print(f"After 4-bit generation: Free: {free/1024**2:.0f} MB  /  Total: {total/1024**2:.0f} MB")

    lpips = compute_lpips(results_dir_16_bit, results_dir_4_bit)
    print(f"lpips: {lpips}")
    assert lpips < expected_lpips * 1.1
