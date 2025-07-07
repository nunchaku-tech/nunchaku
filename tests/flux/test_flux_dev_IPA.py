import numpy as np
import pytest
import torch
import torch.nn.functional as F
from diffusers import FluxPipeline
from diffusers.utils import load_image

from nunchaku import NunchakuFluxTransformer2dModel
from nunchaku.models.IP_adapter.diffusers_adapters import apply_IPA_on_pipe
from nunchaku.models.IP_adapter.utils import get_puild_embed, resize_numpy_image_long
from nunchaku.utils import get_precision, is_turing


@pytest.mark.skipif(is_turing(), reason="Skip tests due to using Turing GPUs")
def test_flux_dev_IPA():
    precision = get_precision()  # auto-detect your precision is 'int4' or 'fp4' based on your GPU
    transformer = NunchakuFluxTransformer2dModel.from_pretrained(f"mit-han-lab/svdq-{precision}-flux.1-dev")

    pipeline = FluxPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-dev", transformer=transformer, torch_dtype=torch.bfloat16
    ).to("cuda")

    pipeline.load_ip_adapter(
        pretrained_model_name_or_path_or_dict="XLabs-AI/flux-ip-adapter-v2",
        weight_name="ip_adapter.safetensors",
        image_encoder_pretrained_model_name_or_path="openai/clip-vit-large-patch14",
    )
    apply_IPA_on_pipe(pipeline, ip_adapter_scale=1.0, repo_id="XLabs-AI/flux-ip-adapter-v2")

    id_image = load_image("https://github.com/ToTheBeginning/PuLID/blob/main/example_inputs/liuyifei.png?raw=true")

    image = pipeline(
        prompt="A woman holding a sign that says 'SVDQuant is fast!",
        ip_adapter_image=id_image.convert("RGB"),
        num_inference_steps=50,
        generator=torch.Generator("cuda"),
    ).images[0]

    id_image = id_image.convert("RGB")
    id_image_numpy = np.array(id_image)
    id_image = resize_numpy_image_long(id_image_numpy, 1024)
    id_embeddings, _ = get_puild_embed(id_image, "cuda")

    output_image = image.convert("RGB")
    output_image_numpy = np.array(output_image)
    output_image = resize_numpy_image_long(output_image_numpy, 1024)
    output_id_embeddings, _ = get_puild_embed(output_image, "cuda")
    cosine_similarities = (
        F.cosine_similarity(id_embeddings.view(32, 2048), output_id_embeddings.view(32, 2048), dim=1).mean().item()
    )
    print(cosine_similarities)
    assert cosine_similarities > 0.70


if __name__ == "__main__":
    test_flux_dev_IPA()
