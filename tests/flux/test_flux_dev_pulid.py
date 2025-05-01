from types import MethodType

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from diffusers.utils import load_image

from nunchaku import NunchakuFluxTransformer2dModel
from nunchaku.models.pulid.pulid_forward import pulid_forward
from nunchaku.models.pulid.utils import resize_numpy_image_long
from nunchaku.pipeline.pipeline_flux_pulid import PuLIDFluxPipeline
from nunchaku.utils import get_precision, is_turing


@pytest.mark.skipif(is_turing(), reason="Skip tests due to using Turing GPUs")
def test_flux_dev_pulid():
    precision = get_precision()  # auto-detect your precision is 'int4' or 'fp4' based on your GPU
    transformer = NunchakuFluxTransformer2dModel.from_pretrained(f"mit-han-lab/svdq-{precision}-flux.1-dev")

    pipeline = PuLIDFluxPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-dev",
        transformer=transformer,
        torch_dtype=torch.bfloat16,
    ).to("cuda")

    pipeline.transformer.forward = MethodType(pulid_forward, pipeline.transformer)

    id_image = load_image("https://github.com/ToTheBeginning/PuLID/blob/main/example_inputs/liuyifei.png?raw=true")

    image = pipeline(
        "A woman holding a sign that says hello world",
        id_image=id_image,
        id_weight=1,
        num_inference_steps=12,
        guidance_scale=3.5,
    ).images[0]

    id_image = id_image.convert("RGB")
    id_image_numpy = np.array(id_image)
    id_image = resize_numpy_image_long(id_image_numpy, 1024)
    id_embeddings, _ = pipeline.pulid_model.get_id_embedding(id_image)

    output_image = image.convert("RGB")
    output_image_numpy = np.array(output_image)
    output_image = resize_numpy_image_long(output_image_numpy, 1024)
    output_id_embeddings, _ = pipeline.pulid_model.get_id_embedding(output_image)
    cosine_similarities = (
        F.cosine_similarity(id_embeddings.view(32, 2048), output_id_embeddings.view(32, 2048), dim=1).mean().item()
    )
    print(cosine_similarities)
    assert cosine_similarities > 0.93
