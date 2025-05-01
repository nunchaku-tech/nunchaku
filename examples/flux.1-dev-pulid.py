from types import MethodType

import torch
from diffusers.utils import load_image

from nunchaku.models.pulid.pulid_forward import pulid_forward
from nunchaku.models.transformers.transformer_flux import NunchakuFluxTransformer2dModel
from nunchaku.pipeline.pipeline_flux_pulid import PuLIDFluxPipeline
from nunchaku.utils import get_precision

precision = get_precision()
transformer = NunchakuFluxTransformer2dModel.from_pretrained(f"mit-han-lab/svdq-{precision}-flux.1-dev")

pipeline = PuLIDFluxPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-dev",
    transformer=transformer,
    torch_dtype=torch.bfloat16,
).to("cuda")

pipeline.transformer.forward = MethodType(pulid_forward, pipeline.transformer)

id_image = load_image("https://github.com/ToTheBeginning/PuLID/blob/main/example_inputs/liuyifei.png?raw=true")

image = pipeline(
    "A woman holding a sign that says 'SVDQuant is fast!",
    id_image=id_image,
    id_weight=1,
    num_inference_steps=12,
    guidance_scale=3.5,
).images[0]
image.save("flux.1-dev-pulid.png")
