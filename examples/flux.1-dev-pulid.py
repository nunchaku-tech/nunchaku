from nunchaku.pipeline.pipeline_flux_pulid import PuLIDFluxPipeline
from nunchaku.models.pulid.pulid_forward import forward
import torch
from types import MethodType
from nunchaku.models.transformers.transformer_flux import NunchakuFluxTransformer2dModel
from diffusers.utils import load_image


transformer = NunchakuFluxTransformer2dModel.from_pretrained("mit-han-lab/svdq-int4-flux.1-dev")

pipe = PuLIDFluxPipeline.from_pretrained("black-forest-labs/FLUX.1-dev",transformer=transformer, torch_dtype=torch.bfloat16,).to('cuda')

pipe.transformer.forward = MethodType(forward, pipe.transformer)

id_image = load_image("https://github.com/ToTheBeginning/PuLID/blob/main/example_inputs/liuyifei.png?raw=true")

image = pipe(
    "A woman holding a sign that says hello world",
    id_image=id_image,
    id_weight=1,
    num_inference_steps=12,
    guidance_scale=3.5).images[0]
image.save("./flux.1-dev-pulid.png")
