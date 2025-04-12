from nunchaku.models.pulid.pipeline_flux import PuLIDFluxPipeline
from nunchaku.models.pulid.pulid_forward import forward
import torch
from types import MethodType
from nunchaku.models.transformers.transformer_flux import NunchakuFluxTransformer2dModel


transformer = NunchakuFluxTransformer2dModel.from_pretrained("mit-han-lab/svdq-int4-flux.1-dev")

cache_dir = "/workspace/svdquant/pins/model_cache"


pipe = PuLIDFluxPipeline.from_pretrained("black-forest-labs/FLUX.1-dev",transformer=transformer, torch_dtype=torch.bfloat16, cache_dir=cache_dir).to('cuda')

pipe.transformer.forward = MethodType(forward, pipe.transformer)

image = pipe("A man holding a sign that says hello world", id_image='/workspace/svdquant/pins/man.png', id_weight = 1, num_inference_steps=12, guidance_scale=3.5).images[0]
image.save("/workspace/svdquant/mysvdquant/lecun.png")