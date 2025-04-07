import torch
from diffusers import FluxPipeline

from nunchaku.models.transformers.transformer_flux_FB_cache import NunchakuFluxTransformer2dModel
from nunchaku.models.transformers.DFB_cache import *

device = torch.device("cuda")

threshold_multi = 0.01
threshold_single = 0.01
adaptive_th = 0.9
USE_FBCahe = True
graph_save_path = "./results/FB_cahce.png"

out_base = f"results/flux_t"  # 기본 파일 경로

transformer,m = NunchakuFluxTransformer2dModel.from_pretrained("mit-han-lab/svdq-int4-flux.1-dev")


pipeline = FluxPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-dev", transformer=transformer, torch_dtype=torch.bfloat16
).to("cuda")

#prompts = "A panda holding a sign that says hello world. Very realistic photo."


#prompts = ["A panda drinking beer. Very realistic photo.",
           #"A Welsh Corgi holding a sign that says hello world. Very realistic photo."]


prompts = ["A panda drinking beer. Very realistic photo.",
           "A Welsh Corgi holding a sign that says hello world. Very realistic photo.",
       "A dragon holding a sign that says hello world. Very realistic photo."
           ]



with FBTransformerCacheContext() as fb_ctx:
    transformer.set_residual_diff_threshold(
        threshold_multi=threshold_multi,
        threshold_single=threshold_single,
        adaptive_th=adaptive_th,
        USE_FBCahe = USE_FBCahe
        )
    image = pipeline(prompts, num_inference_steps=50, guidance_scale=3.5)
    
    for idx, img in enumerate(image[0]):
        out_path = f"{out_base}_batch_{idx+1}.png"
        safe_save(img, out_path)