import torch
from diffusers import FluxPipeline
from diffusers.utils import load_image

from nunchaku import NunchakuFluxTransformer2dModel
from nunchaku.models.IP_adapter.diffusers_adapters import apply_IPA_on_pipe
from nunchaku.utils import get_precision

precision = get_precision()
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

IP_image = load_image("https://github.com/ToTheBeginning/PuLID/blob/main/example_inputs/liuyifei.png?raw=true")

image = pipeline(
    prompt="A woman holding a sign that says 'SVDQuant is fast!",
    ip_adapter_image=IP_image.convert("RGB"),
    num_inference_steps=50,
    generator=torch.Generator("cuda"),
).images[0]

image.save(f"flux.1-dev-IP-adapter-{precision}.png")
