import torch
from diffusers import FluxPipeline

from nunchaku import NunchakuFluxTransformer2dModel
from nunchaku.utils import get_precision

precision = get_precision()  # auto-detect your precision is 'int4' or 'fp4' based on your GPU
transformer = NunchakuFluxTransformer2dModel.from_pretrained(
    f"mit-han-lab/nunchaku-flux.1-dev/svdq-{precision}_r32-flux.1-dev.safetensors"
)
pipeline = FluxPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-dev", transformer=transformer, torch_dtype=torch.bfloat16
).to("cuda")

### LoRA Related Code ###
transformer.update_lora_params(
    "./loras/mage.safetensors"
)  # Path to your LoRA safetensors, can also be a remote HuggingFace path
transformer.set_lora_strength(1)  # Your LoRA strength here
### End of LoRA Related Code ###

prompt = "masterful impressionism oil painting titled 'the violinist', the composition follows the rule of thirds, placing the violinist centrally in the frame. the subject is a young woman with fair skin and light blonde hair is styled in a long, flowing hairstyle with natural waves. she is dressed in an opulent, luxurious silver silk gown with a high waist and intricate gold detailing along the bodice. the gown's texture is smooth and reflective. she holds a violin under her chin, her right hand poised to play, and her left hand supporting the neck of the instrument. she wears a delicate gold necklace with small, sparkling gemstones that catch the light. her beautiful eyes focused on the viewer. the background features an elegantly furnished room with classical late 19th century decor. to the left, there is a large, ornate portrait of a man in a dark suit, set in a gilded frame. below this, a wooden desk with a closed book. to the right, a red upholstered chair with a wooden frame is partially visible. the room is bathed in natural light streaming through a window with red curtains, creating a warm, inviting atmosphere. the lighting highlights the violinist, casting soft shadows that enhance the depth and realism of the scene, highly aesthetic, harmonious colors, impressioniststrokes, <lora:style-impressionist_strokes-flux-by_daalis:1.0> <lora:image_upgrade-flux-by_zeronwo7829:1.0>"

image = pipeline(
    prompt,
    num_inference_steps=20,
    guidance_scale=3.5,
    generator=torch.Generator().manual_seed(42),
).images[0]
image.save(f"flux.1-dev-{precision}-1.png")
