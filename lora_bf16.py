import torch
from diffusers.pipelines import FluxPipeline

model_id = "black-forest-labs/FLUX.1-dev"

pipe = FluxPipeline.from_pretrained(model_id, torch_dtype=torch.bfloat16)

pipe.load_lora_weights("./converted_mage.safetensors", adapter_name="default")
pipe.set_adapters(["default"], adapter_weights=[1])

# with open("flux.txt", "w") as f:
#     f.write(str(pipe.transformer) + "\n")

pipe.enable_model_cpu_offload()

prompt = "masterful impressionism oil painting titled 'the violinist', the composition follows the rule of thirds, placing the violinist centrally in the frame. the subject is a young woman with fair skin and light blonde hair is styled in a long, flowing hairstyle with natural waves. she is dressed in an opulent, luxurious silver silk gown with a high waist and intricate gold detailing along the bodice. the gown's texture is smooth and reflective. she holds a violin under her chin, her right hand poised to play, and her left hand supporting the neck of the instrument. she wears a delicate gold necklace with small, sparkling gemstones that catch the light. her beautiful eyes focused on the viewer. the background features an elegantly furnished room with classical late 19th century decor. to the left, there is a large, ornate portrait of a man in a dark suit, set in a gilded frame. below this, a wooden desk with a closed book. to the right, a red upholstered chair with a wooden frame is partially visible. the room is bathed in natural light streaming through a window with red curtains, creating a warm, inviting atmosphere. the lighting highlights the violinist, casting soft shadows that enhance the depth and realism of the scene, highly aesthetic, harmonious colors, impressioniststrokes, <lora:style-impressionist_strokes-flux-by_daalis:1.0> <lora:image_upgrade-flux-by_zeronwo7829:1.0>"
image = pipe(
    prompt=prompt,
    guidance_scale=3.5,
    height=1024,
    width=1024,
    num_inference_steps=20,
    max_sequence_length=512,
    generator=torch.Generator().manual_seed(42),
).images[0]
image.save("bf16_1.png")
