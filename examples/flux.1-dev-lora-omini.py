import torch
from nunchaku.pipeline.pipeline_flux_omini import OminiFluxDownsizeAllInOnePipeline

from nunchaku import NunchakuOminiFluxTransformer2dModel
from nunchaku.utils import get_precision
from nunchaku.lora.flux.compose import compose_lora
from PIL import Image
import os
import math
import logging
# Configure logging with more detailed format and DEBUG level
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Set environment variable for C++ logging
os.environ["SPDLOG_LEVEL"] = "debug"
os.environ["SPDLOG_PATTERN"] = "%Y-%m-%d %H:%M:%S.%e [%l] [%t] %v"

# Function to rescale image dimensions while maintaining aspect ratio
# and ensuring dimensions are divisible by `div` (typically for model constraints).
def rescale_with_max_pixels(
    image_size: tuple[int, int], # Original (width, height)
    max_pixels: int = 768**2,    # Maximum number of pixels for the rescaled image
    div: int = 64                # Rescaled dimensions must be divisible by this number
) -> tuple[int, int]:
    w, h = image_size
    aspect = w / h

    # Calculate new dimensions based on constraining width first
    w_1 = math.sqrt(max_pixels * aspect) // div * div
    h_1 = (w_1 / aspect) // div * div

    # Calculate new dimensions based on constraining height first
    h_2 = math.sqrt(max_pixels / aspect) // div * div
    w_2 = (h_2 * aspect) // div * div

    # Choose the set of dimensions that better preserves the aspect ratio
    if abs(w_1 / h_1 - aspect) < abs(w_2 / h_2 - aspect):
        w_, h_ = w_1, h_1
    else:
        w_, h_ = w_2, h_2

    return int(w_), int(h_)


# Automatically detect precision based on GPU capabilities ('int4' or 'fp4')
precision = get_precision()  # auto-detect your precision is 'int4' or 'fp4' based on your GPU
print(f"Precision: {precision}")

# Load the NunchakuOminiFluxTransformer2dModel from a pretrained path corresponding to the detected precision.
# This model uses Nunchaku's C++ backend for acceleration.
transformer = NunchakuOminiFluxTransformer2dModel.from_pretrained(f"mit-han-lab/svdq-{precision}-flux.1-dev")

# Initialize the OminiFluxDownsizeAllInOnePipeline with the loaded Nunchaku transformer.
# This pipeline is a custom version designed for OminiFlux and potentially includes downsampling logic.
# It uses bfloat16 for mixed-precision inference and is moved to the CUDA device.
pipeline = OminiFluxDownsizeAllInOnePipeline.from_pretrained(
    "black-forest-labs/FLUX.1-dev", transformer=transformer, torch_dtype=torch.bfloat16
).to("cuda")

### LoRA Related Code ###
# Example of composing multiple LoRAs with different strengths.
# `compose_lora` likely merges these LoRAs into a single state dictionary.
composed_lora = compose_lora(
    [
        ("/mnt/sdb/lc/world_engine/output/fluxdev-empty-lora-1024-64-downsize-flux_shift-lr5e-5-2-balanced-8gpu-downsize-retrain/checkpoint-6200/design.safetensors", 1.125),
        ("/mnt/sdb/lc/ckpt/shortcut-8-step-v2.safetensors", 1.2),
    ]
)  # set your lora strengths here when using composed lora

# Update the transformer model with the composed LoRA parameters.
# The `update_lora_params` method handles applying these to both quantized and unquantized parts of the model.
transformer.update_lora_params(composed_lora)
 # Your LoRA strength here # This comment seems incomplete or refers to a removed line
### End of LoRA Related Code ###

# Alternative: Load and apply a single LoRA
# transformer.update_lora_params(
#     "/mnt/sdb/lc/world_engine/output/fluxdev-empty-lora-1024-64-downsize-flux_shift-lr5e-5-2-balanced-8gpu-downsize-retrain/checkpoint-6200/design.safetensors"
# )  # Path to your LoRA safetensors, can also be a remote HuggingFace path
# transformer.set_lora_strength(1)  # Y # This comment also seems incomplete

# The prompt for image generation.
prompt="a high quality interior design photo for a large size modern living room, within the unobstructed flow layout room:  a rectangular wood coffee table with a minimalist design and natural finish. a large area rug with a distressed pattern in neutral tones, featuring a soft texture and vintage style. a modern, white sectional sofa with soft fabric and clean lines, featuring multiple pillows in neutral tones. small round black side table next to the sofa, minimalist design, possibly metal. a collection of throw pillows with different textures and patterns, including checkered and geometric designs in neutral tones like beige, brown, and white. glass vase with long, green stems and small decorative yellow flowers. also features books and a black candle for added interest. a long, beige console table positioned behind the sofa, supporting decorative items such as books and a vase. pair of armchairs with dark gray and white stripes, featuring wooden legs and a modern design. two abstract paintings with white, black, and beige tones, framed in a minimalist style, creating a focal point on the . a large, monochrome abstract print with a simple wooden frame. it hangs on the  above the sofa, adding an artistic focal point. white, soft knitted throw blanket draped over the sofa, adding texture and warmth. two modern pendant lights with brass accents hanging over the kitchen , featuring a sleek, cylindrical design.  on either side of the , painted in dark green, displaying various decorative items. "


# Path to the input image for image-to-image or conditioning.
image_path = "/mnt/sdb/lc/test_outputs/case9/3d7b8a68-4692-4cd5-973e-4c8a1df5c48d.webp"
image = Image.open(image_path)

# Get original image dimensions and rescale them using the defined function.
image_size = image.size
image_size_ = rescale_with_max_pixels(image_size, 1024**2)

# Run the generation pipeline.
# - `image`: The input conditional image.
# - `width`, `height`: Target dimensions for the output image (rescaled).
# - `num_images_per_prompt`: Number of images to generate.
# - `num_inference_steps`: Number of denoising steps.
# - `guidance_scale`: Classifier-free guidance scale.
# - `prompt`: The text prompt.
# - `generator`: PyTorch random number generator for reproducibility.
# - `downsize`: Boolean indicating if downsampling logic in the pipeline should be used (specific to this custom pipeline).
for i in range(10):
    image_ = pipeline(
        image=image,
        width=image_size_[0],
        height=image_size_[1],
        num_images_per_prompt=1,
        num_inference_steps=6,
        guidance_scale=6,
        prompt=prompt,
        generator=torch.Generator(device="cuda").manual_seed(0),
        return_dict=True,
        downsize=True, # This flag likely controls how conditional image latents are prepared
    )[0][0]
    # print(image_)

# Save the generated image.
image_.save(f"flux.1-dev-omini-lora-{precision}.png")
