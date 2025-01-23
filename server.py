import os
import time
import uuid
from typing import List
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
from diffusers import FluxPipeline
from nunchaku.models.transformer_flux import NunchakuFluxTransformer2dModel
from safety_checker.censor import check_safety

app = FastAPI(title="FLUX Image Generation API")

MODEL_ID = "black-forest-labs/FLUX.1-schnell"
MODEL_CACHE = "model-cache"
QUANT_MODEL_PATH = "mit-han-lab/svdq-int4-flux.1-schnell"

class ImageRequest(BaseModel):
    prompt: str = "a photo of an astronaut riding a horse on mars"
    width: int = 1024
    height: int = 1024
    num_inference_steps: int = 4
    seed: int | None = None
    safety_checker_adj: float = 0.5  # Controls sensitivity of NSFW detection

class ImageResponse(BaseModel):
    image_path: str

pipe = None

@app.on_event("startup")
async def startup_event():
    global pipe
    print("Loading FLUX pipeline...")
    transformer = NunchakuFluxTransformer2dModel.from_pretrained(QUANT_MODEL_PATH)
    pipe = FluxPipeline.from_pretrained(
        MODEL_ID,
        transformer=transformer,
        torch_dtype=torch.bfloat16
    ).to("cuda")

def find_nearest_valid_dimensions(width: float, height: float) -> tuple[int, int]:
    """Find the nearest dimensions that are multiples of 8 and their product is divisible by 65536."""
    start_w = round(width)
    start_h = round(height)
    
    def is_valid(w: int, h: int) -> bool:
        return w % 8 == 0 and h % 8 == 0 and (w * h) % 65536 == 0
    
    # Find nearest multiple of 8 for each dimension
    nearest_w = round(start_w / 8) * 8
    nearest_h = round(start_h / 8) * 8
    
    # Search in a spiral pattern from the nearest multiples of 8
    offset = 0
    while offset < 100:  # Limit search to reasonable range
        for w in range(nearest_w - offset * 8, nearest_w + offset * 8 + 1, 8):
            if w <= 0:
                continue
            for h in range(nearest_h - offset * 8, nearest_h + offset * 8 + 1, 8):
                if h <= 0:
                    continue
                if is_valid(w, h):
                    return w, h
        offset += 1
    
    # If no valid dimensions found, return the nearest multiples of 8
    return nearest_w, nearest_h

@app.post("/predict", response_model=ImageResponse)
async def predict(request: ImageRequest):
    print(f"Request: {request}")
    if pipe is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
        
    seed = request.seed if request.seed is not None else int.from_bytes(os.urandom(2), "big")
    print(f"Using seed: {seed}")

    generator = torch.Generator("cuda").manual_seed(seed)
    
    # Find nearest valid dimensions
    width, height = find_nearest_valid_dimensions(request.width, request.height)
    print(f"Original dimensions: {request.width}x{request.height}")
    print(f"Adjusted dimensions: {width}x{height}")

    with torch.inference_mode():
        output = pipe(
            prompt=request.prompt,
            generator=generator,
            width=width,
            height=height,
            num_inference_steps=request.num_inference_steps,
        )

    # Check for NSFW content
    image = output.images[0]
    concepts, has_nsfw = check_safety([image], request.safety_checker_adj)
    if has_nsfw[0]:
        raise HTTPException(status_code=400, detail="Generated image contains NSFW content")

    # Create unique filename using timestamp and UUID
    timestamp = int(time.time())
    random_id = str(uuid.uuid4())[:8]
    output_path = f"/tmp/out-{timestamp}-{random_id}.jpg"
    
    output.images[0].save(output_path, format='JPEG', quality=95)
    
    return ImageResponse(image_path=output_path)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)