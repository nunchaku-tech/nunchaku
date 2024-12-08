import os
import time
import uuid
from typing import List
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
from diffusers import FluxPipeline
from nunchaku.models.transformer_flux import NunchakuFluxTransformer2dModel

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

@app.post("/predict", response_model=ImageResponse)
async def predict(request: ImageRequest):
    if pipe is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
        
    seed = request.seed if request.seed is not None else int.from_bytes(os.urandom(2), "big")
    print(f"Using seed: {seed}")

    generator = torch.Generator("cuda").manual_seed(seed)
    
    with torch.inference_mode():
        output = pipe(
            prompt=request.prompt,
            generator=generator,
            width=request.width,
            height=request.height,
            num_inference_steps=request.num_inference_steps,
        )

    # Create unique filename using timestamp and UUID
    timestamp = int(time.time())
    random_id = str(uuid.uuid4())[:8]
    output_path = f"/tmp/out-{timestamp}-{random_id}.jpg"
    
    output.images[0].save(output_path, format='JPEG', quality=95)
    
    return ImageResponse(image_path=output_path)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)