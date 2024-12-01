import os
import time
import uuid
from typing import List

import torch
from cog import BasePredictor, Input, Path
from nunchaku.pipelines import flux as nunchaku_flux

MODEL_ID = "black-forest-labs/FLUX.1-schnell"
MODEL_CACHE = "model-cache"
QUANT_MODEL_PATH = "mit-han-lab/svdq-int4-flux.1-schnell"

class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""
        print("Loading FLUX pipeline...")
        self.pipe = nunchaku_flux.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.bfloat16,
            qmodel_path=QUANT_MODEL_PATH,
            # cache_dir=MODEL_CACHE,
            # local_files_only=False,
        )
        self.pipe.enable_sequential_cpu_offload()

    @torch.inference_mode()
    def predict(
        self,
        prompt: str = Input(
            description="Input prompt",
            default="a photo of an astronaut riding a horse on mars",
        ),
        width: int = Input(
            description="Width of output image",
            default=1024,
        ),
        height: int = Input(
            description="Height of output image",
            default=1024,
        ),
        num_inference_steps: int = Input(
            description="Number of denoising steps", ge=1, le=10, default=4
        ),
        seed: int = Input(
            description="Random seed. Leave blank to randomize the seed", default=None
        ),
    ) -> List[Path]:
        """Run a single prediction on the model"""
        if seed is None:
            seed = int.from_bytes(os.urandom(2), "big")
        print(f"Using seed: {seed}")

        generator = torch.Generator("cuda").manual_seed(seed)
        
        output = self.pipe(
            prompt=prompt,
            generator=generator,
            width=width,
            height=height,
            num_inference_steps=num_inference_steps,
        )

        # Create unique filename using timestamp and UUID
        timestamp = int(time.time())
        random_id = str(uuid.uuid4())[:8]
        output_path = f"/tmp/out-{timestamp}-{random_id}.jpg"
        
        output.images[0].save(output_path, format='JPEG', quality=95)
        
        return [Path(output_path)]