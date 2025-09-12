from diffusers import QwenImageControlNetPipeline, QwenImageControlNetModel
from diffusers.utils import load_image
from typing import Union, List, Optional
from PIL import Image
from nunchaku.models.transformers.transformer_qwenimage import NunchakuQwenImageTransformer2DModel
import torch
from nunchaku.utils import get_gpu_memory, get_precision

class PatchedQwenImageControlNetPipeline(QwenImageControlNetPipeline):
    def _get_qwen_prompt_embeds(
        self,
        prompt: Union[str, List[str]] = None,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        device = device or self._execution_device
        dtype = dtype or self.text_encoder.dtype

        prompt = [prompt] if isinstance(prompt, str) else prompt

        template = self.prompt_template_encode
        drop_idx = self.prompt_template_encode_start_idx
        txt = [template.format(e) for e in prompt]
        txt_tokens = self.tokenizer(
            txt, max_length=self.tokenizer_max_length + drop_idx, padding=True, truncation=True, return_tensors="pt"
        ).to(device) # <--- The fix
        encoder_hidden_states = self.text_encoder(
            input_ids=txt_tokens.input_ids,
            attention_mask=txt_tokens.attention_mask,
            output_hidden_states=True,
        )
        hidden_states = encoder_hidden_states.hidden_states[-1]
        split_hidden_states = self._extract_masked_hidden(hidden_states, txt_tokens.attention_mask)
        split_hidden_states = [e[drop_idx:] for e in split_hidden_states]
        attn_mask_list = [torch.ones(e.size(0), dtype=torch.long, device=e.device) for e in split_hidden_states]
        max_seq_len = max([e.size(0) for e in split_hidden_states])
        prompt_embeds = torch.stack(
            [torch.cat([u, u.new_zeros(max_seq_len - u.size(0), u.size(1))]) for u in split_hidden_states]
        )
        encoder_attention_mask = torch.stack(
            [torch.cat([u, u.new_zeros(max_seq_len - u.size(0))]) for u in attn_mask_list]
        )

        prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)

        return prompt_embeds, encoder_attention_mask

model_name = "Qwen/Qwen-Image"
rank = 32  # you can also use rank=128 model to improve the quality

# Load components with correct dtype
controlnet = QwenImageControlNetModel.from_pretrained(
    "InstantX/Qwen-Image-ControlNet-Union",
    torch_dtype=torch.bfloat16
)
transformer = NunchakuQwenImageTransformer2DModel.from_pretrained(
    f"nunchaku-tech/nunchaku-qwen-image/svdq-{get_precision()}_r{rank}-qwen-image.safetensors"
)

# Create pipeline
pipeline = PatchedQwenImageControlNetPipeline.from_pretrained(
       model_name, 
       transformer=transformer, 
       controlnet=controlnet,
       torch_dtype=torch.bfloat16
)

if get_gpu_memory() > 18:
    pipeline.enable_model_cpu_offload()
else:
    # use per-layer offloading for low VRAM. This only requires 3-4GB of VRAM.
    transformer.set_offload(True)
    pipeline._exclude_from_cpu_offload.append("transformer")
    pipeline.enable_sequential_cpu_offload()

control_image = load_image("https://huggingface.co/InstantX/Qwen-Image-ControlNet-Union/resolve/main/conds/depth.png")

# Generate with control
image = pipeline(
       prompt="A swanky, minimalist living room with a huge floor-to-ceiling window letting in loads of natural light. A beige couch with white cushions sits on a wooden floor, with a matching coffee table in front. The walls are a soft, warm beige, decorated with two framed botanical prints. A potted plant chills in the corner near the window. Sunlight pours through the leaves outside, casting cool shadows on the floor.",
       negative_prompt=" ",
       control_image=control_image,
       controlnet_conditioning_scale=1.0,
       num_inference_steps=30,
       true_cfg_scale=4.0
   ).images[0]

# Save the result
image.save(f"controlnet_depth_r{rank}.png")

