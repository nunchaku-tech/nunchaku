import torch
from diffusers import FluxPipeline

from nunchaku import NunchakuFluxTransformer2dModel


device = torch.device("cuda")


transformer,m = NunchakuFluxTransformer2dModel.from_pretrained("mit-han-lab/svdq-int4-flux.1-dev")


hidden_states = torch.randn(2, 4096, 3072, dtype=torch.bfloat16).to(device)
encoder_hidden_states = torch.randn(2, 512, 3072, dtype=torch.bfloat16).to(device)
temb = torch.randn(2, 3072, dtype=torch.bfloat16).to(device)

rotary_emb_img = torch.randn(2, 4096, 64, 1, 2, dtype=torch.float32).to(device)
rotary_emb_txt = torch.randn(2, 512, 64, 1, 2, dtype=torch.float32).to(device)
rotary_emb_single = torch.randn(2, 4608, 64, 1, 2, dtype=torch.float32).to(device)

'''
hidden_states = torch.randn(1, 4096, 3072, dtype=torch.bfloat16).to(device)
encoder_hidden_states = torch.randn(1, 512, 3072, dtype=torch.bfloat16).to(device)
temb = torch.randn(1, 3072, dtype=torch.bfloat16).to(device)

rotary_emb_img = torch.randn(1, 4096, 64, 1, 2, dtype=torch.float32).to(device)
rotary_emb_txt = torch.randn(1, 512, 64, 1, 2, dtype=torch.float32).to(device)
rotary_emb_single = torch.randn(1, 4608, 64, 1, 2, dtype=torch.float32).to(device)
'''
out = m.forward(
            hidden_states, encoder_hidden_states, temb, rotary_emb_img, rotary_emb_txt, rotary_emb_single
        )

print(out.shape)