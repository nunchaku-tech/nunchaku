import torch
import torch.nn as nn


class IDEncoder(nn.Module):
    def __init__(self, width=1280, context_dim=2048, num_token=5):
        super().__init__()
        self.num_token = num_token
        self.context_dim = context_dim
        h1 = min((context_dim * num_token) // 4, 1024)
        h2 = min((context_dim * num_token) // 2, 1024)
        self.body = nn.Sequential(
            nn.Linear(width, h1),
            nn.LayerNorm(h1),
            nn.LeakyReLU(),
            nn.Linear(h1, h2),
            nn.LayerNorm(h2),
            nn.LeakyReLU(),
            nn.Linear(h2, context_dim * num_token),
        )

        for i in range(5):
            setattr(
                self,
                f'mapping_{i}',
                nn.Sequential(
                    nn.Linear(1024, 1024),
                    nn.LayerNorm(1024),
                    nn.LeakyReLU(),
                    nn.Linear(1024, 1024),
                    nn.LayerNorm(1024),
                    nn.LeakyReLU(),
                    nn.Linear(1024, context_dim),
                ),
            )

            setattr(
                self,
                f'mapping_patch_{i}',
                nn.Sequential(
                    nn.Linear(1024, 1024),
                    nn.LayerNorm(1024),
                    nn.LeakyReLU(),
                    nn.Linear(1024, 1024),
                    nn.LayerNorm(1024),
                    nn.LeakyReLU(),
                    nn.Linear(1024, context_dim),
                ),
            )

    def forward(self, x, y):
        # x shape [N, C]
        x = self.body(x)
        x = x.reshape(-1, self.num_token, self.context_dim)

        hidden_states = ()
        for i, emb in enumerate(y):
            hidden_state = getattr(self, f'mapping_{i}')(emb[:, :1]) + getattr(self, f'mapping_patch_{i}')(
                emb[:, 1:]
            ).mean(dim=1, keepdim=True)
            hidden_states += (hidden_state,)
        hidden_states = torch.cat(hidden_states, dim=1)

        return torch.cat([x, hidden_states], dim=1)
