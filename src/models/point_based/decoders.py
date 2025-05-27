import torch.nn as nn
import torch
    

class MlpDecoder(nn.Module):
    def __init__(self, latent_dim=4, output_size=1024, output_dim=4):
        super().__init__()
        self.output_size = output_size
        self.output_dim = output_dim
        self.mlp = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(p=0.2),

            nn.Linear(256, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            
            nn.Linear(512, output_size * output_dim)
        )

    def forward(self, z):
        B = z.shape[0]
        out = self.mlp(z)  # (B, output_size * output_dim)
        return out.view(B, self.output_size, self.output_dim)
