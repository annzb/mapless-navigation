from typing import Optional

import torch.nn as nn
import torch
    

class MlpDecoder(nn.Module):
    def __init__(self, latent_dim=4, output_size=1024, output_dim=4, layer_norm: bool = True, dropout: Optional[float] = None, **kwargs):
        super().__init__()
        if not isinstance(output_size, int):
            raise ValueError('output_size must be an integer')
        if output_size <= 0:
            raise ValueError('output_size must be positive')
        if not isinstance(output_dim, int):
            raise ValueError('output_dim must be an integer')
        if output_dim <= 0:
            raise ValueError('output_dim must be positive')
        if not isinstance(latent_dim, int):
            raise ValueError('latent_dim must be an integer')
        if latent_dim <= 0:
            raise ValueError('latent_dim must be positive')
        if not isinstance(layer_norm, bool):
            raise ValueError('layer_norm must be a boolean')
        if dropout is not None and not isinstance(dropout, float):
            raise ValueError('dropout must be a float')
        if dropout is not None and (dropout < 0.0 or dropout > 1.0):
            raise ValueError('dropout must be between 0.0 and 1.0')

        self.output_size = output_size
        self.output_dim = output_dim

        layers = []
        layer_dims = [latent_dim, 128, 256, 512]
        for in_dim, out_dim in zip(layer_dims[:-1], layer_dims[1:]):
            layers.append(nn.Linear(in_dim, out_dim))
            if layer_norm:
                layers.append(nn.LayerNorm(out_dim))
            layers.append(nn.ReLU())
            if dropout is not None:
                layers.append(nn.Dropout(p=dropout))
        layers.append(nn.Linear(layer_dims[-1], output_size * output_dim))
        self.mlp = nn.Sequential(*layers)

    def forward(self, z):
        B = z.shape[0]
        out = self.mlp(z)  # (B, output_size * output_dim)
        return out.view(B, self.output_size, self.output_dim)
