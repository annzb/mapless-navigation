import torch.nn as nn
from torch_geometric.nn import global_max_pool


class PointEncoder(nn.Module):
    def __init__(self, in_dim=4, hidden_dim=64, output_size=64, **kwargs):
        super(PointEncoder, self).__init__(**kwargs)
        self.output_size = int(output_size)
        self.lin  = nn.Linear(in_dim, hidden_dim)
        self.proj = nn.Linear(hidden_dim, output_size * 4)

    def forward(self, flat_pts, batch_idx):
        # flat_pts:   (P_total, 4)
        # batch_idx:  (P_total,) in 0..B-1
        h = self.lin(flat_pts)                        # (P_total, hidden_dim)
        pooled = global_max_pool(h, batch_idx)        # (B, hidden_dim)
        proj = self.proj(pooled)                      # (B, output_size*4)
        B = proj.size(0)                              # infer batch dimension
        return proj.view(B, self.output_size, 4)      # (B, output_size, 4)


class MlpPointEncoder(nn.Module):
    def __init__(self, in_dim=4, hidden_dim=64, output_size=64, **kwargs):
        super(MlpPointEncoder, self).__init__(**kwargs)
        self.output_size = int(output_size)
        self.mlp1 = nn.Linear(in_dim, hidden_dim)
        self.mlp2 = nn.Linear(hidden_dim, hidden_dim)
        self.proj = nn.Linear(hidden_dim, output_size * 4)

    def forward(self, flat_pts, batch_idx):
        # flat_pts:   (P_total, 4)
        # batch_idx:  (P_total,) in 0..B-1
        X = self.mlp1(flat_pts)  # (P_total, hidden_dim)
        X = self.mlp2(X)
        pooled = global_max_pool(X, batch_idx)  # (B, hidden_dim)
        proj = self.proj(pooled)  # (B, output_size*4)
        B = proj.size(0)  # infer batch dimension
        return proj.view(B, self.output_size, 4)  # (B, output_size, 4)
