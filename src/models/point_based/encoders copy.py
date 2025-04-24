import torch.nn as nn
from torch_geometric.nn import global_max_pool
import torch


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


class PointEncoder2(nn.Module):
    def __init__(self, in_dim=4, hidden_dim=256, output_size=4096):
        super().__init__()
        self.output_size = output_size
        
        # Process each point
        self.point_mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Sampler layer to select most important points
        self.sampler = nn.Sequential(
            nn.Linear(256, 1),
            nn.BatchNorm1d(1),
            nn.ReLU()
        )
        
        # Initialize sampler weights with small values
        nn.init.xavier_uniform_(self.sampler[0].weight, gain=0.01)
        nn.init.zeros_(self.sampler[0].bias)
        
        # Process selected points
        self.processor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 4)  # x,y,z + feature
        )

    def forward(self, flat_pts, batch_idx):
        # Process each point
        point_features = self.point_mlp(flat_pts)  # (P_total, hidden_dim)
        
        # Score points
        point_scores = self.sampler(point_features)  # (P_total, 1)
        
        # Get batch size and prepare for vectorized processing
        batch_size = batch_idx.max() + 1
        k = min(self.output_size, len(flat_pts) // batch_size)
        
        # Compute weights for all points at once
        weights = torch.softmax(point_scores.squeeze(-1) / 0.1, dim=0)  # (P_total,)
        
        # Create batch-wise masks and indices
        batch_masks = batch_idx.unsqueeze(1) == torch.arange(batch_size, device=batch_idx.device)  # (P_total, B)
        
        # For each batch, get top-k indices
        topk_indices = []
        for b in range(batch_size):
            batch_mask = batch_masks[:, b]
            batch_weights = weights[batch_mask]
            _, indices = batch_weights.topk(k, dim=0)
            # Convert local indices to global indices
            global_indices = torch.nonzero(batch_mask)[indices].squeeze(-1)
            topk_indices.append(global_indices)
        
        # Stack indices and create selection mask
        topk_indices = torch.stack(topk_indices)  # (B, k)
        selection_mask = torch.zeros_like(weights, dtype=torch.bool)
        selection_mask[topk_indices.flatten()] = True
        
        # Apply selection mask to weights
        selected_weights = weights * selection_mask.float()
        # Normalize weights per batch
        batch_sums = torch.zeros(batch_size, device=weights.device)
        for b in range(batch_size):
            batch_mask = batch_masks[:, b]
            batch_sums[b] = selected_weights[batch_mask].sum()
        selected_weights = selected_weights / (batch_sums[batch_idx] + 1e-8)
        
        # Weighted sum of points and features
        weighted_points = flat_pts * selected_weights.unsqueeze(-1)  # (P_total, 4)
        weighted_features = point_features * selected_weights.unsqueeze(-1)  # (P_total, hidden_dim)
        
        # Gather selected points and features
        selected_points = weighted_points[topk_indices.flatten()].view(batch_size, k, -1)  # (B, k, 4)
        selected_features = weighted_features[topk_indices.flatten()].view(batch_size, k, -1)  # (B, k, hidden_dim)
        
        # Process selected points
        processed = self.processor(selected_features)  # (B, k, 4)
        
        # Reshape to [B, output_size, 4]
        return processed.view(batch_size, self.output_size, 4)

    def check_gradient(self, tensor, name):
        if self.training:  # Only check during training
            if not tensor.requires_grad:
                raise RuntimeError(f"{name} does not require gradients")
            if tensor.grad_fn is None:
                raise RuntimeError(f"{name} is not connected to computation graph")
