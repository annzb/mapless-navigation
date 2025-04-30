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
    

class MlpPointEncoder2(nn.Module):
    def __init__(self, in_dim=4, hidden_dim=128, output_size=1024, output_features=4):
        super().__init__()
        self.output_size = output_size
        self.output_features = output_features
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
        )
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, output_size * output_features),
            nn.ReLU()
        )

    def forward(self, flat_pts, batch_idx):
        X = self.mlp(flat_pts)  # (P_total, hidden_dim)
        pooled = global_max_pool(X, batch_idx)  # (B, hidden_dim)
        proj = self.output_proj(pooled)  # (B, output_size * output_features)
        B = proj.size(0)
        return proj.view(B, self.output_size, self.output_features)


class PointEncoder2(nn.Module):
    def __init__(self, in_dim=4, hidden_dim=16, output_size=4096):
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
        
        # Weighted sum of features while preserving coordinates
        weighted_features = point_features * selected_weights.unsqueeze(-1)  # (P_total, hidden_dim)
        
        # Gather selected points and features
        selected_points = flat_pts[topk_indices.flatten()].view(batch_size, k, -1)  # (B, k, 4)
        selected_features = weighted_features[topk_indices.flatten()].view(batch_size, k, -1)  # (B, k, hidden_dim)
        
        # Process selected features while preserving coordinates
        processed_features = self.processor(selected_features)  # (B, k, 4)
        
        # Combine original coordinates with processed features
        output = torch.cat([
            selected_points[..., :3],  # Original x,y,z coordinates
            processed_features[..., 3:]  # Processed features
        ], dim=-1)
        
        # Reshape to [B, output_size, 4]
        return output.view(batch_size, self.output_size, 4)

    def check_gradient(self, tensor, name):
        if self.training:  # Only check during training
            if not tensor.requires_grad:
                raise RuntimeError(f"{name} does not require gradients")
            if tensor.grad_fn is None:
                raise RuntimeError(f"{name} is not connected to computation graph")
    

def build_encoder(in_num_features: int, out_num_features: int, hidden_num_features_multiplier=32, dropout_rate: float = 0.5, num_layers: int = 2):
    hidden_layers = []

    if num_layers >= 3:
        for i in range(1, num_layers - 1):
            in_f, out_f = hidden_num_features_multiplier * i, hidden_num_features_multiplier * (i + 1)
            layer = nn.Linear(in_f, out_f)
            norm = nn.BatchNorm1d(out_f)
            relu = nn.ReLU()
            drop = nn.Dropout(dropout_rate)
            hidden_layers.extend([layer, norm, relu, drop])

    out_f_1 = hidden_num_features_multiplier if hidden_layers else out_num_features
    in_f_2 = hidden_num_features_multiplier * (num_layers - 1) if hidden_layers else out_f_1

    return nn.Sequential(
        nn.Linear(in_num_features, out_f_1),
        nn.BatchNorm1d(out_f_1),
        nn.ReLU(),
        nn.Dropout(dropout_rate),

        *hidden_layers,

        nn.Linear(in_f_2, out_num_features),
        nn.BatchNorm1d(out_num_features),
        nn.ReLU()
    )


class DualPointEncoder(nn.Module):
    """Encoder that processes spatial and intensity features separately and combines them.
    
    Architecture:
    - Two parallel encoders for spatial and intensity features
    - Feature concatenation and fusion
    - Point selection based on combined features
    - Final processing to output format
    """
    def __init__(self, output_size=4096, dropout_rate=0.5, in_num_intensity_features=1, in_num_spatial_features=3):
        super().__init__()
        self.output_size = output_size
        hidden_dim = 32
        num_features = in_num_intensity_features + in_num_spatial_features

        # Point selection layer
        self.sampler = nn.Sequential(
            nn.Linear(num_features, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, 1)  # Remove BatchNorm1d and ReLU from final layer
        )
        
        # Feature fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(num_features, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            nn.Linear(hidden_dim, 4)  # x,y,z + occupancy
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights using Xavier uniform initialization."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=1.0)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, spatial_features, intensity_features, batch_idx):
        # Combine features
        combined_features = torch.cat([spatial_features, intensity_features], dim=-1)
        
        # Score points
        point_scores = self.sampler(combined_features)
        
        # Select top-k points
        batch_size = batch_idx.max() + 1
        k = min(self.output_size, len(batch_idx) // batch_size)
        
        # Compute weights and select points
        weights = torch.softmax(point_scores.squeeze(-1) / 2.0, dim=0)
        
        # Create batch-wise masks
        batch_masks = batch_idx.unsqueeze(1) == torch.arange(batch_size, device=batch_idx.device)
        
        # Select and process points
        selected_points = []
        selected_features = []
        
        for b in range(batch_size):
            batch_mask = batch_masks[:, b]
            batch_weights = weights[batch_mask]
            _, indices = batch_weights.topk(k, dim=0)
            
            # Create selection mask
            mask = torch.zeros_like(batch_weights, dtype=torch.bool)
            mask[indices] = True
            
            # Normalize weights for selected points
            selected_weights = batch_weights * mask.float()
            selected_weights = selected_weights / (selected_weights.sum() + 1e-8)
            
            # Get batch features
            batch_features = combined_features[batch_mask]
            
            # Weighted sum of features
            weighted_features = batch_features * selected_weights.unsqueeze(-1)
            
            # Select only top-k points
            selected_features.append(weighted_features[indices])
        
        # Stack and process
        selected_features = torch.stack(selected_features)
        
        # Final processing
        output = self.fusion(selected_features)
        
        return output.view(batch_size, self.output_size, 4)
