import torch.nn as nn
from torch_geometric.nn import global_max_pool
import torch
    

class MlpPointEncoder(nn.Module):
    def __init__(self, in_dim=4, hidden_dim=128, output_size=1024, output_features=4):
        super().__init__()
        self.output_size = output_size
        self.output_features = output_features
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, output_size * output_features),
            nn.ReLU()
        )
        nn.init.xavier_uniform_(self.output_proj[0].weight, gain=1.0)
        if self.output_proj[0].bias is not None:
            nn.init.zeros_(self.output_proj[0].bias)

    def forward(self, flat_pts, batch_idx):
        X = self.mlp(flat_pts)  # (P_total, hidden_dim)
        pooled = global_max_pool(X, batch_idx)  # (B, hidden_dim)
        proj = self.output_proj(pooled)  # (B, output_size * output_features)
        B = proj.size(0)
        return proj.view(B, self.output_size, self.output_features)


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
