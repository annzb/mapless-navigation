from models.base import RadarOccupancyModel
from models.point_based.polar_to_cartesian import PolarToCartesianPoints
from models.point_based.downsampling import TrainedDownsampling
from models.point_based.pointnet import PointNet

import torch
import torch.nn as nn
from torch_geometric.nn import global_max_pool


class PointEncoder(nn.Module):
    def __init__(self, in_dim=4, hidden_dim=64, out_dim=128, **kwargs):
        super(PointEncoder, self).__init__(**kwargs)
        self.lin  = nn.Linear(in_dim, hidden_dim)
        self.proj = nn.Linear(hidden_dim, out_dim)
        self.out_dim = out_dim

    def forward(self, flat_pts, batch_idx, batch_size=1, **kwargs):
        """
        flat_pts:   (P_total, 4)
        batch_idx:  (P_total,) each element in 0..B-1
        returns:    (B, out_dim)  fixed-size embedding per sample
        """
        h = self.lin(flat_pts)                      # (P_total, hidden_dim)
        # use torch_geometric’s global_max_pool to aggregate per-sample
        pooled = global_max_pool(h, batch_idx)      # (B, hidden_dim)
        proj = self.proj(pooled)                    # (B, out_dim)
        result = proj.view(batch_size, int(self.out_dim / 4), 4)
        return result


class PointBaseline(RadarOccupancyModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = 'baseline_v1.0'
        self.encoder = PointEncoder(in_dim=4, hidden_dim=128, out_dim=256)

    def forward(self, X, batch_size=1, debug=False, **kwargs):
        flat_pts, batch_idx = X
        embeddings = self.encoder(flat_pts, batch_idx, batch_size=batch_size, **kwargs)
        embeddings_flat, embeddings_flat_indices = self.merge_batches(embeddings)
        probs = self.apply_sigmoid(embeddings_flat)

        if debug:
            return embeddings_flat, embeddings_flat_indices, probs
        return probs, embeddings_flat_indices


class SinglePointnet(RadarOccupancyModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pointnet = PointNet()
        self.name = 'single_pointnet_v1.0'

    def forward(self, X, debug=False):
        points, batch_indices = X
        predicted_log_odds = self.pointnet(points)
        predicted_probabilities = self.apply_sigmoid(predicted_log_odds)
        if not debug:
            return predicted_probabilities, batch_indices
        return predicted_log_odds, predicted_probabilities, batch_indices


class PointOccupancyModel(RadarOccupancyModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        num_points = self.radar_config.num_azimuth_bins * self.radar_config.num_range_bins * self.radar_config.num_elevation_bins
        self.polar_to_cartesian = PolarToCartesianPoints(radar_config=self.radar_config)
        self.down = TrainedDownsampling(num_points, retain_fraction=0.05)
        self.pointnet = PointNet()
        self.name = 'cart+down+pointnet_v1.1'

    def forward(self, polar_frames, debug=False):
        cartesian_points = self.polar_to_cartesian(polar_frames)                       # [B, 153600, 4]
        downsampled_points = self.down(cartesian_points)                               # [B, 7680, 4]
        points = downsampled_points[..., :3]                                           # [B, 7680, 3]
        features = downsampled_points[..., 3:]                                         # [B, 7680, 1]
        log_odds = self.pointnet(points, features)                                     # [B, 7680, 4]

        predicted_batch_flat, predicted_batch_indices = self.merge_batches(log_odds)   # [B * 7680, 4], [B * 7680]
        predicted_probabilities = self.apply_sigmoid(predicted_batch_flat)             # [B * 7680, 4

        if not debug:
            return predicted_probabilities, predicted_batch_indices
        return cartesian_points, downsampled_points, log_odds, predicted_batch_flat, predicted_batch_indices, predicted_probabilities
