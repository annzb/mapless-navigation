import torch
import torch.nn as nn

from models.base import RadarOccupancyModel
from models.point_based.encoders import MlpPointEncoder, DualPointEncoder, build_encoder
from models.point_based.polar_to_cartesian import PolarToCartesianPoints
from models.point_based.downsampling import TrainedDownsampling
from models.point_based.pointnet import PointNet, PointNet2Spatial, SinglePointEncoder


class Baseline(RadarOccupancyModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = 'encoder_pointnet_v1.2'
        self.encoder = MlpPointEncoder(in_dim=4, hidden_dim=128, output_size=4096, output_features=4)
        self.pointnet = PointNet2Spatial(num_features=1, dropout=0.2)

    def forward(self, X, debug=False, **kwargs):
        flat_pts, batch_idx = X

        embeddings = self.encoder(flat_pts, batch_idx)
        self.check_gradient(embeddings, "Encoder output")

        predicted_log_odds = self.pointnet(coords=embeddings[..., :3], features=embeddings[..., 3:])
        self.check_gradient(predicted_log_odds, "PointNet output")

        predicted_log_odds_flat, predicted_flat_indices = self.merge_batches(predicted_log_odds)
        self.check_gradient(predicted_log_odds_flat, "Merged predictions")

        probs = self.apply_sigmoid(predicted_log_odds_flat)
        self.check_gradient(probs, "Final probabilities")

        if debug:
            return embeddings, predicted_log_odds, probs, predicted_flat_indices
        return probs, predicted_flat_indices
    

class RegressionBaseline(RadarOccupancyModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = 'gregression_baseline_v1.0'
        num_features = 128
        encoded_cloud_size = 1024
        
        self.encoder = SinglePointEncoder(output_size=encoded_cloud_size, output_dim=num_features)
        self.decoder = nn.Sequential(
            nn.Linear(num_features * 2 + 3, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)  # Predict per-point occupancy logits
        )

    def forward(self, X, debug=False, **kwargs):
        flat_pts, batch_idx = X

        encoded_clouds = []
        for sample_idx in range(self.batch_size):
            cloud = flat_pts[batch_idx == sample_idx]
            encoded = self.encoder(cloud)
            pooled_mean = encoded.mean(dim=0, keepdim=True)
            pooled_max = encoded.max(dim=0, keepdim=True).values
            latent = torch.cat([pooled_mean, pooled_max], dim=-1)  # (1, num_features*2)
            encoded_clouds.append(latent)

        encoded_clouds = torch.cat(encoded_clouds, dim=0)  # (B, latent_dim)

        B, latent_dim = encoded_clouds.shape
        S = self.support_coords.shape[0]
        expanded_support_coords = self.support_coords.unsqueeze(0).expand(B, -1, -1) # support_coords: (S, 3) → (B, S, 3)
        expanded_latent = encoded_clouds.unsqueeze(1).expand(-1, S, -1) # encoded_clouds: (B, latent_dim) → (B, S, latent_dim)
        decoder_input = torch.cat([expanded_support_coords, expanded_latent], dim=-1)  # (B, S, latent_dim + 3)
        decoder_input = decoder_input.reshape(B * S, latent_dim + 3) # Reshape to (B*S, latent_dim+3) for MLP
        clouds_logits = self.decoder(decoder_input).reshape(B, S)
        clouds_probs = torch.sigmoid(clouds_logits)

        cloud_probs_flat, cloud_probs_indices = self.merge_batches(clouds_probs)
        self.check_gradient(cloud_probs_flat, "Merged predictions")

        if debug:
            return encoded_clouds, clouds_logits, clouds_probs, cloud_probs_flat, cloud_probs_indices
        return cloud_probs_flat, cloud_probs_indices

    
class DualBranchPointnet(RadarOccupancyModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = 'dual_branch_pointnet_v1.0'
        self.spatial_encoder = build_encoder(in_num_features=3, out_num_features=64, hidden_num_features_multiplier=16, num_layers=3, dropout_rate=0.1)
        self.intensity_encoder = build_encoder(in_num_features=1, out_num_features=64, hidden_num_features_multiplier=16, num_layers=3, dropout_rate=0.1)
        self.encoder = DualPointEncoder(output_size=4096, dropout_rate=0.1, in_num_intensity_features=64, in_num_spatial_features=64)
        self.pointnet = PointNet()

    def forward(self, X, debug=False, **kwargs):
        flat_pts, batch_idx = X

        spatial_features = self.spatial_encoder(flat_pts[..., :3])
        intensity_features = self.intensity_encoder(flat_pts[..., 3:])
        embeddings = self.encoder(spatial_features, intensity_features, batch_idx)
        self.check_gradient(embeddings, "Encoder output")

        predicted_log_odds = self.pointnet(coords=embeddings[..., :3], features=embeddings[..., 3:])
        self.check_gradient(predicted_log_odds, "PointNet output")

        predicted_log_odds_flat, predicted_flat_indices = self.merge_batches(predicted_log_odds)
        self.check_gradient(predicted_log_odds_flat, "Merged predictions")

        probs = self.apply_sigmoid(predicted_log_odds_flat)
        self.check_gradient(probs, "Final probabilities")

        if debug:
            return spatial_features, intensity_features, embeddings, predicted_log_odds, probs, predicted_flat_indices
        return probs, predicted_flat_indices
    

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
