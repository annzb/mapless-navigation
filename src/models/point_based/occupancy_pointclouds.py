from models.base import RadarOccupancyModel
from models.point_based.encoders import PointEncoder, MlpPointEncoder2, PointEncoder2, DualPointEncoder, build_encoder
from models.point_based.polar_to_cartesian import PolarToCartesianPoints
from models.point_based.downsampling import TrainedDownsampling
from models.point_based.pointnet import PointNet, PointNet2, PointNet2Spatial
import torch


class PointBaseline(RadarOccupancyModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = 'baseline_v1.0'
        self.encoder = PointEncoder(in_dim=4, hidden_dim=256, output_size=4096)

    def forward(self, X, debug=False, **kwargs):
        flat_pts, batch_idx = X
        embeddings = self.encoder(flat_pts, batch_idx, **kwargs)
        embeddings_flat, embeddings_flat_indices = self.merge_batches(embeddings)
        probs = self.apply_sigmoid(embeddings_flat)

        if debug:
            return embeddings_flat, embeddings_flat_indices, probs
        return probs, embeddings_flat_indices


class BasePointnet(RadarOccupancyModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = 'encoder+pointnet_v1.0'
        self.encoder = PointEncoder(in_dim=4, hidden_dim=256, output_size=4096)
        self.pointnet = PointNet()

    def forward(self, X, debug=False, **kwargs):
        flat_pts, batch_idx = X
        embeddings = self.encoder(flat_pts, batch_idx, **kwargs)
        predicted_log_odds = self.pointnet(coords=embeddings[..., :3], features=embeddings[..., 3:])
        predicted_log_odds_flat, predicted_flat_indices = self.merge_batches(predicted_log_odds)
        probs = self.apply_sigmoid(predicted_log_odds_flat)

        if debug:
            return embeddings, predicted_log_odds, probs, predicted_flat_indices
        return probs, predicted_flat_indices


class MlpPointnet(RadarOccupancyModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = 'test_encoder+pointnet_v1.0'
        self.encoder = PointEncoder2(in_dim=4, hidden_dim=256, output_size=4096)
        self.pointnet = PointNet()

    def forward(self, X, debug=False, **kwargs):
        flat_pts, batch_idx = X
        
        # Encoder
        embeddings = self.encoder(flat_pts, batch_idx, **kwargs)
        self.check_gradient(embeddings, "Encoder output")
            
        # Split coordinates and features
        coords = embeddings[..., :3]
        features = embeddings[..., 3:]
        self.check_gradient(coords, "Coordinates")
        self.check_gradient(features, "Features")
            
        # PointNet
        predicted_log_odds = self.pointnet(coords=coords, features=features)
        self.check_gradient(predicted_log_odds, "PointNet output")
            
        # Merge batches
        predicted_log_odds_flat, predicted_flat_indices = self.merge_batches(predicted_log_odds)
        self.check_gradient(predicted_log_odds_flat, "Merged predictions")
            
        # Apply sigmoid
        probs = self.apply_sigmoid(predicted_log_odds_flat)
        self.check_gradient(probs, "Final probabilities")

        if debug:
            return embeddings, predicted_log_odds, probs, predicted_flat_indices
        return probs, predicted_flat_indices
    

class EncoderPointnet(RadarOccupancyModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = 'encoder_pointnet_v1.2'
        self.encoder = MlpPointEncoder2(in_dim=4, hidden_dim=128, output_size=4096, output_features=4)
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
