from models.base import RadarOccupancyModel
from models.point_based.encoders import PointEncoder, MlpPointEncoder
from models.point_based.polar_to_cartesian import PolarToCartesianPoints
from models.point_based.downsampling import TrainedDownsampling
from models.point_based.pointnet import PointNet


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
        self.name = 'mlp_encoder+pointnet_v1.0'
        self.encoder = MlpPointEncoder(in_dim=4, hidden_dim=256, output_size=4096)
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
