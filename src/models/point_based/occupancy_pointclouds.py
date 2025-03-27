from models.base import RadarOccupancyModel
from models.point_based.polar_to_cartesian import PolarToCartesianPoints
from models.point_based.downsampling import TrainedDownsampling
from models.point_based.pointnet import PointNet


class PointOccupancyModel(RadarOccupancyModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        num_points = self.radar_config.num_azimuth_bins * self.radar_config.num_range_bins * self.radar_config.num_elevation_bins
        self.polar_to_cartesian = PolarToCartesianPoints(radar_config=self.radar_config)
        self.down = TrainedDownsampling(num_points, retain_fraction=0.05)
        self.pointnet = PointNet()
        self.name = 'cart+down+pointnet_v1.1'

    def forward(self, polar_frames):
        cartesian_points = self.polar_to_cartesian(polar_frames)                       # [B, 153600, 4]
        downsampled_points = self.down(cartesian_points)                               # [B, 7680, 4]
        points = downsampled_points[..., :3]                                           # [B, 7680, 3]
        features = downsampled_points[..., 3:]                                         # [B, 7680, 1]
        log_odds = self.pointnet(points, features)                                     # [B, 7680, 4]

        predicted_batch_flat, predicted_batch_indices = self.merge_batches(log_odds)   # [B * 7680, 4], [B * 7680]
        predicted_probabilities = self.apply_sigmoid(predicted_batch_flat)             # [B * 7680, 4
        return predicted_probabilities, predicted_batch_indices
