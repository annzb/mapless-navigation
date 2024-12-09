# - some initial architectures to test the approaches. For the radar encoder, we want to try 1) layers using a direct conversion to cartesian space 2) CNN layers 3) fourier layers (if applicable to images that are already transformed using FFT) 4) attention layers 5) combinations of those. Don't use too many layers since we just want to test how it runs
# - some actual layer that will do the mapping like a normal fully connected layer or an lstm layer, whatever is more appropriate. We will try mapping the output of this layer with the occupancy odds of the groundtruth.
# - the function to start training
# - the function to evaluate a model by converting the occupancy odds into probabilities and calculating the error.


import torch
import torch.nn as nn
import torch.nn.functional as F
from Pointnet_Pointnet2_pytorch.models.pointnet2_utils import PointNetSetAbstraction, PointNetFeaturePropagation


class PolarToCartesian(nn.Module):
    def __init__(self, radar_config):
        super(PolarToCartesian, self).__init__()
        self.radar_config = radar_config

        self.azimuth_scale = nn.Parameter(torch.ones(radar_config.num_azimuth_bins))
        self.azimuth_bias = nn.Parameter(torch.zeros(radar_config.num_azimuth_bins))
        self.azimuth_cos_weight = nn.Parameter(torch.ones(radar_config.num_azimuth_bins))
        self.azimuth_sin_weight = nn.Parameter(torch.ones(radar_config.num_azimuth_bins))

        self.elevation_scale = nn.Parameter(torch.ones(radar_config.num_elevation_bins))
        self.elevation_bias = nn.Parameter(torch.zeros(radar_config.num_elevation_bins))
        self.elevation_cos_weight = nn.Parameter(torch.ones(radar_config.num_elevation_bins))
        self.elevation_sin_weight = nn.Parameter(torch.ones(radar_config.num_elevation_bins))

        self.range_scale = nn.Parameter(torch.ones(radar_config.num_range_bins))
        self.range_bias = nn.Parameter(torch.zeros(radar_config.num_range_bins))

    def forward(self, polar_frames):
        batch_size = polar_frames.shape[0]

        azimuths = torch.tensor(self.radar_config.clipped_azimuth_bins, device=polar_frames.device)
        azimuths = azimuths * self.azimuth_scale + self.azimuth_bias

        ranges = torch.linspace(0, self.radar_config.num_range_bins * self.radar_config.range_bin_width, self.radar_config.num_range_bins, device=polar_frames.device)
        ranges = ranges * self.range_scale + self.range_bias

        elevations = torch.tensor(self.radar_config.clipped_elevation_bins, device=polar_frames.device)
        elevations = elevations * self.elevation_scale + self.elevation_bias

        azimuths_grid, ranges_grid, elevations_grid = torch.meshgrid(azimuths, ranges, elevations, indexing="ij")
        azimuths_grid = azimuths_grid.unsqueeze(0).expand(batch_size, -1, -1, -1)
        ranges_grid = ranges_grid.unsqueeze(0).expand(batch_size, -1, -1, -1)
        elevations_grid = elevations_grid.unsqueeze(0).expand(batch_size, -1, -1, -1)

        cos_azimuths = self.azimuth_cos_weight.view(1, -1, 1, 1) * torch.cos(azimuths_grid)
        sin_azimuths = self.azimuth_sin_weight.view(1, -1, 1, 1) * torch.sin(azimuths_grid)
        cos_elevations = self.elevation_cos_weight.view(1, 1, 1, -1) * torch.cos(elevations_grid)
        sin_elevations = self.elevation_sin_weight.view(1, 1, 1, -1) * torch.sin(elevations_grid)

        x = ranges_grid * cos_elevations * cos_azimuths
        y = ranges_grid * cos_elevations * sin_azimuths
        z = ranges_grid * sin_elevations
        x = x.flatten(start_dim=1).unsqueeze(-1)
        y = y.flatten(start_dim=1).unsqueeze(-1)
        z = z.flatten(start_dim=1).unsqueeze(-1)

        intensity = polar_frames.flatten(start_dim=1, end_dim=3).unsqueeze(-1)
        cartesian_points = torch.cat((x, y, z, intensity), dim=-1)

        return cartesian_points  # [B, 151040, 4]


class PointNet(nn.Module):
    def __init__(self):
        super(PointNet, self).__init__()
        # PointNet++ MSG backbone
        self.sa1 = PointNetSetAbstraction(1024, 0.1, 32, 9 + 3, [32, 32, 64], False)
        self.sa2 = PointNetSetAbstraction(256, 0.2, 32, 64 + 3, [64, 64, 128], False)
        self.sa3 = PointNetSetAbstraction(64, 0.4, 32, 128 + 3, [128, 128, 256], False)
        self.sa4 = PointNetSetAbstraction(16, 0.8, 32, 256 + 3, [256, 256, 512], False)
        self.fp4 = PointNetFeaturePropagation(768, [256, 256])
        self.fp3 = PointNetFeaturePropagation(384, [256, 256])
        self.fp2 = PointNetFeaturePropagation(320, [256, 128])
        self.fp1 = PointNetFeaturePropagation(128, [128, 128, 128])

        # Prediction layers
        self.conv1 = nn.Conv1d(128, 128, 1)
        self.bn1 = nn.BatchNorm1d(128)
        self.drop1 = nn.Dropout(0.5)
        self.conv2 = nn.Conv1d(128, 1, 1)  # Single channel for log odds

    def forward(self, point_clouds):
        # Split into coordinates and features
        xyz = point_clouds[..., :3].permute(0, 2, 1)  # [B, 3, N_points]
        features = point_clouds[..., 3:].permute(0, 2, 1)  # [B, 1, N_points]

        # Step 2: Hierarchical feature extraction
        l0_xyz, l0_points = xyz, features
        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        l4_xyz, l4_points = self.sa4(l3_xyz, l3_points)

        # Step 3: Feature upsampling
        l3_points = self.fp4(l3_xyz, l4_xyz, l3_points, l4_points)
        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)
        l0_points = self.fp1(l0_xyz, l1_xyz, None, l1_points)

        # Step 4: Predict occupancy log odds
        x = self.drop1(F.relu(self.bn1(self.conv1(l0_points))))
        log_odds = self.conv2(x)  # Shape: [B, 1, N_points]
        return log_odds.squeeze(1)  # Shape: [B, N_points]


class RadarOccupancyModel(nn.Module):
    def __init__(self, radar_config):
        super(RadarOccupancyModel, self).__init__()
        self.radar_config = radar_config
        self.polar_to_cartesian = PolarToCartesian(radar_config)
        self.pointnet = PointNet()

    def forward(self, polar_frames):
        cartesian_point_clouds = self.polar_to_cartesian(polar_frames)  # Shape: [B, N_points, 4]
        print('cartesian_point_clouds.shape', cartesian_point_clouds.shape)
        log_odds = self.pointnet(cartesian_point_clouds)  # Shape: [B, N_points]
        print('log_odds.shape', log_odds.shape)
        return log_odds
