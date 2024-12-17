# - some initial architectures to test the approaches. For the radar encoder, we want to try 1) layers using a direct conversion to cartesian space 2) CNN layers 3) fourier layers (if applicable to images that are already transformed using FFT) 4) attention layers 5) combinations of those. Don't use too many layers since we just want to test how it runs
# - some actual layer that will do the mapping like a normal fully connected layer or an lstm layer, whatever is more appropriate. We will try mapping the output of this layer with the occupancy odds of the groundtruth.
# - the function to start training
# - the function to evaluate a model by converting the occupancy odds into probabilities and calculating the error.


import torch
import torch.nn as nn
import torch.nn.functional as F
from Pointnet_Pointnet2_pytorch.models.pointnet2_utils import PointNetSetAbstraction, PointNetFeaturePropagation


class SphericalFourierTransform(nn.Module):
    def __init__(self, num_azimuth_bins, num_elevation_bins):
        super(SphericalFourierTransform, self).__init__()
        self.num_azimuth_bins = num_azimuth_bins
        self.num_elevation_bins = num_elevation_bins

    def forward(self, polar_frames):
        # Perform FFT on the azimuth and elevation dimensions
        sft_output = torch.fft.rfft2(polar_frames, dim=(-2, -1))
        magnitude = torch.abs(sft_output)
        return torch.cat((polar_frames, magnitude), dim=-1)


class TrainedDropout(nn.Module):
    def __init__(self, num_points, retain_fraction=0.5):
        super(TrainedDropout, self).__init__()
        self.retain_fraction = retain_fraction
        self.dropout_weights = nn.Parameter(torch.full((num_points,), 0.5))

    def forward(self, points):
        batch_size, num_points, _ = points.shape
        probabilities = torch.sigmoid(self.dropout_weights)
        num_retain = int(num_points * self.retain_fraction)
        _, indices = torch.topk(probabilities, num_retain, largest=True, sorted=False)
        keep_mask = torch.zeros(batch_size, num_points, device=points.device)
        keep_mask[:, indices] = 1
        keep_mask = keep_mask.bool()
        downsampled_points = points[keep_mask].view(batch_size, num_retain, -1)  # Reshape to [B, num_retain, 4]
        return downsampled_points


class PolarToCartesian(nn.Module):
    def __init__(self, radar_config, dropout=0.5):
        super(PolarToCartesian, self).__init__()
        self.radar_config = radar_config
        self.dropout = dropout

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
        # x = x.flatten(start_dim=1).unsqueeze(-1)
        # y = y.flatten(start_dim=1).unsqueeze(-1)
        # z = z.flatten(start_dim=1).unsqueeze(-1)
        #
        # intensity = polar_frames.flatten(start_dim=1, end_dim=3).unsqueeze(-1)
        # cartesian_points = torch.cat((x, y, z, intensity), dim=-1)
        cartesian_points = torch.cat((
            x.flatten(1).unsqueeze(-1),  # [B, N, 1]
            y.flatten(1).unsqueeze(-1),  # [B, N, 1]
            z.flatten(1).unsqueeze(-1),  # [B, N, 1]
            polar_frames.flatten(1, -2)  # [B, N, num_features]
        ), dim=-1)
        return cartesian_points  # [B, N, 5]


class CrossAttentionTransformer(nn.Module):
    def __init__(self, embed_dim, num_heads, num_layers):
        super(CrossAttentionTransformer, self).__init__()
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(embed_dim, num_heads, dim_feedforward=embed_dim * 4, batch_first=True)
            for _ in range(num_layers)
        ])

    def forward(self, cartesian_points):
        # Split xyz (for positional encoding) and other features
        xyz = cartesian_points[:, :, :3]  # [B, N, 3]
        features = cartesian_points[:, :, 3:]  # [B, N, num_features]

        # Compute positional encodings
        positional_encodings = self._positional_encoding(xyz)
        combined_features = torch.cat((features, positional_encodings), dim=-1)  # [B, N, num_features + PE]

        # Pass through transformer layers
        for layer in self.layers:
            combined_features = layer(combined_features)

        return combined_features  # Output features with spatial awareness

    def _positional_encoding(self, coords, num_frequencies=6):
        frequencies = torch.linspace(1.0, 2**num_frequencies, num_frequencies).to(coords.device)
        encodings = torch.cat([torch.sin(coords * freq) for freq in frequencies], dim=-1)
        return encodings



class PointNet(nn.Module):
    def __init__(self):
        super(PointNet, self).__init__()
        # PointNet++ MSG backbone
        self.sa1 = PointNetSetAbstraction(1180, 0.2, 16, 4, [32, 32, 64], False)
        self.sa2 = PointNetSetAbstraction(295, 0.4, 16, 64 + 3, [64, 64, 128], False)
        self.sa3 = PointNetSetAbstraction(59, 0.6, 16, 128 + 3, [128, 128, 256], False)
        # self.sa4 = PointNetSetAbstraction(59, 0.8, 16, 256, [256, 256, 512], False)
        # self.fp4 = PointNetFeaturePropagation(768, [256, 256])
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
        # print('l0_xyz, l0_points', l0_xyz.shape, l0_points.shape)
        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)
        # print('l1_xyz, l1_points', l1_xyz.shape, l1_points.shape)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        # l4_xyz, l4_points = self.sa4(l3_xyz, l3_points)

        # Step 3: Feature upsampling
        # l3_points = self.fp4(l3_xyz, l4_xyz, l3_points, l4_points)
        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)
        l0_points = self.fp1(l0_xyz, l1_xyz, None, l1_points)

        # Step 4: Predict occupancy log odds
        x = self.drop1(F.relu(self.bn1(self.conv1(l0_points))))
        log_odds = self.conv2(x)  # Shape: [B, 1, N_points]
        return log_odds.squeeze(1)  # Shape: [B, N_points]


class RadarOccupancyModel(nn.Module):
    def __init__(self, radar_config, radar_point_downsample_rate=0.5, occupancy_threshold=0.5, trans_embed_dim=128, trans_num_heads=4, trans_num_layers=2):
        super(RadarOccupancyModel, self).__init__()
        self.occupancy_threshold = occupancy_threshold
        self.num_radar_points = radar_config.num_azimuth_bins * radar_config.num_elevation_bins * radar_config.num_range_bins
        # print('self.num_radar_points', self.num_radar_points)
        self.radar_config = radar_config
        self.sft = SphericalFourierTransform(radar_config.num_azimuth_bins, radar_config.num_elevation_bins)
        self.polar_to_cartesian = PolarToCartesian(radar_config)
        self.transformer = CrossAttentionTransformer(trans_embed_dim, trans_num_heads, trans_num_layers)
        # self.radar_downsample_1 = TrainedDropout(self.num_radar_points, radar_point_downsample_rate)
        # self.radar_downsample_2 = TrainedDropout(int(self.num_radar_points * (1 - radar_point_downsample_rate)), radar_point_downsample_rate)
        # self.pointnet = PointNet()

    def forward(self, polar_frames):
        expanded_frames = self.sft(polar_frames)
        print('expanded_frames.shape', expanded_frames.shape)
        cartesian_points = self.polar_to_cartesian(expanded_frames)
        print('cartesian_points.shape', cartesian_points.shape)

        # downsampled_radar_clouds = self.radar_downsample_1(cartesian_radar_clouds)
        # downsampled_radar_clouds = self.radar_downsample_2(downsampled_radar_clouds)
        # print('downsampled_radar_clouds.shape', downsampled_radar_clouds.shape)

        transformed_features = self.transformer(cartesian_points)
        print('transformed_features.shape', transformed_features.shape)
        return transformed_features

        # log_odds = self.pointnet(transformed_features)
        # print('log_odds.shape', log_odds.shape)
        #
        # probabilities = torch.sigmoid(log_odds)
        # print('probabilities.shape', probabilities.shape)
        #
        # keep_mask = probabilities > self.occupancy_threshold  # Shape: [B, reduced_N]
        # filtered_points = downsampled_radar_clouds[keep_mask]  # Filtered points [M, 4]
        # filtered_probs = probabilities[keep_mask]  # Filtered probabilities [M]
        # print('filtered_points.shape', filtered_points.shape)
        # print('filtered_probs.shape', filtered_probs.shape)
        # return filtered_points, filtered_probs

