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
        sft_output = torch.fft.rfft2(polar_frames, dim=(-3, -1))
        magnitude = torch.abs(sft_output)
        return magnitude # [B 128 118 6]
        # print('sft_output.shape', sft_output.shape)
        # return torch.cat((polar_frames, magnitude), dim=-1)


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

        x = ranges_grid * cos_elevations * sin_azimuths
        y = ranges_grid * cos_elevations * cos_azimuths
        z = ranges_grid * sin_elevations
        x = x.flatten(start_dim=1).unsqueeze(-1)
        y = y.flatten(start_dim=1).unsqueeze(-1)
        z = z.flatten(start_dim=1).unsqueeze(-1)

        intensity = polar_frames.flatten(start_dim=1, end_dim=3).unsqueeze(-1)
        cartesian_points = torch.cat((x, y, z, intensity), dim=-1)

        return cartesian_points  # [B, N, 4]


class Downsampling(nn.Module):
    def __init__(self, input_channels, output_channels_rate=1.0, point_reduction_rate=2, pool_size=2, num_layers=2, padding=0):
        """
        Args:
            input_channels: Number of input feature channels (e.g., 4 for [x, y, z, intensity]).
            output_channels: Number of feature channels after convolution.
            kernel_size: Size of the convolution kernel.
            pool_size: Size of the pooling window.
            num_layers: Number of convolution-pooling layers.
        """
        super(Downsampling, self).__init__()
        kernel_size = point_reduction_rate
        stride = int(point_reduction_rate / 2) or 2

        layers = []
        in_channels = input_channels
        for _ in range(num_layers):
            output_channels = int(in_channels * output_channels_rate)
            layers.append(nn.Conv1d(in_channels, output_channels, kernel_size=kernel_size, stride=stride, padding=padding))
            layers.append(nn.ReLU())
            layers.append(nn.MaxPool1d(pool_size))
            in_channels = output_channels
        self.downsampling = nn.Sequential(*layers)

    def forward(self, x):
        x = x.permute(0, 2, 1)  # Change to [B, input_channels, N] for Conv1d
        x = self.downsampling(x)
        x = x.permute(0, 2, 1)  # Back to [B, N / (pool_size ** num_layers), output_channels]
        return x


class PointNet(nn.Module):
    def __init__(self):
        super(PointNet, self).__init__()
        # PointNet++ MSG backbone
        self.sa1 = PointNetSetAbstraction(1180, 0.2, 16, 32, [32, 32, 64], False)
        self.sa2 = PointNetSetAbstraction(590, 0.4, 16, 64 + 3, [64, 64, 128], False)
        self.sa3 = PointNetSetAbstraction(295, 0.6, 16, 128 + 3, [128, 128, 256], False)
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
        features = point_clouds[..., 3:].permute(0, 2, 1)  # [B, n_features, N_points]

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
        log_odds = self.conv2(x)  # [B, 1, N_points]
        log_odds = log_odds.permute(0, 2, 1) # [B, N_points, 1]
        return torch.cat((point_clouds[..., :3], log_odds), dim=-1)


class RadarOccupancyModel(nn.Module):
    def __init__(self, radar_config, radar_point_downsample_rate=0.5, trans_embed_dim=128, trans_num_heads=4, trans_num_layers=2):
        super(RadarOccupancyModel, self).__init__()
        self.num_radar_points = radar_config.num_azimuth_bins * radar_config.num_elevation_bins * radar_config.num_range_bins
        self.radar_config = radar_config

        embed_dim = radar_config.num_elevation_bins
        num_heads = embed_dim // 2
        num_layers = 4
        encoder = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads)
        self.transformer = nn.TransformerEncoder(encoder, num_layers=num_layers)

        self.polar_to_cartesian = PolarToCartesian(radar_config)
        self.down = Downsampling(input_channels=4, output_channels_rate=2, point_reduction_rate=4, pool_size=2, num_layers=3, padding=1)
        self.pointnet = PointNet()

    def forward(self, polar_frames):
        batch_size = polar_frames.shape[0]
        # print('input shape:', polar_frames.shape)
        # [B, 128, 118, 10]
        reshaped_frames = polar_frames.view(batch_size, self.radar_config.num_azimuth_bins * self.radar_config.num_range_bins, self.radar_config.num_elevation_bins)
        # [B, 15104, 10]
        # print('view shape:', reshaped_frames.shape)
        transformed_frames = self.transformer(reshaped_frames)
        # [B, 15104, 10]
        # print('transformed_frames shape:', transformed_frames.shape)
        transformed_frames = transformed_frames.view(batch_size, self.radar_config.num_azimuth_bins, self.radar_config.num_range_bins, self.radar_config.num_elevation_bins)
        # [B, 128, 118, 10]
        # print('transformed_frames view shape:', transformed_frames.shape)

        cartesian_points = self.polar_to_cartesian(transformed_frames)
        # [B, 151040, 4]
        # print('cartesian_points shape:', cartesian_points.shape)
        less_points = self.down(cartesian_points)
        # [B, 2360, 32]
        # print('less_points shape:', less_points.shape)

        log_odds = self.pointnet(less_points)
        # [B, 2360, 4]
        # print('output shape:', log_odds.shape)
        probabilities = torch.sigmoid(log_odds)
        return probabilities
