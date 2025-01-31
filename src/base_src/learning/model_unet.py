import torch
import torch.nn as nn
import torch.nn.functional as F
from model_polar import RadarOccupancyModel


class AttentionAggregator(nn.Module):
    def __init__(self, input_dim, output_dim, num_heads=1):
        """
        Attention-based aggregator for mapping N inputs to M outputs.

        Args:
            input_dim (int): Dimensionality of input features (per point).
            output_dim (int): Dimensionality of output features (per point).
            num_heads (int): Number of attention heads (default=1).
        """
        super().__init__()
        self.num_heads = num_heads
        self.query = nn.Parameter(torch.randn(output_dim, input_dim))  # [M, Input_Dim]
        self.attention = nn.MultiheadAttention(embed_dim=input_dim, num_heads=num_heads, batch_first=True)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape [Batch, N_points, Input_Dim].

        Returns:
            torch.Tensor: Output tensor of shape [Batch, M, Input_Dim].
        """
        batch_size = x.shape[0]
        query = self.query.unsqueeze(0).expand(batch_size, -1, -1)
        output, _ = self.attention(query, x, x)
        return output


# class PolarToCartesianGrid(nn.Module):
#     def __init__(self, radar_config, attention_dim=8, num_heads=1):
#         """
#         A module to convert a polar 3D grid ([B, 1, El, R, Az]) directly into a Cartesian grid using attention-based aggregation.
#
#         Args:
#             radar_config: Configuration object containing grid and radar parameters.
#             attention_dim (int): Dimensionality of attention intermediate features.
#             num_heads (int): Number of attention heads in the aggregator.
#         """
#         super().__init__()
#         self.radar_config = radar_config
#
#         # Learnable parameters for polar-to-Cartesian transformation
#         self.azimuth_scale = nn.Parameter(torch.ones(radar_config.num_azimuth_bins), requires_grad=True)
#         self.azimuth_bias = nn.Parameter(torch.zeros(radar_config.num_azimuth_bins), requires_grad=True)
#         self.range_scale = nn.Parameter(torch.ones(radar_config.num_range_bins), requires_grad=True)
#         self.range_bias = nn.Parameter(torch.zeros(radar_config.num_range_bins), requires_grad=True)
#         self.elevation_scale = nn.Parameter(torch.ones(radar_config.num_elevation_bins), requires_grad=True)
#         self.elevation_bias = nn.Parameter(torch.zeros(radar_config.num_elevation_bins), requires_grad=True)
#
#         # Attention-based aggregation module
#         self.aggregate_intensity = AttentionAggregator(
#             input_dim=1,  # Input feature dimension (intensity)
#             output_dim=radar_config.num_grid_voxels,  # Flattened voxel grid size
#             num_heads=num_heads
#         )
#
#     def forward(self, polar_frames):
#         """
#         Converts a batch of polar grids ([B, 1, El, R, Az]) into Cartesian voxel grids ([B, 1, Z, Y, X]).
#
#         Args:
#             polar_frames (torch.Tensor): Polar grid of shape [B, 1, El, R, Az].
#
#         Returns:
#             torch.Tensor: Cartesian voxel grid of shape [B, 1, Z, Y, X].
#         """
#         batch_size = polar_frames.shape[0]
#         polar_intensities = polar_frames.flatten(start_dim=2)
#         polar_intensities = polar_intensities.permute(0, 2, 1)
#         aggregated_values = self.aggregate_intensity(polar_intensities)
#         voxel_grids = aggregated_values.view(batch_size, 1, *self.radar_config.grid_size[::-1])  # [B, 1, Z, Y, X]
#         return voxel_grids


# class PolarToCartesianGrid(nn.Module):
#     def __init__(self, radar_config, num_hidden_dim=8):
#         """
#         A module to convert a polar 3D grid ([B, 1, El, R, Az]) directly into a Cartesian grid with learnable aggregation.
#
#         Args:
#             radar_config: Configuration object containing grid and radar parameters.
#             num_hidden_dim (int): Number of hidden dimensions in the convolutional layers.
#         """
#         super().__init__()
#         self.radar_config = radar_config
#
#         self.azimuth_scale = nn.Parameter(torch.ones(radar_config.num_azimuth_bins), requires_grad=True)
#         self.azimuth_bias = nn.Parameter(torch.zeros(radar_config.num_azimuth_bins), requires_grad=True)
#         self.range_scale = nn.Parameter(torch.ones(radar_config.num_range_bins), requires_grad=True)
#         self.range_bias = nn.Parameter(torch.zeros(radar_config.num_range_bins), requires_grad=True)
#         self.elevation_scale = nn.Parameter(torch.ones(radar_config.num_elevation_bins), requires_grad=True)
#         self.elevation_bias = nn.Parameter(torch.zeros(radar_config.num_elevation_bins), requires_grad=True)
#
#         self.aggregate_intensity = nn.Sequential(
#             nn.Conv3d(1, num_hidden_dim, kernel_size=3, padding=1),
#             nn.BatchNorm3d(num_hidden_dim),
#             nn.Conv3d(num_hidden_dim, num_hidden_dim, kernel_size=3, padding=1),
#             nn.BatchNorm3d(num_hidden_dim),
#             nn.Conv3d(num_hidden_dim, 1, kernel_size=1),
#             nn.BatchNorm3d(1),
#             nn.ReLU()
#         )
#
#     def forward(self, polar_frames):
#         """
#         Converts a batch of polar grids ([B, 1, El, R, Az]) into Cartesian voxel grids ([B, 1, Z, Y, X]).
#
#         Args:
#             polar_frames (torch.Tensor): Polar grid of shape [B, 1, El, R, Az].
#
#         Returns:
#             torch.Tensor: Cartesian voxel grid of shape [B, 1, Z, Y, X].
#         """
#         batch_size = polar_frames.shape[0]
#         azimuths = torch.tensor(self.radar_config.clipped_azimuth_bins, device=polar_frames.device)
#         azimuths = azimuths * self.azimuth_scale + self.azimuth_bias
#         ranges = torch.linspace(0, self.radar_config.num_range_bins * self.radar_config.range_bin_width, self.radar_config.num_range_bins, device=polar_frames.device,)
#         ranges = ranges * self.range_scale + self.range_bias
#         elevations = torch.tensor(self.radar_config.clipped_elevation_bins, device=polar_frames.device)
#         elevations = elevations * self.elevation_scale + self.elevation_bias
#
#         elevations_grid, ranges_grid, azimuths_grid = torch.meshgrid(elevations, ranges, azimuths, indexing="ij")
#         elevations_grid = elevations_grid.unsqueeze(0).unsqueeze(0).expand(batch_size, -1, -1, -1, -1)
#         ranges_grid = ranges_grid.unsqueeze(0).unsqueeze(0).expand(batch_size, -1, -1, -1, -1)
#         azimuths_grid = azimuths_grid.unsqueeze(0).unsqueeze(0).expand(batch_size, -1, -1, -1, -1)
#
#         x = ranges_grid * torch.cos(elevations_grid) * torch.sin(azimuths_grid)
#         y = ranges_grid * torch.cos(elevations_grid) * torch.cos(azimuths_grid)
#         z = ranges_grid * torch.sin(elevations_grid)
#         x = x.flatten(start_dim=2)
#         y = y.flatten(start_dim=2)
#         z = z.flatten(start_dim=2)
#         xmin, _, ymin, _, zmin, _ = self.radar_config.point_range
#         x_indices = ((x - xmin) / self.radar_config.grid_resolution).long()
#         y_indices = ((y - ymin) / self.radar_config.grid_resolution).long()
#         z_indices = ((z - zmin) / self.radar_config.grid_resolution).long()
#         flat_indices = (
#                 z_indices * self.radar_config.grid_size[1] * self.radar_config.grid_size[0]
#                 + y_indices * self.radar_config.grid_size[0]
#                 + x_indices
#         )
#         aggregated_intensity = self.aggregate_intensity(polar_frames)
#         intensity = aggregated_intensity.flatten(start_dim=2)
#
#         voxel_grids = torch.zeros((batch_size, 1, *self.radar_config.grid_size[::-1]), device=polar_frames.device)
#         voxel_grids = voxel_grids.view(batch_size, -1)  # Flatten to [B, Z * Y * X]
#         voxel_grids.scatter_add_(1, flat_indices.unsqueeze(0).expand(batch_size, -1), intensity.squeeze(1))
#         voxel_grids = voxel_grids.view(batch_size, 1, *self.radar_config.grid_size[::-1])  # [B, 1, Z, Y, X]
#         return voxel_grids


class PolarToCartesianGrid(nn.Module):
    def __init__(self, radar_config, device):
        """
        A parameter-free module to convert a polar 3D grid ([B, 1, El, R, Az]) directly into a Cartesian grid.

        Args:
            radar_config: Configuration object containing grid and radar parameters.
        """
        super().__init__()
        self.radar_config = radar_config

        azimuths = torch.tensor(radar_config.clipped_azimuth_bins, device=device)
        ranges = torch.linspace(0, radar_config.num_range_bins * radar_config.range_bin_width, radar_config.num_range_bins, device=device)
        elevations = torch.tensor(radar_config.clipped_elevation_bins, device=device)
        elevations_grid, ranges_grid, azimuths_grid = torch.meshgrid(elevations, ranges, azimuths, indexing="ij")
        x = ranges_grid * torch.cos(elevations_grid) * torch.sin(azimuths_grid)
        y = ranges_grid * torch.cos(elevations_grid) * torch.cos(azimuths_grid)
        z = ranges_grid * torch.sin(elevations_grid)
        x_flat = x.flatten()
        y_flat = y.flatten()
        z_flat = z.flatten()
        xmin, _, ymin, _, zmin, _ = radar_config.point_range
        x_indices = ((x_flat - xmin) / radar_config.grid_resolution).long()
        y_indices = ((y_flat - ymin) / radar_config.grid_resolution).long()
        z_indices = ((z_flat - zmin) / radar_config.grid_resolution).long()
        self.flat_voxel_indices = (
            z_indices * radar_config.grid_size[1] * radar_config.grid_size[0]
            + y_indices * radar_config.grid_size[0]
            + x_indices
        )

    def forward(self, polar_frames):
        """
        Converts a batch of polar grids ([B, 1, El, R, Az]) into Cartesian voxel grids ([B, 1, Z, Y, X]).

        Args:
            polar_frames (torch.Tensor): Polar grid of shape [B, 1, El, R, Az].

        Returns:
            torch.Tensor: Cartesian voxel grid of shape [B, 1, Z, Y, X].
        """
        batch_size = polar_frames.shape[0]
        flat_voxel_indices = self.flat_voxel_indices.unsqueeze(0).expand(batch_size, self.radar_config.num_radar_points)
        intensities = polar_frames.flatten(start_dim=2).squeeze(1)
        voxel_grids = torch.zeros((batch_size, 1, *self.radar_config.grid_size[::-1]), device=polar_frames.device)
        voxel_grids = voxel_grids.view(batch_size, -1)
        voxel_grids.scatter_add_(1, flat_voxel_indices, intensities)
        voxel_grids = voxel_grids.view(batch_size, 1, *self.radar_config.grid_size[::-1])  # [B, 1, Z, Y, X]
        return voxel_grids


# class CloudsToGrids(nn.Module):
#     def __init__(self, radar_config, num_hidden_dim=8):
#         """
#         Initializes the CloudsToGrids layer with an IntensityAggregation layer.
#
#         Args:
#             radar_config: Configuration object containing voxel size, grid size, and other parameters.
#         """
#         super().__init__()
#         self.voxel_size = radar_config.grid_resolution
#         self.grid_size = radar_config.grid_size  # (X, Y, Z)
#         self.point_range = radar_config.point_range  # (xmin, xmax, ymin, ymax, zmin, zmax)
#         self.aggregate_intensity = nn.Sequential(
#             nn.Linear(radar_config.num_radar_points, num_hidden_dim),
#             nn.ReLU(),
#             nn.Linear(num_hidden_dim, radar_config.num_grid_voxels)
#         )
#
#     def forward(self, clouds):
#         """
#         Converts a batch of point clouds to voxel grids.
#
#         Args:
#             clouds (torch.Tensor): Point cloud of shape [B, N_points, 4] (X, Y, Z, intensity).
#
#         Returns:
#             torch.Tensor: Voxel grid of shape [B, X, Y, Z, 1].
#         """
#         batch_size = clouds.shape[0]
#         intensity = clouds[..., 3]
#         flat_voxel_values = self.aggregate_intensity(intensity)
#         voxel_grid = flat_voxel_values.view(batch_size, self.grid_size[0], self.grid_size[1], self.grid_size[2], -1)
#         return voxel_grid


class GridReshape(nn.Module):
    def __init__(self, channels):
        """
        A convolutional module to transform a tensor of shape
        [B, C, 16, 28, 270] into [B, C, 16, 32, 256].

        Args:
            in_channels (int): Number of input channels (C).
            out_channels (int): Number of output channels (C).
        """
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(
                in_channels=channels, out_channels=channels,
                kernel_size=(1, 1, 8), dilation=(1, 1, 2)
            ),
            nn.BatchNorm3d(channels),
            nn.ConvTranspose3d(channels, channels, kernel_size=(1, 5, 1)),
            nn.BatchNorm3d(channels)
            # nn.ReLU()
        )

    def forward(self, x):
        return self.conv(x)


# no size change
class ConvBlock(nn.Module):
    def __init__(self, in_c, out_c):
            super().__init__()
            self.conv1 = nn.Conv3d(in_c, out_c, kernel_size=3, padding=1)
            self.bn1 = nn.BatchNorm3d(out_c)
            self.conv2 = nn.Conv3d(out_c, out_c, kernel_size=3, padding=1)
            self.bn2 = nn.BatchNorm3d(out_c)
            self.relu = nn.ReLU()

    def forward(self, inputs):
        x = self.conv1(inputs)
        x = self.bn1(x)
        # x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        # x = self.relu(x)
        return x


class EncoderBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv = ConvBlock(in_c, out_c)
        self.pool = nn.MaxPool3d(2)

    def forward(self, inputs):
        x = self.conv(inputs)
        p = self.pool(x)
        return x, p


class DecoderBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.up = nn.ConvTranspose3d(in_c, out_c, kernel_size=2, stride=2, padding=0)
        self.conv = ConvBlock(out_c+out_c, out_c)

    def forward(self, inputs, skip):
        x = self.up(inputs)
        x = torch.cat((x, skip), dim=1)
        x = self.conv(x)
        return x


class Unet1C3DPolar(RadarOccupancyModel):
    def __init__(self, radar_config, device, *args, **kwargs):
        super().__init__(radar_config, *args, **kwargs)
        self.name = 'grid+unet+logits+pad_v1.0'

        self.polar_to_cartesian = PolarToCartesianGrid(radar_config, device)
        # self.reshape = GridReshape(1)

        self.e1 = EncoderBlock(1, 32)
        self.e2 = EncoderBlock(32, 64)
        self.e3 = EncoderBlock(64, 128)

        self.bottleneck = ConvBlock(128, 256)

        self.d1 = DecoderBlock(256, 128)
        self.d2 = DecoderBlock(128, 64)
        self.d3 = DecoderBlock(64, 32)

        self.output = nn.Sequential(
            nn.Conv3d(32, 64, kernel_size=1),
            nn.BatchNorm3d(64),
            # nn.ReLU(),
            nn.Conv3d(64, 1, kernel_size=1),
            nn.Sigmoid()
        )

    def add_padding(self, grids, padding=(0, 0, 0)):
        """
        Adds padding to the input grids.

        Args:
            grids (torch.Tensor): Input tensor of shape [B, C, A, B, C].
            padding (tuple): Padding sizes for dimensions A, B, C in the format (a, b, c).

        Returns:
            torch.Tensor: Padded tensor.
        """
        pad_a, pad_b, pad_c = padding
        padded_grids = F.pad(grids, (0, pad_c, 0, pad_b, 0, pad_a))
        return padded_grids

    def remove_padding(self, grids, padding=(0, 0, 0)):
        """
        Removes padding from the input grids.

        Args:
            grids (torch.Tensor): Padded tensor of shape [B, C, A+p, B+q, C+r].
            padding (tuple): Padding sizes for dimensions A, B, C in the format (a, b, c).

        Returns:
            torch.Tensor: Tensor with padding removed.
        """
        pad_a, pad_b, pad_c = padding
        cropped_grids = grids[..., pad_a:, pad_b:, pad_c:]
        return cropped_grids


    def forward(self, polar_frames):                                      # [B, Az, R, El]
        polar_frames = polar_frames.unsqueeze(-1).permute(0, 4, 3, 2, 1)  # [B, 1, El, R, Az]
        cartesian_grids = self.polar_to_cartesian(polar_frames)           # [B, 1, 16, 28, 270]
        grids = self.add_padding(cartesian_grids, padding=(0, 4, 2))      # [B, 1, 16, 32, 272]
        # cartesian_grids = cartesian_grids.permute(0, 4, 3, 2, 1)  # [B, 1, 16, 28, 270]
        # grids = self.reshape(cartesian_grids)                             # [B, 1, 16, 32, 256]
        s1, p1 = self.e1(grids)                                           # [B, 32, 16, 32, 256], [B, 32, 8, 16, 128]
        s2, p2 = self.e2(p1)                                              # [B, 64, 8, 16, 128],  [B, 64, 4, 8, 64]
        s3, p3 = self.e3(p2)                                              # [B, 128, 4, 8, 64],   [B, 128, 2, 4, 32]
        b = self.bottleneck(p3)                                           # [B, 256, 2, 4, 32]
        o1 = self.d1(b, s3)                                               # [B, 128, 4, 8, 64]
        o2 = self.d2(o1, s2)                                              # [B, 64, 8, 16, 128]
        o3 = self.d3(o2, s1)                                              # [B, 32, 16, 32, 256]
        output = self.output(o3)                                          # [B, 1, 16, 32, 256]
        output = self.remove_padding(output, padding=(0, 4, 2)).squeeze(1).permute(0, 3, 2, 1)  # [B, 270, 28, 16]
        return output
