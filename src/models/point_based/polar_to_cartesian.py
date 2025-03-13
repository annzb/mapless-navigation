import torch
import torch.nn as nn


class PolarToCartesianPoints(nn.Module):
    def __init__(self, radar_config):
        super(PolarToCartesianPoints, self).__init__()
        self.radar_config = radar_config

        # self.azimuth_scale = nn.Parameter(torch.ones(radar_config.num_azimuth_bins), requires_grad=True)
        # self.azimuth_bias = nn.Parameter(torch.zeros(radar_config.num_azimuth_bins), requires_grad=True)
        # self.azimuth_cos_weight = nn.Parameter(torch.ones(radar_config.num_azimuth_bins), requires_grad=True)
        # self.azimuth_sin_weight = nn.Parameter(torch.ones(radar_config.num_azimuth_bins), requires_grad=True)

        # self.elevation_scale = nn.Parameter(torch.ones(radar_config.num_elevation_bins))
        # self.elevation_bias = nn.Parameter(torch.zeros(radar_config.num_elevation_bins))
        # self.elevation_cos_weight = nn.Parameter(torch.ones(radar_config.num_elevation_bins))
        # self.elevation_sin_weight = nn.Parameter(torch.ones(radar_config.num_elevation_bins))

        # self.range_scale = nn.Parameter(torch.ones(radar_config.num_range_bins), requires_grad=True)
        # self.range_bias = nn.Parameter(torch.zeros(radar_config.num_range_bins), requires_grad=True)

    def forward(self, polar_frames):
        batch_size = polar_frames.shape[0]

        azimuths = torch.tensor(self.radar_config.clipped_azimuth_bins, device=polar_frames.device)
        # azimuths = azimuths * self.azimuth_scale + self.azimuth_bias

        ranges = torch.linspace(0, self.radar_config.num_range_bins * self.radar_config.range_bin_width, self.radar_config.num_range_bins, device=polar_frames.device)
        # ranges = ranges * self.range_scale + self.range_bias

        elevations = torch.tensor(self.radar_config.clipped_elevation_bins, device=polar_frames.device)
        # elevations = elevations * self.elevation_scale + self.elevation_bias

        elevations_grid, azimuths_grid, ranges_grid = torch.meshgrid(elevations, azimuths, ranges, indexing="ij")
        elevations_grid = elevations_grid.unsqueeze(0).expand(batch_size, -1, -1, -1)
        azimuths_grid = azimuths_grid.unsqueeze(0).expand(batch_size, -1, -1, -1)
        ranges_grid = ranges_grid.unsqueeze(0).expand(batch_size, -1, -1, -1)

        # cos_azimuths = self.azimuth_cos_weight.view(1, -1, 1, 1) * torch.cos(azimuths_grid)
        # sin_azimuths = self.azimuth_sin_weight.view(1, -1, 1, 1) * torch.sin(azimuths_grid)
        cos_azimuths = torch.cos(azimuths_grid)
        sin_azimuths = torch.sin(azimuths_grid)
        # cos_elevations = self.elevation_cos_weight.view(1, 1, 1, -1) * torch.cos(elevations_grid)
        # sin_elevations = self.elevation_sin_weight.view(1, 1, 1, -1) * torch.sin(elevations_grid)
        cos_elevations = torch.cos(elevations_grid)
        sin_elevations = torch.sin(elevations_grid)

        x = ranges_grid * cos_elevations * sin_azimuths
        y = ranges_grid * cos_elevations * cos_azimuths
        z = ranges_grid * sin_elevations
        x = x.flatten(start_dim=1).unsqueeze(-1)
        y = y.flatten(start_dim=1).unsqueeze(-1)
        z = z.flatten(start_dim=1).unsqueeze(-1)

        intensity = polar_frames.flatten(start_dim=1, end_dim=3).unsqueeze(-1)
        cartesian_points = torch.cat((x, y, z, intensity), dim=-1)

        return cartesian_points  # [B, N, 4]
