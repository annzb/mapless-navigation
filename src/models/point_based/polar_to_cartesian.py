import torch
import torch.nn as nn

class PolarToCartesianPoints(nn.Module):
    def __init__(self, radar_config):
        super(PolarToCartesianPoints, self).__init__()
        self.radar_config = radar_config
        print(f'Radar params: {radar_config.num_elevation_bins} elevation bins, {radar_config.num_azimuth_bins} azimuth bins, {radar_config.num_range_bins} range bins')
        self.register_buffer("azimuths", torch.tensor(radar_config.clipped_azimuth_bins))
        self.register_buffer("elevations", torch.tensor(radar_config.clipped_elevation_bins))
        self.register_buffer("ranges", torch.linspace(0, radar_config.num_range_bins * radar_config.range_bin_width, radar_config.num_range_bins))

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

        # elevations = self.elevations * self.elevation_scale + self.elevation_bias
        # azimuths = self.azimuths * self.azimuth_scale + self.azimuth_bias
        # ranges = self.ranges * self.range_scale + self.range_bias
        elevations, azimuths, ranges = self.elevations, self.azimuths, self.ranges

        elevations_grid, azimuths_grid, ranges_grid = torch.meshgrid(elevations, azimuths, ranges, indexing="ij")
        ranges_grid = ranges_grid.expand(batch_size, -1, -1, -1)

        elevations_grid = elevations_grid.expand(batch_size, -1, -1, -1)
        cos_elevations = torch.cos(elevations_grid) # * self.elevation_cos_weight.view(1, 1, 1, -1)
        sin_elevations = torch.sin(elevations_grid) # * self.elevation_sin_weight.view(1, 1, 1, -1)

        azimuths_grid = azimuths_grid.expand(batch_size, -1, -1, -1)
        cos_azimuths = torch.cos(azimuths_grid) # * self.azimuth_cos_weight.view(1, -1, 1, 1)
        sin_azimuths = torch.sin(azimuths_grid) # * self.azimuth_sin_weight.view(1, -1, 1, 1)

        x = (ranges_grid * cos_elevations * sin_azimuths).reshape(batch_size, -1, 1)
        y = (ranges_grid * cos_elevations * cos_azimuths).reshape(batch_size, -1, 1)
        z = (ranges_grid * sin_elevations).reshape(batch_size, -1, 1)
        intensity = polar_frames.reshape(batch_size, -1, 1)
        cartesian_points = torch.cat((x, y, z, intensity), dim=-1)
        return cartesian_points
