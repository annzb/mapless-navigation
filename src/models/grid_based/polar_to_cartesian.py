import torch
import torch.nn as nn


class PolarToCartesianGrids(nn.Module):
    def __init__(self, radar_config, device):
        """
        A parameter-free module to convert a polar 3D grid ([B, 1, El, R, Az]) directly into a Cartesian grid.

        Args:
            radar_config: Configuration object containing grid and radar parameters.
        """
        super(PolarToCartesianGrids, self).__init__()
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
