import torch
import torch.nn as nn


class RadarOccupancyModel(nn.Module):
    def __init__(self, radar_config, batch_size=1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.radar_config = radar_config
        self.batch_size = batch_size
        self.name = 'radar_occupancy_model'
        self._init_points()

    def _init_points(self):
        el_bins = torch.tensor(self.radar_config.clipped_elevation_bins, dtype=torch.float16)
        az_bins = torch.tensor(self.radar_config.clipped_azimuth_bins, dtype=torch.float16)
        r_bins = torch.linspace(
            0,
            (self.radar_config.num_range_bins - 1) * self.radar_config.range_bin_width,
            self.radar_config.num_range_bins,
            dtype=torch.float16
        )
        el_grid, az_grid, r_grid = torch.meshgrid(el_bins, az_bins, r_bins, indexing="ij")
        x = r_grid * torch.cos(el_grid) * torch.sin(az_grid)
        y = r_grid * torch.cos(el_grid) * torch.cos(az_grid)
        z = r_grid * torch.sin(el_grid)

        support_coords = torch.stack((x, y, z), dim=-1).reshape(-1, 3)
        self.register_buffer("support_coords", support_coords)
    
    def apply_sigmoid(self, pcl_batch_flat):
        coords = pcl_batch_flat[:, :3]
        probs = torch.sigmoid(pcl_batch_flat[:, 3])
        return torch.cat((coords, probs.unsqueeze(-1)), dim=-1)

    def merge_batches(self, pcl_batch):
        B, N, C = pcl_batch.shape
        pcl_batch_flat = pcl_batch.reshape(B * N, C)
        batch_indices = torch.arange(B, device=pcl_batch.device).unsqueeze(1).repeat(1, N).view(-1)
        return pcl_batch_flat, batch_indices
    
    def check_gradient(self, tensor, name):
        if not self.training:
            return
        if not tensor.requires_grad:
            raise RuntimeError(f"{name} does not require gradients")
        if tensor.grad_fn is None:
            raise RuntimeError(f"{name} is not connected to computation graph")
