import torch
import torch.nn as nn


class RadarOccupancyModel(nn.Module):
    def __init__(self, radar_config, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.radar_config = radar_config
        self.name = 'radar_occupancy_model'

    def apply_sigmoid(self, pcl_batch_flat):
        """
        Apply sigmoid to the probability values in a merged batch.
        Args:
            pcl_batch_flat (torch.Tensor): A merged tensor of shape (B*N, 4) where the last column represents raw probability logits.
        Returns:
            torch.Tensor: A merged tensor of shape (B*N, 4) with the probability column passed through the sigmoid.
        """
        coords = pcl_batch_flat[:, :3]
        probs = torch.sigmoid(pcl_batch_flat[:, 3])
        return torch.cat((coords, probs.unsqueeze(-1)), dim=-1)

    def merge_batches(self, pcl_batch):
        """
        Merge the batch and point dimensions.
        Args:
            pcl_batch (torch.Tensor): Input tensor of shape (B, N, 4).
        Returns:
            merged (torch.Tensor): Tensor of shape (B*N, 4) with the merged data.
            batch_indices (torch.Tensor): Tensor of shape (B*N, ) containing the original batch index for each point.
        """
        B, N, C = pcl_batch.shape
        pcl_batch_flat = pcl_batch.view(B * N, C)
        batch_indices = torch.arange(B, device=pcl_batch.device).unsqueeze(1).repeat(1, N).view(-1)
        return pcl_batch_flat, batch_indices
