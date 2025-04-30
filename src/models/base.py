import torch
import torch.nn as nn


class RadarOccupancyModel(nn.Module):
    def __init__(self, radar_config, batch_size=1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.radar_config = radar_config
        self.batch_size = batch_size
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
        pcl_batch_flat = pcl_batch.reshape(B * N, C)
        batch_indices = torch.arange(B, device=pcl_batch.device).unsqueeze(1).repeat(1, N).view(-1)
        return pcl_batch_flat, batch_indices
    
    def check_gradient(self, tensor, name):
        # tensor.register_hook(lambda g: print(g.norm()))
        if not self.training:
            return
        if not tensor.requires_grad:
            raise RuntimeError(f"{name} does not require gradients")
        # Just check if the tensor is connected to the computation graph
        if tensor.grad_fn is None:
            raise RuntimeError(f"{name} is not connected to computation graph")
