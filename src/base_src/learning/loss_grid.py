import torch
import torch.nn as nn
import torch.nn.functional as F


def sparse_tensor_loss(predicted_grid, groundtruth_cloud, grid_resolution, point_range):
    """
    Computes a sparse-aware loss using sparse tensor representations for the ground truth.

    Args:
        predicted_grid (torch.Tensor): Dense predicted occupancy grid of shape [B, X, Y, Z].
        groundtruth_cloud (torch.Tensor): Sparse ground truth point cloud of shape [B, N, 4] (x, y, z, prob).
        grid_resolution (float): Size of each voxel.
        point_range (tuple): (xmin, xmax, ymin, ymax, zmin, zmax).

    Returns:
        torch.Tensor: Sparse-aware loss.
    """
    B, X, Y, Z = predicted_grid.shape
    xmin, xmax, ymin, ymax, zmin, zmax = point_range

    positions = groundtruth_cloud[..., :3]
    probabilities = groundtruth_cloud[..., 3]
    voxel_indices = ((positions - torch.tensor([xmin, ymin, zmin], device=positions.device)) / grid_resolution).long()
    x_idx, y_idx, z_idx = voxel_indices[..., 0], voxel_indices[..., 1], voxel_indices[..., 2]
    x_idx = x_idx.clamp(0, X - 1)
    y_idx = y_idx.clamp(0, Y - 1)
    z_idx = z_idx.clamp(0, Z - 1)

    sparse_indices = torch.stack((torch.arange(B, device=positions.device).repeat_interleave(x_idx.size(1)), x_idx.flatten(), y_idx.flatten(), z_idx.flatten()), dim=1)
    sparse_values = probabilities.flatten()
    sparse_groundtruth = torch.sparse_coo_tensor(sparse_indices.T, sparse_values, size=(B, X, Y, Z), device=positions.device)
    dense_groundtruth = sparse_groundtruth.to_dense()

    loss = F.binary_cross_entropy_with_logits(predicted_grid, dense_groundtruth, reduction='mean')
    return loss


class SparseBceLoss(nn.Module):
    def __init__(self, alpha=1.0, beta=1.0, matching_temperature=1, distance_threshold=1.0):
        super(SoftMatchingLossScaled, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.temperature = matching_temperature
        self.distance_threshold = distance_threshold
        self.big_dist = distance_threshold * 2


def sparse_bce_loss(predicted_grid, groundtruth_cloud, grid_resolution, point_range):
    """
    Computes BCE loss focusing only on occupied voxels.

    Args:
        predicted_grid (torch.Tensor): Predicted grid of shape [B, X, Y, Z].
        groundtruth_cloud (torch.Tensor): Ground truth point clouds of shape [B, N, 4] (x, y, z, prob).
        grid_resolution (float): Voxel size.
        point_range (tuple): (xmin, xmax, ymin, ymax, zmin, zmax).

    Returns:
        torch.Tensor: Sparse BCE loss.
    """
    # Convert ground truth cloud to sparse grid
    groundtruth_grid = clouds_to_grids(groundtruth_cloud, grid_resolution, point_range)

    # Mask for occupied voxels in the ground truth
    occupied_mask = (groundtruth_grid > 0).float()

    # Apply BCE loss only to occupied voxels
    loss = F.binary_cross_entropy(predicted_grid, groundtruth_grid, reduction='none')
    sparse_loss = (loss * occupied_mask).sum() / occupied_mask.sum()
    return sparse_loss

