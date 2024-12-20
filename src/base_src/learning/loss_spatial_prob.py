import torch
import torch.nn as nn
import torch.nn.functional as F


def match_pointclouds(true_xyz, pred_xyz, max_distance=float('inf')):
    """
    Matches true points to predicted points within a maximum distance.

    Args:
        true_xyz (torch.Tensor): Ground truth points of shape [N_true, 3].
        pred_xyz (torch.Tensor): Predicted points of shape [N_pred, 3].
        max_distance (float): Maximum allowable distance for matching.

    Returns:
        matched_true_xyz (torch.Tensor): Matched true points, or empty tensor if no matches.
        matched_pred_xyz (torch.Tensor): Matched predicted points, or empty tensor if no matches.
        matched_true_idx (torch.Tensor): Indices of matched true points, or empty tensor if no matches.
        matched_pred_idx (torch.Tensor): Indices of matched predicted points, or empty tensor if no matches.
    """
    # if true_xyz.size(0) == 0 or pred_xyz.size(0) == 0:
    #     return (
    #         torch.empty((0, 3), device=true_xyz.device),
    #         torch.empty((0, 3), device=pred_xyz.device),
    #         torch.empty((0,), dtype=torch.long, device=true_xyz.device),
    #         torch.empty((0,), dtype=torch.long, device=pred_xyz.device),
    #     )
    dists = torch.cdist(true_xyz, pred_xyz)  # [N_true, N_pred]
    valid_mask = dists <= max_distance
    dists[~valid_mask] = float('inf')

    matched_true_idx = []
    matched_pred_idx = []
    for i in range(dists.size(0)):
        if valid_mask[i].any():  # Check if there are any valid matches for this true point
            min_dist, min_idx = dists[i].min(dim=0)
            if min_dist != float('inf'):  # Valid match found
                matched_true_idx.append(i)
                matched_pred_idx.append(min_idx.item())
                dists[:, min_idx] = float('inf')  # Invalidate the matched predicted point

    if not matched_true_idx:
        return (
            torch.empty((0, 3), device=true_xyz.device),
            torch.empty((0, 3), device=pred_xyz.device),
            torch.empty((0,), dtype=torch.long, device=true_xyz.device),
            torch.empty((0,), dtype=torch.long, device=pred_xyz.device),
        )
    matched_true_idx = torch.tensor(matched_true_idx, dtype=torch.long, device=true_xyz.device)
    matched_pred_idx = torch.tensor(matched_pred_idx, dtype=torch.long, device=pred_xyz.device)
    matched_true_xyz = true_xyz[matched_true_idx]
    matched_pred_xyz = pred_xyz[matched_pred_idx]
    return matched_true_xyz, matched_pred_xyz, matched_true_idx, matched_pred_idx


class SpatialProbLoss(nn.Module):
    def __init__(self, occupancy_threshold=0.5, point_match_radius=1.0):
        """
        Initializes the loss class with threshold and radius parameters.

        Args:
            occupancy_threshold (float): Threshold to filter occupied points based on probabilities.
            point_match_radius (float): Maximum distance for matching points.
        """
        super(SpatialProbLoss, self).__init__()
        self.occupancy_threshold = occupancy_threshold
        self.point_match_radius = point_match_radius

    def forward(self, pred_cloud, true_cloud):
        """
        Computes the spatial-probability loss.

        Args:
            pred_cloud (torch.Tensor): Predicted point cloud of shape [N_pred, 4] (XYZP).
            true_cloud (torch.Tensor): Ground truth point cloud of shape [N_true, 4] (XYZP).

        Returns:
            torch.Tensor: Combined spatial and probability loss.
        """
        pred_occupied = pred_cloud[pred_cloud[:, -1] >= self.occupancy_threshold]
        true_occupied = true_cloud[true_cloud[:, -1] >= self.occupancy_threshold]
        pred_xyz, true_xyz = pred_occupied[:, :3], true_occupied[:, :3]
        pred_probs, true_probs = pred_occupied[:, 3], true_occupied[:, 3]

        matched_true_xyz, matched_pred_xyz, matched_true_idx, matched_pred_idx = match_pointclouds(true_xyz, pred_xyz, max_distance=self.point_match_radius)
        unmatched_mask = torch.ones(true_xyz.size(0), device=true_xyz.device, dtype=torch.bool)
        unmatched_mask[matched_true_idx] = False
        num_unmatched_points = unmatched_mask.sum()
        matched_distances = torch.norm(matched_true_xyz - matched_pred_xyz, dim=-1)
        spatial_error = matched_distances.mean() + self.point_match_radius * 10 * num_unmatched_points
        prob_error = F.mse_loss(true_probs[matched_true_idx], pred_probs[matched_pred_idx]) + num_unmatched_points

        loss = spatial_error + prob_error
        return loss
