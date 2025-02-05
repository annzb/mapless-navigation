import torch
import torch.nn as nn

from base_classes import PointcloudOccupancyLoss


class ChamferBceLoss(PointcloudOccupancyLoss):
    def __init__(self, **kwargs):
        """
        Combined Chamfer Distance and Binary Cross-Entropy loss for 3D point clouds with occupancy probabilities.

        Args:
            spatial_weight (float): Weighting factor for Chamfer Distance loss.
            probability_weight (float): Weighting factor for BCE loss.
        """
        super().__init__(**kwargs)
        self.bce_loss = nn.BCELoss()

    def chamfer_distance(self, pred, true):
        pred_xyz = pred[:, :3]
        true_xyz = true[:, :3]
        dist_matrix = torch.cdist(pred_xyz, true_xyz, p=2) ** 2
        min_dist_pred_to_true = torch.min(dist_matrix, dim=1)[0]
        min_dist_true_to_pred = torch.min(dist_matrix, dim=0)[0]
        chamfer_loss = min_dist_pred_to_true.mean() + min_dist_true_to_pred.mean()
        return chamfer_loss

    def occupancy_bce_loss(self, pred, true):
        pred_probs = pred[:, 3].unsqueeze(1)
        true_probs = true[:, 3].unsqueeze(1)
        dist_matrix = torch.cdist(pred[:, :3], true[:, :3], p=2)
        weights = torch.softmax(-dist_matrix, dim=1)
        matched_true_probs = (weights @ true_probs).squeeze(1)
        return self.bce_loss(pred_probs, matched_true_probs)

    def calc(self, pred, true):
        """
        Computes the combined loss.

        Args:
            pred (Tensor): Predicted point cloud of shape (N, 4).
            true (Tensor): Ground truth point cloud of shape (M, 4).

        Returns:
            Tensor: Weighted sum of Chamfer Distance and BCE loss.
        """
        chamfer_loss = self.chamfer_distance(pred, true)
        bce_loss = self.occupancy_bce_loss(pred, true)
        total_loss = self.spatial_weight * chamfer_loss + self.probability_weight * bce_loss
        return total_loss
