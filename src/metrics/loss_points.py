import torch
import torch.nn as nn

from metrics.base import PointcloudOccupancyLoss


class ChamferBceLoss(PointcloudOccupancyLoss):
    def __init__(self, spatial_weight=1.0, probability_weight=1.0, **kwargs):
        """
        Combined Chamfer Distance and Binary Cross-Entropy loss for 3D point clouds with occupancy probabilities.

        Args:
            spatial_weight (float): Weighting factor for Chamfer Distance loss.
            probability_weight (float): Weighting factor for BCE loss.
        """
        super().__init__(**kwargs)
        self.spatial_weight = spatial_weight
        self.probability_weight = probability_weight
        self.bce_loss = nn.BCELoss()

    def chamfer_distance(self, pred_list, true_list):
        """
        Computes the Chamfer Distance loss for variable-sized ground truth clouds.

        Args:
            pred_list (list of Tensors): List of predicted clouds [(N, 4), (N, 4), ...] with fixed N.
            true_list (list of Tensors): List of per-sample ground truth clouds [(M1, 4), (M2, 4), ...]

        Returns:
            Tensor: Chamfer Distance loss (scalar).
        """
        chamfer_losses = []
        for pred, true in zip(pred_list, true_list):
            pred_xyz = pred[:, :3]
            true_xyz = true[:, :3]
            dist_matrix = torch.cdist(pred_xyz, true_xyz, p=2) ** 2
            if torch.isnan(dist_matrix).any():
                print("NaN detected in dist_matrix before softmax!")
                print(dist_matrix)
                raise RuntimeError("NaN in dist_matrix!")
            if torch.isinf(dist_matrix).any():
                print("Infinity detected in dist_matrix before softmax!")
                print(dist_matrix)
                raise RuntimeError("Infinity in dist_matrix!")
            if (dist_matrix.abs() > 1e6).any():
                print("Very large values detected in dist_matrix before softmax!")
                print(dist_matrix)
            min_dist_pred_to_true = torch.min(dist_matrix, dim=1)[0]
            min_dist_true_to_pred = torch.min(dist_matrix, dim=0)[0]
            chamfer_loss = min_dist_pred_to_true.mean() + min_dist_true_to_pred.mean()
            chamfer_losses.append(chamfer_loss)
        loss = torch.stack(chamfer_losses).mean()
        return loss

    def occupancy_bce_loss(self, pred, true_list):
        """
        Computes BCE loss between predicted and true occupancy probabilities for variable-sized ground truth.

        Args:
            pred (list of Tensors): List of predicted clouds [(N, 4), (N, 4), ...] with fixed N.
            true_list (list of Tensors): List of per-sample ground truth clouds [(M1, 4), (M2, 4), ...]

        Returns:
            Tensor: BCE loss (scalar).
        """
        bce_losses = []
        temperature = 5.0  # Adjust softmax temperature
        for pred, true in zip(pred, true_list):
            pred_probs = pred[:, 3].unsqueeze(-1)
            true_probs = true[:, 3].unsqueeze(-1)
            dist_matrix = torch.cdist(pred[:, :3], true[:, :3], p=2)
            dist_matrix = torch.clamp(dist_matrix, min=1e-3, max=10.0)
            weights = torch.softmax(-dist_matrix / temperature, dim=1)
            if torch.isnan(weights).any():
                raise RuntimeError("NaNs detected in weights after softmax!")
            matched_true_probs = torch.bmm(weights.unsqueeze(0), true_probs.unsqueeze(0)).squeeze(0).squeeze(-1)
            bce_losses.append(self.bce_loss(pred_probs.squeeze(-1), matched_true_probs))
        return torch.stack(bce_losses).mean()

    def calc(self, pred, true):
        """
        Computes the combined Chamfer + BCE loss.

        Args:
            pred (list of Tensors): List of predicted clouds [(N, 4), (N, 4), ...] with fixed N.
            true (list of Tensors): List of ground truth point clouds with variable sizes.

        Returns:
            Tensor: Weighted sum of Chamfer Distance and BCE loss.
        """
        chamfer_loss = self.chamfer_distance(pred, true)
        bce_loss = self.occupancy_bce_loss(pred, true)
        total_loss = self.spatial_weight * chamfer_loss + self.probability_weight * bce_loss
        return total_loss
