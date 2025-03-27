import torch
import torch.nn as nn

from metrics.base import PointcloudOccupancyLoss


class ChamferBceLoss(PointcloudOccupancyLoss):
    def __init__(self, spatial_weight=1.0, probability_weight=1.0, **kwargs):
        super().__init__(**kwargs)
        self.spatial_weight = spatial_weight
        self.probability_weight = probability_weight
        self.bce_loss = nn.BCELoss()

    def _calc_unmatched_loss(self, y_pred_values, y_true_values, y_pred_matched_indices, y_true_matched_indices):
        """
        For points in the original clouds that were not matched, computes a penalty based on the nearest neighbor distance.
        Args:
            orig_pred (torch.Tensor): Original y_pred values, shape (P, 4).
            orig_true (torch.Tensor): Original y_true values, shape (Q, 4).
            mapping_pred_indices (torch.Tensor): Indices in orig_pred that were matched.
            mapping_true_indices (torch.Tensor): Indices in orig_true that were matched.
        Returns:
            A scalar penalty value.
        """
        device, pred_size = y_pred_values.device, y_pred_values.size(0)
        all_pred_idx = torch.arange(pred_size, device=device)
        all_true_idx = torch.arange(pred_size, device=device)
        mask_pred = torch.zeros(pred_size, dtype=torch.bool, device=device)
        mask_pred[y_pred_matched_indices] = True
        mask_true = torch.zeros(y_true_values.size(0), dtype=torch.bool, device=device)
        mask_true[y_true_matched_indices] = True
        unmatched_pred = all_pred_idx[~mask_pred]
        unmatched_true = all_true_idx[~mask_true]
        loss_pred = torch.tensor(0.0, device=device)
        if unmatched_pred.numel() > 0:
            dists_pred = torch.cdist(y_pred_values[unmatched_pred][:, :3], y_true_values[:, :3], p=2) ** 2
            min_dists_pred, _ = dists_pred.min(dim=1)
            loss_pred = min_dists_pred.mean()
        loss_true = torch.tensor(0.0, device=device)
        if unmatched_true.numel() > 0:
            dists_true = torch.cdist(y_true_values[unmatched_true][:, :3], y_pred_values[:, :3], p=2) ** 2
            min_dists_true, _ = dists_true.min(dim=1)
            loss_true = min_dists_true.mean()
        return (loss_pred + loss_true) / 2.0

    def _calc_chamfer_loss(self, y_pred_values_mapped, y_true_values_mapped, batch_indices_mapped):
        """
        Computes the chamfer loss for already matched point clouds.
        Assumes that mapped_y_pred and mapped_y_true are tensors of shape (K, 4)
        and mapped_batch (of shape (K,)) indicates the batch each pair belongs to.
        """
        sq_dists = torch.sum((y_pred_values_mapped[:, :3] - y_true_values_mapped[:, :3]) ** 2, dim=1)
        batch_size = int(torch.max(batch_indices_mapped).item() + 1)
        batch_loss = torch.zeros(batch_size, device=y_pred_values_mapped.device)
        batch_loss = batch_loss.index_add(0, batch_indices_mapped, sq_dists)
        count = torch.zeros(batch_size, device=y_pred_values_mapped.device)
        count = count.index_add(0, batch_indices_mapped, torch.ones_like(sq_dists))
        mean_loss_per_batch = batch_loss / (count + 1e-8)
        return mean_loss_per_batch.mean()

    def _calc_occupancy_bce_loss(self, y_pred_values_mapped, y_true_values_mapped):
        """
        Computes the occupancy BCE loss on matched point clouds.
        Assumes that mapped_y_pred and mapped_y_true are tensors of shape (K, 4) where column 3 holds occupancy probabilities.
        """
        return self.bce_loss(y_pred_values_mapped[:, 3], y_true_values_mapped[:, 3])

    def _calc(self, y_pred, y_true, **kwargs):
        """
        Calculates the overall loss. First, we map the clouds using match_chamfer (via map_clouds) so that each matched pair comes from the same batch.
        Then we compute the chamfer loss and occupancy BCE loss on the mapped clouds, and an unmatched loss on the original clouds.
        All operations are tensor-based and differentiable.
        Validation checks are performed to ensure that each loss term is differentiable.
        """
        y_pred_mapped, y_true_mapped, mapped_indices, y_pred_matched_indices, y_true_matched_indices = self._point_mapper.get_mapped_clouds()

        chamfer_loss = self._calc_chamfer_loss(y_pred_mapped, y_true_mapped, mapped_indices)
        bce_loss = self._calc_occupancy_bce_loss(y_pred_mapped, y_true_mapped)
        unmatched_loss = self._calc_unmatched_loss(y_pred[0], y_true[0], y_pred_matched_indices, y_true_matched_indices)
        if not chamfer_loss.requires_grad:
            raise RuntimeError("chamfer_loss does not require gradients!")
        if not bce_loss.requires_grad:
            raise RuntimeError("bce_loss does not require gradients!")
        if not unmatched_loss.requires_grad:
            raise RuntimeError("unmatched_loss does not require gradients!")

        total_loss = self.spatial_weight * chamfer_loss + self.probability_weight * bce_loss + unmatched_loss
        return total_loss
