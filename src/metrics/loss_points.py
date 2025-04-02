import torch
import torch.nn as nn

from metrics.base import PointcloudOccupancyLoss


class SpatialBceLoss(PointcloudOccupancyLoss):
    def __init__(self, spatial_weight=1.0, probability_weight=1.0, **kwargs):
        super().__init__(**kwargs)
        self.spatial_weight = spatial_weight
        self.probability_weight = probability_weight
        self.bce_loss = nn.BCELoss()

    def _calc_unmatched_loss(self, y_pred_values, y_true_values, y_pred_mapped_mask, y_true_mapped_mask):
        """
        Computes a penalty for unmatched occupied points using nearest neighbor distances.
        Handles empty inputs safely.
        """
        device = y_pred_values.device
        if y_pred_values.numel() == 0 and y_true_values.numel() == 0:
            return torch.tensor(0.0, device=device, requires_grad=True)

        unmatched_pred = torch.nonzero(~y_pred_mapped_mask, as_tuple=False).squeeze(1) if y_pred_mapped_mask.numel() > 0 else torch.empty(0, dtype=torch.long, device=device)
        unmatched_true = torch.nonzero(~y_true_mapped_mask, as_tuple=False).squeeze(1) if y_true_mapped_mask.numel() > 0 else torch.empty(0, dtype=torch.long, device=device)
        loss_pred = loss_true = 0.0

        if unmatched_pred.numel() > 0 and y_true_values.numel() > 0:
            dists_pred = torch.cdist(y_pred_values[unmatched_pred][:, :3], y_true_values[:, :3], p=2) ** 2
            loss_pred = dists_pred.min(dim=1)[0].mean()

        if unmatched_true.numel() > 0 and y_pred_values.numel() > 0:
            dists_true = torch.cdist(y_true_values[unmatched_true][:, :3], y_pred_values[:, :3], p=2) ** 2
            loss_true = dists_true.min(dim=1)[0].mean()

        loss_pred = loss_pred if isinstance(loss_pred, torch.Tensor) else torch.tensor(0.0, device=device, requires_grad=True)
        loss_true = loss_true if isinstance(loss_true, torch.Tensor) else torch.tensor(0.0, device=device, requires_grad=True)
        return (loss_pred + loss_true) / 2.0

    def _calc_chamfer_loss(self, y_pred_values_mapped, y_true_values_mapped, batch_indices_mapped):
        """
        Computes Chamfer loss between matched points. Returns 0 if there are no matched points.
        """
        if y_pred_values_mapped.numel() == 0 or y_true_values_mapped.numel() == 0:
            return torch.tensor(0.0, device=y_pred_values_mapped.device, requires_grad=True)

        sq_dists = torch.sum((y_pred_values_mapped[:, :3] - y_true_values_mapped[:, :3]) ** 2, dim=1)
        batch_size = int(torch.max(batch_indices_mapped).item() + 1) if batch_indices_mapped.numel() > 0 else 1
        batch_loss = torch.zeros(batch_size, device=y_pred_values_mapped.device)
        batch_loss = batch_loss.index_add(0, batch_indices_mapped, sq_dists)
        count = torch.zeros(batch_size, device=y_pred_values_mapped.device)
        count = count.index_add(0, batch_indices_mapped, torch.ones_like(sq_dists))
        mean_loss_per_batch = batch_loss / (count + 1e-8)
        return mean_loss_per_batch.mean()

    def _calc_occupancy_bce_loss(self, y_pred_values_mapped, y_true_values_mapped):
        """
        Computes BCE loss on occupancy. Returns 0 if there are no matched points.
        """
        if y_pred_values_mapped.numel() == 0 or y_true_values_mapped.numel() == 0:
            return torch.tensor(0.0, device=y_pred_values_mapped.device, requires_grad=True)
        return self.bce_loss(y_pred_values_mapped[:, 3], y_true_values_mapped[:, 3])

    def _calc(self, y_pred, y_true, data_buffer=None, *args, **kwargs):
        """
        Calculates the overall loss. First, we map the clouds using match_chamfer (via map_clouds) so that each matched pair comes from the same batch.
        Then we compute the chamfer loss and occupancy BCE loss on the mapped clouds, and an unmatched loss on the original clouds.
        All operations are tensor-based and differentiable.
        Validation checks are performed to ensure that each loss term is differentiable.
        """
        if data_buffer.occupied_only():
            y_pred, y_true = data_buffer.occupied_data()
            y_pred_values_mapped, y_true_values_mapped, batch_indices_mapped = data_buffer.occupied_mapped_clouds()
            y_pred_mapped_mask, y_true_mapped_mask = data_buffer.occupied_mapped_masks()
        else:
            y_pred_values_mapped, y_true_values_mapped, batch_indices_mapped = data_buffer.mapped_clouds()
            y_pred_mapped_mask, y_true_mapped_mask = data_buffer.mapped_masks()

        chamfer_loss = self._calc_chamfer_loss(y_pred_values_mapped, y_true_values_mapped, batch_indices_mapped)
        bce_loss = self._calc_occupancy_bce_loss(y_pred_values_mapped, y_true_values_mapped)
        unmatched_loss = self._calc_unmatched_loss(y_pred[0], y_true[0], y_pred_mapped_mask, y_true_mapped_mask)
        if not chamfer_loss.requires_grad:
            raise RuntimeError("chamfer_loss does not require gradients!")
        if not bce_loss.requires_grad:
            raise RuntimeError("bce_loss does not require gradients!")
        if not unmatched_loss.requires_grad:
            raise RuntimeError("unmatched_loss does not require gradients!")

        total_loss = self.spatial_weight * chamfer_loss + self.probability_weight * bce_loss + unmatched_loss
        return total_loss
