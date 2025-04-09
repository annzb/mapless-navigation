import torch
import torch.nn as nn

from metrics.base import PointcloudOccupancyLoss


class SpatialBceLoss(PointcloudOccupancyLoss):
    def __init__(self, unmatched_point_spatial_penalty, spatial_weight=1.0, probability_weight=1.0, **kwargs):
        super().__init__(**kwargs)
        self.spatial_penalty = unmatched_point_spatial_penalty
        self.spatial_weight = spatial_weight
        self.probability_weight = probability_weight
        self.bce_loss = nn.BCELoss(reduction='none')

    def _calc_spatial_loss(self, y_pred, y_true, data_buffer):
        device = y_pred[0].device
        (y_pred_values, y_pred_batch_indices), (y_true_values, y_true_batch_indices) = y_pred, y_true

        if not y_pred_values.requires_grad:
            raise RuntimeError("Predicted values do not require gradients!")

        losses = []
        if self.occupied_only:
            pred_matched, true_matched, batch_indices = data_buffer.get_occupied_mapped_data(y_pred, y_true)
            _, true_occ_mask = data_buffer.occupied_mask()
            _, mapped_true_mask = data_buffer.occupied_mapped_mask()
        else:
            pred_matched, true_matched, batch_indices = data_buffer.get_mapped_data(y_pred, y_true)
            _, true_occ_mask = torch.ones_like(y_true_batch_indices, dtype=torch.bool), torch.ones_like(
                y_true_batch_indices, dtype=torch.bool)
            _, mapped_true_mask = data_buffer.mapped_mask()

        unmatched_true_mask = true_occ_mask & ~mapped_true_mask  # Only those not matched
        unmatched_true_batch_indices = y_true_batch_indices[unmatched_true_mask]

        for b in range(self._batch_size):
            matched_b = batch_indices == b
            pred_b = pred_matched[matched_b]
            true_b = true_matched[matched_b]

            if pred_b.numel() > 0 and not pred_b.requires_grad:
                raise RuntimeError("Matched predicted values do not require gradients!")

            matched_loss = torch.tensor(0.0, device=device, requires_grad=True)
            if pred_b.size(0) > 0 and true_b.size(0) > 0:
                dists = torch.cdist(pred_b[:, :3], true_b[:, :3], p=2) ** 2
                loss_pred = dists.min(dim=1)[0].mean()
                loss_true = dists.min(dim=0)[0].mean()
                matched_loss = (loss_pred + loss_true) / 2.0

            num_unmatched = (unmatched_true_batch_indices == b).sum()
            unmatched_loss = num_unmatched * self.spatial_penalty
            total = matched_loss + unmatched_loss

            if not total.requires_grad:
                raise RuntimeError("Spatial loss term does not require gradients!")

            losses.append(total)

        return torch.stack(losses).mean()

    def _calc_occupancy_bce_loss(self, y_pred, y_true, data_buffer):
        device = y_pred[0].device
        (y_pred_values, y_pred_batch_indices), (y_true_values, y_true_batch_indices) = y_pred, y_true

        if not y_pred_values.requires_grad:
            raise RuntimeError("Predicted values do not require gradients!")

        losses = []

        if self.occupied_only:
            pred_matched, true_matched, batch_indices = data_buffer.get_occupied_mapped_data(y_pred, y_true)
            _, true_occ_mask = data_buffer.occupied_mask()
            _, mapped_true_mask = data_buffer.occupied_mapped_mask()
        else:
            pred_matched, true_matched, batch_indices = data_buffer.get_mapped_data(y_pred, y_true)
            _, true_occ_mask = torch.ones_like(y_true_batch_indices, dtype=torch.bool), torch.ones_like(
                y_true_batch_indices, dtype=torch.bool)
            _, mapped_true_mask = data_buffer.mapped_mask()

        unmatched_true_mask = true_occ_mask & ~mapped_true_mask
        unmatched_true_batch_indices = y_true_batch_indices[unmatched_true_mask]

        for b in range(self._batch_size):
            matched_b = batch_indices == b
            pred_b = pred_matched[matched_b]
            true_b = true_matched[matched_b]

            if pred_b.numel() > 0 and not pred_b.requires_grad:
                raise RuntimeError("Matched predicted values do not require gradients!")

            bce_loss_matched = torch.tensor(0.0, device=device, requires_grad=True)
            if pred_b.numel() > 0 and true_b.numel() > 0:
                bce_loss_matched = self.bce_loss(pred_b[:, 3], true_b[:, 3]).mean()

            num_unmatched = (unmatched_true_batch_indices == b).sum()
            if num_unmatched > 0:
                fake_preds = torch.zeros(num_unmatched, device=device, requires_grad=True)
                true_targets = torch.ones(num_unmatched, device=device)
                bce_loss_unmatched = self.bce_loss(fake_preds, true_targets).mean()
            else:
                bce_loss_unmatched = torch.tensor(0.0, device=device, requires_grad=True)

            total_loss = bce_loss_matched + bce_loss_unmatched
            if not total_loss.requires_grad:
                raise RuntimeError("Occupancy BCE loss term does not require gradients!")

            losses.append(total_loss)

        return torch.stack(losses).mean()

    def _calc(self, y_pred, y_true, data_buffer=None, *args, **kwargs):
        if not y_pred[0].requires_grad:
            raise RuntimeError("y_pred_values does not require gradients at the start of _calc")

        spatial_loss = self._calc_spatial_loss(y_pred, y_true, data_buffer)
        bce_loss = self._calc_occupancy_bce_loss(y_pred, y_true, data_buffer)

        return self.spatial_weight * spatial_loss + self.probability_weight * bce_loss
