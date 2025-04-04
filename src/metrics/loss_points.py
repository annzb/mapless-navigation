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

    def _calc_spatial_loss(
            self,
            y_pred_values, y_pred_batch_indices, y_pred_mapped_mask,
            y_true_values, y_true_batch_indices, y_true_mapped_mask
    ):
        device = y_pred_values.device
        losses = []

        for b in range(self._batch_size):
            pred_b_mask = (y_pred_batch_indices == b)
            true_b_mask = (y_true_batch_indices == b)

            pred_values_b = y_pred_values[pred_b_mask]
            true_values_b = y_true_values[true_b_mask]

            pred_mapped_b = y_pred_mapped_mask[pred_b_mask]
            true_mapped_b = y_true_mapped_mask[true_b_mask]

            pred_matched = pred_values_b[pred_mapped_b]
            true_matched = true_values_b[true_mapped_b]

            matched_loss = torch.tensor(0.0, device=device)
            if pred_matched.size(0) > 0 and true_matched.size(0) > 0:
                dists = torch.cdist(pred_matched[:, :3], true_matched[:, :3], p=2) ** 2
                loss_pred = dists.min(dim=1)[0].mean()
                loss_true = dists.min(dim=0)[0].mean()
                matched_loss = (loss_pred + loss_true) / 2.0

            num_unmatched_true = (~true_mapped_b).sum()
            unmatched_loss = num_unmatched_true * self.spatial_penalty
            total = matched_loss + unmatched_loss
            losses.append(total if total.requires_grad else torch.tensor(0.0, device=device, requires_grad=True))

        return torch.stack(losses).mean()

    def _calc_occupancy_bce_loss(
            self,
            y_pred_values, y_pred_batch_indices, y_pred_mapped_mask,
            y_true_values, y_true_batch_indices, y_true_mapped_mask
    ):
        device = y_pred_values.device
        losses = []

        for b in range(self._batch_size):
            # Select current batch
            pred_b_mask = (y_pred_batch_indices == b)
            true_b_mask = (y_true_batch_indices == b)

            pred_values_b = y_pred_values[pred_b_mask]
            true_values_b = y_true_values[true_b_mask]

            pred_mapped_b = y_pred_mapped_mask[pred_b_mask]
            true_mapped_b = y_true_mapped_mask[true_b_mask]

            pred_matched = pred_values_b[pred_mapped_b]
            true_matched = true_values_b[true_mapped_b]

            # Matched BCE
            bce_loss_matched = torch.tensor(0.0, device=device)
            if pred_matched.numel() > 0 and true_matched.numel() > 0:
                bce_loss_matched = self.bce_loss(pred_matched[:, 3], true_matched[:, 3]).mean()

            # Unmatched true points: assume true occupancy = 1, pred occupancy = 0
            unmatched_true_mask = ~true_mapped_b
            if unmatched_true_mask.sum() > 0:
                num_unmatched = unmatched_true_mask.sum()
                fake_preds = torch.zeros(num_unmatched, device=device)
                true_targets = torch.ones(num_unmatched, device=device)
                bce_loss_unmatched = self.bce_loss(fake_preds, true_targets).mean()
            else:
                bce_loss_unmatched = torch.tensor(0.0, device=device, requires_grad=True)

            total_loss = bce_loss_matched + bce_loss_unmatched
            losses.append(total_loss if total_loss.requires_grad else torch.tensor(0.0, device=device, requires_grad=True))

        return torch.stack(losses).mean()

    def _calc(self, y_pred, y_true, data_buffer=None, *args, **kwargs):
        """
        Computes total loss using spatial distance and BCE for matched and unmatched points,
        calculated per-batch using data_buffer.occupied_only().
        """
        (y_pred_values, y_pred_batch_indices), (y_true_values, y_true_batch_indices) = y_pred, y_true
        full_pred_occupied_mask, full_true_occupied_mask = data_buffer.occupied_mask()
        full_pred_mapped_mask, full_true_mapped_mask = data_buffer.mapped_mask()

        if self.occupied_only:
            (y_pred_values, y_pred_batch_indices), (y_true_values, y_true_batch_indices) = data_buffer.get_occupied_data(y_pred, y_true)
            pred_occ_idx = torch.nonzero(full_pred_occupied_mask, as_tuple=False).squeeze(1)
            true_occ_idx = torch.nonzero(full_true_occupied_mask, as_tuple=False).squeeze(1)
            y_pred_mapped_mask = full_pred_mapped_mask[pred_occ_idx]
            y_true_mapped_mask = full_true_mapped_mask[true_occ_idx]
        else:
            y_pred_mapped_mask = full_pred_mapped_mask
            y_true_mapped_mask = full_true_mapped_mask

        spatial_loss = self._calc_spatial_loss(
            y_pred_values, y_pred_batch_indices, y_pred_mapped_mask,
            y_true_values, y_true_batch_indices, y_true_mapped_mask
        )
        bce_loss = self._calc_occupancy_bce_loss(
            y_pred_values, y_pred_batch_indices, y_pred_mapped_mask,
            y_true_values, y_true_batch_indices, y_true_mapped_mask
        )
        if not spatial_loss.requires_grad:
            raise RuntimeError("spatial_loss does not require gradients!")
        if not bce_loss.requires_grad:
            raise RuntimeError("bce_loss does not require gradients!")

        total_loss = self.spatial_weight * spatial_loss + self.probability_weight * bce_loss
        return total_loss
