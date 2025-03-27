import torch

from metrics.base import PointcloudOccupancyMetric


class IoU(PointcloudOccupancyMetric):
    def _calc(self, y_pred, y_true, **kwargs):
        y_pred_mapped, y_true_mapped, mapped_indices, y_pred_matched_indices, y_true_matched_indices = self._point_mapper.get_mapped_clouds()

        true_values, true_indices = y_true
        batch_size = y_pred.shape[0]
        batch_iou_scores = []

        for batch_idx in range(batch_size):
            pred_coords = y_pred[batch_idx][:, :3]
            gt_coords = true_values[true_indices == batch_idx][:, :3]

            if pred_coords.shape[0] == 0 or gt_coords.shape[0] == 0:
                batch_iou_scores.append(torch.tensor(0.0, device=pred_coords.device))
                continue

            pairwise_distances = torch.cdist(pred_coords.unsqueeze(0), gt_coords.unsqueeze(0), p=2).squeeze(0)
            intersection = (
                (pairwise_distances.min(dim=1).values <= self.max_point_distance).sum() +
                (pairwise_distances.min(dim=0).values <= self.max_point_distance).sum()
            ) // 2

            union = pred_coords.shape[0] + gt_coords.shape[0] - intersection
            iou_score = intersection / union if union > 0 else torch.tensor(0.0, device=pred_coords.device)
            batch_iou_scores.append(iou_score)

        return torch.stack(batch_iou_scores).mean().item()


class WeightedChamfer(PointcloudOccupancyMetric):
    def _calc(self, y_pred, y_true):
        y_pred_mapped, y_true_mapped, mapped_indices, y_pred_matched_indices, y_true_matched_indices = self._point_mapper.get_mapped_clouds()
        batch_size = y_pred.shape[0]
        batch_chamfer_distances = []

        for batch_idx in range(batch_size):
            pred_sample = y_pred[batch_idx]
            gt_sample = true_values[true_indices == batch_idx]

            pred_coords = pred_sample[:, :3]
            pred_probs = pred_sample[:, 3]
            gt_coords = gt_sample[:, :3]
            gt_probs = gt_sample[:, 3]

            if pred_coords.shape[0] == 0 or gt_coords.shape[0] == 0:
                batch_chamfer_distances.append(torch.tensor(0.0, device=pred_coords.device))
                continue

            pairwise_distances = torch.cdist(pred_coords.unsqueeze(0), gt_coords.unsqueeze(0), p=2).squeeze(0)
            forward_distances, _ = pairwise_distances.min(dim=1)
            forward_chamfer = torch.sum(pred_probs * forward_distances ** 2) / torch.sum(pred_probs)
            backward_distances, _ = pairwise_distances.min(dim=0)
            backward_chamfer = torch.sum(gt_probs * backward_distances ** 2) / torch.sum(gt_probs)
            total_chamfer = backward_chamfer + forward_chamfer
            batch_chamfer_distances.append(total_chamfer)

        return torch.stack(batch_chamfer_distances).mean().item()
