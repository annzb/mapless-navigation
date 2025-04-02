import torch
from sklearn.metrics import average_precision_score, roc_auc_score

from metrics.base import PointcloudOccupancyMetric


class IoU(PointcloudOccupancyMetric):
    def _calc(self, y_pred, y_true, data_buffer=None, *args, **kwargs):
        y_pred_indices_occupied, y_true_indices_occupied = data_buffer.occupied_indices()
        pred_mask = y_pred_indices_occupied
        true_mask = y_true_indices_occupied
        intersection = (pred_mask & true_mask).sum().float()
        union = (pred_mask | true_mask).sum().float()
        if union == 0:
            return torch.tensor(1.0, device=intersection.device) if intersection == 0 else torch.tensor(0.0, device=intersection.device)
        return intersection / union


class Precision(PointcloudOccupancyMetric):
    def _calc(self, y_pred, y_true, data_buffer=None, *args, **kwargs):
        pred_mask, true_mask = data_buffer.occupied_indices()
        tp = (pred_mask & true_mask).sum().float()
        fp = (pred_mask & ~true_mask).sum().float()
        return tp / (tp + fp + 1e-8)


class Recall(PointcloudOccupancyMetric):
    def _calc(self, y_pred, y_true, data_buffer=None, *args, **kwargs):
        pred_mask, true_mask = data_buffer.occupied_indices()
        tp = (pred_mask & true_mask).sum().float()
        fn = (~pred_mask & true_mask).sum().float()
        return tp / (tp + fn + 1e-8)


class F1(PointcloudOccupancyMetric):
    def _calc(self, y_pred, y_true, data_buffer=None, *args, **kwargs):
        pred_mask, true_mask = data_buffer.occupied_indices()
        tp = (pred_mask & true_mask).sum().float()
        fp = (pred_mask & ~true_mask).sum().float()
        fn = (~pred_mask & true_mask).sum().float()
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        return 2 * precision * recall / (precision + recall + 1e-8)


class ChamferDistance(PointcloudOccupancyMetric):
    def _calc(self, y_pred, y_true, data_buffer=None, *args, **kwargs):
        y_pred_values_mapped, y_true_values_mapped, _ = data_buffer.mapped_clouds()
        sq_dists = torch.sum((y_pred_values_mapped[:, :3] - y_true_values_mapped[:, :3]) ** 2, dim=1)
        return sq_dists.mean()


class OccupancyMSE(PointcloudOccupancyMetric):
    def _calc(self, y_pred, y_true, data_buffer=None, *args, **kwargs):
        y_pred_values_mapped, y_true_values_mapped, _ = data_buffer.mapped_clouds()
        return ((y_pred_values_mapped[:, 3] - y_true_values_mapped[:, 3]) ** 2).mean()


class AUROC(PointcloudOccupancyMetric):
    def _calc(self, y_pred, y_true, data_buffer=None, *args, **kwargs):
        y_pred_values_mapped, y_true_values_mapped, _ = data_buffer.mapped_clouds()
        y_score = y_pred_values_mapped[:, 3].detach().cpu().numpy()
        y_true_binary = (y_true_values_mapped[:, 3] >= data_buffer.occupancy_threshold()).cpu().numpy()
        if y_true_binary.sum() == 0 or y_true_binary.sum() == len(y_true_binary):
            return torch.tensor(1.0)
        score = roc_auc_score(y_true_binary, y_score)
        return torch.tensor(score)


class AUPRC(PointcloudOccupancyMetric):
    def _calc(self, y_pred, y_true, data_buffer=None, *args, **kwargs):
        y_pred_values_mapped, y_true_values_mapped, _ = data_buffer.mapped_clouds()
        y_score = y_pred_values_mapped[:, 3].detach().cpu().numpy()
        y_true_binary = (y_true_values_mapped[:, 3] >= data_buffer.occupancy_threshold()).cpu().numpy()
        if y_true_binary.sum() == 0:
            return torch.tensor(1.0)
        score = average_precision_score(y_true_binary, y_score)
        return torch.tensor(score)
