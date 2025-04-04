import torch
from sklearn.metrics import average_precision_score, roc_auc_score

from metrics.base import PointcloudOccupancyMetric


class IoU(PointcloudOccupancyMetric):
    def _calc(self, y_pred, y_true, data_buffer=None, *args, **kwargs):
        y_pred_occupied_mask, y_true_occupied_mask = data_buffer.occupied_masks()
        y_pred_occupied_mapped_mask, y_true_occupied_mapped_mask = data_buffer.occupied_mapped_masks()

        intersection = (y_pred_occupied_mapped_mask & y_true_occupied_mapped_mask).sum().float()
        union = y_pred_occupied_mask.sum().float() + y_true_occupied_mask.sum().float() - intersection
        score = intersection / (union + 1e-8)
        return score


class Precision(PointcloudOccupancyMetric):
    def _calc(self, y_pred, y_true, data_buffer=None, *args, **kwargs):
        pred_mask, true_mask = data_buffer.occupied_masks()
        tp = (pred_mask & true_mask).sum().float()
        fp = (pred_mask & ~true_mask).sum().float()
        return tp / (tp + fp + 1e-8)


class Recall(PointcloudOccupancyMetric):
    def _calc(self, y_pred, y_true, data_buffer=None, *args, **kwargs):
        pred_mask, true_mask = data_buffer.occupied_masks()
        tp = (pred_mask & true_mask).sum().float()
        fn = (~pred_mask & true_mask).sum().float()
        return tp / (tp + fn + 1e-8)


class F1(PointcloudOccupancyMetric):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._precision = Precision(**kwargs)
        self._recall = Recall(**kwargs)

    def _calc(self, y_pred, y_true, data_buffer=None, *args, **kwargs):
        precision = self._precision(y_pred, y_true, data_buffer=data_buffer, *args, **kwargs)
        recall = self._recall(y_pred, y_true, data_buffer=data_buffer, *args, **kwargs)
        return 2 * precision * recall / (precision + recall + 1e-8)


class ChamferDistance(PointcloudOccupancyMetric):
    def _calc(self, y_pred, y_true, data_buffer=None, *args, **kwargs):
        if data_buffer.occupied_only():
            y_pred_values, y_true_values, _ = data_buffer.occupied_mapped_clouds()
        else:
            y_pred_values, y_true_values, _ = data_buffer.mapped_clouds()

        if y_pred_values.numel() == 0 or y_true_values.numel() == 0:
            return torch.tensor(0.0, device=y_pred_values.device)
        sq_dists = torch.sum((y_pred_values[:, :3] - y_true_values[:, :3]) ** 2, dim=1)
        return sq_dists.mean()


class OccupancyMSE(PointcloudOccupancyMetric):
    def _calc(self, y_pred, y_true, data_buffer=None, *args, **kwargs):
        if data_buffer.occupied_only():
            y_pred_values, y_true_values, _ = data_buffer.occupied_mapped_clouds()
        else:
            y_pred_values, y_true_values, _ = data_buffer.mapped_clouds()

        if y_pred_values.numel() == 0 or y_true_values.numel() == 0:
            return torch.tensor(0.0, device=y_pred_values.device)
        return ((y_pred_values[:, 3] - y_true_values[:, 3]) ** 2).mean()


class AUROC(PointcloudOccupancyMetric):
    def _calc(self, y_pred, y_true, data_buffer=None, *args, **kwargs):
        if data_buffer.occupied_only():
            y_pred_values, y_true_values, _ = data_buffer.occupied_mapped_clouds()
        else:
            y_pred_values, y_true_values, _ = data_buffer.mapped_clouds()

        if y_pred_values.numel() == 0 or y_true_values.numel() == 0:
            return torch.tensor(1.0)

        y_score = y_pred_values[:, 3].detach().cpu().numpy()
        y_true_binary = (y_true_values[:, 3] >= data_buffer.occupancy_threshold()).cpu().numpy()
        if y_true_binary.sum() == 0 or y_true_binary.sum() == len(y_true_binary):
            return torch.tensor(1.0)
        score = roc_auc_score(y_true_binary, y_score)
        return torch.tensor(score)


class AUPRC(PointcloudOccupancyMetric):
    def _calc(self, y_pred, y_true, data_buffer=None, *args, **kwargs):
        if data_buffer.occupied_only():
            y_pred_values, y_true_values, _ = data_buffer.occupied_mapped_clouds()
        else:
            y_pred_values, y_true_values, _ = data_buffer.mapped_clouds()

        if y_pred_values.numel() == 0 or y_true_values.numel() == 0:
            return torch.tensor(1.0)
        y_score = y_pred_values[:, 3].detach().cpu().numpy()
        y_true_binary = (y_true_values[:, 3] >= data_buffer.occupancy_threshold()).cpu().numpy()
        if y_true_binary.sum() == 0:
            return torch.tensor(1.0)
        score = average_precision_score(y_true_binary, y_score)
        return torch.tensor(score)
