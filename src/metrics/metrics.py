import torch
from sklearn.metrics import average_precision_score, roc_auc_score

from metrics.base import PointcloudOccupancyMetric


class OccupancyRatio(PointcloudOccupancyMetric):
    def _calc(self, y_pred, y_true, data_buffer=None, *args, **kwargs):
        pred_occupied_mask, true_occupied_mask = data_buffer.occupied_mask()
        pred_ratio = pred_occupied_mask.float().mean()
        true_ratio = true_occupied_mask.float().mean()
        score = 1.0 - torch.abs(pred_ratio - true_ratio)
        return score


class IoU(PointcloudOccupancyMetric):
    def _calc(self, y_pred, y_true, data_buffer=None, *args, **kwargs):
        pred_occ_mask, true_occ_mask = data_buffer.occupied_mask()
        matched_pairs = data_buffer.occupied_mapping()
        if matched_pairs is None or matched_pairs.size(0) == 0:
            return torch.tensor(0.0)
        pred_idxs = matched_pairs[:, 0]
        true_idxs = matched_pairs[:, 1]
        intersection_mask = pred_occ_mask[pred_idxs] & true_occ_mask[true_idxs]
        intersection = intersection_mask.sum().float()
        union = pred_occ_mask.sum().float() + true_occ_mask.sum().float() - intersection
        return intersection / (union + 1e-8)


class Precision(PointcloudOccupancyMetric):
    def _calc(self, y_pred, y_true, data_buffer=None, *args, **kwargs):
        pred_occ_mask, true_occ_mask = data_buffer.occupied_mask()
        matched_pairs = data_buffer.occupied_mapping()
        if matched_pairs is None or matched_pairs.size(0) == 0:
            return torch.tensor(0.0)
        pred_idxs = matched_pairs[:, 0]
        true_idxs = matched_pairs[:, 1]
        tp = (pred_occ_mask[pred_idxs] & true_occ_mask[true_idxs]).sum().float()
        fp = (pred_occ_mask[pred_idxs] & ~true_occ_mask[true_idxs]).sum().float()
        return tp / (tp + fp + 1e-8)


class Recall(PointcloudOccupancyMetric):
    def _calc(self, y_pred, y_true, data_buffer=None, *args, **kwargs):
        pred_occ_mask, true_occ_mask = data_buffer.occupied_mask()
        matched_pairs = data_buffer.occupied_mapping()
        if matched_pairs is None or matched_pairs.size(0) == 0:
            return torch.tensor(0.0)
        pred_idxs = matched_pairs[:, 0]
        true_idxs = matched_pairs[:, 1]
        tp = (pred_occ_mask[pred_idxs] & true_occ_mask[true_idxs]).sum().float()
        fn = (~pred_occ_mask[pred_idxs] & true_occ_mask[true_idxs]).sum().float()
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
        if self.occupied_only:
            y_pred_values, y_true_values, _ = data_buffer.get_occupied_mapped_data(y_pred, y_true)
        else:
            y_pred_values, y_true_values, _ = data_buffer.get_mapped_data(y_pred, y_true)

        if y_pred_values.numel() == 0 or y_true_values.numel() == 0:
            return torch.tensor(0.0, device=y_pred_values.device)
        sq_dists = torch.sum((y_pred_values[:, :3] - y_true_values[:, :3]) ** 2, dim=1)
        return sq_dists.mean()


class OccupancyMSE(PointcloudOccupancyMetric):
    def _calc(self, y_pred, y_true, data_buffer=None, *args, **kwargs):
        if self.occupied_only:
            y_pred_values, y_true_values, _ = data_buffer.get_occupied_mapped_data(y_pred, y_true)
        else:
            y_pred_values, y_true_values, _ = data_buffer.get_mapped_data(y_pred, y_true)

        if y_pred_values.numel() == 0 or y_true_values.numel() == 0:
            return torch.tensor(0.0, device=y_pred_values.device)
        return ((y_pred_values[:, 3] - y_true_values[:, 3]) ** 2).mean()


class AUROC(PointcloudOccupancyMetric):
    def _calc(self, y_pred, y_true, data_buffer=None, *args, **kwargs):
        if self.occupied_only:
            y_pred_values, y_true_values, _ = data_buffer.get_occupied_mapped_data(y_pred, y_true)
        else:
            y_pred_values, y_true_values, _ = data_buffer.get_mapped_data(y_pred, y_true)

        if y_pred_values.numel() == 0 or y_true_values.numel() == 0:
            return torch.tensor(1.0)

        y_score = y_pred_values[:, 3].detach().cpu().numpy()
        y_true_binary = y_true_values[:, 3].detach().cpu().numpy()

        if y_true_binary.min() == y_true_binary.max():
            return torch.tensor(1.0)
        score = roc_auc_score(y_true_binary, y_score)
        return torch.tensor(score)


class AUPRC(PointcloudOccupancyMetric):
    def _calc(self, y_pred, y_true, data_buffer=None, *args, **kwargs):
        if self.occupied_only:
            y_pred_values, y_true_values, _ = data_buffer.get_occupied_mapped_data(y_pred, y_true)
        else:
            y_pred_values, y_true_values, _ = data_buffer.get_mapped_data(y_pred, y_true)

        if y_pred_values.numel() == 0 or y_true_values.numel() == 0:
            return torch.tensor(1.0)
        y_score = y_pred_values[:, 3].detach().cpu().numpy()
        y_true_binary = y_true_values[:, 3].detach().cpu().numpy()
        if y_true_binary.sum() == 0:
            return torch.tensor(1.0)
        score = average_precision_score(y_true_binary, y_score)
        return torch.tensor(score)


class CoordinateError(PointcloudOccupancyMetric):
    """Metric to evaluate spatial accuracy of point predictions.
    
    This metric computes the mean Euclidean distance between matched points
    in the prediction and ground truth point clouds. Lower values indicate
    better spatial accuracy.
    """
    
    def _calc(self, y_pred, y_true, data_buffer=None, *args, **kwargs):
        # Get mapped points using the buffer
        if self.occupied_only:
            pred_matched, true_matched, _ = data_buffer.get_occupied_mapped_data(y_pred, y_true)
        else:
            pred_matched, true_matched, _ = data_buffer.get_mapped_data(y_pred, y_true)
            
        if pred_matched.numel() == 0 or true_matched.numel() == 0:
            return torch.tensor(float('inf'), device=pred_matched.device)
            
        # Compute Euclidean distances between matched points
        sq_dists = torch.sum((pred_matched[:, :3] - true_matched[:, :3]) ** 2, dim=1)
        return torch.sqrt(sq_dists.mean())


class ProbabilityError(PointcloudOccupancyMetric):
    """Metric to evaluate occupancy probability prediction accuracy.
    
    This metric computes the mean absolute error between predicted and ground truth
    occupancy probabilities for matched points. Lower values indicate better
    probability prediction accuracy.
    """
    
    def _calc(self, y_pred, y_true, data_buffer=None, *args, **kwargs):
        # Get mapped points using the buffer
        if self.occupied_only:
            pred_matched, true_matched, _ = data_buffer.get_occupied_mapped_data(y_pred, y_true)
        else:
            pred_matched, true_matched, _ = data_buffer.get_mapped_data(y_pred, y_true)
            
        if pred_matched.numel() == 0 or true_matched.numel() == 0:
            return torch.tensor(1.0, device=pred_matched.device)
            
        # Compute absolute error between predicted and ground truth probabilities
        abs_errors = torch.abs(pred_matched[:, 3] - true_matched[:, 3])
        return abs_errors.mean()
