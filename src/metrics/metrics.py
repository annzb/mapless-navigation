import torch
import torch.nn.functional as F

from metrics.base import PointcloudOccupancyMetric
from metrics.loss_points import PointLoss2


class MatchedPointRatio(PointcloudOccupancyMetric):
    def _calc(self, y_pred, y_true, data_buffer=None, *args, **kwargs):
        y_pred_batch_indices, y_true_batch_indices = y_pred[1], y_true[1]
        target_masks = self._calc_matched_masks(y_pred_batch_indices, y_true_batch_indices, data_buffer)
        matched_ratios = self._calc_matching_ratios(y_pred_batch_indices, y_true_batch_indices, data_buffer, target_masks)
        return matched_ratios.mean()
    

class UnmatchedLossMetric(PointcloudOccupancyMetric, PointLoss2):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.negative = True

    def _calc(self, y_pred, y_true, data_buffer=None, *args, **kwargs):
        return self._calc_unmatched_loss(y_pred, y_true, data_buffer, *args, **kwargs)
    

class SpatialLossMetric(UnmatchedLossMetric):
    def _calc(self, y_pred, y_true, data_buffer=None, *args, **kwargs):
        spatial_loss, _ = self._calc_matched_loss(y_pred, y_true, data_buffer, *args, **kwargs)
        return spatial_loss


class OccupancyLossMetric(UnmatchedLossMetric):
    def _calc(self, y_pred, y_true, data_buffer=None, *args, **kwargs):
        _, occupancy_loss = self._calc_matched_loss(y_pred, y_true, data_buffer, *args, **kwargs)
        return occupancy_loss


# class OccupancyPrecisionRecall(PointcloudOccupancyMetric):
#     def _calc(self, y_pred, y_true, data_buffer=None, *args, **kwargs):
#         mapping = data_buffer.mapping()
#         pred_matched_mask = torch.zeros(len(y_pred[0]), dtype=torch.bool, device=y_pred[0].device)
#         pred_matched_mask[mapping[:, 0]] = True

#         precision = pred_matched_mask.float().mean()
#         recall = mapping.size(0) / (len(y_true[0]) + 1e-8)

#         return {'precision': precision, 'recall': recall}
    

# class UnmatchedPointRatio(PointcloudOccupancyMetric):
#     def _calc(self, y_pred, y_true, data_buffer=None, *args, **kwargs):
#         pred_mask, true_mask = data_buffer.mapped_mask()
#         unmatched_pred_ratio = (~pred_mask).float().mean()
#         unmatched_true_ratio = (~true_mask).float().mean()
#         return (unmatched_pred_ratio + unmatched_true_ratio) / 2


# class IoU(PointcloudOccupancyMetric):
#     def _calc(self, y_pred, y_true, data_buffer=None, *args, **kwargs):
#         pred_occ_mask, true_occ_mask = data_buffer.occupied_mask()
#         matched_pairs = data_buffer.occupied_mapping()
#         if matched_pairs is None or matched_pairs.size(0) == 0:
#             return torch.tensor(0.0)
#         pred_idxs = matched_pairs[:, 0]
#         true_idxs = matched_pairs[:, 1]
#         intersection_mask = pred_occ_mask[pred_idxs] & true_occ_mask[true_idxs]
#         intersection = intersection_mask.sum().float()
#         union = pred_occ_mask.sum().float() + true_occ_mask.sum().float() - intersection
#         return intersection / (union + 1e-8)


# class Precision(PointcloudOccupancyMetric):
#     def _calc(self, y_pred, y_true, data_buffer=None, *args, **kwargs):
#         pred_occ_mask, true_occ_mask = data_buffer.occupied_mask()
#         matched_pairs = data_buffer.occupied_mapping()
#         if matched_pairs is None or matched_pairs.size(0) == 0:
#             return torch.tensor(0.0)
#         pred_idxs = matched_pairs[:, 0]
#         true_idxs = matched_pairs[:, 1]
#         tp = (pred_occ_mask[pred_idxs] & true_occ_mask[true_idxs]).sum().float()
#         fp = (pred_occ_mask[pred_idxs] & ~true_occ_mask[true_idxs]).sum().float()
#         return tp / (tp + fp + 1e-8)


# class Recall(PointcloudOccupancyMetric):
#     def _calc(self, y_pred, y_true, data_buffer=None, *args, **kwargs):
#         pred_occ_mask, true_occ_mask = data_buffer.occupied_mask()
#         matched_pairs = data_buffer.occupied_mapping()
#         if matched_pairs is None or matched_pairs.size(0) == 0:
#             return torch.tensor(0.0)
#         pred_idxs = matched_pairs[:, 0]
#         true_idxs = matched_pairs[:, 1]
#         tp = (pred_occ_mask[pred_idxs] & true_occ_mask[true_idxs]).sum().float()
#         fn = (~pred_occ_mask[pred_idxs] & true_occ_mask[true_idxs]).sum().float()
#         return tp / (tp + fn + 1e-8)


# class F1(PointcloudOccupancyMetric):
#     def __init__(self, **kwargs):
#         super().__init__(**kwargs)
#         self._precision = Precision(**kwargs)
#         self._recall = Recall(**kwargs)

#     def _calc(self, y_pred, y_true, data_buffer=None, *args, **kwargs):
#         precision = self._precision(y_pred, y_true, data_buffer=data_buffer, *args, **kwargs)
#         recall = self._recall(y_pred, y_true, data_buffer=data_buffer, *args, **kwargs)
#         return 2 * precision * recall / (precision + recall + 1e-8)


# class ChamferDistance(PointcloudOccupancyMetric):
#     def _calc(self, y_pred, y_true, data_buffer=None, *args, **kwargs):
#         if self.occupied_only:
#             y_pred_values, y_true_values, _ = data_buffer.get_occupied_mapped_data(y_pred, y_true)
#         else:
#             y_pred_values, y_true_values, _ = data_buffer.get_mapped_data(y_pred, y_true)

#         if y_pred_values.numel() == 0 or y_true_values.numel() == 0:
#             return torch.tensor(0.0, device=y_pred_values.device)
#         sq_dists = torch.sum((y_pred_values[:, :3] - y_true_values[:, :3]) ** 2, dim=1)
#         return sq_dists.mean()


# class OccupancyMSE(PointcloudOccupancyMetric):
#     def _calc(self, y_pred, y_true, data_buffer=None, *args, **kwargs):
#         if self.occupied_only:
#             y_pred_values, y_true_values, _ = data_buffer.get_occupied_mapped_data(y_pred, y_true)
#         else:
#             y_pred_values, y_true_values, _ = data_buffer.get_mapped_data(y_pred, y_true)

#         if y_pred_values.numel() == 0 or y_true_values.numel() == 0:
#             return torch.tensor(0.0, device=y_pred_values.device)
#         return ((y_pred_values[:, 3] - y_true_values[:, 3]) ** 2).mean()


# class AUROC(PointcloudOccupancyMetric):
#     def _calc(self, y_pred, y_true, data_buffer=None, *args, **kwargs):
#         if self.occupied_only:
#             y_pred_values, y_true_values, _ = data_buffer.get_occupied_mapped_data(y_pred, y_true)
#         else:
#             y_pred_values, y_true_values, _ = data_buffer.get_mapped_data(y_pred, y_true)

#         if y_pred_values.numel() == 0 or y_true_values.numel() == 0:
#             return torch.tensor(1.0)

#         y_score = y_pred_values[:, 3].detach().cpu().numpy()
#         y_true_binary = y_true_values[:, 3].detach().cpu().numpy()

#         if y_true_binary.min() == y_true_binary.max():
#             return torch.tensor(1.0)
#         score = roc_auc_score(y_true_binary, y_score)
#         return torch.tensor(score)


# class AUPRC(PointcloudOccupancyMetric):
#     def _calc(self, y_pred, y_true, data_buffer=None, *args, **kwargs):
#         if self.occupied_only:
#             y_pred_values, y_true_values, _ = data_buffer.get_occupied_mapped_data(y_pred, y_true)
#         else:
#             y_pred_values, y_true_values, _ = data_buffer.get_mapped_data(y_pred, y_true)

#         if y_pred_values.numel() == 0 or y_true_values.numel() == 0:
#             return torch.tensor(1.0)
#         y_score = y_pred_values[:, 3].detach().cpu().numpy()
#         y_true_binary = y_true_values[:, 3].detach().cpu().numpy()
#         if y_true_binary.sum() == 0:
#             return torch.tensor(1.0)
#         score = average_precision_score(y_true_binary, y_score)
#         return torch.tensor(score)


# class CoordinateError(PointcloudOccupancyMetric):
#     """Metric to evaluate spatial accuracy of point predictions.
    
#     This metric computes the mean Euclidean distance between matched points
#     in the prediction and ground truth point clouds. Lower values indicate
#     better spatial accuracy.
#     """
    
#     def _calc(self, y_pred, y_true, data_buffer=None, *args, **kwargs):
#         # Get mapped points using the buffer
#         if self.occupied_only:
#             pred_matched, true_matched, _ = data_buffer.get_occupied_mapped_data(y_pred, y_true)
#         else:
#             pred_matched, true_matched, _ = data_buffer.get_mapped_data(y_pred, y_true)
            
#         if pred_matched.numel() == 0 or true_matched.numel() == 0:
#             return torch.tensor(float('inf'), device=pred_matched.device)
            
#         # Compute Euclidean distances between matched points
#         sq_dists = torch.sum((pred_matched[:, :3] - true_matched[:, :3]) ** 2, dim=1)
#         return torch.sqrt(sq_dists.mean())


# class ProbabilityError(PointcloudOccupancyMetric):
#     """Metric to evaluate occupancy probability prediction accuracy.
    
#     This metric computes the mean absolute error between predicted and ground truth
#     occupancy probabilities for matched points. Lower values indicate better
#     probability prediction accuracy.
#     """
    
#     def _calc(self, y_pred, y_true, data_buffer=None, *args, **kwargs):
#         # Get mapped points using the buffer
#         if self.occupied_only:
#             pred_matched, true_matched, _ = data_buffer.get_occupied_mapped_data(y_pred, y_true)
#         else:
#             pred_matched, true_matched, _ = data_buffer.get_mapped_data(y_pred, y_true)
            
#         if pred_matched.numel() == 0 or true_matched.numel() == 0:
#             return torch.tensor(1.0, device=pred_matched.device)
            
#         # Compute absolute error between predicted and ground truth probabilities
#         abs_errors = torch.abs(pred_matched[:, 3] - true_matched[:, 3])
#         return abs_errors.mean()
