import torch
import torch.nn.functional as F
# from pytorch3d.loss import chamfer_distance
from sklearn.metrics import average_precision_score, roc_auc_score

from metrics.base import PointcloudOccupancyMetric


class OccupancyRatio(PointcloudOccupancyMetric):
    def _calc(self, y_pred, y_true, data_buffer=None, *args, **kwargs):
        pred_occupied_mask, true_occupied_mask = data_buffer.occupied_mask()
        pred_ratio = pred_occupied_mask.float().mean()
        true_ratio = true_occupied_mask.float().mean()
        score = 1.0 - torch.abs(pred_ratio - true_ratio)
        return score
    

class MatchedPointRatio(PointcloudOccupancyMetric):
    def _calc(self, y_pred, y_true, data_buffer=None, *args, **kwargs):
        (y_pred_values, y_pred_batch_indices), (y_true_values, y_true_batch_indices) = y_pred, y_true
        mapped_mask, mapped_mask_other = data_buffer.mapped_mask()
        pred_occ_mask, true_occ_mask = data_buffer.occupied_mask()
        
        matched_ratios = []
        
        for b in range(self._batch_size):
            pred_mask = (y_pred_batch_indices == b) & ~mapped_mask
            true_mask = (y_true_batch_indices == b) & ~mapped_mask_other
            
            if data_buffer._match_occupied_only:
                pred_mask &= pred_occ_mask
                true_mask &= true_occ_mask
            
            n_pred, n_true = (y_pred_batch_indices == b).sum(), (y_true_batch_indices == b).sum()
            total_points = (n_pred + n_true).float() + 1e-6
            n_unmatched_pred = pred_mask.sum().float()
            n_unmatched_true = true_mask.sum().float()
            unmatched_ratio = (n_unmatched_pred + n_unmatched_true) / total_points
            matched_ratios.append(1.0 - unmatched_ratio)

        return torch.stack(matched_ratios).mean()
    

class NegativeUnmatchedLoss(PointcloudOccupancyMetric):
    def _calc(self, y_pred, y_true, data_buffer=None, *args, **kwargs):
        (y_pred_values, y_pred_batch_indices), (y_true_values, y_true_batch_indices) = y_pred, y_true
        mapped_mask, mapped_mask_other = data_buffer.mapped_mask()
        pred_occ_mask, true_occ_mask = data_buffer.occupied_mask()
        
        unmatched_losses = []
        
        for b in range(self._batch_size):
            pred_mask = (y_pred_batch_indices == b) & ~mapped_mask
            true_mask = (y_true_batch_indices == b) & ~mapped_mask_other
            
            if data_buffer._match_occupied_only:
                pred_mask &= pred_occ_mask
                true_mask &= true_occ_mask
            
            n_pred, n_true = (y_pred_batch_indices == b).sum(), (y_true_batch_indices == b).sum()
            total_points = (n_pred + n_true).float() + 1e-6
            n_unmatched_pred = pred_mask.sum().float()
            n_unmatched_true = true_mask.sum().float()
            unmatched_ratio = (n_unmatched_pred + n_unmatched_true) / total_points
            unmatched_loss = unmatched_ratio * self.max_distance

            if pred_mask.any():
                gradient_anchor = y_pred_values[pred_mask, 3].mean()
            elif true_mask.any():
                gradient_anchor = y_true_values[true_mask, 3].mean()
            else:
                gradient_anchor = y_pred_values[:, 3].mean()  # fallback
            
            unmatched_losses.append(unmatched_loss + gradient_anchor * 0.0)  # maintains gradient flow

        return -torch.stack(unmatched_losses).mean()
    

class NegativeSpatialLoss(PointcloudOccupancyMetric):
    def _calc(self, y_pred, y_true, data_buffer=None, *args, **kwargs):
        (y_pred_values, y_pred_batch_indices), (y_true_values, y_true_batch_indices) = y_pred, y_true
        
        mapping = data_buffer.mapping()
        if mapping.numel() == 0:
            spatial_loss = y_pred_values[:, :3].mean() * 0.0
            return -spatial_loss
            
        pred_matched = y_pred_values[mapping[:, 0]]
        true_matched = y_true_values[mapping[:, 1]]
        batch_indices = y_pred_batch_indices[mapping[:, 0]]
        
        spatial_losses = []
        for b in range(self._batch_size):
            batch_mask = batch_indices == b
            pred_b = pred_matched[batch_mask]
            true_b = true_matched[batch_mask]
            
            if pred_b.size(0) == 0 or true_b.size(0) == 0:
                spatial_loss = y_pred_values[:, :3].mean() * 0.0
                spatial_losses.append(spatial_loss)
                continue
                
            spatial_loss = torch.norm(pred_b[:, :3] - true_b[:, :3], dim=1).mean()
            spatial_losses.append(spatial_loss)
            
        return -torch.stack(spatial_losses).mean()


class NegativeOccupancyLoss(PointcloudOccupancyMetric):
    def _calc(self, y_pred, y_true, data_buffer=None, *args, **kwargs):
        (y_pred_values, y_pred_batch_indices), (y_true_values, y_true_batch_indices) = y_pred, y_true
        
        mapping = data_buffer.mapping()
        if mapping.numel() == 0:
            prob_loss = y_pred_values[:, 3].mean() * 0.0
            return -prob_loss
            
        pred_matched = y_pred_values[mapping[:, 0]]
        true_matched = y_true_values[mapping[:, 1]]
        batch_indices = y_pred_batch_indices[mapping[:, 0]]
        
        prob_losses = []
        for b in range(self._batch_size):
            batch_mask = batch_indices == b
            pred_b = pred_matched[batch_mask]
            true_b = true_matched[batch_mask]
            
            if pred_b.size(0) == 0 or true_b.size(0) == 0:
                prob_loss = y_pred_values[:, 3].mean() * 0.0
                prob_losses.append(prob_loss)
                continue
                
            prob_loss = torch.norm(pred_b[:, 3:4] - true_b[:, 3:4], dim=1).mean()
            prob_losses.append(prob_loss)
            
        return -torch.stack(prob_losses).mean()

    

# class ChamferMetric(PointcloudOccupancyMetric):
#     def _calc(self, y_pred, y_true, data_buffer=None, *args, **kwargs):
#         pred_coords, pred_batch_idx = y_pred
#         true_coords, true_batch_idx = y_true
#         cd_total = 0.0
#         for b in range(self._batch_size):
#             pred_b = pred_coords[pred_batch_idx == b, :3].unsqueeze(0)  # [1, N_pred, 3]
#             true_b = true_coords[true_batch_idx == b, :3].unsqueeze(0)  # [1, N_true, 3]
#             cd, _ = chamfer_distance(pred_b, true_b)
#             cd_total += cd
#         return cd_total / self._batch_size


class OccupancyPrecisionRecall(PointcloudOccupancyMetric):
    def _calc(self, y_pred, y_true, data_buffer=None, *args, **kwargs):
        mapping = data_buffer.mapping()
        pred_matched_mask = torch.zeros(len(y_pred[0]), dtype=torch.bool, device=y_pred[0].device)
        pred_matched_mask[mapping[:, 0]] = True

        precision = pred_matched_mask.float().mean()
        recall = mapping.size(0) / (len(y_true[0]) + 1e-8)

        return {'precision': precision, 'recall': recall}
    

class UnmatchedPointRatio(PointcloudOccupancyMetric):
    def _calc(self, y_pred, y_true, data_buffer=None, *args, **kwargs):
        pred_mask, true_mask = data_buffer.mapped_mask()
        unmatched_pred_ratio = (~pred_mask).float().mean()
        unmatched_true_ratio = (~true_mask).float().mean()
        return (unmatched_pred_ratio + unmatched_true_ratio) / 2


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
