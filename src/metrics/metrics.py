import numpy as np
import torch
import torch.nn.functional as F
from sklearn.cluster import DBSCAN

from metrics.base import PointcloudOccupancyMetric
from metrics.loss_points import DistanceLoss, DistanceOccupancyLoss, PointLoss2
from utils import data_transforms


# class MatchedPointRatio(PointcloudOccupancyMetric):
#     def _calc(self, y_pred, y_true, data_buffer=None, *args, **kwargs):
#         pred_points = data_transforms.collapse_close_points(y_pred[0], d=self.same_point_distance_limit)
#         if self.occupied_only:
#             _, true_occ_mask = data_buffer.occupied_mask()
#             true_points = y_true[0][true_occ_mask]
#         else:
#             true_points = y_true[0]
#         if len(pred_points) == 0 or len(true_points) == 0:
#             return 0.0 

#         dists = torch.cdist(true_points[:, :3], pred_points[:, :3])
#         candidate_mask = dists <= self.max_distance
#         candidate_indices = candidate_mask.nonzero(as_tuple=False)
#         if candidate_indices.numel() == 0:
#             return 0.0

#         distances = dists[candidate_mask]
#         pairs_with_dist = torch.cat([candidate_indices, distances.unsqueeze(1)], dim=1)
#         sorted_pairs = pairs_with_dist[torch.argsort(pairs_with_dist[:, 2])]
#         matched_true = set()
#         matched_pred = set()
#         num_matched = 0

#         for i in range(sorted_pairs.size(0)):
#             t_idx, p_idx = int(sorted_pairs[i, 0]), int(sorted_pairs[i, 1])
#             if t_idx not in matched_true and p_idx not in matched_pred:
#                 matched_true.add(t_idx)
#                 matched_pred.add(p_idx)
#                 num_matched += 1
#         total_points = len(true_points) + len(pred_points)
#         if total_points == 0:
#             return 1.0
        
#         return num_matched * 2 / total_points


class DbscanMetric(PointcloudOccupancyMetric):
    def tp_fp_fn(self, cluster_counts):
        tp_list, fp_list, fn_list = [], [], []
        for counts in cluster_counts:
            pred_mask, true_mask = counts[:, 0] > 0, counts[:, 1] > 0
            tp_list.append((pred_mask & true_mask).sum())
            fp_list.append((pred_mask & ~true_mask).sum())
            fn_list.append((~pred_mask & true_mask).sum())
        tp = torch.stack(tp_list).to(self._device)
        fp = torch.stack(fp_list).to(self._device)
        fn = torch.stack(fn_list).to(self._device)
        return tp, fp, fn
    
class DbscanRecall(DbscanMetric):
    def _calc(self, y_pred, y_true, data_buffer=None, *args, **kwargs):
        batch_tp, batch_fp, batch_fn = self.tp_fp_fn(data_buffer.cluster_counts())
        return (batch_tp / (batch_tp + batch_fn  + 1e-8)).mean()
    
class DbscanPrecision(DbscanMetric):
    def _calc(self, y_pred, y_true, data_buffer=None, *args, **kwargs):
        batch_tp, batch_fp, batch_fn = self.tp_fp_fn(data_buffer.cluster_counts())
        return (batch_tp / (batch_tp + batch_fp  + 1e-8)).mean()
    
class DbscanF1(DbscanMetric):
    def _calc(self, y_pred, y_true, data_buffer=None, *args, **kwargs):
        batch_tp, batch_fp, batch_fn = self.tp_fp_fn(data_buffer.cluster_counts())
        precision = (batch_tp / (batch_tp + batch_fp)).mean()
        recall = (batch_tp / (batch_tp + batch_fn)).mean()
        return 2 * precision * recall / (precision + recall + 1e-8)
    
class DbscanPurity(DbscanMetric):
    def _calc(self, y_pred, y_true, data_buffer=None, *args, **kwargs):
        purity_list = []
        for counts in data_buffer.cluster_counts():
            if counts.numel() == 0:
                continue
            pred = counts[:, 0].float()
            true = counts[:, 1].float()
            max_val = torch.maximum(pred, true)
            min_val = torch.minimum(pred, true)
            mask = max_val > 0
            purity = torch.zeros_like(max_val)
            purity[mask] = min_val[mask] / max_val[mask]
            purity_list.append(purity.mean())
        if not purity_list:
            return torch.tensor(1.0, device=self._device)
        return torch.stack(purity_list).mean()

class DbscanReduction(DbscanMetric):
    def _calc(self, y_pred, y_true, data_buffer=None, *args, **kwargs):
        true_cloud, true_batch_indices = y_true
        if self.occupied_only:
            _, occ_mask = data_buffer.occupied_mask()
            true_cloud = true_cloud[occ_mask]
            true_batch_indices = true_batch_indices[occ_mask]

        reductions = []
        for b, counts in enumerate(data_buffer.cluster_counts()):
            num_clusters = counts.shape[0]
            batch_mask = true_batch_indices == b
            num_true = batch_mask.sum().item()

            if num_true == 0:
                reductions.append(torch.tensor(1.0, device=self._device))
            else:
                diff = abs(num_clusters - num_true)
                reduction = 1 - diff / num_true
                reductions.append(torch.tensor(reduction, device=self._device))
        return torch.stack(reductions).mean()

# class DbscanRecall(PointcloudOccupancyMetric):
#     def _calc(self, y_pred, y_true, data_buffer=None, *args, **kwargs):
#         pred_cloud, true_cloud = y_pred[0], y_true[0]
#         if pred_cloud.numel() == 0 and true_cloud.numel() == 0:
#             return {'recall': 1.0, 'reduction': 1.0, 'purity': 1.0}

#         pred_np = pred_cloud.detach().cpu().numpy()
#         true_np = true_cloud.detach().cpu().numpy()
#         N_pred, N_true = len(pred_np), len(true_np)

#         merged_coords = np.vstack([pred_np[:, :3], true_np[:, :3]])
#         merged_labels = np.array([0]*N_pred + [1]*N_true)  # 0=pred, 1=true

#         clustering = DBSCAN(eps=self.same_point_distance_limit, min_samples=1).fit(merged_coords)
#         cluster_ids = clustering.labels_

#         num_clusters = cluster_ids.max() + 1 if cluster_ids.max() >= 0 else 0

#         match_count = 0
#         purity_sum = 0
#         for cluster_id in range(num_clusters):
#             mask = cluster_ids == cluster_id
#             cluster_labels = merged_labels[mask]
#             n_pred = np.sum(cluster_labels == 0)
#             n_true = np.sum(cluster_labels == 1)
#             if n_pred > 0 and n_true > 0:
#                 match_count += 1
#             if n_pred + n_true > 0:
#                 purity = min(n_pred, n_true) / max(n_pred, n_true)
#                 purity_sum += purity

#         # Strategy 1: Match-based recall
#         recall = match_count / max(N_true, 1)

#         # Strategy 2: Cluster reduction ratio
#         reduction = 1.0 - abs(num_clusters - N_true) / max(N_true, 1)

#         # Strategy 3: Average cluster purity
#         avg_purity = purity_sum / max(num_clusters, 1)

#         return {
#             'recall': recall,
#             'reduction': reduction,
#             'purity': avg_purity
#         }
    

# class DbscanPurity(PointcloudOccupancyMetric):
#     def _calc(self, y_pred, y_true, data_buffer=None, *args, **kwargs):
#         pred_cloud, true_cloud = y_pred[0], y_true[0]
#         if pred_cloud.numel() == 0 and true_cloud.numel() == 0:
#             return {'recall': 1.0, 'reduction': 1.0, 'purity': 1.0}

#         pred_np = pred_cloud.detach().cpu().numpy()
#         true_np = true_cloud.detach().cpu().numpy()
#         N_pred, N_true = len(pred_np), len(true_np)

#         merged_coords = np.vstack([pred_np[:, :3], true_np[:, :3]])
#         merged_labels = np.array([0]*N_pred + [1]*N_true)  # 0=pred, 1=true

#         clustering = DBSCAN(eps=self.same_point_distance_limit, min_samples=1).fit(merged_coords)
#         cluster_ids = clustering.labels_

#         num_clusters = cluster_ids.max() + 1 if cluster_ids.max() >= 0 else 0

#         match_count = 0
#         purity_sum = 0
#         for cluster_id in range(num_clusters):
#             mask = cluster_ids == cluster_id
#             cluster_labels = merged_labels[mask]
#             n_pred = np.sum(cluster_labels == 0)
#             n_true = np.sum(cluster_labels == 1)
#             if n_pred > 0 and n_true > 0:
#                 match_count += 1
#             if n_pred + n_true > 0:
#                 purity = min(n_pred, n_true) / max(n_pred, n_true)
#                 purity_sum += purity

#         # Strategy 1: Match-based recall
#         recall = match_count / max(N_true, 1)

#         # Strategy 2: Cluster reduction ratio
#         reduction = 1.0 - abs(num_clusters - N_true) / max(N_true, 1)

#         # Strategy 3: Average cluster purity
#         avg_purity = purity_sum / max(num_clusters, 1)

#         return {
#             'recall': recall,
#             'reduction': reduction,
#             'purity': avg_purity
#         }
    

# class DbscanMatch(PointcloudOccupancyMetric):
#     def _calc(self, y_pred, y_true, data_buffer=None, *args, **kwargs):
#         pred_coud, true_cloud = y_pred[1], y_true[1]
#         merged_cloud = np.vstack([pred_coud.detach().cpu().numpy(), true_cloud.detach().cpu().numpy()])
#         coords, probs = merged_cloud[:, :3], merged_cloud[:, 3]
#         clustering = DBSCAN(eps=self.same_point_distance_limit, min_samples=1).fit(coords)
#         labels = clustering.labels_

#         reduced = []
#         for label in np.unique(labels):
#             mask = labels == label
#             cluster_coords = coords[mask]
#             cluster_probs = probs[mask]
#             center = cluster_coords.mean(axis=0)
#             total_prob = cluster_probs.sum()
#             reduced.append(np.append(center, total_prob))
#         collapsed_np = np.vstack(reduced)
#         collapsed_np[:, 3] = np.clip(collapsed_np[:, 3], 0.0, 1.0)
#         collapsed_cloud = torch.tensor(collapsed_np, dtype=pred_coud.dtype, device=pred_coud.device)
            


class DistanceLossFpFnMetric(DistanceOccupancyLoss, PointcloudOccupancyMetric):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.negative = True
        self._subloss_type = 1  # fn + fp

    def _calc(self, y_pred, y_true, data_buffer=None, *args, **kwargs):
        losses, loss_types = super()._calc(y_pred, y_true, data_buffer, verbose_return=True)
        if self._subloss_type in loss_types:
            return losses[loss_types == self._subloss_type].mean()
        return None
    
class DistanceLossFnMetric(DistanceLossFpFnMetric):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._subloss_type = 2  # fn

class DistanceLossFpMetric(DistanceLossFpFnMetric):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._subloss_type = 3  # fp

class DistanceLossOccupancyMetric(DistanceLossFpFnMetric):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._subloss_type = 4


class UnmatchedLossFpFnMetric(PointcloudOccupancyMetric, PointLoss2):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._subloss_type = 1  # fn + fp

    def _calc(self, y_pred, y_true, data_buffer=None, *args, **kwargs):
        unmatched_loss, subloss_type = self._calc_unmatched_loss(y_pred, y_true, data_buffer, *args, **kwargs)
        if self._subloss_type in subloss_type:
            return unmatched_loss[subloss_type == self._subloss_type].mean()
        return None
    
class UnmatchedLossFnMetric(UnmatchedLossFpFnMetric):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._subloss_type = 2  # fn

class UnmatchedLossFpMetric(UnmatchedLossFpFnMetric):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._subloss_type = 3  # fp


class UnmatchedLossMetric(PointcloudOccupancyMetric, PointLoss2):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.negative = True
        self.score_multiplier = self._unmatched_weight

    def _calc(self, y_pred, y_true, data_buffer=None, *args, **kwargs):
        unmatched_loss, _ = self._calc_unmatched_loss(y_pred, y_true, data_buffer, *args, **kwargs)
        return unmatched_loss.mean()

class SpatialLossMetric(UnmatchedLossMetric):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.score_multiplier = self._spatial_weight

    def _calc(self, y_pred, y_true, data_buffer=None, *args, **kwargs):
        spatial_loss, _ = self._calc_matched_loss(y_pred, y_true, data_buffer, *args, **kwargs)
        return spatial_loss

class OccupancyLossMetric(UnmatchedLossMetric):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.score_multiplier = self._occupancy_weight

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
