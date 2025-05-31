import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F


sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from metrics.base import PointcloudOccupancyLoss


class MsePointLoss(PointcloudOccupancyLoss):
    def _calc(self, y_pred, y_true, data_buffer=None, *args, **kwargs):
        y_pred_values, y_pred_batch_indices = y_pred
        y_true_values, y_true_batch_indices = y_true

        losses = []
        for b in range(self._batch_size):
            pred_batch_mask = (y_pred_batch_indices == b)
            true_batch_mask = (y_true_batch_indices == b)
            pred_probs = y_pred_values[:, 3]
            true_probs = y_true_values[:, 3]
            pred_avg = (pred_probs * pred_batch_mask.float()).sum() / (pred_batch_mask.sum() + 1e-8)
            true_avg = (true_probs * true_batch_mask.float()).sum() / (true_batch_mask.sum() + 1e-8)
            loss = (pred_avg - true_avg) ** 2
            losses.append(loss)

        return torch.stack(losses).mean()


class PointLoss(PointcloudOccupancyLoss):
    def __init__(self, spatial_weight=1.0, probability_weight=1.0, unmatched_weight=1.0, unmatched_pred_weight=1.0, unmatched_true_weight=1.0, **kwargs):
        super().__init__(**kwargs)
        self.spatial_weight = spatial_weight
        self.probability_weight = probability_weight
        self.unmatched_weight = unmatched_weight
        self.unmatched_pred_weight = unmatched_pred_weight
        self.unmatched_true_weight = unmatched_true_weight

    def _calc_matched_loss(self, y_pred, y_true, data_buffer):
        (y_pred_values, y_pred_batch_indices), (y_true_values, y_true_batch_indices) = y_pred, y_true
        
        mapping = data_buffer.mapping()
        if mapping.numel() == 0:
            spatial_loss = y_pred_values[:, :3].mean() * 0.0
            prob_loss = y_pred_values[:, 3].mean() * 0.0
            return spatial_loss, prob_loss
            
        pred_matched = y_pred_values[mapping[:, 0]]
        true_matched = y_true_values[mapping[:, 1]]
        batch_indices = y_pred_batch_indices[mapping[:, 0]]
        
        spatial_losses, prob_losses = [], []
        for b in range(self._batch_size):
            batch_mask = batch_indices == b
            pred_b = pred_matched[batch_mask]
            true_b = true_matched[batch_mask]
            
            if pred_b.size(0) == 0 or true_b.size(0) == 0:
                spatial_loss = y_pred_values[:, :3].mean() * 0.0
                prob_loss = y_pred_values[:, 3].mean() * 0.0
                spatial_losses.append(spatial_loss)
                prob_losses.append(prob_loss)
                continue
                
            spatial_loss = torch.norm(pred_b[:, :3] - true_b[:, :3], dim=1).mean()
            prob_loss = torch.norm(pred_b[:, 3:4] - true_b[:, 3:4], dim=1).mean()

            spatial_losses.append(spatial_loss)
            prob_losses.append(prob_loss)
            
        return torch.stack(spatial_losses).mean(), torch.stack(prob_losses).mean()

    def _calc_unmatched_loss(self, y_pred, y_true, data_buffer):
        (y_pred_values, y_pred_batch_indices), (y_true_values, y_true_batch_indices) = y_pred, y_true
        
        target_masks = self._calc_unmatched_masks(y_pred_batch_indices, y_true_batch_indices, data_buffer)
        unmatched_ratios = self._calc_matching_ratios(y_pred_batch_indices, y_true_batch_indices, data_buffer, target_masks)
        unmatched_losses = []
        
        for b in range(self._batch_size):
            unmatched_loss = unmatched_ratios[b] * self.max_distance
            gradient_anchor = y_pred_values[:, 3].mean()
            unmatched_losses.append(unmatched_loss + gradient_anchor * 0.0)  # maintains gradient flow

        return torch.stack(unmatched_losses).mean()
        
    def _calc(self, y_pred, y_true, data_buffer=None, *args, **kwargs):
        spatial_loss, probability_loss = self._calc_matched_loss(y_pred, y_true, data_buffer)
        unmatched_loss = self._calc_unmatched_loss(y_pred, y_true, data_buffer)
        total_loss = self.spatial_weight * spatial_loss + self.probability_weight * probability_loss + self.unmatched_weight * unmatched_loss
        return total_loss


class PointLoss2(PointLoss):
    def __init__(self, fn_fp_weight=1.0, fn_weight=1.0, fp_weight=1.0, **kwargs):
        super().__init__(**kwargs)
        self.fn_fp_weight = fn_fp_weight
        self.fn_weight = fn_weight
        self.fp_weight = fp_weight
    
    def _calc_unmatched_loss(self, y_pred, y_true, data_buffer):
        (y_pred_values, y_pred_batch_indices), (y_true_values, y_true_batch_indices) = y_pred, y_true
        target_masks = self._calc_unmatched_masks(y_pred_batch_indices, y_true_batch_indices, data_buffer)
        unmatched_ratios = self._calc_matching_ratios_soft(y_pred, y_true, data_buffer, target_masks)
        if self.occupied_only:
            pred_occ_mask, true_occ_mask = data_buffer.occupied_mask()
        else:
            pred_occ_mask, true_occ_mask = torch.ones_like(y_pred_batch_indices, dtype=torch.bool, device=self.device), torch.ones_like(y_true_batch_indices, dtype=torch.bool, device=self.device)
        unmatched_losses = []

        for b in range(self._batch_size):
            pred_mask, true_mask = target_masks[b]
            pred_unmatched, true_unmatched = y_pred_values[pred_mask], y_true_values[true_mask]
            pred_all, true_all = y_pred_values[(y_pred_batch_indices == b) & pred_occ_mask], y_true_values[(y_true_batch_indices == b) & true_occ_mask]

            if pred_unmatched.size(0) > 0 and true_unmatched.size(0) > 0:  # FPs and FNs
                dist_loss = torch.cdist(pred_unmatched[:, :3], true_unmatched[:, :3]).mean() * self.fn_fp_weight

            elif true_unmatched.size(0) > 0 and pred_all.size(0) > 0:  # FNs only
                dist_loss = torch.cdist(pred_all[:, :3], true_unmatched[:, :3]).mean() * self.fn_weight

            elif pred_unmatched.size(0) > 0 and true_all.size(0) > 0:  # FPs only
                dist_loss = torch.cdist(pred_unmatched[:, :3], true_all[:, :3]).mean() * self.fp_weight
            
            else: # no unmatched points
                dist_loss = y_pred_values[:, :3].mean() * 0.0

            unmatched_loss = unmatched_ratios[b] * dist_loss
            unmatched_losses.append(unmatched_loss)

        return torch.stack(unmatched_losses).mean()
    
    # def _calc_matched_loss(self, y_pred, y_true, data_buffer):
    #     (y_pred_values, y_pred_batch_indices), (y_true_values, y_true_batch_indices) = y_pred, y_true
    #     mapping = data_buffer.mapping()

    #     batch_spatial_losses, batch_prob_losses = [], []

    #     for b in range(self._batch_size):
    #         pred_b = y_pred_values[y_pred_batch_indices == b]
    #         true_b = y_true_values[y_true_batch_indices == b]
    #         total_count = pred_b.size(0) + true_b.size(0)

    #         if total_count == 0:
    #             # Nothing to do; skip
    #             continue

    #         if mapping.numel() > 0:
    #             # Select only matched pairs for this batch
    #             batch_mask = (y_pred_batch_indices[mapping[:, 0]] == b)
    #             matched_pred = y_pred_values[mapping[batch_mask, 0]]
    #             matched_true = y_true_values[mapping[batch_mask, 1]]
    #         else:
    #             matched_pred = matched_true = None

    #         if matched_pred is None or matched_pred.size(0) == 0:
    #             # No matches in this batch → penalize with a default large loss using actual predictions
    #             # This ensures a gradient path and trains the model to increase matches
    #             spatial_loss = pred_b[:, :3].pow(2).sum(dim=1).mean()
    #             prob_loss = pred_b[:, 3].pow(2).mean()
    #         else:
    #             # Compute mean error across matched pairs, but scale by total number of points
    #             spatial_error = torch.norm(matched_pred[:, :3] - matched_true[:, :3], dim=1).sum()
    #             prob_error = torch.norm(matched_pred[:, 3:4] - matched_true[:, 3:4], dim=1).sum()

    #             spatial_loss = spatial_error / (total_count + 1e-6)
    #             prob_loss = prob_error / (total_count + 1e-6)

    #         batch_spatial_losses.append(spatial_loss)
    #         batch_prob_losses.append(prob_loss)

    #     return torch.stack(batch_spatial_losses).mean(), torch.stack(batch_prob_losses).mean()
        

class RegressionPointLoss(PointcloudOccupancyLoss):
    def __init__(self, alpha=0.25, gamma=2.0, **kwargs):
        super().__init__(**kwargs)
        self.alpha = alpha
        self.gamma = gamma
        self._init_points()

    def _init_points(self):
        el_bins = torch.tensor(self.radar_config.clipped_elevation_bins, dtype=torch.float16)
        az_bins = torch.tensor(self.radar_config.clipped_azimuth_bins, dtype=torch.float16)
        r_bins = torch.linspace(
            0,
            (self.radar_config.num_range_bins - 1) * self.radar_config.range_bin_width,
            self.radar_config.num_range_bins,
            dtype=torch.float16
        )
        el_grid, az_grid, r_grid = torch.meshgrid(el_bins, az_bins, r_bins, indexing="ij")
        x = r_grid * torch.cos(el_grid) * torch.sin(az_grid)
        y = r_grid * torch.cos(el_grid) * torch.cos(az_grid)
        z = r_grid * torch.sin(el_grid)

        self.support_coords = torch.stack((x, y, z), dim=-1).reshape(-1, 3)

    def _points_to_grid(self, points: torch.Tensor, batch_idx: torch.Tensor) -> torch.Tensor:
        support_coords = self.support_coords.to(self.device)
        points_xyz = points[:, :3]
        points_p = points[:, 3]

        support_voxels = torch.round(support_coords / self.grid_resolution).to(dtype=torch.int32)
        point_voxels = torch.round(points_xyz / self.grid_resolution).to(dtype=torch.int32)
        coeffs = torch.tensor([1_000_000, 1_000, 1], device=self.device, dtype=torch.int32)
        flat_support = (support_voxels * coeffs).sum(dim=1)
        flat_points = (point_voxels * coeffs).sum(dim=1)
        support_index_map = dict(zip(flat_support.tolist(), range(support_coords.shape[0])))
        output = torch.zeros(support_coords.shape[0], dtype=torch.float16, device=self.device)

        for i in range(points.shape[0]):
            key = flat_points[i].item()
            if key in support_index_map:
                output[support_index_map[key]] = points_p[i].half()

        return output
    
    def _calc(self, y_pred, y_true, data_buffer=None, *args, **kwargs):
        y_true = self._points_to_grid(*y_true)
        bce = F.binary_cross_entropy_with_logits(y_pred[0], y_true, reduction='none')
        focal = self.alpha * (1 - torch.exp(-bce)) ** self.gamma * bce
        return focal.mean()
