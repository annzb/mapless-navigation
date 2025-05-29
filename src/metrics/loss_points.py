import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F


sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from metrics.base import PointcloudOccupancyLoss
from metrics.data_buffer import ChamferPointDataBuffer


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
    def __init__(self, spatial_weight=1.0, probability_weight=1.0, max_distance=10.0, unmatched_weight=1.0, unmatched_pred_weight=1.0, unmatched_true_weight=1.0, **kwargs):
        super().__init__(**kwargs)
        self.spatial_weight = spatial_weight
        self.probability_weight = probability_weight
        self.max_distance = max_distance
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
