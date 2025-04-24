import os
import sys

import torch
import torch.nn as nn


sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.metrics.base import PointcloudOccupancyLoss
from src.metrics.data_buffer import ChamferPointDataBuffer


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


class ChamferPointLoss(PointcloudOccupancyLoss):
    def __init__(self, spatial_weight=1.0, probability_weight=1.0, max_distance=10.0, occupied_only=False, unmatched_weight=1.0, **kwargs):
        super().__init__(**kwargs)
        self.spatial_weight = spatial_weight
        self.probability_weight = probability_weight
        self.max_distance = max_distance
        self.occupied_only = occupied_only
        self.unmatched_weight = unmatched_weight
        # self.bce_loss = nn.BCELoss(reduction='none')
        
    def _calc_matched_loss(self, y_pred, y_true, data_buffer):
        (y_pred_values, y_pred_batch_indices), (y_true_values, y_true_batch_indices) = y_pred, y_true
        
        mapping = data_buffer.mapping()
        if mapping.numel() == 0:
            return y_pred_values[:, 0].mean() * 0.0 + self.max_distance, y_pred_values[:, 3].mean() * 0.0 + 1.0
            
        pred_matched = y_pred_values[mapping[:, 0]]
        true_matched = y_true_values[mapping[:, 1]]
        batch_indices = y_pred_batch_indices[mapping[:, 0]]
        
        spatial_losses = []
        prob_losses = []
        for b in range(self._batch_size):
            batch_mask = batch_indices == b
            pred_b = pred_matched[batch_mask]
            true_b = true_matched[batch_mask]
            
            if pred_b.size(0) == 0 or true_b.size(0) == 0:
                spatial_loss = pred_b[:, 0].mean() * 0.0 + self.max_distance
                prob_loss = pred_b[:, 3].mean() * 0.0 + 1.0 if pred_b.size(0) > 0 else true_b[:, 3].mean() * 0.0 + 1.0
                spatial_losses.append(spatial_loss)
                prob_losses.append(prob_loss)
                continue
                
            distances = torch.norm(pred_b[:, :3] - true_b[:, :3], dim=1)
            spatial_loss = distances.mean()
            spatial_losses.append(spatial_loss)
            
            prob_distances = torch.norm(pred_b[:, 3:4] - true_b[:, 3:4], dim=1)
            prob_loss = prob_distances.mean()
            prob_losses.append(prob_loss)
            
        return torch.stack(spatial_losses).mean(), torch.stack(prob_losses).mean()
        
    def _calc_unmatched_loss(self, y_pred, y_true, data_buffer):
        (y_pred_values, y_pred_batch_indices), (y_true_values, y_true_batch_indices) = y_pred, y_true
        mapped_mask, mapped_mask_other = data_buffer.mapped_mask()
        unmatched_mask = ~mapped_mask_other
        
        if self.occupied_only:
            occupied_mask = y_true_values[:, 3] >= self._occupancy_threshold
            unmatched_mask = unmatched_mask & occupied_mask
            
        losses = []
        for b in range(self._batch_size):
            batch_mask = y_true_batch_indices == b
            if not batch_mask.any():
                losses.append(y_pred_values[:, 0].mean() * 0.0)
                continue
                
            batch_unmatched_mask = batch_mask & unmatched_mask
            if not batch_unmatched_mask.any():
                losses.append(y_pred_values[:, 0].mean() * 0.0)
                continue
                
            n_unmatched = batch_unmatched_mask.sum().item()
            batch_loss = y_pred_values[:, 0].mean() * 0.0 + self.max_distance * n_unmatched
            losses.append(batch_loss)
            
        return torch.stack(losses).mean()
        
    def _calc(self, y_pred, y_true, data_buffer=None, *args, **kwargs):
        spatial_loss, probability_loss = self._calc_matched_loss(y_pred, y_true, data_buffer)
        unmatched_loss = self._calc_unmatched_loss(y_pred, y_true, data_buffer)
        total_loss = self.spatial_weight * spatial_loss + self.probability_weight * probability_loss + self.unmatched_weight * unmatched_loss
        return total_loss
