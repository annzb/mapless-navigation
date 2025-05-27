import os
import sys

import torch
import torch.nn as nn


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
    def __init__(self, spatial_weight=1.0, probability_weight=1.0, max_distance=10.0, unmatched_weight=1.0, 
                 unmatched_pred_weight=1.0, unmatched_true_weight=1.0, **kwargs):
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
        mapped_mask, mapped_mask_other = data_buffer.mapped_mask()
        pred_occ_mask, true_occ_mask = data_buffer.occupied_mask()
        
        unmatched_losses = []
        for b in range(self._batch_size):
            pred_mask = (y_pred_batch_indices == b) & ~mapped_mask
            true_mask = (y_true_batch_indices == b) & ~mapped_mask_other
            
            if data_buffer._match_occupied_only:
                pred_mask &= pred_occ_mask
                true_mask &= true_occ_mask
            
            n_unmatched_pred = pred_mask.sum().float()
            n_unmatched_true = true_mask.sum().float()
            
            total_points = ((y_pred_batch_indices == b).sum() + (y_true_batch_indices == b).sum()).float() + 1e-6
            unmatched_ratio = (n_unmatched_pred + n_unmatched_true) / total_points
            unmatched_loss = unmatched_ratio * self.max_distance

            if pred_mask.any():
                gradient_anchor = y_pred_values[pred_mask, 3].mean()
            elif true_mask.any():
                gradient_anchor = y_true_values[true_mask, 3].mean()
            else:
                gradient_anchor = y_pred_values[:, 3].mean()  # fallback
            
            unmatched_losses.append(unmatched_loss + gradient_anchor * 0.0)  # maintains gradient flow

        return torch.stack(unmatched_losses).mean()
        
    def _calc(self, y_pred, y_true, data_buffer=None, *args, **kwargs):
        spatial_loss, probability_loss = self._calc_matched_loss(y_pred, y_true, data_buffer)
        unmatched_loss = self._calc_unmatched_loss(y_pred, y_true, data_buffer)
        total_loss = self.spatial_weight * spatial_loss + self.probability_weight * probability_loss + self.unmatched_weight * unmatched_loss
        return total_loss
