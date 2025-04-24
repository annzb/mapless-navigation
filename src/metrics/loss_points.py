import torch
import torch.nn as nn

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


class ChamferPointLoss(PointcloudOccupancyLoss):
    """Loss function that combines Chamfer distance for coordinates with BCE for probabilities.
    
    This loss function:
    1. Uses Chamfer distance to match points spatially
    2. Uses BCE loss to match occupancy probabilities
    3. Properly handles the mappings from ChamferPointDataBuffer
    4. Maintains gradient flow for both coordinate and probability learning
    """
    
    def __init__(self, spatial_weight=1.0, probability_weight=1.0, max_distance=10.0, occupied_only=False, **kwargs):
        """Initialize the loss function.
        
        Args:
            spatial_weight: Weight for the spatial (Chamfer) loss term
            probability_weight: Weight for the probability (BCE) loss term
            max_distance: Maximum possible distance in the space (used for unmatched batches)
            occupied_only: Whether to only match occupied points
            **kwargs: Additional arguments passed to parent class
        """
        super().__init__(**kwargs)
        self.spatial_weight = spatial_weight
        self.probability_weight = probability_weight
        self.max_distance = max_distance
        self.occupied_only = occupied_only
        self.bce_loss = nn.BCELoss(reduction='none')
        
    def _calc_spatial_loss(self, y_pred, y_true, data_buffer):
        """Calculate the spatial loss using Chamfer distance.
        
        Args:
            y_pred: Tuple of (predicted points, batch indices)
            y_true: Tuple of (true points, batch indices)
            data_buffer: ChamferPointDataBuffer instance
            
        Returns:
            Mean spatial loss across batches
        """
        (y_pred_values, y_pred_batch_indices), (y_true_values, y_true_batch_indices) = y_pred, y_true
        
        print("\nDEBUG: Starting spatial loss calculation")
        print(f"DEBUG: y_pred shape: {y_pred_values.shape}, y_true shape: {y_true_values.shape}")
        print(f"DEBUG: batch indices shapes - pred: {y_pred_batch_indices.shape}, true: {y_true_batch_indices.shape}")
        
        # Get masks and mapping
        if self.occupied_only:
            mapped_mask, mapped_mask_other = data_buffer.occupied_mapped_mask()
            mapping = data_buffer.occupied_mapping()
        else:
            mapped_mask, mapped_mask_other = data_buffer.mapped_mask()
            mapping = data_buffer.mapping()
            
        print(f"DEBUG: Mapping obtained - exists: {mapping is not None}, size: {mapping.shape if mapping is not None else 'None'}")
            
        if mapping is None or mapping.numel() == 0:
            print("DEBUG: No valid mapping found, returning max distance")
            # Create a loss that maintains gradient flow
            return y_pred_values[:, 0].mean() * 0.0 + self.max_distance
            
        # Get mapped points using indices
        pred_matched = y_pred_values[mapping[:, 0]]
        true_matched = y_true_values[mapping[:, 1]]
        batch_indices = y_pred_batch_indices[mapping[:, 0]]
        
        print(f"DEBUG: Matched points shapes - pred: {pred_matched.shape}, true: {true_matched.shape}")
        print(f"DEBUG: Batch indices after mapping: {batch_indices}")
        
        losses = []
        for b in range(self._batch_size):
            print(f"\nDEBUG: Processing batch {b}")
            # Get points for current batch
            batch_mask = batch_indices == b
            pred_b = pred_matched[batch_mask]
            true_b = true_matched[batch_mask]
            
            print(f"DEBUG: Points in batch {b} - pred: {pred_b.shape}, true: {true_b.shape}")
            
            if pred_b.size(0) == 0 or true_b.size(0) == 0:
                print(f"DEBUG: Empty batch {b}, using fallback loss")
                # If both are empty, just return max_distance without gradient
                if pred_b.size(0) == 0 and true_b.size(0) == 0:
                    batch_loss = torch.tensor(self.max_distance, device=y_pred_values.device)
                # Otherwise use the non-empty tensor to maintain gradient flow
                elif pred_b.size(0) > 0:
                    batch_loss = pred_b[:, 0].mean() * 0.0 + self.max_distance
                else:
                    batch_loss = true_b[:, 0].mean() * 0.0 + self.max_distance
                losses.append(batch_loss)
                continue
                
            # Compute pairwise distances
            dists = torch.cdist(pred_b[:, :3], true_b[:, :3], p=2)
            print(f"DEBUG: Distance matrix shape for batch {b}: {dists.shape}")
            print(f"DEBUG: Distance matrix values: min={dists.min().item():.4f}, max={dists.max().item():.4f}")
            
            # Chamfer distance: min distance in both directions
            loss_pred = dists.min(dim=1)[0].mean()  # For each pred point, distance to closest true point
            loss_true = dists.min(dim=0)[0].mean()  # For each true point, distance to closest pred point
            
            print(f"DEBUG: Batch {b} losses - pred: {loss_pred.item():.4f}, true: {loss_true.item():.4f}")
            
            # Total spatial loss for this batch
            batch_loss = (loss_pred + loss_true) / 2.0
            print(f"DEBUG: Batch {b} final loss: {batch_loss.item():.4f}")
            losses.append(batch_loss)
            
        final_loss = torch.stack(losses).mean()
        print(f"\nDEBUG: Final spatial loss: {final_loss.item():.4f}")
        return final_loss
        
    def _calc_probability_loss(self, y_pred, y_true, data_buffer):
        """Calculate the probability loss using BCE.
        
        Args:
            y_pred: Tuple of (predicted points, batch indices)
            y_true: Tuple of (true points, batch indices)
            data_buffer: ChamferPointDataBuffer instance
            
        Returns:
            Mean probability loss across batches
        """
        (y_pred_values, y_pred_batch_indices), (y_true_values, y_true_batch_indices) = y_pred, y_true
        
        # Get masks and mapping
        if self.occupied_only:
            mapped_mask, mapped_mask_other = data_buffer.occupied_mapped_mask()
            mapping = data_buffer.occupied_mapping()
        else:
            mapped_mask, mapped_mask_other = data_buffer.mapped_mask()
            mapping = data_buffer.mapping()
            
        if mapping is None or mapping.numel() == 0:
            # Create a loss that maintains gradient flow
            return y_pred_values[:, 3].mean() * 0.0 + 1.0
            
        # Get mapped points using indices
        pred_matched = y_pred_values[mapping[:, 0]]
        true_matched = y_true_values[mapping[:, 1]]
        batch_indices = y_pred_batch_indices[mapping[:, 0]]
        
        losses = []
        for b in range(self._batch_size):
            # Get points for current batch
            batch_mask = batch_indices == b
            pred_b = pred_matched[batch_mask]
            true_b = true_matched[batch_mask]
            
            if pred_b.size(0) == 0 or true_b.size(0) == 0:
                # Create a loss that maintains gradient flow
                batch_loss = pred_b[:, 3].mean() * 0.0 + 1.0 if pred_b.size(0) > 0 else true_b[:, 3].mean() * 0.0 + 1.0
                losses.append(batch_loss)
                continue
                
            # BCE loss on occupancy probabilities
            bce_loss = self.bce_loss(pred_b[:, 3], true_b[:, 3]).mean()
            losses.append(bce_loss)
            
        return torch.stack(losses).mean()
        
    def _calc(self, y_pred, y_true, data_buffer=None, *args, **kwargs):
        """Calculate the total loss.
        
        Args:
            y_pred: Tuple of (predicted points, batch indices)
            y_true: Tuple of (true points, batch indices)
            data_buffer: ChamferPointDataBuffer instance
            *args, **kwargs: Additional arguments
            
        Returns:
            Total loss combining spatial and probability terms
        """
        spatial_loss = self._calc_spatial_loss(y_pred, y_true, data_buffer)
        probability_loss = self._calc_probability_loss(y_pred, y_true, data_buffer)
        total_loss = self.spatial_weight * spatial_loss + self.probability_weight * probability_loss
        return total_loss
