import numpy as np
import torch

from abc import ABC, abstractmethod
from collections import defaultdict
from sklearn.cluster import DBSCAN

from utils import param_validation as validate


class OccupancyDataBuffer(ABC):
    def __init__(self, occupancy_threshold: float, **kwargs):
        if not isinstance(occupancy_threshold, (int, float)):
            raise ValueError('occupancy_threshold must be a number')
        if occupancy_threshold < 0 or occupancy_threshold > 1:
            raise ValueError('occupancy_threshold must be between 0 and 1')
        
        self._occupancy_threshold = occupancy_threshold
        self._occupied_mask = None

    def occupancy_threshold(self) -> float:
        return self._occupancy_threshold

    def occupied_mask(self):
        return self._occupied_mask

    def _validate_input(self, y, y_other=None, **kwargs):
        if y is None:
            raise ValueError('Input y cannot be None')
        if not isinstance(y, tuple) or len(y) != 2:
            raise ValueError('Input y must be a tuple of (points, batch_indices)')
        if y[0].numel() == 0:
            raise ValueError('Input points cannot be empty')

    @abstractmethod
    def filter_occupied(self, y, **kwargs):
        raise NotImplementedError()

    def create_masks(self, y, y_other=None, **kwargs):
        self._validate_input(y, y_other=y_other, **kwargs)
        self._occupied_mask = self.filter_occupied(y, y_other=y_other, **kwargs)


class GridOccupancyDataBuffer(OccupancyDataBuffer):
    def _validate_input(self, y, y_other=None, **kwargs):
        super()._validate_input(y, y_other=y_other, **kwargs)
        if y.shape != y_other.shape:
            raise ValueError(f"Grid shape mismatch: {y.shape} != {y_other.shape}")

    def filter_occupied(self, y, y_other=None, **kwargs):
        if y_other is None:
            raise ValueError('y_other not provided')
        mask = y >= self._occupancy_threshold
        mask_other = y_other >= self._occupancy_threshold
        return mask, mask_other


class PointOccupancyDataBuffer(OccupancyDataBuffer):
    def __init__(self, max_point_distance: float, **kwargs):
        super().__init__(**kwargs)
        if not isinstance(max_point_distance, (int, float)):
            raise ValueError('max_point_distance must be a number')
        if max_point_distance <= 0:
            raise ValueError('max_point_distance must be positive')
        
        self._max_point_distance = max_point_distance

    def max_point_distance(self) -> float:
        return self._max_point_distance

    def _validate_input(self, y, y_other=None, **kwargs):
        super()._validate_input(y, y_other=y_other, **kwargs)
        y_values, y_batch_indices = y
        if y_values.shape[0] != y_batch_indices.shape[0]:
            raise ValueError(f"Points and batch indices size mismatch: {y_values.shape[0]} != {y_batch_indices.shape[0]}")

    def filter_occupied(self, y, **kwargs):
        cloud_values, _ = y
        mask = cloud_values[:, -1] >= self._occupancy_threshold
        return mask


class MappedPointOccupancyDataBuffer(PointOccupancyDataBuffer):
    def __init__(self, occupied_only: bool, same_point_distance_limit: float, **kwargs):
        super().__init__(**kwargs)
        if not isinstance(occupied_only, bool):
            raise ValueError('occupied_only must be a boolean')
        self._occupied_only = occupied_only
        self._same_point_distance_limit = validate.validate_positive_number(same_point_distance_limit, 'same_point_distance_limit')
        
        self._mapped_mask = None
        self._occupied_mapped_mask = None
        self._mapping = None
        self._occupied_mapping = None
        self._soft_assignment = None
        self._cluster_counts = None

    def occupied_only(self) -> float:
        return self._occupied_only
    
    def same_point_distance_limit(self) -> float:
        return self._same_point_distance_limit

    def _validate_input(self, y, y_other=None, **kwargs):
        super()._validate_input(y, y_other=y_other, **kwargs)
        if y_other is None:
            raise ValueError('y_other not provided')
        super()._validate_input(y_other)

    def filter_occupied(self, y, y_other=None, **kwargs):
        if y_other is None:
            raise ValueError('y_other not provided.')
        y_values, _ = y
        y_values_other, _ = y_other
        mask = y_values[:, -1] >= self._occupancy_threshold
        mask_other = y_values_other[:, -1] >= self._occupancy_threshold
        return mask, mask_other

    def create_masks(self, y, y_other=None, **kwargs):
        self._validate_input(y, y_other=y_other, **kwargs)
        super().create_masks(y, y_other=y_other, **kwargs)
            
        (y_values, y_batch_indices), (y_values_other, y_batch_indices_other) = y, y_other
        occupied_mask, occupied_mask_other = self._occupied_mask
        mapped_mask = torch.zeros_like(occupied_mask, dtype=torch.bool)
        mapped_mask_other = torch.zeros_like(occupied_mask_other, dtype=torch.bool)
        
        # Initialize empty mapping
        self._mapping = torch.zeros((0, 2), device=y_values.device, dtype=torch.long)
    
        if self._occupied_only:
            # When matching only occupied points, filter inputs
            y_values_masked = y_values[occupied_mask]
            y_values_other_masked = y_values_other[occupied_mask_other]
            y_batch_indices_masked = y_batch_indices[occupied_mask]
            y_batch_indices_other_masked = y_batch_indices_other[occupied_mask_other]
            
            # Store indices for mapping back to original tensors
            occupied_indices = torch.nonzero(occupied_mask).squeeze(1)
            occupied_indices_other = torch.nonzero(occupied_mask_other).squeeze(1)
            
            # Get mapping for filtered points
            filtered_mapping = self.match_points(
                y_values_masked, y_batch_indices_masked,
                y_values_other_masked, y_batch_indices_other_masked
            )
            
            if filtered_mapping.numel() > 0:
                # Convert filtered indices back to original indices
                self._mapping = torch.stack([
                    occupied_indices[filtered_mapping[:, 0]],
                    occupied_indices_other[filtered_mapping[:, 1]]
                ], dim=1)
                
                # Update mapped masks
                mapped_mask[self._mapping[:, 0]] = True
                mapped_mask_other[self._mapping[:, 1]] = True
        else:
            mapping = self.match_points(
                y_values, y_batch_indices,
                y_values_other, y_batch_indices_other
            )
            
            if mapping.numel() > 0:
                self._mapping = mapping[:, :2].to(torch.long)  # Only keep indices, convert to long
                mapped_mask[self._mapping[:, 0]] = True
                mapped_mask_other[self._mapping[:, 1]] = True
                    
        self._mapped_mask = (mapped_mask, mapped_mask_other)
        self._cluster_clouds(y, y_other, **kwargs)
    def mapped_mask(self):
        """Get the pair of masks indicating mapped points.
        
        Returns:
            Tuple of (mask, mask_other) where each mask is a boolean tensor indicating which points are mapped to the other cloud.
        """
        return self._mapped_mask

    def mapping(self):
        """Get the soft mapping between points.
        
        Returns:
            Tensor of shape (N_matches, 2) containing the soft mapping weights between matched points.
        """
        return self._mapping

    @abstractmethod
    def match_points(self, cloud_values_1, batch_indices_1, cloud_values_2, batch_indices_2, **kwargs):
        """Compute soft mapping between points in two clouds.
        
        Args:
            cloud_values_1: Points from first cloud.
            batch_indices_1: Batch indices for first cloud.
            cloud_values_2: Points from second cloud.
            batch_indices_2: Batch indices for second cloud.
            **kwargs: Additional arguments.
            
        Returns:
            Tensor of shape (N_matches, 2) containing the soft mapping weights between matched points.
        """
        raise NotImplementedError()
    
    def _cluster_clouds(self, y, y_other=None, **kwargs):       
        (y_cloud, y_batch_indices), (y_cloud_other, y_batch_indices_other) = y, y_other
        cluster_counts = []

        for b in y_batch_indices.unique():
            y_mask, y_other_mask = y_batch_indices == b, y_batch_indices_other == b
            if self._occupied_only:
                y_other_mask = y_other_mask & self._occupied_mask[1]   # current loss doesn't predict occupancy probability so no thresholding for predicted points
            y_batch_cloud, y_other_batch_cloud = y_cloud[y_mask].detach().cpu().numpy(), y_cloud_other[y_other_mask].detach().cpu().numpy()

            N_y, N_y_other = len(y_batch_cloud), len(y_other_batch_cloud)
            if N_y_other == 0:  # No empty predictions expected but filtered ground truth cloud can be empty
                merged_points = y_batch_cloud[:, :3]
            else:
                merged_points = np.vstack([y_batch_cloud[:, :3], y_other_batch_cloud[:, :3]])

            labels = np.zeros(N_y + N_y_other, dtype=np.int32)
            labels[N_y:] = 1
            clustering = DBSCAN(eps=self._same_point_distance_limit, min_samples=1).fit(merged_points)
            cluster_ids = clustering.labels_
            num_clusters = cluster_ids.max() + 1
            sample_counts = np.zeros((num_clusters, 2), dtype=np.int32)

            for label in (0, 1):
                label_mask = labels == label
                cluster_ids_label = cluster_ids[label_mask]
                bincount = np.bincount(cluster_ids_label, minlength=num_clusters)
                sample_counts[:, label] = bincount
            cluster_counts.append(torch.from_numpy(sample_counts).to(y_cloud.device))

        self._cluster_counts = cluster_counts

    def cluster_counts(self):
        return self._cluster_counts


class ChamferPointDataBuffer(MappedPointOccupancyDataBuffer):
    def match_points(self, cloud_values_1, batch_indices_1, cloud_values_2, batch_indices_2, **kwargs):
        # Handle empty input cases
        if cloud_values_1.size(0) == 0 or cloud_values_2.size(0) == 0:
            return torch.zeros((0, 2), device=cloud_values_1.device, dtype=torch.long)
            
        # Get unique batch indices
        unique_batches = torch.unique(batch_indices_1)
        
        # Pre-allocate lists for matches
        all_matches = []
        
        # Process each batch
        for batch_idx in unique_batches:
            # Get points for this batch using boolean indexing
            mask1 = (batch_indices_1 == batch_idx)
            mask2 = (batch_indices_2 == batch_idx)
            
            if not (mask1.any() and mask2.any()):
                continue
                
            # Get points for this batch (only xyz coordinates)
            batch_points1 = cloud_values_1[mask1, :3]
            batch_points2 = cloud_values_2[mask2, :3]
            
            # Compute pairwise distances efficiently
            # Use squared Euclidean distance to avoid sqrt
            dists = torch.cdist(batch_points1, batch_points2, p=2)
            
            # Find nearest neighbors in one operation
            best_12 = dists.argmin(dim=1)
            best_21 = dists.argmin(dim=0)
            
            # Find mutual matches efficiently
            idx1 = torch.arange(batch_points1.size(0), device=cloud_values_1.device)
            mutual_mask = (best_21[best_12] == idx1)
            
            if not mutual_mask.any():
                continue
                
            # Get matched indices
            matched_idx1 = idx1[mutual_mask]
            matched_idx2 = best_12[mutual_mask]
            
            # Filter by distance threshold efficiently
            valid_dist_mask = dists[matched_idx1, matched_idx2] <= self._max_point_distance
            matched_idx1 = matched_idx1[valid_dist_mask]
            matched_idx2 = matched_idx2[valid_dist_mask]
            
            if len(matched_idx1) > 0:
                # Convert batch-relative indices to global indices efficiently
                global_idx1 = torch.nonzero(mask1)[matched_idx1].squeeze(1)
                global_idx2 = torch.nonzero(mask2)[matched_idx2].squeeze(1)
                
                # Add matches to list
                batch_matches = torch.stack([global_idx1, global_idx2], dim=1)
                all_matches.append(batch_matches)
        
        if not all_matches:
            return torch.zeros((0, 2), device=cloud_values_1.device, dtype=torch.long)
            
        # Concatenate all matches efficiently
        return torch.cat(all_matches, dim=0)


class SinkhornPointDataBuffer(MappedPointOccupancyDataBuffer): pass
    # """Buffer that uses Sinkhorn algorithm for soft point matching.
    
    # This buffer implements soft matching between points using the Sinkhorn algorithm.
    # It computes a doubly stochastic matrix that represents the soft assignment
    # between points in the two clouds. Points are only matched within the same batch.
    # """
    
    # def __init__(self, temperature=0.1, n_iters=20, *args, **kwargs):
    #     super().__init__(*args, **kwargs)
    #     self._soft_assignment = None
    #     self._mapping = None
    #     self.temperature = temperature
    #     self._num_iterations = n_iters

    # def soft_assignment(self):
    #     """Get the soft assignment matrix.
        
    #     Returns:
    #         Tensor of shape (N1, N2) containing the soft assignment weights
    #         between points in the two clouds.
    #     """
    #     return self._soft_assignment

    # def match_points(self, y_values, y_batch_indices, y_values_other, y_batch_indices_other):
    #     """Match points between two point clouds using Sinkhorn algorithm.
        
    #     Args:
    #         y_values: Points from first cloud.
    #         y_batch_indices: Batch indices for first cloud.
    #         y_values_other: Points from second cloud.
    #         y_batch_indices_other: Batch indices for second cloud.
            
    #     Returns:
    #         Tensor of shape (N, 3) containing indices of matched points and their soft weights.
    #     """
    #     if y_values.shape[0] == 0 or y_values_other.shape[0] == 0:
    #         return torch.zeros((0, 3), device=y_values.device)
            
    #     # Get unique batch indices
    #     unique_batches = torch.unique(y_batch_indices)
    #     all_matches = []
        
    #     # Process each batch separately
    #     for batch_idx in unique_batches:
    #         # Get points for this batch
    #         mask1 = (y_batch_indices == batch_idx)
    #         mask2 = (y_batch_indices_other == batch_idx)
            
    #         if not (mask1.any() and mask2.any()):
    #             continue
                
    #         # Get points for this batch
    #         batch_points1 = y_values[mask1]
    #         batch_points2 = y_values_other[mask2]
            
    #         # Compute pairwise distances for this batch
    #         distances = torch.cdist(batch_points1[:, :3], batch_points2[:, :3])
            
    #         # Initialize soft assignment matrix for this batch
    #         soft_assignment = torch.exp(-distances / self.temperature)
            
    #         # Apply Sinkhorn algorithm
    #         for _ in range(self._num_iterations):
    #             # Row normalization
    #             row_sum = soft_assignment.sum(dim=1, keepdim=True)
    #             row_sum = torch.where(row_sum == 0, torch.ones_like(row_sum), row_sum)
    #             soft_assignment = soft_assignment / row_sum
                
    #             # Column normalization
    #             col_sum = soft_assignment.sum(dim=0, keepdim=True)
    #             col_sum = torch.where(col_sum == 0, torch.ones_like(col_sum), col_sum)
    #             soft_assignment = soft_assignment / col_sum
            
    #         # Store soft assignment for this batch
    #         if self._soft_assignment is None:
    #             self._soft_assignment = torch.zeros((y_values.shape[0], y_values_other.shape[0]), 
    #                                               device=y_values.device)
            
    #         # Convert batch-relative indices to global indices
    #         global_idx1 = torch.nonzero(mask1).squeeze(1)
    #         global_idx2 = torch.nonzero(mask2).squeeze(1)
            
    #         # Update the full soft assignment matrix for this batch
    #         for i, gi1 in enumerate(global_idx1):
    #             for j, gi2 in enumerate(global_idx2):
    #                 self._soft_assignment[gi1, gi2] = soft_assignment[i, j]
            
    #         # Find mutual nearest neighbors based on distances
    #         best_12 = torch.argmin(distances, dim=1)  # For each point in cloud 1, closest in cloud 2
    #         best_21 = torch.argmin(distances, dim=0)  # For each point in cloud 2, closest in cloud 1
            
    #         # Find mutual matches
    #         idx1 = torch.arange(batch_points1.shape[0], device=y_values.device)
    #         mutual_mask = (best_21[best_12] == idx1)
            
    #         if not mutual_mask.any():
    #             continue
            
    #         # Get matched indices
    #         matched_idx1 = idx1[mutual_mask]
    #         matched_idx2 = best_12[mutual_mask]
            
    #         # Get soft weights for the matches
    #         soft_weights = soft_assignment[matched_idx1, matched_idx2]
            
    #         # Convert to global indices and add to matches with soft weights
    #         batch_matches = torch.stack([
    #             global_idx1[matched_idx1],
    #             global_idx2[matched_idx2],
    #             soft_weights
    #         ], dim=1)
    #         all_matches.append(batch_matches)
        
    #     if not all_matches:
    #         return torch.zeros((0, 3), device=y_values.device)
            
    #     return torch.cat(all_matches, dim=0)

    # def create_masks(self, y, y_other=None, **kwargs):
    #     """Create all necessary masks for the input clouds.
        
    #     Args:
    #         y: Tuple of (points, batch_indices) for first cloud.
    #         y_other: Tuple of (points, batch_indices) for second cloud.
    #         **kwargs: Additional arguments.
    #     """
    #     self._validate_input(y, y_other=y_other, **kwargs)
    #     super().create_masks(y, y_other=y_other, **kwargs)
            
    #     (y_values, y_batch_indices), (y_values_other, y_batch_indices_other) = y, y_other
    #     occupied_mask, occupied_mask_other = self._occupied_mask
    #     mapped_mask = torch.zeros_like(occupied_mask, dtype=torch.bool)
    #     mapped_mask_other = torch.zeros_like(occupied_mask_other, dtype=torch.bool)
        
    #     # Initialize empty mapping
    #     self._mapping = torch.zeros((0, 3), device=y_values.device)
    
    #     if self._occupied_only:
    #         # When matching only occupied points, filter inputs
    #         y_values_masked = y_values[occupied_mask]
    #         y_values_other_masked = y_values_other[occupied_mask_other]
    #         y_batch_indices_masked = y_batch_indices[occupied_mask]
    #         y_batch_indices_other_masked = y_batch_indices_other[occupied_mask_other]
            
    #         # Store indices for mapping back to original tensors
    #         occupied_indices = torch.nonzero(occupied_mask).squeeze(1)
    #         occupied_indices_other = torch.nonzero(occupied_mask_other).squeeze(1)
            
    #         # Get mapping for filtered points
    #         filtered_mapping = self.match_points(
    #             y_values_masked, y_batch_indices_masked,
    #             y_values_other_masked, y_batch_indices_other_masked
    #         )
            
    #         if filtered_mapping.numel() > 0:
    #             # Convert filtered indices back to original indices
    #             self._mapping = torch.stack([
    #                 occupied_indices[filtered_mapping[:, 0]],
    #                 occupied_indices_other[filtered_mapping[:, 1]],
    #                 filtered_mapping[:, 2]  # Keep soft weights
    #             ], dim=1)
                
    #             # Update mapped masks
    #             mapped_mask[self._mapping[:, 0].long()] = True
    #             mapped_mask_other[self._mapping[:, 1].long()] = True
    #     else:
    #         mapping = self.match_points(
    #             y_values, y_batch_indices,
    #             y_values_other, y_batch_indices_other
    #         )
            
    #         if mapping.numel() > 0:
    #             self._mapping = mapping
    #             mapped_mask[mapping[:, 0].long()] = True
    #             mapped_mask_other[mapping[:, 1].long()] = True
                    
    #     self._mapped_mask = (mapped_mask, mapped_mask_other)
