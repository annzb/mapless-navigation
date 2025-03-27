from abc import ABC, abstractmethod

import torch


class OccupancyDataBuffer(ABC):
    def __init__(self, occupied_only=False, occupancy_threshold=0.5, **kwargs):
        self._occupied_only = occupied_only
        self._occupancy_threshold = occupancy_threshold
        for k, v in kwargs.items():
            setattr(self, k, v)
        self._occupied_data = None
        self._occupied_indices = None

    def occupied_only(self):
        return self._occupied_only

    def occupancy_threshold(self):
        return self._occupancy_threshold

    def occupied_data(self):
        return self._occupied_data

    def occupied_indices(self):
        return self._occupied_indices

    @abstractmethod
    def filter_occupied(self, y, *args, **kwargs):
        raise NotImplementedError()

    def __call__(self, y_pred, y_true, *args, **kwargs):
        if self._occupied_only:
            y_pred, y_pred_occupied_indices = self.filter_occupied(y_pred)
            y_true, y_true_occupied_indices = self.filter_occupied(y_true)
            self._occupied_data = (y_pred, y_true)
            self._occupied_indices = (y_pred_occupied_indices, y_true_occupied_indices)
        return y_pred, y_true


class PointOccupancyDataBuffer(OccupancyDataBuffer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._mapped_clouds = None
        self._mapped_indices = None

    def mapped_clouds(self):
        return self._mapped_clouds

    def mapped_indices(self):
        return self._mapped_indices

    @abstractmethod
    def match_points(self, cloud_values_1, batch_indices_1, cloud_values_2, batch_indices_2, **kwargs):
        """
        Matches two point clouds using a mutual similarity metric.
        Args:
            cloud_values_1 (torch.Tensor): First point cloud tensor of shape (M1, 4), where the first 3 columns are spatial coordinates.
            batch_indices_1 (torch.Tensor): Batch indices for cloud_values_1, shape (M1,).
            cloud_values_2 (torch.Tensor): Second point cloud tensor of shape (M2, 4).
            batch_indices_2 (torch.Tensor): Batch indices for cloud_values_2, shape (M2,).
        Returns:
            mapping (torch.Tensor): A tensor of shape (K, 2), where each row contains [index_in_pc1, index_in_pc2] for a mutual match.
        """
        raise NotImplementedError()

    def filter_occupied(self, clouds_flat, *args, **kwargs):
        cloud_values, cloud_batch_indices = clouds_flat
        mask = cloud_values[:, -1] >= self._occupancy_threshold
        return (cloud_values[mask], cloud_batch_indices[mask]), mask

    def __call__(self, y_pred, y_true, *args, **kwargs):
        (y_pred_values, y_pred_batch_indices), (y_true_values, y_true_batch_indices) = super().__call__(y_pred, y_true, *args, **kwargs)
        mapping = self.match_points(y_pred_values, y_pred_batch_indices, y_true_values, y_true_batch_indices)
        mapping_1, mapping_2 = mapping[:, 0], mapping[:, 1]
        y_pred_values_mapped = torch.index_select(y_pred_values, 0, mapping_1)
        y_true_values_mapped = torch.index_select(y_true_values, 0, mapping_2)
        batch_indices_mapped = torch.index_select(y_pred_batch_indices, 0, mapping_1)
        self._mapped_clouds = (y_pred_values_mapped, y_true_values_mapped, batch_indices_mapped)

        matches_mask_1 = torch.zeros(y_pred_values.shape[0], dtype=torch.bool, device=y_pred_values.device)
        matches_mask_1[mapping_1] = True
        matches_mask_2 = torch.zeros(y_true_values.shape[0], dtype=torch.bool, device=y_true_values.device)
        matches_mask_2[mapping_2] = True
        self._mapped_indices = (matches_mask_1, matches_mask_2)

        return y_pred_values_mapped, y_true_values_mapped, batch_indices_mapped


# class ChamferMapping(PointcloudMapping):
#     def match_points(self, cloud_values_1, batch_indices_1, cloud_values_2, batch_indices_2, large_val=1e9, **kwargs):
#         """
#         Matches two point clouds using mutual nearest neighbors.
#         Args:
#             ...
#             large_val (float): A large number used to mask out distances for points from different batches.
#         Note:
#             The matching operations (using argmin) are inherently non-differentiable with respect to the point coordinates.
#             However, once the mapping is computed, the subsequent gather operations preserve gradients on the input clouds.
#         """
#         dists = torch.cdist(cloud_values_1[:, :3], cloud_values_2[:, :3], p=2)
#         valid_mask = (batch_indices_1.unsqueeze(1) == batch_indices_2.unsqueeze(0))
#         dists = dists.masked_fill(~valid_mask, large_val)
#         best_12 = dists.argmin(dim=1)
#         best_21 = dists.argmin(dim=0)
#         idx1 = torch.arange(cloud_values_1.size(0), device=cloud_values_1.device)
#         mutual_mask = (best_21[best_12] == idx1)
#         final_idx1 = idx1[mutual_mask]
#         final_idx2 = best_12[mutual_mask]
#         mapping = torch.stack((final_idx1, final_idx2), dim=1)
#         return mapping
#
#
# class SinkhornMapping(PointcloudMapping):
#     def __init__(self, n_iters=20, temperature=0.1, **kwargs):
#         super().__init__(**kwargs)
#         if not isinstance(n_iters, int) or n_iters < 1:
#             raise ValueError("n_iters must be a positive integer.")
#         self._n_iters = n_iters
#         if not isinstance(temperature, (float, int)) or temperature <= 0:
#             raise ValueError("temperature must be a positive number.")
#         self._temperature = float(temperature)
#
#     def _sinkhorn_normalization(self, log_alpha):
#         """
#         Applies Sinkhorn normalization to a log affinity matrix to produce a doubly stochastic matrix.
#         Args:
#             log_alpha (torch.Tensor): Log affinity matrix of shape (M, N).
#         Returns:
#             torch.Tensor: A doubly stochastic matrix of shape (M, N).
#         """
#         for _ in range(self._n_iters):
#             log_alpha = log_alpha - torch.logsumexp(log_alpha, dim=1, keepdim=True)
#             log_alpha = log_alpha - torch.logsumexp(log_alpha, dim=0, keepdim=True)
#         return torch.exp(log_alpha)
#
#     def match_points(self, cloud_values_1, batch_indices_1, cloud_values_2, batch_indices_2, **kwargs):
#         """
#         Implements a differentiable matching using the Sinkhorn algorithm.
#         This function computes a soft assignment between two point clouds (each given as a flattened tensor with corresponding batch indices). Points from different batches are masked out.
#         After computing a soft assignment matrix via Sinkhorn normalization, a hard mapping is extracted using argmax in a way analogous to the chamfer mapping method.
#         """
#         # Compute pairwise Euclidean distances (using only the first 3 columns)
#         cost = torch.cdist(cloud_values_1[:, :3], cloud_values_2[:, :3], p=2)
#         # Convert cost to log affinity (similarity) scaled by temperature.
#         log_alpha = -cost / self._temperature
#         # Only allow matches within the same batch: mask out other pairs.
#         valid_mask = (batch_indices_1.unsqueeze(1) == batch_indices_2.unsqueeze(0))
#         log_alpha = log_alpha.masked_fill(~valid_mask, -1e9)
#         # Apply Sinkhorn normalization to get a soft assignment matrix.
#         assignment = self._sinkhorn_normalization(log_alpha)
#         # Convert the soft assignment to a discrete mapping similar to ChamferMapping:
#         best_12 = assignment.argmax(dim=1)
#         best_21 = assignment.argmax(dim=0)
#         idx1 = torch.arange(cloud_values_1.size(0), device=cloud_values_1.device)
#         mutual_mask = (best_21[best_12] == idx1)
#         final_idx1 = idx1[mutual_mask]
#         final_idx2 = best_12[mutual_mask]
#         mapping = torch.stack((final_idx1, final_idx2), dim=1)
#         return mapping
