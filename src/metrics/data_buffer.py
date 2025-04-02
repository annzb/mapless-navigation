from abc import ABC, abstractmethod

import torch


class OccupancyDataBuffer(ABC):
    def __init__(self, occupied_only=False, occupancy_threshold=0.5, **kwargs):
        self._occupied_only = occupied_only
        self._occupancy_threshold = occupancy_threshold
        for k, v in kwargs.items():
            setattr(self, k, v)
        self._occupied_data = None
        self._occupied_masks = None

    def occupied_only(self) -> bool:
        return self._occupied_only

    def occupancy_threshold(self) -> float:
        return self._occupancy_threshold

    def occupied_data(self) -> tuple or None:
        """
        :returns: None or a tuple (y_pred_occupied, y_true_occupied), where each is a tuple (y_*_values_occupied, y_*_indices_occupied), or points and their in-batch sample indices.
        Sizes: (((N_pred_occ, 4), N_pred_occ), ((N_true_occ, 4), N_true_occ)), N_pred_occ <= N_pred, N_true_occ <= N_true.
        """
        return self._occupied_data

    def occupied_masks(self) -> tuple or None:
        """
        :returns: None or a tuple (y_pred_occupied_mask, y_true_occupied_mask).
        Sizes: (N_pred, N_true) of the original point clouds.
        """
        return self._occupied_masks

    @abstractmethod
    def filter_occupied(self, y, *args, **kwargs):
        raise NotImplementedError()

    def __call__(self, y_pred, y_true, *args, **kwargs):
        y_pred_occupied, y_pred_occupied_mask = self.filter_occupied(y_pred)
        y_true_occupied, y_true_occupied_mask = self.filter_occupied(y_true)
        self._occupied_data = (y_pred_occupied, y_true_occupied)
        self._occupied_masks = (y_pred_occupied_mask, y_true_occupied_mask)
        if not y_pred_occupied[0].requires_grad:
            raise ValueError("Predicted tensor does not require gradient after filtering.")
        return y_pred_occupied, y_true_occupied


class PointOccupancyDataBuffer(OccupancyDataBuffer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._mapped_clouds = None
        self._mapped_masks = None
        self._occupied_mapped_clouds = None
        self._occupied_mapped_masks = None

    def mapped_clouds(self) -> tuple or None:
        """
        :returns: None or a tuple (y_pred_values_mapped, y_true_values_mapped, indices_mapped), where the first two are point clouds of the same size and the third one is their in-batch sample indices.
        Sizes: ((N_mapped, 4), (N_mapped, 4), N_mapped), N_mapped <= min(N_pred, N_true).
        """
        return self._mapped_clouds

    def mapped_masks(self) -> tuple or None:
        """
        :returns: None or a tuple (y_pred_mapped_mask, y_true_mapped_mask) of boolean masks.
        Sizes: (N_pred, N_true) of the original point clouds.
        """
        return self._mapped_masks

    def occupied_mapped_clouds(self) -> tuple or None:
        """
        :returns: None or a tuple (y_pred_values_occupied_mapped, y_true_values_occupied_mapped, indices_occupied_mapped) corresponding to points that are both occupied and matched.
        Sizes: ((N_occupied_mapped, 4), (N_occupied_mapped, 4), N_occupied_mapped), N_occupied_mapped <= min(N_pred_occ, N_true_occ, N_mapped).
        """
        return self._occupied_mapped_clouds

    def occupied_mapped_masks(self) -> tuple or None:
        """
        :returns: None or a tuple (y_pred_occupied_mapped_mask, y_true_occupied_mapped_mask) of boolean masks.
        Sizes: (N_pred, N_true) of the original point clouds.
        """
        return self._occupied_mapped_masks

    @abstractmethod
    def match_points(self, cloud_values_1, batch_indices_1, cloud_values_2, batch_indices_2, **kwargs):
        raise NotImplementedError()

    def filter_occupied(self, clouds_flat, *args, **kwargs):
        cloud_values, cloud_indices = clouds_flat
        mask = cloud_values[:, -1] >= self._occupancy_threshold
        return (cloud_values[mask], cloud_indices[mask]), mask

    def __call__(self, y_pred, y_true, *args, **kwargs):
        super().__call__(y_pred, y_true, *args, **kwargs)
        (y_pred_values, y_pred_indices), (y_true_values, y_true_indices) = y_pred, y_true

        mapping = self.match_points(y_pred_values, y_pred_indices, y_true_values, y_true_indices)
        mapping_pred, mapping_true = mapping[:, 0], mapping[:, 1]
        y_pred_values_mapped = torch.index_select(y_pred_values, 0, mapping_pred)
        y_true_values_mapped = torch.index_select(y_true_values, 0, mapping_true)
        indices_mapped = torch.index_select(y_pred_indices, 0, mapping_pred)
        self._mapped_clouds = (y_pred_values_mapped, y_true_values_mapped, indices_mapped)
        if not y_pred_values_mapped.requires_grad:
            raise ValueError("Predicted tensor does not require gradient after mapping.")

        y_pred_mapped_mask = torch.zeros(y_pred_values.shape[0], dtype=torch.bool, device=y_pred_values.device)
        y_pred_mapped_mask[mapping_pred] = True
        y_true_mapped_mask = torch.zeros(y_true_values.shape[0], dtype=torch.bool, device=y_true_values.device)
        y_true_mapped_mask[mapping_true] = True
        self._mapped_masks = (y_pred_mapped_mask, y_true_mapped_mask)

        if self._occupied_data is not None and self._occupied_masks is not None:
            (y_pred_occupied_values, y_pred_occupied_indices), (y_true_occupied_values, y_true_occupied_indices) = self.occupied_data()
            y_pred_occupied_mask, y_true_occupied_mask = self.occupied_masks()
            y_pred_occ_indices, y_true_occ_indices = torch.nonzero(y_pred_occupied_mask, as_tuple=False).squeeze(1), torch.nonzero(y_true_occupied_mask, as_tuple=False).squeeze(1)
            self._occupied_mapped_masks = (y_pred_mapped_mask[y_pred_occ_indices], y_true_mapped_mask[y_true_occ_indices])
            self._occupied_mapped_clouds = (
                y_pred_occupied_values[self._occupied_mapped_masks[0]],
                y_true_occupied_values[self._occupied_mapped_masks[1]],
                y_pred_occupied_indices[self._occupied_mapped_masks[0]]
            )
            if not self._occupied_mapped_clouds[0].requires_grad:
                raise ValueError("Predicted tensor does not require gradient after mapping.")

        return y_pred, y_true


class ChamferPointDataBuffer(PointOccupancyDataBuffer):
    def __init__(self, large_val=1e9, **kwargs):
        super().__init__(**kwargs)
        if not isinstance(large_val, (float, int)) or large_val <= 0:
            raise ValueError("large_val must be a positive number.")
        self._large_val = large_val

    def match_points(self, cloud_values_1, batch_indices_1, cloud_values_2, batch_indices_2, **kwargs):
        """
        Matches mutually nearest points across batches using Chamfer distance.
        """
        dists = torch.cdist(cloud_values_1[:, :3], cloud_values_2[:, :3], p=2)
        valid_mask = (batch_indices_1.unsqueeze(1) == batch_indices_2.unsqueeze(0))
        dists = dists.masked_fill(~valid_mask, self._large_val)
        best_12 = dists.argmin(dim=1)
        best_21 = dists.argmin(dim=0)
        idx1 = torch.arange(cloud_values_1.size(0), device=cloud_values_1.device)
        mutual_mask = (best_21[best_12] == idx1)
        final_idx1 = idx1[mutual_mask]
        final_idx2 = best_12[mutual_mask]
        return torch.stack((final_idx1, final_idx2), dim=1)


class SinkhornPointDataBuffer(ChamferPointDataBuffer):
    def __init__(self, n_iters=20, temperature=0.1, **kwargs):
        if not isinstance(n_iters, int) or n_iters < 1:
            raise ValueError("n_iters must be a positive integer.")
        if not isinstance(temperature, (float, int)) or temperature <= 0:
            raise ValueError("temperature must be a positive number.")
        super().__init__(**kwargs)
        self._n_iters = n_iters
        self._temperature = float(temperature)
        self._soft_assignment = None

    def soft_assignment(self):
        """Returns the last computed soft assignment matrix."""
        return self._soft_assignment

    def _sinkhorn_normalization(self, log_alpha):
        """
        Iteratively normalizes log_alpha to produce a doubly stochastic matrix.
        """
        for _ in range(self._n_iters):
            log_alpha = log_alpha - torch.logsumexp(log_alpha, dim=1, keepdim=True)
            log_alpha = log_alpha - torch.logsumexp(log_alpha, dim=0, keepdim=True)
        return torch.exp(log_alpha)

    def match_points(self, cloud_values_1, batch_indices_1, cloud_values_2, batch_indices_2, **kwargs):
        """
        Computes soft and hard point matches using the Sinkhorn algorithm.
        """
        cost = torch.cdist(cloud_values_1[:, :3], cloud_values_2[:, :3], p=2)
        log_alpha = -cost / self._temperature
        valid_mask = (batch_indices_1.unsqueeze(1) == batch_indices_2.unsqueeze(0))
        log_alpha = log_alpha.masked_fill(~valid_mask, -self._large_val)
        assignment = self._sinkhorn_normalization(log_alpha)
        self._soft_assignment = assignment
        best_12 = assignment.argmax(dim=1)
        best_21 = assignment.argmax(dim=0)
        idx1 = torch.arange(cloud_values_1.size(0), device=cloud_values_1.device)
        mutual_mask = (best_21[best_12] == idx1)
        final_idx1 = idx1[mutual_mask]
        final_idx2 = best_12[mutual_mask]
        return torch.stack((final_idx1, final_idx2), dim=1)
