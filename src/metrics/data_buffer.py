from abc import ABC, abstractmethod

import torch


class OccupancyDataBuffer(ABC):
    def __init__(self, occupancy_threshold=0.5, **kwargs):
        self._occupancy_threshold = occupancy_threshold
        for k, v in kwargs.items():
            setattr(self, k, v)
        # self._occupied_data = None
        # self._occupied_masks = None
        self._occupied_mask = None

    def occupancy_threshold(self) -> float:
        return self._occupancy_threshold

    def occupied_mask(self):
        """
        :returns: None or a Tensor of bool.
        Size: N_points of the original point cloud.
        """
        return self._occupied_mask

    @abstractmethod
    def filter_occupied(self, y, **kwargs):
        raise NotImplementedError()

    def create_masks(self, y, **kwargs):
        self._occupied_mask = self.filter_occupied(y, **kwargs)

    @abstractmethod
    def get_occupied_data(self, y, **kwargs):
        raise NotImplementedError()

    # def occupied_only(self) -> bool:
    #     return self._occupied_only

    # def occupied_data(self) -> tuple or None:
    #     """
    #     :returns: None or a tuple (y_pred_occupied, y_true_occupied), where each is a tuple (y_*_values_occupied, y_*_indices_occupied), or points and their in-batch sample indices.
    #     Sizes: (((N_pred_occ, 4), N_pred_occ), ((N_true_occ, 4), N_true_occ)), N_pred_occ <= N_pred, N_true_occ <= N_true.
    #     """
    #     return self._occupied_data

    # def occupied_masks(self) -> tuple or None:
    #     """
    #     :returns: None or a tuple (y_pred_occupied_mask, y_true_occupied_mask).
    #     Sizes: (N_pred, N_true) of the original point clouds.
    #     """
    #     return self._occupied_masks

    # def create_masks(self, y, **kwargs):
        # y_pred_occupied, y_pred_occupied_mask = self.filter_occupied(y_pred)
        # y_true_occupied, y_true_occupied_mask = self.filter_occupied(y_true)
        # self._occupied_data = (y_pred_occupied, y_true_occupied)
        # y_pred_occupied_mask = self.filter_occupied(y_pred)
        # y_true_occupied_mask = self.filter_occupied(y_true)
        # self._occupied_masks = (y_pred_occupied_mask, y_true_occupied_mask)
        # if not y_pred_occupied[0].requires_grad:
        #     raise ValueError("Predicted tensor does not require gradient after filtering.")
        # return (y_pred[0][y_pred_occupied_mask], y_pred[1][y_pred_occupied_mask]), (y_true[0][y_true_occupied_mask], y_true[1][y_true_occupied_mask])


class PointOccupancyDataBuffer(OccupancyDataBuffer):
    def filter_occupied(self, y, **kwargs):
        cloud_values, _ = y
        mask = cloud_values[:, -1] >= self._occupancy_threshold
        return mask

    def get_occupied_data(self, y, **kwargs):
        """
        :returns: a tuple (y_values_occupied, y_indices_occupied), or points and their in-batch sample indices.
        Sizes: ((N_points_occ, 4), N_points_occ), N_points_occ <= N_points.
        """
        if self._occupied_mask is None:
            raise ValueError('Occupied mask not created.')
        y_values, y_batch_indices = y
        return y_values[self._occupied_mask], y_batch_indices[self._occupied_mask]


class MappedPointOccupancyDataBuffer(PointOccupancyDataBuffer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # self._mapped_clouds = None
        # self._occupied_mapped_clouds = None
        self._mapped_mask = None
        self._occupied_mapped_mask = None
        # self._mapped_mask = None
        # self._occupied_mapped_mask = None

    def filter_occupied(self, y, y_other=None, **kwargs):
        if y_other is None:
            raise ValueError('y_other not provided.')
        y_values, _ = y
        y_values_other, _ = y_other
        mask = y_values[:, -1] >= self._occupancy_threshold
        mask_other = y_values_other[:, -1] >= self._occupancy_threshold
        return mask, mask_other

    def get_occupied_data(self, y, y_other=None, **kwargs) -> tuple:
        """
        :returns: a tuple (y_occupied, y_occupied_other), where each is a tuple (y_values_occupied, y_indices_occupied), or points and their in-batch sample indices.
        Sizes: (((N_points_occ, 4), N_points_occ), ((N_points_occ_other, 4), N_points_occ_other)), N_points_occ <= N_points, N_points_occ_other <= N_points_other.
        """
        if y_other is None:
            raise ValueError('y_other not provided.')
        if self._occupied_mask is None:
            raise ValueError('Occupied masks not created.')
        (y_values, y_batch_indices), (y_values_other, y_batch_indices_other) = y, y_other
        occupied_mask, occupied_mask_other = self._occupied_mask
        return (y_values[occupied_mask], y_batch_indices[occupied_mask]), (y_values_other[occupied_mask_other], y_batch_indices_other[occupied_mask_other])

    def mapped_mask(self) -> tuple or None:
        """
        :returns: None or a tuple (mapped_mask, y_other_mapped_mask) of boolean masks.
        Sizes: (N_points, N_other) of the original point clouds.
        """
        return self._mapped_mask

    def occupied_mapped_mask(self) -> tuple or None:
        """
        :returns: None or a tuple (mapped_mask, y_other_mapped_mask) of boolean masks.
        Sizes: (N_points, N_other) of the original point clouds.
        """
        return self._occupied_mapped_mask

    @abstractmethod
    def match_points(self, cloud_values_1, batch_indices_1, cloud_values_2, batch_indices_2, **kwargs):
        raise NotImplementedError()

    def create_masks(self, y, y_other=None, **kwargs):
        super().create_masks(y, y_other=y_other, **kwargs)
        (y_values, y_batch_indices), (y_values_other, y_batch_indices_other) = y, y_other

        mapping = self.match_points(y_values, y_batch_indices, y_values_other, y_batch_indices_other)
        mapping, mapping_other = mapping[:, 0], mapping[:, 1]
        mapped_mask = torch.zeros(y_values.shape[0], dtype=torch.bool, device=y_values.device)
        mapped_mask[mapping] = True
        mapped_mask_other = torch.zeros(y_values_other.shape[0], dtype=torch.bool, device=y_values_other.device)
        mapped_mask_other[mapping_other] = True
        self._mapped_mask = (mapped_mask, mapped_mask_other)

        if self._occupied_mask is not None:
            occupied_mask, occupied_mask_other = self._occupied_mask
            occupied_mapped_mask, occupied_mapped_mask_other = torch.zeros_like(occupied_mask), torch.zeros_like(occupied_mask_other)
            occupied_mapped_mask[occupied_mask & mapped_mask] = True
            occupied_mapped_mask_other[occupied_mask_other & mapped_mask_other] = True
            self._occupied_mapped_mask = (occupied_mapped_mask, occupied_mapped_mask_other)

    def get_mapped_data(self, y, y_other=None, **kwargs) -> tuple:
        """
        :returns: a tuple (y_mapped, y_mapped_other, mapped_batch_indices).
        Sizes: ((N_points_mapped, 4), (N_points_mapped, 4), N_points_mapped), N_points_mapped <= N_points, N_points_mapped <= N_points_other.
        """
        if y_other is None:
            raise ValueError('y_other not provided.')
        if self._mapped_mask is None:
            raise ValueError('Mapped masks not created.')
        (y_values, y_batch_indices), (y_values_other, y_batch_indices_other) = y, y_other
        mapped_mask, mapped_mask_other = self._mapped_mask
        return y_values[mapped_mask], y_values_other[mapped_mask_other], torch.index_select(y_batch_indices, 0, torch.nonzero(mapped_mask, as_tuple=False).squeeze(1))

    def get_occupied_mapped_data(self, y, y_other=None, **kwargs) -> tuple:
        """
        :returns: a tuple (y_occupied_mapped, y_occupied_mapped_other, occupied_mapped_batch_indices).
        Sizes: ((N_points_occupied_mapped, 4), (N_points_occupied_mapped, 4), N_points_occupied_mapped), N_points_occupied_mapped <= N_points, N_points_occupied_mapped <= N_points_other.
        """
        if y_other is None:
            raise ValueError('y_other not provided.')
        if self._occupied_mapped_mask is None:
            raise ValueError('Occupied mapped masks not created.')
        (y_values, y_batch_indices), (y_values_other, y_batch_indices_other) = y, y_other
        occupied_mapped_mask, occupied_mapped_mask_other = self._occupied_mapped_mask
        return y_values[occupied_mapped_mask], y_values_other[occupied_mapped_mask_other], torch.index_select(y_batch_indices, 0, torch.nonzero(occupied_mapped_mask, as_tuple=False).squeeze(1))

    # def mapped_clouds(self) -> tuple or None:
    #     """
    #     :returns: None or a tuple (y_pred_values_mapped, y_true_values_mapped, indices_mapped), where the first two are point clouds of the same size and the third one is their in-batch sample indices.
    #     Sizes: ((N_mapped, 4), (N_mapped, 4), N_mapped), N_mapped <= min(N_pred, N_true).
    #     """
    #     return self._mapped_clouds

    # def occupied_mapped_clouds(self) -> tuple or None:
    #     """
    #     :returns: None or a tuple (y_pred_values_occupied_mapped, y_true_values_occupied_mapped, indices_occupied_mapped) corresponding to points that are both occupied and matched.
    #     Sizes: ((N_occupied_mapped, 4), (N_occupied_mapped, 4), N_occupied_mapped), N_occupied_mapped <= min(N_pred_occ, N_true_occ, N_mapped).
    #     """
    #     return self._occupied_mapped_clouds

    # def mapped_masks(self) -> tuple or None:
    #     """
    #     :returns: None or a tuple (y_pred_mapped_mask, y_true_mapped_mask) of boolean masks.
    #     Sizes: (N_pred, N_true) of the original point clouds.
    #     """
    #     return self._mapped_masks
    #
    # def occupied_mapped_masks(self) -> tuple or None:
    #     """
    #     :returns: None or a tuple (y_pred_occupied_mapped_mask, y_true_occupied_mapped_mask) of boolean masks.
    #     Sizes: (N_pred, N_true) of the original point clouds.
    #     """
    #     return self._occupied_mapped_masks

    # def mapped_mask(self):
    #     """
    #     :returns: None or a Tensor of bool.
    #     Size: N_points of the original point cloud.
    #     """
    #     return self._mapped_mask
    #
    # def occupied_mapped_mask(self):
    #     """
    #     :returns: None or a Tensor of bool.
    #     Size: N_points of the original point cloud.
    #     """
    #     return self._occupied_mapped_mask

    # def filter_occupied(self, y, *args, **kwargs):
    #     cloud_values, _ = y
    #     mask = cloud_values[:, -1] >= self._occupancy_threshold
    #     # return (cloud_values[mask], cloud_indices[mask]), mask
    #     return mask

    # def create_masks(self, y_pred, y_true, *args, **kwargs):
    #     super().create_masks(y_pred, y_true, *args, **kwargs)
    #     (y_pred_values, y_pred_indices), (y_true_values, y_true_indices) = y_pred, y_true
    #
    #     mapping = self.match_points(y_pred_values, y_pred_indices, y_true_values, y_true_indices)
    #     mapping_pred, mapping_true = mapping[:, 0], mapping[:, 1]
    #     # y_pred_values_mapped = torch.index_select(y_pred_values, 0, mapping_pred)
    #     # y_true_values_mapped = torch.index_select(y_true_values, 0, mapping_true)
    #     # indices_mapped = torch.index_select(y_pred_indices, 0, mapping_pred)
    #     # self._mapped_clouds = (y_pred_values_mapped, y_true_values_mapped, indices_mapped)
    #     # if not y_pred_values_mapped.requires_grad:
    #     #     raise ValueError("Predicted tensor does not require gradient after mapping.")
    #
    #     y_pred_mapped_mask = torch.zeros(y_pred_values.shape[0], dtype=torch.bool, device=y_pred_values.device)
    #     y_pred_mapped_mask[mapping_pred] = True
    #     y_true_mapped_mask = torch.zeros(y_true_values.shape[0], dtype=torch.bool, device=y_true_values.device)
    #     y_true_mapped_mask[mapping_true] = True
    #     self._mapped_masks = (y_pred_mapped_mask, y_true_mapped_mask)
    #
    #     # if self._occupied_data is not None and self._occupied_masks is not None:
    #     if self._occupied_masks is not None:
    #         # (y_pred_occupied_values, y_pred_occupied_indices), (y_true_occupied_values, y_true_occupied_indices) = self.occupied_data()
    #         y_pred_occupied_mask, y_true_occupied_mask = self.occupied_masks()
    #         y_pred_occupied_mapped_mask, y_true_occupied_mapped_mask = torch.zeros_like(y_pred_occupied_mask), torch.zeros_like(y_true_occupied_mask)
    #         y_pred_occupied_mapped_mask[y_pred_occupied_mask & y_pred_mapped_mask] = True
    #         y_true_occupied_mapped_mask[y_true_occupied_mask & y_true_mapped_mask] = True
    #         self._occupied_mapped_masks = (y_pred_occupied_mapped_mask, y_true_occupied_mapped_mask)
    #         # self._occupied_mapped_clouds = (
    #         #     y_pred_occupied_values[y_pred_occupied_mapped_mask[y_pred_occupied_mask]],
    #         #     y_true_occupied_values[y_true_occupied_mapped_mask[y_true_occupied_mask]],
    #         #     y_pred_occupied_indices[y_pred_occupied_mapped_mask[y_pred_occupied_mask]],
    #         # )
    #         # if not self._occupied_mapped_clouds[0].requires_grad:
    #         #     raise ValueError("Predicted tensor does not require gradient after mapping.")
    #
    #     # return y_pred, y_true



class ChamferPointDataBuffer(MappedPointOccupancyDataBuffer):
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


class SinkhornPointDataBuffer(MappedPointOccupancyDataBuffer):
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
