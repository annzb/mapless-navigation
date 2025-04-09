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

    # @abstractmethod
    # def get_occupied_data(self, y, **kwargs):
    #     raise NotImplementedError()


class PointOccupancyDataBuffer(OccupancyDataBuffer):
    def filter_occupied(self, y, **kwargs):
        cloud_values, _ = y
        mask = cloud_values[:, -1] >= self._occupancy_threshold
        return mask

    # def get_occupied_data(self, y, **kwargs):
    #     """
    #     :returns: a tuple (y_values_occupied, y_indices_occupied), or points and their in-batch sample indices.
    #     Sizes: ((N_points_occ, 4), N_points_occ), N_points_occ <= N_points.
    #     """
    #     if self._occupied_mask is None:
    #         raise ValueError('Occupied mask not created.')
    #     y_values, y_batch_indices = y
    #     return y_values[self._occupied_mask], y_batch_indices[self._occupied_mask]


class MappedPointOccupancyDataBuffer(PointOccupancyDataBuffer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._mapped_mask = None
        self._occupied_mapped_mask = None
        self._mapping = None  # stores matched index pairs (N_matches, 2)
        self._occupied_mapping = None  # stores matched index pairs for occupied points

    def filter_occupied(self, y, y_other=None, **kwargs):
        if y_other is None:
            raise ValueError('y_other not provided.')
        y_values, _ = y
        y_values_other, _ = y_other
        mask = y_values[:, -1] >= self._occupancy_threshold
        mask_other = y_values_other[:, -1] >= self._occupancy_threshold
        return mask, mask_other

    # def get_occupied_data(self, y, y_other=None, **kwargs) -> tuple:
    #     if y_other is None:
    #         raise ValueError('y_other not provided.')
    #     if self._occupied_mask is None:
    #         raise ValueError('Occupied masks not created.')
    #     (y_values, y_batch_indices), (y_values_other, y_batch_indices_other) = y, y_other
    #     occupied_mask, occupied_mask_other = self._occupied_mask
    #     return (y_values[occupied_mask], y_batch_indices[occupied_mask]), (y_values_other[occupied_mask_other], y_batch_indices_other[occupied_mask_other])

    # def mapped_mask(self):
    #     return self._mapped_mask
    #
    # def occupied_mapped_mask(self):
    #     return self._occupied_mapped_mask
    #
    # def mapping(self):
    #     return self._mapping
    #
    # def occupied_mapping(self):
    #     return self._occupied_mapping

    # def create_masks(self, y, y_other=None, **kwargs):
    #     super().create_masks(y, y_other=y_other, **kwargs)
    #     (y_values, y_batch_indices), (y_values_other, y_batch_indices_other) = y, y_other
    #
    #     assert y_values.shape[0] == y_batch_indices.shape[0], "y_values and y_batch_indices size mismatch"
    #     assert y_values_other.shape[0] == y_batch_indices_other.shape[0], "y_values_other and y_batch_indices_other size mismatch"
    #
    #     mapping = self.match_points(y_values, y_batch_indices, y_values_other, y_batch_indices_other)
    #     assert mapping.dim() == 2 and mapping.size(1) == 2, "Mapping shape should be [N_matches, 2]"
    #     assert mapping[:, 0].max() < y_values.shape[0], "Invalid indices in mapping[:,0]"
    #     assert mapping[:, 1].max() < y_values_other.shape[0], "Invalid indices in mapping[:,1]"
    #     self._mapping = mapping
    #
    #     mapped_mask = torch.zeros(y_values.shape[0], dtype=torch.bool, device=y_values.device)
    #     mapped_mask_other = torch.zeros(y_values_other.shape[0], dtype=torch.bool, device=y_values_other.device)
    #     mapped_mask[mapping[:, 0]] = True
    #     mapped_mask_other[mapping[:, 1]] = True
    #     self._mapped_mask = (mapped_mask, mapped_mask_other)
    #
    #     assert mapped_mask.sum() == mapped_mask_other.sum(), f"Mismatch in number of matched points: {mapped_mask.sum()} vs {mapped_mask_other.sum()}"
    #
    #     if self._occupied_mask is not None:
    #         occupied_mask, occupied_mask_other = self._occupied_mask
    #         pred_occ = occupied_mask[mapping[:, 0]]
    #         true_occ = occupied_mask_other[mapping[:, 1]]
    #         both_occupied = pred_occ & true_occ
    #         occupied_mapping = mapping[both_occupied]
    #         self._occupied_mapping = occupied_mapping
    #
    #         occupied_mapped_mask = torch.zeros_like(occupied_mask)
    #         occupied_mapped_mask_other = torch.zeros_like(occupied_mask_other)
    #         occupied_mapped_mask[occupied_mapping[:, 0]] = True
    #         occupied_mapped_mask_other[occupied_mapping[:, 1]] = True
    #         self._occupied_mapped_mask = (occupied_mapped_mask, occupied_mapped_mask_other)
    #         assert occupied_mapped_mask.sum() == occupied_mapped_mask_other.sum(), f"Occupied matched masks mismatch: {occupied_mapped_mask.sum()} vs {occupied_mapped_mask_other.sum()}"
    #
    # def get_mapped_data(self, y, y_other=None, **kwargs) -> tuple:
    #     if y_other is None:
    #         raise ValueError('y_other not provided.')
    #     if self._mapping is None:
    #         raise ValueError('Mapping not created.')
    #     (y_values, y_batch_indices), (y_values_other, _) = y, y_other
    #     return (
    #         y_values[self._mapping[:, 0]],
    #         y_values_other[self._mapping[:, 1]],
    #         y_batch_indices[self._mapping[:, 0]]
    #     )
    #
    # def get_occupied_mapped_data(self, y, y_other=None, **kwargs) -> tuple:
    #     if y_other is None:
    #         raise ValueError('y_other not provided.')
    #     if self._occupied_mapping is None:
    #         raise ValueError('Occupied mapping not created.')
    #     (y_values, y_batch_indices), (y_values_other, _) = y, y_other
    #     return (
    #         y_values[self._occupied_mapping[:, 0]],
    #         y_values_other[self._occupied_mapping[:, 1]],
    #         y_batch_indices[self._occupied_mapping[:, 0]]
    #     )


class ChamferPointDataBuffer(MappedPointOccupancyDataBuffer):
    def __init__(self, large_val=1e9, **kwargs):
        super().__init__(**kwargs)
        if not isinstance(large_val, (float, int)) or large_val <= 0:
            raise ValueError("large_val must be a positive number.")
        self._large_val = large_val

    def match_points(self, cloud_values_1, batch_indices_1, cloud_values_2, batch_indices_2, **kwargs):
        matches = []
        batch_ids = torch.unique(batch_indices_1)
        for b in batch_ids:
            mask1 = (batch_indices_1 == b)
            mask2 = (batch_indices_2 == b)
            points1_b = cloud_values_1[mask1, :3]
            points2_b = cloud_values_2[mask2, :3]
            if points1_b.size(0) == 0 or points2_b.size(0) == 0:
                continue
            dists_b = torch.cdist(points1_b, points2_b, p=2)
            best_12_b = dists_b.argmin(dim=1)
            best_21_b = dists_b.argmin(dim=0)
            idx1_b = torch.arange(points1_b.size(0), device=points1_b.device)
            mutual_mask_b = (best_21_b[best_12_b] == idx1_b)
            final_idx1_b = idx1_b[mutual_mask_b]
            final_idx2_b = best_12_b[mutual_mask_b]
            assert mutual_mask_b.sum() == mutual_mask_b.nonzero(as_tuple=True)[0].shape[0], "Mismatch in mutual matching calculation"

            global_idx1 = mask1.nonzero(as_tuple=False).squeeze(1)[final_idx1_b]
            global_idx2 = mask2.nonzero(as_tuple=False).squeeze(1)[final_idx2_b]
            batch_matches = torch.stack((global_idx1, global_idx2), dim=1)
            matches.append(batch_matches)
            assert (batch_indices_1[global_idx1] == batch_indices_2[global_idx2]).all(), "Cross-batch contamination detected!"

        if matches:
            return torch.cat(matches, dim=0)
        else:
            return torch.empty((0, 2), dtype=torch.long, device=cloud_values_1.device)


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
        assert mutual_mask.sum() == mutual_mask.nonzero(as_tuple=True)[0].shape[0], "Mismatch in mutual matching calculation"
        final_idx1 = idx1[mutual_mask]
        final_idx2 = best_12[mutual_mask]
        assert (batch_indices_1[final_idx1] == batch_indices_2[final_idx2]).all(), "Cross-batch contamination detected!"
        return torch.stack((final_idx1, final_idx2), dim=1)
