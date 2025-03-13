from abc import abstractmethod

import torch
import torch.nn as nn


class BaseCriteria:
    def __init__(self, **kwargs):
        super().__init__()
        self.default_value = 0.0
        for k, v in kwargs.items():
            setattr(self, k, v)

    def calc(self, y_pred, y_true):
        return self.default_value

    def forward(self, y_pred, y_true):
        if len(y_pred) == 0 or len(y_true) == 0:
            return self.default_value
        return self.calc(y_pred, y_true)


class BaseMetric(BaseCriteria):
    def __init__(self, name='', **kwargs):
        super().__init__(**kwargs)
        self.total_score = 0.0
        self.best_score = 0.0
        self.name = (f'{name}_' if name else '') + self.__class__.__name__.lower()
        self._scaled = False

    def reset_epoch(self):
        if self.total_score > self.best_score:
            self.best_score = self.total_score
        self.total_score = 0.0
        self._scaled = False

    def reset(self):
        self.reset_epoch()
        self.best_score = 0.0

    def scale_score(self, n_samples):
        if self._scaled:
            raise RuntimeError(f'Metric {self.name} already scaled')
        self.total_score /= n_samples
        self._scaled = True

    def __call__(self, y_pred, y_true):
        score = self.forward(y_pred, y_true)
        self.total_score += score
        return score


class BaseLoss(BaseCriteria, nn.Module):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.default_value = torch.tensor(float('inf'), device=self.device, requires_grad=True)


class OccupancyCriteria(BaseCriteria):
    def __init__(self, occupancy_threshold=0.5, occupied_only=False, **kwargs):
        super().__init__(**kwargs)
        self.occupancy_threshold = occupancy_threshold
        self.occupied_only = occupied_only

    @abstractmethod
    def filter_occupied(self, y_pred, y_true):
        raise NotImplementedError

    def forward(self, y_pred, y_true):
        if self.occupied_only:
            y_pred, y_true = self.filter_occupied(y_pred, y_true)
        return super().forward(y_pred, y_true)


class PointcloudOccupancyCriteria(OccupancyCriteria):
    def mask_nans(self, y_pred, y_true):
        mask = ~torch.isnan(y_true).any(dim=-1)  # [B, M]
        y_true_filtered = [t[m] for t, m in zip(y_true, mask)]
        return y_pred, y_true_filtered

    def filter_occupied(self, y_pred, y_true):
        """
        Filters occupied points where occupancy probability is >= `occupancy_threshold`.

        Args:
            y_pred (Tensor): Predicted point cloud of shape (B, N, 4).
            y_true (list of Tensors): List of ground truth point clouds [(M1, 4), (M2, 4), ...]

        Returns:
            Tuple: Filtered `y_pred` (list of tensors) and `y_true` (filtered list).
        """
        y_pred_filtered = [yp[yp[:, -1] >= self.occupancy_threshold] for yp in y_pred]
        y_true_filtered = [yt[yt[:, -1] >= self.occupancy_threshold] for yt in y_true]
        return y_pred_filtered, y_true_filtered

    def forward(self, y_pred, y_true):
        y_pred, y_true = self.mask_nans(y_pred, y_true)
        return super().forward(y_pred, y_true)


class GridOccupancyCriteria(OccupancyCriteria):
    def filter_occupied(self, y_pred, y_true):
        y_pred = y_pred[y_pred[:, :, -1] >= self.occupancy_threshold]
        y_true = y_true[y_true[:, :, -1] >= self.occupancy_threshold]
        return y_pred, y_true


class PointcloudOccupancyMetric(BaseMetric, PointcloudOccupancyCriteria): pass
class PointcloudOccupancyLoss(BaseLoss, PointcloudOccupancyCriteria): pass
class GridOccupancyMetric(BaseMetric, GridOccupancyCriteria): pass
class GridOccupancyLoss(BaseLoss, GridOccupancyCriteria): pass
