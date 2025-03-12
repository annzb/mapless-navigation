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
        if y_pred.size(0) == 0 or y_true.size(0) == 0:
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
    def filter_occupied(self, y_pred, y_true):
        y_pred = y_pred[y_pred[:, -1] >= self.occupancy_threshold]
        y_true = y_true[y_true[:, -1] >= self.occupancy_threshold]
        return y_pred, y_true


class GridOccupancyCriteria(OccupancyCriteria):
    def filter_occupied(self, y_pred, y_true):
        y_pred = y_pred[y_pred[:, -1] >= self.occupancy_threshold]
        y_true = y_true[y_true[:, -1] >= self.occupancy_threshold]
        return y_pred, y_true


class PointcloudOccupancyMetric(BaseMetric, PointcloudOccupancyCriteria): pass
class PointcloudOccupancyLoss(BaseLoss, PointcloudOccupancyCriteria): pass
class GridOccupancyMetric(BaseMetric, GridOccupancyCriteria): pass
class GridOccupancyLoss(BaseLoss, GridOccupancyCriteria): pass
