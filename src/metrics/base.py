from abc import abstractmethod

import torch
import torch.nn as nn

from metrics.data_buffer import OccupancyDataBuffer, PointOccupancyDataBuffer


class BaseCriteria:
    def __init__(self, **kwargs):
        super().__init__()
        self.default_value = 0.0
        for k, v in kwargs.items():
            setattr(self, k, v)

    def _validate_input(self, y_pred, y_true, *args, **kwargs):
        return len(y_pred) != 0 and len(y_true) != 0

    def _calc(self, y_pred, y_true, *args, **kwargs):
        return self.default_value

    def forward(self, y_pred, y_true, *args, **kwargs):
        if self._validate_input(y_pred, y_true, *args, **kwargs):
            return self.default_value
        return self._calc(y_pred, y_true, *args, **kwargs)


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

    def __call__(self, y_pred, y_true, *args, **kwargs):
        score = self.forward(y_pred, y_true, *args, **kwargs)
        self.total_score += score
        return score


class BaseLoss(BaseCriteria, nn.Module):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.default_value = torch.tensor(float('inf'), device=self.device, requires_grad=True)


class OccupancyCriteria(BaseCriteria):
    def __init__(self, data_buffer=None, **kwargs):
        super().__init__(**kwargs)
        self._data_buffer = data_buffer
        if data_buffer is None or not isinstance(data_buffer, OccupancyDataBuffer):
            raise TypeError('data_buffer must be an instance of OccupancyDataBuffer')

    def _calc(self, y_pred, y_true, *args, **kwargs):
        value = super()._calc(y_pred, y_true, *args, **kwargs)
        if self._data_buffer.occupied_data() is None:
            raise ValueError('Occupancy data not available')
        return value


class PointcloudOccupancyCriteria(OccupancyCriteria):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if not isinstance(self._data_buffer, PointOccupancyDataBuffer):
            raise TypeError('data_buffer must be an instance of PointOccupancyDataBuffer')

    def _calc(self, y_pred, y_true, *args, **kwargs):
        value = super()._calc(y_pred, y_true, *args, **kwargs)
        if self._data_buffer.mapped_clouds() is None:
            raise ValueError('Mapped clouds not available')
        return value


class GridOccupancyCriteria(OccupancyCriteria): pass
    # def filter_occupied(self, y_pred, y_true, *args, **kwargs):
    #     y_pred = y_pred[y_pred[:, :, -1] >= self.occupancy_threshold]
    #     y_true = y_true[y_true[:, :, -1] >= self.occupancy_threshold]
    #     return y_pred, y_true


class PointcloudOccupancyMetric(BaseMetric, PointcloudOccupancyCriteria): pass
class PointcloudOccupancyLoss(BaseLoss, PointcloudOccupancyCriteria): pass
class GridOccupancyMetric(BaseMetric, GridOccupancyCriteria): pass
class GridOccupancyLoss(BaseLoss, GridOccupancyCriteria): pass
