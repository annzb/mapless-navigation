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
        return len(y_pred) != 0 and len(y_true) != 0, 'Empty inputs.'

    def _calc(self, y_pred, y_true, *args, **kwargs):
        return self.default_value

    def forward(self, y_pred, y_true, *args, **kwargs):
        valid, error = self._validate_input(y_pred, y_true, *args, **kwargs)
        if valid:
            return self._calc(y_pred, y_true, *args, **kwargs)
        raise RuntimeError(error)


class BaseLoss(BaseCriteria, nn.Module):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.default_value = torch.tensor(float('inf'), device=self.device, requires_grad=True)


class BaseMetric(BaseCriteria):
    def __init__(self, name='', **kwargs):
        super().__init__(**kwargs)
        self.total_score = 0.0
        self.best_score = 0.0
        self.name = (f'{name}_' if name else '') + self.__class__.__name__.lower()
        self._scaled = False

    def reset_epoch(self):
        # if self.total_score > self.best_score:
        #     self.best_score = self.total_score
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
        if self.total_score > self.best_score:
            self.best_score = self.total_score

    def __call__(self, y_pred, y_true, *args, **kwargs):
        score = self.forward(y_pred, y_true, *args, **kwargs)
        self.total_score += score
        return score


class OccupancyCriteria(BaseCriteria):
    def _validate_input(self, y_pred, y_true, data_buffer=None, *args, **kwargs):
        valid, error = super()._validate_input(y_pred, y_true, data_buffer=data_buffer, *args, **kwargs)
        if valid:
            if data_buffer is None or not isinstance(data_buffer, OccupancyDataBuffer):
                valid, error = False, f'Data buffer not available. Expected instance of {OccupancyDataBuffer.__name__}, got {type(data_buffer).__name__}'
        if valid:
            if data_buffer.occupied_only() and (data_buffer.occupied_data() is None or data_buffer.occupied_masks() is None):
                valid, error = False, 'Occupancy data not available in data buffer.'
        return valid, error


class PointcloudOccupancyCriteria(OccupancyCriteria):
    def _validate_input(self, y_pred, y_true, data_buffer=None, *args, **kwargs):
        valid, error = super()._validate_input(y_pred, y_true, data_buffer=data_buffer, *args, **kwargs)
        if valid:
            if data_buffer is None or not isinstance(data_buffer, PointOccupancyDataBuffer):
                valid, error = False, f'Data buffer not available. Expected instance of {PointOccupancyDataBuffer.__name__}, got {type(data_buffer).__name__}'
        if valid:
            if data_buffer.mapped_clouds() is None or data_buffer.mapped_masks() is None:
                valid, error = False, 'Mapped clouds not available in data buffer.'
        if valid:
            if data_buffer.occupied_mapped_clouds() is None or data_buffer.occupied_mapped_masks() is None:
                valid, error = False, 'Occupied mapped clouds not available in data buffer.'
        return valid, error


class GridOccupancyCriteria(OccupancyCriteria): pass
    # def filter_occupied(self, y_pred, y_true, *args, **kwargs):
    #     y_pred = y_pred[y_pred[:, :, -1] >= self.occupancy_threshold]
    #     y_true = y_true[y_true[:, :, -1] >= self.occupancy_threshold]
    #     return y_pred, y_true


class PointcloudOccupancyMetric(BaseMetric, PointcloudOccupancyCriteria): pass
class PointcloudOccupancyLoss(BaseLoss, PointcloudOccupancyCriteria): pass
class GridOccupancyMetric(BaseMetric, GridOccupancyCriteria): pass
class GridOccupancyLoss(BaseLoss, GridOccupancyCriteria): pass
