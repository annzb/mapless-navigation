import torch
import torch.nn as nn

from metrics.data_buffer import OccupancyDataBuffer, PointOccupancyDataBuffer, MappedPointOccupancyDataBuffer


class BaseCriteria:
    def __init__(self, batch_size=0, **kwargs):
        super().__init__()
        self.default_value = 0.0
        for k, v in kwargs.items():
            setattr(self, k, v)
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        self._batch_size = batch_size

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
        nn.Module.__init__(self)
        BaseCriteria.__init__(self, **kwargs)
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
            if data_buffer.occupied_mask() is None:
                valid, error = False, 'Occupancy data not available in data buffer.'
        return valid, error


class PointcloudOccupancyCriteria(OccupancyCriteria):
    def _validate_input(self, y_pred, y_true, data_buffer=None, occupied_only=False, *args, **kwargs):
        valid, error = super()._validate_input(y_pred, y_true, data_buffer=data_buffer, occupied_only=occupied_only, *args, **kwargs)
        if valid:
            if data_buffer is None or not isinstance(data_buffer, (PointOccupancyDataBuffer, MappedPointOccupancyDataBuffer)):
                valid, error = False, f'Data buffer not available. Expected instance of {PointOccupancyDataBuffer.__name__} or {MappedPointOccupancyDataBuffer.__name__}, got {type(data_buffer).__name__}'
        if valid:
            if isinstance(data_buffer, MappedPointOccupancyDataBuffer):
                if data_buffer.mapped_mask() is None:
                    valid, error = False, 'Mapped masks not available in data buffer.'
                if data_buffer.occupied_mapped_mask() is None:
                    valid, error = False, 'Occupied mapped masks not available in data buffer.'
                if data_buffer.mapping() is None:
                    valid, error = False, 'Point mapping not available in data buffer.'
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
