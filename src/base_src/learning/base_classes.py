from abc import abstractmethod

import torch
import torch.nn as nn


class BaseCriteria:
    def __init__(self, **kwargs):
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
    def __call__(self, y_pred, y_true):
        return self.forward(y_pred, y_true)


class BaseLoss(BaseCriteria, nn.Module):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.default_value = float('inf')

    def forward(self, y_pred, y_true):
        return torch.tensor(super().forward(y_pred, y_true), requires_grad=True)


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


class RadarOccupancyModel(nn.Module):
    def __init__(self, radar_config, *args, **kwargs):
        super().__init__()
        self.radar_config = radar_config

    def apply_sigmoid(self, pcl_batch):
        coords = pcl_batch[..., :3]
        probs = pcl_batch[..., 3]
        probs = torch.sigmoid(probs)
        batch = torch.cat((coords, probs.unsqueeze(-1)), dim=-1)
        return batch
