from abc import abstractmethod

import torch
import torch.nn as nn


class BaseLoss(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        for k, v in kwargs.items():
            setattr(self, k, v)

    @abstractmethod
    def forward(self, y_pred, y_true):
        return 0


class BaseMetric:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

    @abstractmethod
    def __call__(self, y_pred, y_true):
        return 0


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
