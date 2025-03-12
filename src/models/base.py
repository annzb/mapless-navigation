import torch
import torch.nn as nn


class RadarOccupancyModel(nn.Module):
    def __init__(self, radar_config, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.radar_config = radar_config
        self.name = 'radar_occupancy_model'

    def apply_sigmoid(self, pcl_batch):
        coords = pcl_batch[..., :3]
        probs = pcl_batch[..., 3]
        probs = torch.sigmoid(probs)
        batch = torch.cat((coords, probs.unsqueeze(-1)), dim=-1)
        return batch
