import torch
torch.autograd.set_detect_anomaly(True)

import metrics as metric_defs
from dataset import RadarDataset
from loss_points import ChamferBceLoss as PointsLoss
from model_polar import PointnetOccupancyModel as PointsModel
from model_manager import ModelManager
from utils import get_local_params


class PointModelManager(ModelManager):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


    def _define_types(self):
        self._dataset_type = RadarDataset
        self._model_type = PointsModel
        self._optimizer_type = torch.optim.Adam
        self._loss_type = PointsLoss
        self._metric_types = (metric_defs.IoU, metric_defs.WeightedChamfer)


def run():
    SHUFFLE_RUNS = True
    RANDOM_SEEED = 42
    SAVE_MODEL_PREFIX = "model_best_points.pth"

    LOSS_SPATIAL_WEIGHT = 0.5
    LOSS_PROBABILITY_WEIGHT = 1.0
    OCCUPANCY_THRESHOLD = 0.6
    EVAL_OVER_OCCUPIED_POINTS_ONLY = True
    POINT_MATCH_RADIUS = 0.5
    LEARNING_RATE = 0.01
    N_EPOCHS = 1

    local_params = get_local_params()

    mm = PointModelManager(
        dataset_path=local_params['dataset_path'], dataset_part=local_params['dataset_part'], batch_size=local_params['batch_size'], shuffle_dataset_runs=SHUFFLE_RUNS,
        device_name=local_params['device_name'], logger=local_params['logger'], random_state=RANDOM_SEEED, save_model_name=SAVE_MODEL_PREFIX,
        occupancy_threshold=OCCUPANCY_THRESHOLD, evaluate_over_occupied_points_only=EVAL_OVER_OCCUPIED_POINTS_ONLY,
        loss_spatial_weight=LOSS_SPATIAL_WEIGHT, loss_probability_weight=LOSS_PROBABILITY_WEIGHT, max_point_distance=POINT_MATCH_RADIUS,
        learning_rate=LEARNING_RATE, n_epochs=N_EPOCHS
    )
    mm.train()
    mm.evaluate()
    mm.logger.finish()


if __name__ == '__main__':
    run()
