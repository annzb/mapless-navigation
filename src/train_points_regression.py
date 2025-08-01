import torch
torch.autograd.set_detect_anomaly(True)

from metrics import metrics as metric_defs
from utils.dataset import RadarDataset
from metrics import PointLoss2 as PointLoss, ChamferPointDataBuffer as PointDataBuffer
from models import RegressionBaseline as PointModel
from src.training.model_manager import ModelManager
from utils import get_local_params


class PointModelManager(ModelManager):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _define_types(self):
        self._dataset_type = RadarDataset
        self._model_type = PointModel
        self._optimizer_type = torch.optim.Adam
        self._data_buffer_type = PointDataBuffer
        self._loss_type = PointLoss
        self._metric_types = (
            metric_defs.MatchedPointRatio,
            metric_defs.NegativeOccupancyLoss,
            metric_defs.NegativeSpatialLoss,
            metric_defs.NegativeUnmatchedLoss
        )


def run():
    SHUFFLE_RUNS = True
    RANDOM_SEEED = 42
    SESSION_NAME = 'regression_baseline'
    LOSS_SPATIAL_WEIGHT = 1.0
    LOSS_PROBABILITY_WEIGHT = 1.0
    UNMATCHED_WEIGHT = 10.0
    OCCUPANCY_THRESHOLD = 0.6
    EVAL_OVER_OCCUPIED_POINTS_ONLY = True
    MAX_POINT_DISTANCE = 2
    LEARNING_RATE = 0.001
    RADAR_POINT_INTENSITY_THRESHOLD = 0

    local_params = get_local_params()

    mm = PointModelManager(
        max_point_distance=MAX_POINT_DISTANCE,
        radar_point_intensity_threshold=RADAR_POINT_INTENSITY_THRESHOLD,
        shuffle_dataset_runs=SHUFFLE_RUNS, random_state=RANDOM_SEEED,
        learning_rate=LEARNING_RATE,
        occupancy_threshold=OCCUPANCY_THRESHOLD, evaluate_over_occupied_points_only=EVAL_OVER_OCCUPIED_POINTS_ONLY,
        loss_spatial_weight=LOSS_SPATIAL_WEIGHT, loss_probability_weight=LOSS_PROBABILITY_WEIGHT,
        session_name=SESSION_NAME, loss_unmatched_weight=UNMATCHED_WEIGHT,
        **local_params
    )
    mm.train()
    mm.evaluate()
    mm.logger.finish()


if __name__ == '__main__':
    run()
