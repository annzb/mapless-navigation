import torch
torch.autograd.set_detect_anomaly(True)

from metrics import metrics as metric_defs
from utils.dataset import RadarDataset
from metrics import PointLoss as PointLoss, ChamferPointDataBuffer as PointDataBuffer
from models import EncoderPointnet as PointModel
from model_manager import ModelManager
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
            metric_defs.OccupancyRatio,
        #     metric_defs.IoU,
        #     metric_defs.ChamferDistance,
        #     metric_defs.Precision,
        #     metric_defs.Recall,
        #     metric_defs.F1,
        #     metric_defs.OccupancyMSE,
        #     metric_defs.AUROC,
        #     metric_defs.AUPRC
        )


def run():
    SHUFFLE_RUNS = True
    RANDOM_SEEED = 42
    SESSION_NAME = 'multiencoder_pointnet_chamfer'
    LOSS_SPATIAL_WEIGHT = 1.0
    LOSS_PROBABILITY_WEIGHT = 1.0
    UNMATCHED_WEIGHT = 0.01
    OCCUPANCY_THRESHOLD = 0.6
    EVAL_OVER_OCCUPIED_POINTS_ONLY = True
    # POINT_MATCH_RADIUS = 0.25
    NO_MATCH_DISTANCE_PENALTY=10
    LEARNING_RATE = 0.01

    local_params = get_local_params()

    mm = PointModelManager(
        shuffle_dataset_runs=SHUFFLE_RUNS, random_state=RANDOM_SEEED,
        learning_rate=LEARNING_RATE,
        occupancy_threshold=OCCUPANCY_THRESHOLD, evaluate_over_occupied_points_only=EVAL_OVER_OCCUPIED_POINTS_ONLY,
        loss_spatial_penalty=NO_MATCH_DISTANCE_PENALTY, loss_spatial_weight=LOSS_SPATIAL_WEIGHT, loss_probability_weight=LOSS_PROBABILITY_WEIGHT,
        session_name=SESSION_NAME, loss_unmatched_pred_weight=UNMATCHED_WEIGHT, loss_unmatched_true_weight=UNMATCHED_WEIGHT,
        **local_params
    )
    mm.train()
    mm.evaluate()
    mm.logger.finish()


if __name__ == '__main__':
    run()
