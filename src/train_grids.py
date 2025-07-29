import torch
torch.autograd.set_detect_anomaly(True)

from metrics import metrics as metric_defs
from utils.dataset import GridRadarDataset
from metrics import DistanceLoss as PointLoss, GridOccupancyDataBuffer as GridDataBuffer
from models import GenerativeBaseline as PointModel
from src.training.model_manager import ModelManager
from utils.params import get_params


class GridModelManager(ModelManager):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _define_types(self):
        self._dataset_type = GridRadarDataset
        self._model_type = PointModel
        self._optimizer_type = torch.optim.Adam
        self._data_buffer_type = GridDataBuffer
        self._loss_type = PointLoss
        self._metric_types = (
            metric_defs.MatchedPointRatio,
            metric_defs.DistanceLossFpFnMetric,
            metric_defs.DistanceLossFpMetric,
            metric_defs.DistanceLossFnMetric
            # metric_defs.OccupancyLossMetric,
            # metric_defs.SpatialLossMetric,
            # metric_defs.UnmatchedLossMetric,
            # metric_defs.UnmatchedLossFpFnMetric,
            # metric_defs.UnmatchedLossFnMetric,
            # metric_defs.UnmatchedLossFpMetric
        )


def run():
    SESSION_NAME = 'generative_overfit'

    params = get_params()
    params['n_epochs'], params['batch_size'] = 1000, 1
    
    # params['loss_params']['spatial_weight'] = 1.0
    # params['loss_params']['occupancy_weight'] = 1.0
    # params['loss_params']['unmatched_weight'] = 1.0
    params['loss_params']['fn_fp_weight'] = 1.0
    params['loss_params']['fn_weight'] = 1.0
    params['loss_params']['fp_weight'] = 1.0

    mm = GridModelManager(session_name=SESSION_NAME, **params)
    mm.train()
    mm.evaluate()
    mm.logger.finish()


if __name__ == '__main__':
    run()
