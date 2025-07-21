import torch
from typing import Optional
torch.autograd.set_detect_anomaly(True)

from metrics import metrics as metric_defs
from utils.dataset import RadarDataset
from metrics import DistanceLoss as PointLoss, ChamferPointDataBuffer as PointDataBuffer
from models import GenerativeBaseline as PointModel
from model_manager import ModelManager
from utils.params import get_params


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
            metric_defs.DbscanRecall,
            metric_defs.DbscanPrecision,
            metric_defs.DbscanF1,
            metric_defs.DbscanPurity,
            metric_defs.DbscanReduction,
            metric_defs.DistanceLossFpFnMetric,
            metric_defs.DistanceLossFpMetric,
            metric_defs.DistanceLossFnMetric
        )


def run(
        training_params: Optional[dict] = None, 
        model_params: Optional[dict] = None,
        dataset_params: Optional[dict] = None, 
        optimizer_params: Optional[dict] = None, 
        loss_params: Optional[dict] = None
    ):
    params = get_params()
    if training_params:
        params['training_params'].update(training_params)
    if model_params:
        params['model_params'].update(model_params)
    if dataset_params:
        params['dataset_params'].update(dataset_params)
    if optimizer_params:
        params['optimizer_params'].update(optimizer_params)

    if loss_params:
        params['loss_params'].update(loss_params)
    else:
        params['loss_params']['fn_fp_weight'] = 1.0
        params['loss_params']['fn_weight'] = 1.0
        params['loss_params']['fp_weight'] = 1.0

    mm = PointModelManager(**params)
    mm.train()
    mm.evaluate()
    mm.logger.finish()


if __name__ == '__main__':
    run()
