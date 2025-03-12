import torch
torch.autograd.set_detect_anomaly(True)

from dataset import RadarDatasetGrid
from metrics.loss_grid import SparseBceLoss as GridLoss
from models import GridOccupancyModel as GridModel
from model_manager import ModelManager
from utils import get_local_params


class PointModelManager(ModelManager):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


    def _define_types(self):
        self._dataset_type = RadarDatasetGrid
        self._model_type = GridModel
        self._optimizer_type = torch.optim.Adam
        self._loss_type = GridLoss
        self._metric_types = ()


def run():
    SHUFFLE_RUNS = True
    RANDOM_SEEED = 42
    SAVE_MODEL_PREFIX = "model_best_grid.pth"

    GRID_RESOLUTION = 0.25
    OCCUPANCY_THRESHOLD = 0.6
    EVAL_OVER_OCCUPIED_POINTS_ONLY = True
    LEARNING_RATE = 0.01
    N_EPOCHS = 100

    local_params = get_local_params()

    mm = PointModelManager(
        dataset_path=local_params['dataset_path'], dataset_part=local_params['dataset_part'], batch_size=local_params['batch_size'], shuffle_dataset_runs=SHUFFLE_RUNS,
        device_name=local_params['device_name'], logger=local_params['logger'], random_state=RANDOM_SEEED, save_model_name=SAVE_MODEL_PREFIX,
        occupancy_threshold=OCCUPANCY_THRESHOLD, evaluate_over_occupied_points_only=EVAL_OVER_OCCUPIED_POINTS_ONLY,
        grid_voxel_size=GRID_RESOLUTION,
        learning_rate=LEARNING_RATE, n_epochs=N_EPOCHS
    )
    mm.train()
    mm.evaluate()
    mm.logger.finish()


if __name__ == '__main__':
    run()

