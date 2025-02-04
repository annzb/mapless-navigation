import os.path
from pprint import pprint
from abc import ABC, abstractmethod

import numpy as np
import torch
import wandb
torch.autograd.set_detect_anomaly(True)

from base_classes import BaseLoss, BaseMetric, RadarOccupancyModel
import metrics as metric_defs
from dataset import get_dataset, RadarDatasetGrid, RadarDataset
from loss_grid import SparseBceLoss as GridLoss
from loss_points import ChamferBceLoss as PointsLoss
from model_polar import PointnetOccupancyModel as PointsModel
from model_unet import Unet1C3DPolar
from torch.optim.lr_scheduler import LambdaLR


class Logger:
    def __init__(self, print_log=True, loggers=tuple()):
        self.print_log = print_log
        self.loggers = loggers

    def init(self, **kwargs):
        for logger in self.loggers:
            logger.init(**kwargs)

    def log(self, stuff):
        if self.print_log:
            pprint(stuff)
        if isinstance(stuff, dict):
            for logger in self.loggers:
                logger.log(stuff)

    def finish(self, **kwargs):
        for logger in self.loggers:
            logger.finish(**kwargs)


class ModelManager(ABC):

    _modes = 'train', 'val', 'test'

    def __init__(
            self, dataset_path, *args, # data_mode,
            # static params
            grid_voxel_size=1.0,
            batch_size=4,
            dataset_part=1.0,
            shuffle_dataset_runs=True,
            device_name='cpu',
            logger=Logger(),
            random_state=42,

            # overridable params
            # ...
            # loss
            loss_spatial_weight=1.0, loss_probability_weight=1.0,
            # loss, metrics
            occupancy_threshold=0.5, evaluate_over_occupied_points_only=False,
            # metrics
            max_point_distance=1.0,
            # optimizer
            learning_rate=0.01,
            # train loop
             n_epochs=10, save_model_name="model",

            **kwargs

    ):
        self.occupancy_threshold = occupancy_threshold
        self.occupied_only = evaluate_over_occupied_points_only
        self.max_point_distance = max_point_distance
        self.spatial_weight = loss_spatial_weight
        self.probability_weight = loss_probability_weight
        self.learning_rate = learning_rate
        self.logger = logger

        self._define_types()
        self._init_device(device_name)

        self.train_loader, self.val_loader, self.test_loader, self.radar_config = get_dataset(
            dataset_file_path=dataset_path, dataset_type=self._dataset_type,
            batch_size=batch_size, partial=dataset_part, shuffle_runs=shuffle_dataset_runs,
            grid_voxel_size=grid_voxel_size, random_state=random_state
        )
        self.init_model()
        self.init_loss_function(
            occupancy_threshold=self.occupancy_threshold,
            occupied_only=self.occupied_only,
            spatial_weight=self.spatial_weight,
            probability_weight=self.probability_weight
        )
        self.init_metrics(
            occupancy_threshold=self.occupancy_threshold,
            occupied_only=self.occupied_only,
            max_point_distance=self.max_point_distance
        )
        self.init_optimizer(learning_rate=self.learning_rate)

        if self.occupied_only:
            self._filter_cloud = lambda cloud: cloud[cloud[:, 3] >= self.occupancy_threshold]
        else:
            self._filter_cloud = lambda cloud: cloud

        self.n_epochs = n_epochs
        self.save_model_name = save_model_name

        self.logger.init(
            project="radar-occupancy",
            config={
                "dataset": os.path.basename(dataset_path),
                "model": self.model.name,
                "learning_rate": self.learning_rate,
                "epochs": self.n_epochs,
                "dataset_part": dataset_part,
                "batch_size": batch_size,
                "occupancy_threshold": self.occupancy_threshold,
                "point_match_radius": self.max_point_distance,
                "loss": {
                    "name": self.loss_fn.__class__.__name__,
                    "spatial_weight": loss_spatial_weight,
                    "probability_weight": loss_probability_weight
                }
            }
        )
        self._saved_models = set()

    @abstractmethod
    def _define_types(self):
        self._dataset_type = lambda *args, **kwargs: object
        self._model_type = RadarOccupancyModel
        self._optimizer_type = lambda *args, **kwargs: object
        self._loss_type = BaseLoss
        self._metric_types = (BaseMetric, )
        raise NotImplementedError()

    def _init_device(self, device_name):
        self.device = torch.device(device_name)
        print('Using device:', device_name)

    def init_model(self, model_path=None):
        model = self._model_type(self.radar_config)
        model.to(self.device)
        if model_path is not None:
            model.load_state_dict(torch.load(model_path, map_location=self.device))
            model.eval()
        self.model = model

    def init_loss_function(self, **kwargs):
        self.loss_fn = self._loss_type(**kwargs)

    def init_optimizer(self, learning_rate=None, **kwargs):
        self.optimizer = self._optimizer_type(self.model.parameters(), learning_rate=learning_rate)

    def init_metrics(self, **kwargs):
        self.metrics = {}
        for mode in self._modes:
            self.metrics[mode] = (m_t(name=mode, **kwargs) for m_t in self._metric_types)

    def report_metrics(self, mode=None):
        if mode and mode not in self._modes:
            raise ValueError(f'Invalid mode: {mode}, expected one of {self._modes}')
        metrics = self.metrics[mode] if mode else np.concatenate(list(self.metrics.values()), axis=0)
        report = {}
        for metric in metrics:
            report[metric.name] = metric.total_score
            if mode != 'test':
                report[f'best_{metric.name}'] = metric.best_score
        return report

    def apply_metrics(self, y_true, y_pred, mode=None):
        if mode and mode not in self._modes:
            raise ValueError(f'Invalid mode: {mode}, expected one of {self._modes}')
        metrics = self.metrics[mode] if mode else np.concatenate(list(self.metrics.values()), axis=0)
        for metric in metrics:
            metric(y_true, y_pred)

    def scale_metrics(self, n_samples, mode=None):
        if mode and mode not in self._modes:
            raise ValueError(f'Invalid mode: {mode}, expected one of {self._modes}')
        metrics = self.metrics[mode] if mode else np.concatenate(list(self.metrics.values()), axis=0)
        for metric in metrics:
            metric.scale_score(n_samples)

    def reset_metrics_epoch(self, mode=None):
        if mode and mode not in self._modes:
            raise ValueError(f'Invalid mode: {mode}, expected one of {self._modes}')
        metrics = self.metrics[mode] if mode else np.concatenate(list(self.metrics.values()), axis=0)
        for metric in metrics:
            metric.reset_epoch()

    def reset_metrics(self, mode=None):
        if mode and mode not in self._modes:
            raise ValueError(f'Invalid mode: {mode}, expected one of {self._modes}')
        metrics = self.metrics[mode] if mode else np.concatenate(list(self.metrics.values()), axis=0)
        for metric in metrics:
            metric.reset()

    def reset_params(self, **kwargs):
        for k, v in kwargs.items():
            if v is not None:
                setattr(self, k, v)
        self.init_loss_function(
            occupancy_threshold=self.occupancy_threshold,
            occupied_only=self.occupied_only,
            spatial_weight=self.spatial_weight,
            probability_weight=self.probability_weight
        )
        self.init_metrics(
            occupancy_threshold=self.occupancy_threshold,
            occupied_only=self.occupied_only,
            max_point_distance=self.max_point_distance
        )
        self.init_optimizer(learning_rate=self.learning_rate)

    def _train_epoch(self):
        mode = 'train'
        data_loader = self.train_loader
        epoch_loss = 0.0
        self.model.train()
        self.reset_metrics_epoch(mode=mode)

        for radar_frames, lidar_frames, poses in data_loader:
            radar_frames = radar_frames.to(self.device)
            pred_probabilities = self.model(radar_frames)

            batch_loss = 0.0
            for pred_cloud, true_cloud in zip(pred_probabilities, lidar_frames):
                true_cloud.to(self.device)
                pred_cloud_filtered = self._filter_cloud(pred_cloud)
                true_cloud_filtered = self._filter_cloud(true_cloud)
                sample_loss = self.loss_fn(pred_cloud_filtered, true_cloud_filtered)
                batch_loss = batch_loss + sample_loss
                self.apply_metrics(true_cloud_filtered, pred_cloud_filtered, mode=mode)

            epoch_loss += batch_loss.item()
            batch_loss = batch_loss / len(radar_frames)

            self.optimizer.zero_grad()
            batch_loss.backward()
            self.optimizer.step()

        epoch_loss /= len(data_loader)
        self.scale_metrics(n_samples=len(data_loader), mode=mode)
        return epoch_loss

    def _validate_epoch(self):
        mode = 'val'
        data_loader = self.val_loader
        epoch_loss = 0.0
        self.model.eval()
        self.reset_metrics_epoch(mode=mode)

        with torch.no_grad():
            for radar_frames, lidar_frames, poses in data_loader:
                radar_frames = radar_frames.to(self.device)
                pred_probabilities = self.model(radar_frames)

                batch_loss = 0.0
                for pred_cloud, true_cloud in zip(pred_probabilities, lidar_frames):
                    true_cloud.to(self.device)
                    pred_cloud_filtered = self._filter_cloud(pred_cloud)
                    true_cloud_filtered = self._filter_cloud(true_cloud)
                    sample_loss = self.loss_fn(pred_cloud_filtered, true_cloud_filtered)
                    batch_loss = batch_loss + sample_loss
                    self.apply_metrics(true_cloud_filtered, pred_cloud_filtered, mode=mode)
                epoch_loss += batch_loss.item()

            epoch_loss /= len(data_loader)
            self.scale_metrics(n_samples=len(data_loader), mode=mode)

        return epoch_loss

    def _save_model(self, path):
        torch.save(self.model.state_dict(), path)
        self._saved_models.add(path)
        self.logger.log(f'Saved model to {path}')

    def train(self, n_epochs=None, save_model_name=None):
        n_epochs = n_epochs or self.n_epochs
        save_model_name = save_model_name or self.save_model_name
        best_train_loss, best_val_loss = float('inf')
        self.reset_metrics(mode='train')
        self.reset_metrics(mode='val')

        for epoch in range(n_epochs):
            self.logger.log(f"Epoch {epoch + 1}/{n_epochs}")
            train_epoch_loss = self._train_epoch()
            val_epoch_loss = self._validate_epoch()

            if train_epoch_loss < best_train_loss:
                best_train_loss = train_epoch_loss
                self._save_model(save_model_name.replace('.pth', '_train_loss.pth'))
            if val_epoch_loss < best_val_loss:
                best_val_loss = val_epoch_loss
                self._save_model(save_model_name.replace('.pth', '_val_loss.pth'))
            for mode in 'train', 'val':
                for metric in self.metrics[mode]:
                    if metric.total_score > metric.best_score:
                        self._save_model(save_model_name.replace('.pth', f'_{metric.name}.pth'))

            log = {
                "epoch": epoch,
                "train_loss": train_epoch_loss, "best_train_loss": best_train_loss,
                "valid_loss": val_epoch_loss, "best_valid_loss": best_val_loss
            }
            log.update(self.report_metrics(mode='train'))
            log.update(self.report_metrics(mode='val'))
            self.logger.log(log)

    def _evaluate_current_model(self):
        mode = 'test'
        data_loader = self.test_loader
        test_loss = 0.0
        self.model.eval()
        self.reset_metrics(mode=mode)

        with torch.no_grad():
            for radar_frames, lidar_frames, poses in data_loader:
                radar_frames = radar_frames.to(self.device)
                pred_probabilities = self.model(radar_frames)

                batch_loss = 0.0
                for pred_cloud, true_cloud in zip(pred_probabilities, lidar_frames):
                    true_cloud.to(self.device)
                    pred_cloud_filtered = self._filter_cloud(pred_cloud)
                    true_cloud_filtered = self._filter_cloud(true_cloud)
                    sample_loss = self.loss_fn(pred_cloud_filtered, true_cloud_filtered)
                    batch_loss = batch_loss + sample_loss
                    self.apply_metrics(true_cloud_filtered, pred_cloud_filtered, mode=mode)
                test_loss += batch_loss.item()

            test_loss /= len(data_loader)
            self.scale_metrics(n_samples=len(data_loader), mode=mode)

        return test_loss


    def evaluate(self):
        for model_path in self._saved_models:
            self.init_model(model_path)
            model_test_loss = self._evaluate_current_model()

            log = {"model_path": model_path, "test_loss": model_test_loss}
            log.update(self.report_metrics(mode='test'))
            self.logger.log(log)


class PointModelManager(ModelManager):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


    def _define_types(self):
        self._dataset_type = RadarDataset
        self._model_type = PointsModel
        self._optimizer_type = torch.optim.Adam
        self._loss_type = PointsLoss
        self._metric_types = (metric_defs.IoU, metric_defs.WeightedChamfer)



# def init_lr(total_epochs, start_lr, end_lr):
#     def lr_lambda(epoch):
#         return 1.0 - (epoch / total_epochs) * ((start_lr - end_lr) / start_lr)
#     return lr_lambda


def run(use_grid_data=False, octomap_voxel_size=0.25, model_save_name="best_model.pth"):
    OCCUPANCY_THRESHOLD = 0.6
    POINT_MATCH_RADIUS = 0.5
    BATCH_SIZE = 8
    N_EPOCHS = 100
    LEARNING_RATE = 0.01
    LOSS_OVER_OCCUPIED_ONLY = False

    loss_spatial_weight = 1.0
    loss_probability_weight = 1.0
    # loss_matching_temperature = 0.2
    model_save_path = model_save_name

    if os.path.isdir('/media/giantdrive'):
        dataset_path = '/media/giantdrive/coloradar/dataset2.h5'
        device_name = 'cuda:0' if use_grid_data else 'cuda:1'
        dataset_part = 1.0
        logger = Logger(print_log=True, loggers=(wandb, ))
    else:
        dataset_path = '/home/arpg/projects/coloradar_plus_processing_tools/dataset2.h5'
        device_name = 'cpu'
        dataset_part = 0.05
        logger = Logger(print_log=True)

    mm = PointModelManager(dataset_path=dataset_path, device_name=device_name, logger=logger)
    mm.train()
    mm.evaluate()
    mm.logger.finish()


if __name__ == '__main__':
    run(use_grid_data=False, model_save_name="best_point_model.pth")
