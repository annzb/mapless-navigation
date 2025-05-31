import os.path
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Dict

import numpy as np
import torch

torch.autograd.set_detect_anomaly(True)

from metrics import BaseLoss, BaseMetric, OccupancyDataBuffer
from models import RadarOccupancyModel
from utils import Logger
from utils.dataset import get_dataset


class ModelManager(ABC):

    _modes = 'train', 'val', 'test'

    def __init__(
            self, 
            model_save_directory='.',
            session_name=None,
            device_name='cpu',
            logger=Logger(),
            random_seed=1,

            n_epochs=10,
            batch_size=1,

            dataset_params: Dict[str, Any] = {},
            loss_params: Dict[str, Any] = {},
            metric_params: Dict[str, Any] = {},
            optimizer_params: Dict[str, Any] = {},

            # overridable params
            # ...
            # loss
            # loss_spatial_weight=1.0, loss_probability_weight=1.0,
            # loss_unmatched_pred_weight=1.0, loss_unmatched_true_weight=1.0,
            # # loss, metrics
            # occupancy_threshold=0.5, evaluate_over_occupied_points_only=False,
            # # metrics
            # max_point_distance=1.0,
            # # optimizer
            # learning_rate=0.01,
            # train loop
            
            **kwargs
    ):
        self.logger = logger
        self.random_seed, self.n_epochs, self.batch_size = random_seed, n_epochs, batch_size
        self._define_types()
        self._init_device(device_name)

        self.init_data_buffer(**loss_params)
        self.train_loader, self.val_loader, self.test_loader, self.radar_config = get_dataset(
            dataset_type=self._dataset_type, data_buffer=self.data_buffer, 
            logger=self.logger, device=self.device, random_seed=random_seed, batch_size=self.batch_size, 
            **dataset_params
            # dataset_file_path=dataset_path,
            # partial=dataset_part, shuffle_runs=shuffle_dataset_runs,
            # grid_voxel_size=grid_voxel_size, random_state=random_state,
            # intensity_threshold=radar_point_intensity_threshold
        )
        self.init_model()
        self.init_loss_function(device=self.device, batch_size=batch_size, **loss_params
            # occupied_only=evaluate_over_occupied_points_only,
            # max_point_distance=max_point_distance,
            # spatial_weight=loss_spatial_weight,
            # probability_weight=loss_probability_weight,
            # unmatched_pred_weight=loss_unmatched_pred_weight,
            # unmatched_true_weight=loss_unmatched_true_weight,
            # grid_resolution=grid_voxel_size
        )
        self.init_metrics(batch_size=batch_size, **metric_params
            # occupied_only=evaluate_over_occupied_points_only,
            # max_point_distance=max_point_distance
        )
        self.init_optimizer(**optimizer_params)  # learning_rate=learning_rate)

        self.session_name = datetime.now().strftime("%d%B%y").lower()  # current date as DDmonthYY
        if session_name:
            session_name = f'{self.session_name}_{session_name}'
        self.model_save_directory = os.path.join(model_save_directory, session_name)
        os.makedirs(self.model_save_directory, exist_ok=True)
        
        self.logger.init(
            project="radar-occupancy",
            config={
                "model": self.model.name,
                'dataset': dataset_params,
                'training_params': {
                    "epochs": n_epochs,
                    "batch_size": batch_size
                },
                'optimizer': optimizer_params,
                "loss": {
                    "name": self.loss_fn.__class__.__name__,
                    "point_mapping_method": self.data_buffer.__class__.__name__,
                    **loss_params
                },
                'metrics': metric_params
            }
        )
        self._saved_models = set()

    @abstractmethod
    def _define_types(self):
        self._dataset_type = lambda *args, **kwargs: object
        self._model_type = RadarOccupancyModel
        self._optimizer_type = lambda *args, **kwargs: object
        self._data_buffer_type = OccupancyDataBuffer
        self._loss_type = BaseLoss
        self._metric_types = (BaseMetric, )
        raise NotImplementedError()

    def _init_device(self, device_name):
        self.device = torch.device(device_name)
        print('Using device:', device_name)

    def init_model(self, model_path=None):
        model = self._model_type(radar_config=self.radar_config, batch_size=self.batch_size)
        model.to(self.device)
        if model_path is not None:
            model.load_state_dict(torch.load(model_path, map_location=self.device))
        model.eval()
        self.model = model

    def init_data_buffer(self, **kwargs):
        self.data_buffer = self._data_buffer_type(**kwargs)

    def init_loss_function(self, **kwargs):
        self.loss_fn = self._loss_type(**kwargs)

    def init_optimizer(self, learning_rate=None, **kwargs):
        self.optimizer = self._optimizer_type(self.model.parameters(), lr=learning_rate)

    def init_metrics(self, **kwargs):
        self.metrics = {}
        for mode in self._modes:
            self.metrics[mode] = tuple(m_t(name=mode, **kwargs) for m_t in self._metric_types)

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

    def apply_metrics(self, y_pred, y_true, data_buffer, mode=None):
        if mode and mode not in self._modes:
            raise ValueError(f'Invalid mode: {mode}, expected one of {self._modes}')
        metrics = self.metrics[mode] if mode else np.concatenate(list(self.metrics.values()), axis=0)
        for metric in metrics:
            metric(y_pred=y_pred, y_true=y_true, data_buffer=data_buffer)

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

    def _evaluate_current_model(self, mode, data_loader):
        eval_loss = 0.0
        self.model.eval()
        self.reset_metrics_epoch(mode=mode)

        with torch.no_grad():
            for (radar_frames, radar_frame_indices), (lidar_frames, lidar_frame_indices), poses in data_loader:
                radar_frames = radar_frames.to(self.device)
                radar_frame_indices = radar_frame_indices.to(self.device)
                lidar_frames = lidar_frames.to(self.device)
                lidar_frame_indices = lidar_frame_indices.to(self.device)
                pred_frames, pred_indices = self.model((radar_frames, radar_frame_indices))

                self.data_buffer.create_masks(y=(pred_frames, pred_indices), y_other=(lidar_frames, lidar_frame_indices))
                batch_loss = self.loss_fn(y_pred=(pred_frames, pred_indices), y_true=(lidar_frames, lidar_frame_indices), data_buffer=self.data_buffer)
                self.apply_metrics(y_pred=(pred_frames, pred_indices), y_true=(lidar_frames, lidar_frame_indices), data_buffer=self.data_buffer, mode=mode)

                eval_loss += batch_loss.item()

            eval_loss /= len(data_loader)
            self.scale_metrics(n_samples=len(data_loader), mode=mode)

        return eval_loss

    def _train_epoch(self):
        mode = 'train'
        data_loader = self.train_loader
        epoch_loss = 0.0
        self.model.train()
        self.reset_metrics_epoch(mode=mode)

        for (radar_frames, radar_frame_indices), (lidar_frames, lidar_frame_indices), poses in data_loader:
            radar_frames = radar_frames.to(self.device)
            radar_frame_indices = radar_frame_indices.to(self.device)
            lidar_frames = lidar_frames.to(self.device)
            lidar_frame_indices = lidar_frame_indices.to(self.device)
            pred_frames, pred_indices = self.model((radar_frames, radar_frame_indices))

            self.data_buffer.create_masks(y=(pred_frames, pred_indices), y_other=(lidar_frames, lidar_frame_indices))
            batch_loss = self.loss_fn(y_pred=(pred_frames, pred_indices), y_true=(lidar_frames, lidar_frame_indices), data_buffer=self.data_buffer)
            self.apply_metrics(y_pred=(pred_frames, pred_indices), y_true=(lidar_frames, lidar_frame_indices), data_buffer=self.data_buffer, mode=mode)

            epoch_loss += batch_loss.item()
            batch_loss = batch_loss / len(radar_frames)

            self.optimizer.zero_grad(set_to_none=True)
            if torch.isnan(batch_loss).any() or torch.isinf(batch_loss).any():
                raise RuntimeError("Detected NaN or Inf in batch_loss before backward!")
            batch_loss.backward()
            self.optimizer.step()
            for name, param in self.model.named_parameters():
                if param.grad is None:
                    raise RuntimeError(f"Warning: {name} has no gradient!")
                elif torch.isnan(param.grad).any():
                    raise RuntimeError(f"NaN detected in gradient of {name}!")

        epoch_loss /= len(data_loader)
        self.scale_metrics(n_samples=len(data_loader), mode=mode)
        return epoch_loss

    def _save_model(self, filename):
        if not filename.endswith('.pth'):
            filename += '.pth'
        save_path = os.path.join(self.model_save_directory, filename)
        torch.save(self.model.state_dict(), save_path)
        self._saved_models.add(save_path)
        self.logger.log(f'Saved model as {save_path}')

    def train(self, n_epochs=None):
        n_epochs = n_epochs or self.n_epochs
        best_train_loss, best_val_loss = float('inf'), float('inf')
        self.reset_metrics(mode='train')
        self.reset_metrics(mode='val')

        for epoch in range(n_epochs):
            self.logger.log(f"Epoch {epoch + 1}/{n_epochs}")
            train_epoch_loss = self._train_epoch()
            val_epoch_loss = self._evaluate_current_model(mode = 'val', data_loader=self.val_loader)

            if train_epoch_loss < best_train_loss:
                best_train_loss = train_epoch_loss
                self._save_model('best_train_loss')

            if val_epoch_loss < best_val_loss:
                best_val_loss = val_epoch_loss
                self._save_model('best_val_loss')

            for mode in 'train', 'val':
                for metric in self.metrics[mode]:
                    if metric.total_score >= metric.best_score:
                        self._save_model(f'best_{metric.name}')

            log = {
                "epoch": epoch,
                "train_loss": train_epoch_loss, "best_train_loss": best_train_loss,
                "valid_loss": val_epoch_loss, "best_valid_loss": best_val_loss
            }
            log.update(self.report_metrics(mode='train'))
            log.update(self.report_metrics(mode='val'))
            self.logger.log(log)

        self._save_model('last_epoch')

    def evaluate(self):
        mode = 'test'
        for model_path in self._saved_models:
            self.logger.log(f'Evaluating model {os.path.basename(model_path)}')
            self.init_model(model_path)
            self.reset_metrics(mode=mode)
            model_test_loss = self._evaluate_current_model(mode=mode, data_loader=self.test_loader)

            log = {"model_path": model_path, "test_loss": model_test_loss}
            log.update(self.report_metrics(mode=mode))
            self.logger.log(log)
