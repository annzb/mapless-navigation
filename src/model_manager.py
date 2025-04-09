import os.path
from abc import ABC, abstractmethod

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
            loss_spatial_penalty=1, loss_spatial_weight=1.0, loss_probability_weight=1.0,
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
        self.spatial_penalty = loss_spatial_penalty
        self.spatial_weight = loss_spatial_weight
        self.probability_weight = loss_probability_weight
        self.learning_rate = learning_rate
        self.logger = logger

        self._define_types()
        self._init_device(device_name)

        self.init_data_buffer(occupancy_threshold=self.occupancy_threshold)
        self.train_loader, self.val_loader, self.test_loader, self.radar_config = get_dataset(
            dataset_file_path=dataset_path, dataset_type=self._dataset_type,
            batch_size=batch_size, partial=dataset_part, shuffle_runs=shuffle_dataset_runs,
            grid_voxel_size=grid_voxel_size, random_state=random_state,
            occupied_only=self.occupied_only, occupancy_threshold=self.occupancy_threshold,
            data_buffer=self.data_buffer, device=self.device
        )
        self.init_model()
        self.init_loss_function(
            batch_size=batch_size,
            occupancy_threshold=self.occupancy_threshold,
            occupied_only=self.occupied_only,
            unmatched_point_spatial_penalty=loss_spatial_penalty,
            spatial_weight=self.spatial_weight,
            probability_weight=self.probability_weight,
            device=self.device
        )
        self.init_metrics(
            batch_size=batch_size,
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
                    "point_mapping_method": self.data_buffer.__class__.__name__,
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
        self._data_buffer_type = OccupancyDataBuffer
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

    def init_data_buffer(self, **kwargs):
        self.data_buffer = self._data_buffer_type(**kwargs)

    def init_loss_function(self, **kwargs):
        self.loss_fn = self._loss_type(point_mapper=self.data_buffer, **kwargs)

    def init_optimizer(self, learning_rate=None, **kwargs):
        self.optimizer = self._optimizer_type(self.model.parameters(), lr=learning_rate)

    def init_metrics(self, **kwargs):
        self.metrics = {}
        for mode in self._modes:
            self.metrics[mode] = tuple(m_t(name=mode, point_mapper=self.data_buffer, **kwargs) for m_t in self._metric_types)

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

    def _evaluate_current_model(self, mode, data_loader):
        eval_loss = 0.0
        self.model.eval()
        self.reset_metrics_epoch(mode=mode)

        with torch.no_grad():
            for radar_frames, (lidar_frames, lidar_frame_indices), poses in data_loader:
                radar_frames = radar_frames.to(self.device)
                lidar_frames = lidar_frames.to(self.device)
                lidar_frame_indices = lidar_frame_indices.to(self.device)
                pred_frames, pred_indices = self.model(radar_frames)

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

        for radar_frames, (lidar_frames, lidar_frame_indices), poses in data_loader:
            radar_frames = radar_frames.to(self.device)
            lidar_frames = lidar_frames.to(self.device)
            lidar_frame_indices = lidar_frame_indices.to(self.device)
            pred_frames, pred_indices = self.model(radar_frames)

            self.data_buffer.create_masks(y=(pred_frames, pred_indices), y_other=(lidar_frames, lidar_frame_indices))
            batch_loss = self.loss_fn(y_pred=(pred_frames, pred_indices), y_true=(lidar_frames, lidar_frame_indices), data_buffer=self.data_buffer)
            self.apply_metrics(y_pred=(pred_frames, pred_indices), y_true=(lidar_frames, lidar_frame_indices), data_buffer=self.data_buffer, mode=mode)

            epoch_loss += batch_loss.item()
            batch_loss = batch_loss / len(radar_frames)
            torch.cuda.synchronize()

            self.optimizer.zero_grad(set_to_none=True)
            if torch.isnan(batch_loss).any() or torch.isinf(batch_loss).any():
                raise RuntimeError("Detected NaN or Inf in batch_loss before backward!")
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    try:
                        grads = torch.autograd.grad(batch_loss, param, retain_graph=True, allow_unused=True)
                        print(f"{name}: grad computed? {grads[0] is not None}")
                    except Exception as e:
                        print(f"{name}: Error computing grad: {e}")
            batch_loss.backward()
            self.optimizer.step()
            for name, param in self.model.named_parameters():
                if param.grad is None:
                    print(f"Warning: {name} has no gradient!")
                elif torch.isnan(param.grad).any():
                    raise RuntimeError(f"NaN detected in gradient of {name}!")

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
        best_train_loss, best_val_loss = float('inf'), float('inf')
        self.reset_metrics(mode='train')
        self.reset_metrics(mode='val')

        for epoch in range(n_epochs):
            self.logger.log(f"Epoch {epoch + 1}/{n_epochs}")
            train_epoch_loss = self._train_epoch()
            val_epoch_loss = self._evaluate_current_model(mode = 'val', data_loader=self.val_loader)

            if train_epoch_loss < best_train_loss:
                best_train_loss = train_epoch_loss
                self._save_model(save_model_name.replace('.pth', f'_train_loss_epoch{epoch}.pth'))

            if val_epoch_loss < best_val_loss:
                best_val_loss = val_epoch_loss
                self._save_model(save_model_name.replace('.pth', f'_val_loss_epoch{epoch}.pth'))

            for mode in 'train', 'val':
                for metric in self.metrics[mode]:
                    # print(metric.name, metric.total_score, metric.best_score)
                    if metric.total_score >= metric.best_score:
                        self._save_model(save_model_name.replace('.pth', f'_{metric.name}_epoch{epoch}.pth'))

            log = {
                "epoch": epoch,
                "train_loss": train_epoch_loss, "best_train_loss": best_train_loss,
                "valid_loss": val_epoch_loss, "best_valid_loss": best_val_loss
            }
            log.update(self.report_metrics(mode='train'))
            log.update(self.report_metrics(mode='val'))
            self.logger.log(log)


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
