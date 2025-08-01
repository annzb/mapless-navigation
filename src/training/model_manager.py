import os.path
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Union

import numpy as np
import torch

torch.autograd.set_detect_anomaly(True)

from data import get_dataset
from metrics import BaseLoss, BaseMetric, OccupancyDataBuffer
from models import RadarOccupancyModel
from utils import Logger, param_validation as validate


class ModelManager(ABC):

    _modes = 'train', 'val', 'test'

    def __init__(
            self, 
            host_name='unspecified',
            model_save_directory='.',
            device_name='cpu',
            logger=Logger(),
            random_seed=1,

            training_params: Dict[str, Any] = {},
            dataset_params: Dict[str, Any] = {},
            model_params: Dict[str, Any] = {},
            loss_params: Dict[str, Any] = {},
            metric_params: Dict[str, Any] = {},
            optimizer_params: Dict[str, Any] = {},
            
            **kwargs
    ):
        self.logger, self.random_seed = logger, random_seed
        training_params = self._validate_training_params(training_params)
        self.n_epochs, self.checkpoint_interval, self.batch_size = training_params['n_epochs'], training_params['checkpoint_interval'], training_params['batch_size']

        self._define_types()
        self._init_device(device_name)
        self.model_params = model_params

        self.init_data_buffer(**loss_params, **metric_params)
        self.train_loader, self.val_loader, self.test_loader, self.radar_config = get_dataset(
            dataset_type=self._dataset_type, occupancy_threshold=self.data_buffer.occupancy_threshold(),
            logger=self.logger, device=self.device, random_seed=random_seed, batch_size=self.batch_size, 
            **dataset_params
        )
        self.init_model()
        self.init_loss_function(device=self.device, batch_size=self.batch_size, **loss_params)
        self.init_metrics(device=self.device, batch_size=self.batch_size, **loss_params, **metric_params)
        self.init_optimizer(**optimizer_params)
 
        loss_params['loss_type'] = self.loss_fn.__class__.__name__
        loss_params['data_buffer_type'] = self.data_buffer.__class__.__name__
        self.logger.init(
            project="radar-occupancy",
            config={
                "host": host_name,
                **loss_params,
                **training_params,
                **model_params,
                **dataset_params,
                **optimizer_params
            }
        )

        if model_save_directory is None:
            self.model_save_directory = None
        else:
            self.model_save_directory = os.path.join(model_save_directory, self.logger.run_name())
            if os.path.exists(self.model_save_directory):
                self.model_save_directory += "-new"
            os.makedirs(self.model_save_directory, exist_ok=False)
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
    
    def _validate_training_params(self, params: Dict[str, Any]):
        if 'n_epochs' in params:
            validate.validate_positive_int(params['n_epochs'], 'n_epochs')
        if 'checkpoint_interval' in params:
            validate.validate_positive_int(params['checkpoint_interval'], 'checkpoint_interval')
        if 'batch_size' in params:
            validate.validate_positive_int(params['batch_size'], 'batch_size')
        return params
    
    def _init_device(self, device_name):
        self.device = torch.device(device_name)
        print('Using device:', device_name)

    def init_model(self, model_path=None):
        model = self._model_type(radar_config=self.radar_config, batch_size=self.batch_size, **self.model_params)
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

                eval_loss += batch_loss.detach().item()

            eval_loss /= len(data_loader)
            self.scale_metrics(n_samples=len(data_loader), mode=mode)

        return eval_loss

    def _train_epoch(self):
        mode = 'train'
        data_loader = self.train_loader
        epoch_loss, epoch_grad_norm = 0.0, 0.0
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

            epoch_loss += batch_loss.detach().item()
            batch_loss = batch_loss / len(radar_frames)

            self.optimizer.zero_grad(set_to_none=True)
            if torch.isnan(batch_loss).any() or torch.isinf(batch_loss).any():
                raise RuntimeError("Detected NaN or Inf in batch_loss before backward!")
            
            batch_loss.backward()
            grad_norm = 0.0
            for name, param in self.model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    grad_norm += param.grad.norm().item()
            epoch_grad_norm += grad_norm

            self.optimizer.step()
            for name, param in self.model.named_parameters():
                if param.grad is None:
                    raise RuntimeError(f"Warning: {name} has no gradient!")
                elif torch.isnan(param.grad).any():
                    raise RuntimeError(f"NaN detected in gradient of {name}!")

        epoch_loss /= len(data_loader)
        epoch_grad_norm /= len(data_loader)
        self.scale_metrics(n_samples=len(data_loader), mode=mode)
        return epoch_loss, epoch_grad_norm

    def _save_model(self, filename):
        if self.model_save_directory is None:
            self.logger.log(f'Skipping model save for {filename}')
            return

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
            train_epoch_loss, train_grad_norm = self._train_epoch()
            val_epoch_loss = self._evaluate_current_model(mode = 'val', data_loader=self.val_loader)

            if epoch % self.checkpoint_interval == 0:
                self._save_model(f'epoch_{epoch + 1}')

            if train_epoch_loss < best_train_loss:
                best_train_loss = train_epoch_loss
                self._save_model('best_train_loss')

            if val_epoch_loss < best_val_loss:
                best_val_loss = val_epoch_loss
                self._save_model('best_val_loss')

            for mode in 'train', 'val':
                for metric in self.metrics[mode]:
                    if (metric.total_score < metric.best_score) if metric.negative else (metric.total_score > metric.best_score):
                        self._save_model(f'best_{metric.name}')

            log = {
                "epoch": epoch, 'grad_norm': train_grad_norm,
                "train_loss": train_epoch_loss, "best_train_loss": best_train_loss,
                "val_loss": val_epoch_loss, "best_val_loss": best_val_loss
            }
            log.update(self.report_metrics(mode='train'))
            log.update(self.report_metrics(mode='val'))
            self.logger.log(log)

        self._save_model('last_epoch')

    def evaluate(self):
        mode = 'test'
        for model_path in self._saved_models:
            if os.path.basename(model_path).startswith('epoch_'):
                continue
            self.logger.log(f'Evaluating model {os.path.basename(model_path)}')
            self.init_model(model_path)
            self.reset_metrics(mode=mode)
            model_test_loss = self._evaluate_current_model(mode=mode, data_loader=self.test_loader)

            log = {"model_path": model_path, "test_loss": model_test_loss}
            log.update(self.report_metrics(mode=mode))
            self.logger.log(log)
    
    def predict(self, radar_frame: Union[torch.Tensor, np.ndarray], radar_batch_indices: Optional[Union[torch.Tensor, np.ndarray]] = None):
        radar_frame = validate.any_to_tensor(radar_frame, device=self.device)
        if radar_batch_indices is None:
            radar_batch_indices = torch.zeros(radar_frame.shape[0], dtype=torch.int64, device=self.device)
        else:
            radar_batch_indices = validate.any_to_tensor(radar_batch_indices, dtype=torch.int64, device=self.device)

        self.model.eval()
        with torch.no_grad():
            pred_frames, pred_indices = self.model((radar_frame, radar_batch_indices))

        if isinstance(radar_frame, np.ndarray):
            pred_frames, pred_indices = pred_frames.cpu().numpy(), pred_indices.cpu().numpy()
        return pred_frames, pred_indices
        
    def evaluate_batch(
            self, 
            radar_batch: Union[torch.Tensor, np.ndarray], 
            lidar_batch: Union[torch.Tensor, np.ndarray], 
            radar_batch_indices: Optional[Union[torch.Tensor, np.ndarray]] = None,
            lidar_batch_indices: Optional[Union[torch.Tensor, np.ndarray]] = None
    ):
        radar_batch = validate.any_to_tensor(radar_batch, device=self.device)
        lidar_batch = validate.any_to_tensor(lidar_batch, device=self.device)
        if radar_batch_indices is None:
            radar_batch_indices = torch.zeros(radar_batch.shape[0], dtype=torch.int64, device=self.device)
        else:
            radar_batch_indices = validate.any_to_tensor(radar_batch_indices, dtype=torch.int64, device=self.device)
        if lidar_batch_indices is None:
            lidar_batch_indices = torch.zeros(lidar_batch.shape[0], dtype=torch.int64, device=self.device)
        else:
            lidar_batch_indices = validate.any_to_tensor(lidar_batch_indices, dtype=torch.int64, device=self.device)

        mode = 'test'
        self.model.eval()
        self.reset_metrics_epoch(mode=mode)

        with torch.no_grad():
            pred_frames, pred_indices = self.model((radar_batch, radar_batch_indices))

        self.data_buffer.create_masks(y=(pred_frames, pred_indices), y_other=(lidar_batch, lidar_batch_indices))
        self.apply_metrics(y_pred=(pred_frames, pred_indices), y_true=(lidar_batch, lidar_batch_indices), data_buffer=self.data_buffer, mode=mode)
        report = self.report_metrics(mode=mode)
        loss = self.loss_fn(y_pred=(pred_frames, pred_indices), y_true=(lidar_batch, lidar_batch_indices), data_buffer=self.data_buffer).detach().item()
        report['loss'] = loss

        if isinstance(radar_batch, np.ndarray):
            pred_frames, pred_indices = pred_frames.cpu().numpy(), pred_indices.cpu().numpy()
        return (pred_frames, pred_indices), report
