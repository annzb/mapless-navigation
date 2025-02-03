import os.path
from pprint import pprint
from abc import ABC, abstractmethod

import torch
import wandb
torch.autograd.set_detect_anomaly(True)

from base_classes import BaseLoss, BaseMetric, RadarOccupancyModel
import metrics as metric_defs
from dataset import get_dataset, RadarDatasetGrid, RadarDataset
from loss_grid import SparseBceLoss as GridLoss
from loss_points import ChamferBceLoss as PointsLoss
from model_polar import RadarOccupancyModel2
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
        for logger in self.loggers:
            logger.log(stuff)

    def finish(self, **kwargs):
        for logger in self.loggers:
            logger.finish(**kwargs)


class ModelManager(ABC):
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
        self.metrics = (m_t(**kwargs) for m_t in self._metric_types)

    def reset_params(self, **kwargs):
        for k, v in kwargs.items():
            if v is not None:
                setattr(self, k, v)
        self.init_loss_function(**kwargs)
        self.init_metrics(**kwargs)
        self.init_optimizer(**kwargs)


def train(model, optimizer, loss_fn, train_loader, val_loader, device, num_epochs=10, occupied_only=False, save_path="best_model.pth", metrics=tuple(), scheduler=None, logger=Logger()):
    best_val_loss = float('inf')

    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")

        model.train()
        train_scores = {}
        for metric in metrics:
            name = f"train{metric.__class__.__name__}"
            train_scores[name] = {'value': 0.0, 'func': metric}
        train_loss = 0.0

        for radar_frames, lidar_frames, poses in train_loader:
            radar_frames = radar_frames.to(device)
            lidar_frames = [lidar_cloud.to(device) for lidar_cloud in lidar_frames]
            pred_probabilities = model(radar_frames)

            batch_loss = 0.0
            for pred_cloud, true_cloud in zip(pred_probabilities, lidar_frames):
                pred_cloud_filtered = filter_cloud(pred_cloud, model, occupied_only=occupied_only)
                true_cloud_filtered = filter_cloud(true_cloud, model, occupied_only=occupied_only)
                sample_loss = loss_fn(pred_cloud_filtered, true_cloud_filtered)
                # print('sample_loss', sample_loss.item())
                batch_loss = batch_loss + sample_loss
                for metric_name, metric_data in train_scores.items():
                    # print('sample', metric_name, metric_data['func'](pred_cloud_filtered, true_cloud_filtered))
                    train_scores[metric_name]['value'] += metric_data['func'](pred_cloud_filtered, true_cloud_filtered)

            train_loss += batch_loss.item()
            batch_loss = batch_loss / len(radar_frames)
            # print('batch_loss', batch_loss)

            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()
            if scheduler is not None:
                scheduler.step()

        train_loss /= len(train_loader)
        for metric_name, metric_data in train_scores.items():
            train_scores[metric_name]['value'] /= len(train_loader)
            # print('train', metric_name, train_scores[metric_name]['value'])

        # validation
        model.eval()
        val_scores = {}
        for metric in metrics:
            name = f"valid{metric.__class__.__name__}"
            val_scores[name] = {'value': 0.0, 'func': metric}
        val_loss = 0.0

        with torch.no_grad():
            for radar_frames, lidar_frames, poses in val_loader:
                radar_frames = radar_frames.to(device)
                lidar_frames = [lidar_cloud.to(device) for lidar_cloud in lidar_frames]
                pred_probabilities = model(radar_frames)

                for pred_cloud, true_cloud in zip(pred_probabilities, lidar_frames):
                    pred_cloud_filtered = filter_cloud(pred_cloud, model, occupied_only=occupied_only)
                    true_cloud_filtered = filter_cloud(true_cloud, model, occupied_only=occupied_only)
                    val_loss = val_loss + loss_fn(pred_cloud_filtered, true_cloud_filtered).item()
                    for metric_name, metric_data in val_scores.items():
                        val_scores[metric_name]['value'] += metric_data['func'](pred_cloud_filtered, true_cloud_filtered)

        val_loss /= len(val_loader)
        for metric_name, metric_data in val_scores.items():
            val_scores[metric_name]['value'] /= len(val_loader)

        # save model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), save_path)

        log = {"epoch": epoch, "train_loss": train_loss, "valid_loss": val_loss, "best_valid_loss": best_val_loss}
        for scores in (train_scores, val_scores):
            for metric_name, metric_data in scores.items():
                log[metric_name] = metric_data['value']
        logger.log(log)
        # print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Best Val Loss: {best_val_loss:.4f}")


def evaluate(model, test_loader, device, loss_fn, occupied_only=False, metrics=tuple(), logger=Logger()):
    model.eval()
    test_scores = {}
    for metric in metrics:
        name = f"test{metric.__class__.__name__}"
        test_scores[name] = {'value': 0.0, 'func': metric}
    test_loss = 0.0
    with torch.no_grad():
        for radar_frames, lidar_frames, poses in test_loader:
            radar_frames = radar_frames.to(device)
            lidar_frames = [lidar_cloud.to(device) for lidar_cloud in lidar_frames]
            pred_probabilities = model(radar_frames)

            for pred_cloud, true_cloud in zip(pred_probabilities, lidar_frames):
                pred_cloud_filtered = filter_cloud(pred_cloud, model, occupied_only=occupied_only)
                true_cloud_filtered = filter_cloud(true_cloud, model, occupied_only=occupied_only)
                test_loss = test_loss + loss_fn(pred_cloud_filtered, true_cloud_filtered).item()
                for metric_name, metric_data in test_scores.items():
                    test_scores[metric_name]['value'] += metric_data['func'](pred_cloud_filtered, true_cloud_filtered)

    test_loss /= len(test_loader)
    for metric_name, metric_data in test_scores.items():
        test_scores[metric_name]['value'] /= len(test_loader)
    # test_iou /= len(test_loader)
    # test_chamfer /= len(test_loader)
    log = {"test_loss": test_loss}
    for metric_name, metric_data in test_scores.items():
        log[metric_name] = metric_data['value']
    logger.log(log)
    # print(f"Test Loss: {test_loss:.4f}")


def get_model(radar_config, device, occupancy_threshold=0.5, grid=False):
    return Unet1C3DPolar(radar_config, device) if grid else RadarOccupancyModel2(radar_config, occupancy_threshold=occupancy_threshold)


def init_lr(total_epochs, start_lr, end_lr):
    def lr_lambda(epoch):
        return 1.0 - (epoch / total_epochs) * ((start_lr - end_lr) / start_lr)
    return lr_lambda


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

    train_loader, val_loader, test_loader, radar_config = get_dataset(dataset_path, dataset_type=RadarDatasetGrid if use_grid_data else RadarDataset, batch_size=BATCH_SIZE,  partial=dataset_part, grid_voxel_size=octomap_voxel_size)
    # model = RadarOccupancyModel2(radar_config, occupancy_threshold=OCCUPANCY_THRESHOLD)
    device = torch.device(device_name if torch.cuda.is_available() else "cpu")
    model = get_model(radar_config, device, occupancy_threshold=OCCUPANCY_THRESHOLD, grid=use_grid_data)
    print('\ndevice', device)
    model.to(device)

    # lr_lambda = init_lr(N_EPOCHS, LEARNING_RATE, LEARNING_RATE / 10)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    # scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)

    # loss_fn = SoftMatchingLossScaled(alpha=loss_spatial_weight, beta=loss_probability_weight, matching_temperature=loss_matching_temperature, distance_threshold=POINT_MATCH_RADIUS)
    loss_fn = ChamferBceLoss(spatial_weight=loss_spatial_weight, probability_weight=loss_probability_weight)
    metrics = (
        metric_defs.IoU(max_point_distance=POINT_MATCH_RADIUS, probability_threshold=OCCUPANCY_THRESHOLD),
        metric_defs.WeightedChamfer()
    )

    logger.init(
        project="radar-occupancy",
        config={
            "dataset": os.path.basename(dataset_path),
            "model": model.name,
            "learning_rate": LEARNING_RATE,
            "epochs": N_EPOCHS,
            "dataset_part": dataset_part,
            "batch_size": BATCH_SIZE,
            "occupancy_threshold": OCCUPANCY_THRESHOLD,
            "point_match_radius": POINT_MATCH_RADIUS,
            "loss": {
                "name": loss_fn.__class__.__name__,
                "spatial_weight": loss_spatial_weight,
                "probability_weight": loss_probability_weight,
                # "matching_temperature": loss_matching_temperature
            }
        }
    )
    train(model, optimizer, loss_fn, train_loader, val_loader, device, num_epochs=N_EPOCHS, save_path=model_save_path, metrics=metrics, logger=logger, occupied_only=LOSS_OVER_OCCUPIED_ONLY)
    model.load_state_dict(torch.load(model_save_path))
    evaluate(model, test_loader, device, loss_fn, metrics=metrics, logger=logger, occupied_only=LOSS_OVER_OCCUPIED_ONLY)
    logger.log({"best_model_path":  os.path.abspath(model_save_path)})
    logger.finish()


if __name__ == '__main__':
    run(use_grid_data=False, model_save_name="best_point_model.pth")
