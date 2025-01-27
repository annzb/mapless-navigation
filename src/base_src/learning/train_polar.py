import os.path
from pprint import pprint

import torch
import wandb
torch.autograd.set_detect_anomaly(True)

import metrics as metric_defs
from dataset import get_dataset
from loss_spatial_prob import SoftMatchingLossScaled
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


def train(model, optimizer, loss_fn, train_loader, val_loader, device, num_epochs=10, save_path="best_model.pth", metrics=tuple(), scheduler=None, logger=Logger()):
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
                pred_cloud_filtered = model.filter_probs(pred_cloud)
                true_cloud_filtered = model.filter_probs(true_cloud)
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
                    pred_cloud_filtered = model.filter_probs(pred_cloud)
                    true_cloud_filtered = model.filter_probs(true_cloud)
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


def evaluate(model, test_loader, device, loss_fn, metrics=tuple(), logger=Logger()):
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
                pred_cloud_filtered = model.filter_probs(pred_cloud)
                true_cloud_filtered = model.filter_probs(true_cloud)
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


def get_model(radar_config, occupancy_threshold=0.5, grid=False):
    return Unet1C3DPolar(radar_config) if grid else RadarOccupancyModel2(radar_config, occupancy_threshold=occupancy_threshold)


def init_lr(total_epochs, start_lr, end_lr):
    def lr_lambda(epoch):
        return 1.0 - (epoch / total_epochs) * ((start_lr - end_lr) / start_lr)
    return lr_lambda


def run(use_grid_data=False, octomap_voxel_size=0.25, model_save_name="best_model.pth"):
    OCCUPANCY_THRESHOLD = 0.6
    POINT_MATCH_RADIUS = 0.5
    BATCH_SIZE = 4
    N_EPOCHS = 100
    LEARNING_RATE = 0.01

    loss_spatial_weight = 0.5
    loss_probability_weight = 1.0
    loss_matching_temperature = 0.2
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

    train_loader, val_loader, test_loader, radar_config = get_dataset(dataset_path, batch_size=BATCH_SIZE,  partial=dataset_part, occupancy_threshold=OCCUPANCY_THRESHOLD, grid=use_grid_data, grid_voxel_size=octomap_voxel_size)
    # model = RadarOccupancyModel2(radar_config, occupancy_threshold=OCCUPANCY_THRESHOLD)
    model = get_model(radar_config, occupancy_threshold=OCCUPANCY_THRESHOLD, grid=use_grid_data)
    device = torch.device(device_name if torch.cuda.is_available() else "cpu")
    print('\ndevice', device)
    model.to(device)

    # lr_lambda = init_lr(N_EPOCHS, LEARNING_RATE, LEARNING_RATE / 10)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    # scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)

    loss_fn = SoftMatchingLossScaled(alpha=loss_spatial_weight, beta=loss_probability_weight, matching_temperature=loss_matching_temperature, distance_threshold=POINT_MATCH_RADIUS)
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
                "matching_temperature": loss_matching_temperature
            }
        }
    )
    train(model, optimizer, loss_fn, train_loader, val_loader, device, num_epochs=N_EPOCHS, save_path=model_save_path, metrics=metrics) # , scheduler=scheduler)
    model.load_state_dict(torch.load(model_save_path))
    evaluate(model, test_loader, device, loss_fn, metrics=metrics)
    logger.log({"best_model_path":  os.path.abspath(model_save_path)})
    logger.finish()


if __name__ == '__main__':
    run(use_grid_data=False, model_save_name="best_point_model.pth")
