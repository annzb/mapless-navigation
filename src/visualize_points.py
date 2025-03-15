import os
import torch

from utils import get_local_params
from train_points import PointModelManager
from visualize.points import show_radar_pcl


def run():
    MODEL_PATH = ''

    SHUFFLE_RUNS = True
    RANDOM_SEEED = 42
    SAVE_MODEL_PREFIX = "model_best_points.pth"
    LOSS_SPATIAL_WEIGHT = 0.5
    LOSS_PROBABILITY_WEIGHT = 1.0
    OCCUPANCY_THRESHOLD = 0.6
    EVAL_OVER_OCCUPIED_POINTS_ONLY = True
    POINT_MATCH_RADIUS = 0.25
    LEARNING_RATE = 0.01

    local_params = get_local_params()
    mm = PointModelManager(
        dataset_path=local_params['dataset_path'], dataset_part=local_params['dataset_part'], batch_size=local_params['batch_size'], shuffle_dataset_runs=SHUFFLE_RUNS,
        device_name=local_params['device_name'], logger=local_params['logger'], random_state=RANDOM_SEEED, save_model_name=SAVE_MODEL_PREFIX,
        occupancy_threshold=OCCUPANCY_THRESHOLD, evaluate_over_occupied_points_only=EVAL_OVER_OCCUPIED_POINTS_ONLY,
        loss_spatial_weight=LOSS_SPATIAL_WEIGHT, loss_probability_weight=LOSS_PROBABILITY_WEIGHT, max_point_distance=POINT_MATCH_RADIUS,
        learning_rate=LEARNING_RATE, n_epochs=local_params['n_epochs']
    )
    if MODEL_PATH:
        mm.init_model(MODEL_PATH)

    with torch.no_grad():
        for radar_frames, lidar_frames, _ in mm.train_loader:
            radar_frames = radar_frames.to(mm.device)
            radar_points = mm.model.polar_to_cartesian(radar_frames)
            show_radar_pcl(radar_points[0].numpy())
            break

    mm.logger.finish()


if __name__ == '__main__':
    run()
