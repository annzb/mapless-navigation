import numpy as np
import torch
torch.autograd.set_detect_anomaly(True)

from utils import get_local_params
from train_points import PointModelManager
from visualize.points import show_radar_clouds


def run():
    SHUFFLE_RUNS = True
    RANDOM_SEEED = 42
    SESSION_NAME = 'testing'
    LOSS_SPATIAL_WEIGHT = 0.1
    LOSS_PROBABILITY_WEIGHT = 1.0
    OCCUPANCY_THRESHOLD = 0.6
    EVAL_OVER_OCCUPIED_POINTS_ONLY = True
    # POINT_MATCH_RADIUS = 0.25
    NO_MATCH_DISTANCE_PENALTY=100
    LEARNING_RATE = 0.01

    local_params = get_local_params()
    mm = PointModelManager(
        shuffle_dataset_runs=SHUFFLE_RUNS, random_state=RANDOM_SEEED,
        learning_rate=LEARNING_RATE,
        occupancy_threshold=OCCUPANCY_THRESHOLD, evaluate_over_occupied_points_only=EVAL_OVER_OCCUPIED_POINTS_ONLY,
        loss_spatial_penalty=NO_MATCH_DISTANCE_PENALTY, loss_spatial_weight=LOSS_SPATIAL_WEIGHT, loss_probability_weight=LOSS_PROBABILITY_WEIGHT,
        session_name=SESSION_NAME,
        **local_params
    )

    mm.init_model(model_path='/home/arpg/projects/mapless-navigation/trained_models/17april25_testing/best_train_loss.pth')
    idx = 10

    with torch.no_grad():
        input_cloud, gt_cloud, _ = mm.train_loader.dataset[idx]
        input_tensor = torch.from_numpy(input_cloud)
        input_batch_idx = torch.as_tensor(np.zeros(input_cloud.shape[0]), dtype=torch.int64)
        embeddings_flat, embeddings_flat_indices, probs = mm.model((input_tensor, input_batch_idx), batch_size=1, debug=True)
        # pointnet_output_cloud = mm.model.pointnet((input_tensor, input_batch_idx))
        # predicted_cloud = mm.model.apply_sigmoid(pointnet_output_cloud)

    show_radar_clouds(
        clouds=[input_cloud, embeddings_flat.numpy(), probs.numpy()],
        prob_flags=[False, False, True]
    )

    mm.logger.finish()


if __name__ == '__main__':
    run()
