import numpy as np
import os
import torch
torch.autograd.set_detect_anomaly(True)

from utils import get_local_params
from train_points import PointModelManager
from visualize.points import show_radar_clouds


def run():
    SHUFFLE_RUNS = True
    RANDOM_SEEED = 42
    SESSION_NAME = '26may25_generative_baseline'
    LOSS_SPATIAL_WEIGHT = 0.1
    LOSS_PROBABILITY_WEIGHT = 1.0
    OCCUPANCY_THRESHOLD = 0.6
    EVAL_OVER_OCCUPIED_POINTS_ONLY = True
    # POINT_MATCH_RADIUS = 0.25
    NO_MATCH_DISTANCE_PENALTY=100
    LEARNING_RATE = 0.01
    RADAR_POINT_INTENSITY_THRESHOLD = 5000

    local_params = get_local_params()
    mm = PointModelManager(
        radar_point_intensity_threshold=RADAR_POINT_INTENSITY_THRESHOLD,
        shuffle_dataset_runs=SHUFFLE_RUNS, random_state=RANDOM_SEEED,
        learning_rate=LEARNING_RATE,
        occupancy_threshold=OCCUPANCY_THRESHOLD, evaluate_over_occupied_points_only=EVAL_OVER_OCCUPIED_POINTS_ONLY,
        loss_spatial_penalty=NO_MATCH_DISTANCE_PENALTY, loss_spatial_weight=LOSS_SPATIAL_WEIGHT, loss_probability_weight=LOSS_PROBABILITY_WEIGHT,
        session_name=SESSION_NAME,
        **local_params
    )
    model_save_directory = local_params['model_save_directory']

    # model_path='/home/arpg/projects/mapless-navigation/trained_models/29april25_multiencoder_pointnet_chamfer/last_epoch.pth'
    model_path=os.path.join(model_save_directory, SESSION_NAME, 'last_epoch.pth')
    # model_path='/Users/anna/data/coloradar/models/06may25_encoder_pointnet_chamfer_2/best_val_occupancyratio.pth'
    mm.init_model(model_path=model_path)
    idx = 50

    with torch.no_grad():
        input_cloud, gt_cloud, _ = mm.train_loader.dataset[idx]
        input_tensor = torch.from_numpy(input_cloud)
        input_batch_idx = torch.as_tensor(np.zeros(input_cloud.shape[0]), dtype=torch.int64)
        embeddings, predicted_log_odds, probs, predicted_flat_indices = mm.model((input_tensor.to('mps'), input_batch_idx.to('mps')), debug=True)
        embeddings, predicted_log_odds, probs, predicted_flat_indices = embeddings.to('cpu'), predicted_log_odds.to('cpu'), probs.to('cpu'), predicted_flat_indices.to('cpu')
        probs = probs.numpy()
        probs_occupied = probs[probs[:, 3] >= OCCUPANCY_THRESHOLD]
        gt_cloud_occupied = gt_cloud[gt_cloud[:, 3] >= OCCUPANCY_THRESHOLD]

    # title = 'Best Valid Occupancy Ratio Model'
    title = 'Best Valid Loss Model'
    # show_radar_clouds(
    #     clouds=[input_cloud, embeddings[0].numpy(), predicted_log_odds[0].numpy(), probs, gt_cloud, probs_occupied, gt_cloud_occupied],
    #     prob_flags=[False, False, False, True, True, True, True],
    #     window_name=title
    # )
    # show_radar_clouds(
    #     clouds=[probs_occupied, gt_cloud_occupied],
    #     prob_flags=[True, True],
    #     window_name=title
    # )
    # show_radar_clouds(
    #     clouds=[input_cloud, embeddings[0].numpy(), predicted_log_odds[0].numpy(), probs, gt_cloud],
    #     prob_flags=[False, False, False, True, True],
    #     titles=[],
    #     window_name=title
    # )
    # show_radar_clouds(
    #     clouds=[predicted_log_odds[0].numpy(), probs],
    #     prob_flags=[False, True],
    #     titles=['predicted_log_odds', 'probs'],
    #     window_name=title
    # )
    # show_radar_clouds(
    #     clouds=[input_cloud, gt_cloud],
    #     prob_flags=[False, True],
    #     titles=['input_cloud', 'gt_cloud'],
    #     window_name=title
    # )
    show_radar_clouds(
        clouds=[predicted_log_odds[0].numpy(), probs, gt_cloud_occupied],
        prob_flags=[False, True, True],
        titles=['predicted_log_odds', 'probs', 'gt_cloud_occupied'],
        window_name=title
    )
    # show_radar_clouds(
    #     clouds=[embeddings[0].numpy(), predicted_log_odds[0].numpy(), probs],
    #     prob_flags=[False, False, True],
    #     titles=['embeddings', 'predicted_log_odds', 'probs'],
    #     window_name=title
    # )

    mm.logger.finish()


if __name__ == '__main__':
    run()
