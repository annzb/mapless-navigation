import numpy as np
import os
import torch
torch.autograd.set_detect_anomaly(True)

from utils.params import get_params
from train_points_generative import PointModelManager
from visualize.points import show_radar_clouds


def run():
    SESSION_NAME = '03june25_generative_overfit'

    params = get_params()
    mm = PointModelManager(session_name=SESSION_NAME, **params)
    model_save_directory = params['model_save_directory']
    model_path=os.path.join(model_save_directory, SESSION_NAME, 'best_train_loss.pth')
    mm.init_model(model_path=model_path)
    idx = 2

    with torch.no_grad():
        input_cloud, gt_cloud, _ = mm.train_loader.dataset[idx]
        input_tensor = torch.from_numpy(input_cloud)
        input_batch_idx = torch.as_tensor(np.zeros(input_cloud.shape[0]), dtype=torch.int64)
        embeddings, predicted_log_odds, probs, predicted_flat_indices = mm.model((input_tensor.to(mm.device), input_batch_idx.to(mm.device)), debug=True)
        embeddings, predicted_log_odds, probs, predicted_flat_indices = embeddings.to('cpu'), predicted_log_odds.to('cpu'), probs.to('cpu'), predicted_flat_indices.to('cpu')
        probs = probs.numpy()
        probs_occupied = probs[probs[:, 3] >= params['loss_params']['occupancy_threshold']]
        gt_cloud_occupied = gt_cloud[gt_cloud[:, 3] >= params['loss_params']['occupancy_threshold']]

    # title = 'Best Valid Occupancy Ratio Model'
    title = 'Best Train Loss Model'
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
        clouds=[predicted_log_odds.numpy(), probs, gt_cloud_occupied],
        prob_flags=[False, True, True],
        titles=['predicted_log_odds', 'probs', 'gt_cloud_occupied'],
        window_name=title
    )
    # show_radar_clouds(
    #     clouds=[embeddings[0].numpy(), predicted_log_odds.numpy(), probs],
    #     prob_flags=[False, False, True],
    #     titles=['embeddings', 'predicted_log_odds', 'probs'],
    #     window_name=title
    # )

    mm.logger.finish()


if __name__ == '__main__':
    run()
