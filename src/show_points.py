import numpy as np
import os
import torch

from utils import data_transforms
from utils.params import get_params
from train_points_generative import PointModelManager
from visualize.points import show_radar_clouds


def run():
    SESSION_NAME = '03june25_generative_overfit_dloss1_remote'

    params = get_params()
    mm = PointModelManager(session_name=SESSION_NAME, **params)
    model_save_directory = params['model_save_directory']
    model_path=os.path.join(model_save_directory, SESSION_NAME, 'best_train_loss.pth')
    mm.init_model(model_path=model_path)

    match_ratio_metric = mm.metrics['test'][0]
    # for idx in range(10):
    idx = 2

    with torch.no_grad():
        input_cloud, gt_cloud, _ = mm.train_loader.dataset[idx]
        input_tensor = torch.from_numpy(input_cloud)
        input_batch_idx = torch.as_tensor(np.zeros(input_cloud.shape[0]), dtype=torch.int64)
        embeddings, predicted_log_odds, probs, predicted_flat_indices = mm.model((input_tensor.to(mm.device), input_batch_idx.to(mm.device)), debug=True)
        embeddings, predicted_log_odds, probs, predicted_flat_indices = embeddings.to('cpu'), predicted_log_odds.to('cpu'), probs.to('cpu'), predicted_flat_indices.to('cpu')
        # probs_occupied = probs[probs[:, 3] >= params['loss_params']['occupancy_threshold']]
        gt_cloud_occupied = gt_cloud[gt_cloud[:, 3] >= params['loss_params']['occupancy_threshold']]

    match_ratio = match_ratio_metric._calc((probs[:, 0], probs), (gt_cloud_occupied[:, 0], torch.from_numpy(gt_cloud_occupied)), data_buffer=mm.data_buffer)
    probs = data_transforms.collapse_close_points(probs, d=0.007)
    # probs_occupied = probs[probs[:, 3] >= params['loss_params']['occupancy_threshold']]

    # print(f'GT shape {gt_cloud_occupied.shape[0]}')
    # for d in [0.007, 0.0071, 0.0072, 0.0073, 0.0074, 0.0075]:
    #     probs_reduced = data_transforms.collapse_close_points(probs, d=d)
    #     print(f'{d}: {probs_reduced.shape[0]}')

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
        clouds=[predicted_log_odds.numpy(), probs.numpy(), gt_cloud_occupied],
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
