import numpy as np
import os
import torch
from sklearn.cluster import DBSCAN
from tqdm import tqdm

from utils.params import get_params
from train_points_generative import PointModelManager
from visualize.points import show_radar_clouds





def run():
    model_path = '/home/arpg/projects/mapless-navigation/trained_models/28jul25_w3q0/best_val_loss.pth'

    params = get_params()
    params['device_name'] = 'cpu'
    # params['model_params']['encoder_batch_norm'] = True
    # params['model_params']['decoder_layer_norm'] = True
    # params['model_params']['encoder_dropout'] = 0.2
    # params['model_params']['decoder_dropout'] = 0.2
    # params['dataset_params']['partial'] = 1
    # params['loss_params']['occupancy_threshold'] = 0.8

    mm = PointModelManager(**params)
    mm.init_model(model_path=model_path)

    # n_empty_gt_clouds = count_empty_gt_clouds(mm, threshold=0.8) # threshold=params['loss_params']['occupancy_threshold'])
    # print(f'Number of samples: {len(mm.train_loader.dataset)}, number of empty gt clouds: {n_empty_gt_clouds}')
    # best_sample_idx, best_metric_value = find_best_sample(mm, metric_name='loss', loss=True)
    # print(f'Best sample idx: {best_sample_idx}, best loss: {best_metric_value}')
    sample_idx = 0

    input_cloud_np, true_cloud_np, _ = mm.train_loader.dataset[sample_idx]
    y_pred_tensor, report = mm.evaluate_batch(input_cloud_np, true_cloud_np)
    pred_cloud_np = y_pred_tensor[0].cpu().numpy()
    pred_cloud_np_occupied = pred_cloud_np[pred_cloud_np[:, 3] >= params['loss_params']['occupancy_threshold']]
    true_cloud_np_occupied = true_cloud_np[true_cloud_np[:, 3] >= params['loss_params']['occupancy_threshold']]

    for metric_name, metric_value in report.items():
        metric_name_short = metric_name.split('_')[-1]
        print(f'{metric_name_short}: {metric_value}')


    # pred_cloud_reduced = probs # data_transforms.collapse_close_points(probs, d=0.007)
    # y_pred, y_true = (pred_cloud_reduced, torch.zeros(pred_cloud_reduced.shape[0])), (torch.from_numpy(gt_cloud_occupied), torch.zeros(gt_cloud_occupied.shape[0]))
    # merged_cloud = merge_pred_gt(y_pred, y_true)[0]
    # test_distance = 0.05
    # merged_cloud_reduced = data_transforms.collapse_close_points(merged_cloud, d=test_distance)

    # mm.data_buffer._same_point_distance_limit = test_distance
    # mm.data_buffer.create_masks(y_pred, y_true)
    # recall = recall_metric._calc(y_pred, y_true, data_buffer=mm.data_buffer)
    # precision = precision_metric._calc(y_pred, y_true, data_buffer=mm.data_buffer)


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
        clouds=[pred_cloud_np, true_cloud_np, pred_cloud_np_occupied, true_cloud_np_occupied],
        prob_flags=[True, True, True, True],
        titles=['Predicted Probabilities', 'GT Probabilities', 'Predicted Occupied', 'GT Occupied'],
        window_name=title
    )
    # show_radar_clouds(
    #     clouds=[embeddings[0].numpy(), predicted_log_odds.numpy(), probs],
    #     prob_flags=[False, False, True],
    #     titles=['embeddings', 'predicted_log_odds', 'probs'],
    #     window_name=title
    # )

    # With manual F1
    # show_radar_clouds(
    #     clouds=[probs.numpy(), pred_cloud_reduced, gt_cloud_occupied, merged_cloud, merged_cloud_reduced],
    #     prob_flags=[True, True, True, True, True],
    #     titles=['probs', 'pred_cloud_reduced', 'gt_cloud_occupied', 'merged_cloud', 'merged_cloud_reduced'],
    #     window_name=title
    # )

    mm.logger.finish()


if __name__ == '__main__':
    run()
