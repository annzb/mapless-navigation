import numpy as np
import os
import torch
from sklearn.cluster import DBSCAN

from utils import data_transforms
from utils.params import get_params
from train_points_generative import PointModelManager
from visualize.points import show_radar_clouds


def merge_pred_gt(y, y_other):
    (y_cloud, y_batch_indices), (y_cloud_other, y_batch_indices_other) = y, y_other
    merged_points = []

    for b in y_batch_indices.unique():
        y_mask, y_other_mask = y_batch_indices == b, y_batch_indices_other == b
        y_batch_cloud, y_other_batch_cloud = y_cloud[y_mask], y_cloud_other[y_other_mask]
        if len(y_other_batch_cloud) == 0: 
            batch_merged_points = y_batch_cloud
        else:
            batch_merged_points = torch.cat([y_batch_cloud, y_other_batch_cloud])
        merged_points.append(batch_merged_points)

    return merged_points


# def merge_pred_gt(y, y_other, same_point_distance_limit=0.01):
#     (y_cloud, y_batch_indices), (y_cloud_other, y_batch_indices_other) = y, y_other
#     merged_points = []

#     for b in y_batch_indices.unique():
#         y_mask, y_other_mask = y_batch_indices == b, y_batch_indices_other == b
#         y_batch_cloud, y_other_batch_cloud = y_cloud[y_mask].detach().cpu().numpy(), y_cloud_other[y_other_mask].detach().cpu().numpy()

#         N_y, N_y_other = len(y_batch_cloud), len(y_other_batch_cloud)
#         if N_y_other == 0: 
#             batch_merged_points = y_batch_cloud[:, :3]
#         else:
#             batch_merged_points = np.vstack([y_batch_cloud[:, :3], y_other_batch_cloud[:, :3]])
#         merged_points.append(batch_merged_points)

#         labels = np.zeros(N_y + N_y_other, dtype=np.int32)
#         labels[N_y:] = 1
#         clustering = DBSCAN(eps=same_point_distance_limit, min_samples=1).fit(merged_points)
#         cluster_ids = clustering.labels_
#         num_clusters = cluster_ids.max() + 1
#         sample_counts = np.zeros((num_clusters, 2), dtype=np.int32)

#         for label in (0, 1):
#             label_mask = labels == label
#             cluster_ids_label = cluster_ids[label_mask]
#             bincount = np.bincount(cluster_ids_label, minlength=num_clusters)
#             sample_counts[:, label] = bincount
#         cluster_counts.append(torch.from_numpy(sample_counts).to(y_cloud.device))

#     return cluster_counts


def run():
    model_path = '/Users/anna/data/rmodels/sweep1/29_best_train_loss.pth'

    params = get_params()

    # run 29
    # params['training_params']['batch_size'] = 8
    # params['dataset_params']['partial'] = 0.005
    
    mm = PointModelManager(**params)
    mm.init_model(model_path=model_path)

    recall_metric, precision_metric, f1_metric = mm.metrics['test'][:3]
    # for idx in range(10):
    idx = 10

    with torch.no_grad():
        input_cloud, gt_cloud, _ = mm.train_loader.dataset[idx]
        input_tensor = torch.from_numpy(input_cloud)
        input_batch_idx = torch.as_tensor(np.zeros(input_cloud.shape[0]), dtype=torch.int64)
        embeddings, predicted_log_odds, probs, predicted_flat_indices = mm.model((input_tensor.to(mm.device), input_batch_idx.to(mm.device)), debug=True)
        embeddings, predicted_log_odds, probs, predicted_flat_indices = embeddings.to('cpu'), predicted_log_odds.to('cpu'), probs.to('cpu'), predicted_flat_indices.to('cpu')
        # probs_occupied = probs[probs[:, 3] >= params['loss_params']['occupancy_threshold']]
        gt_cloud_occupied = gt_cloud[gt_cloud[:, 3] >= params['loss_params']['occupancy_threshold']]

    # pred_cloud_reduced = probs # data_transforms.collapse_close_points(probs, d=0.007)
    # y_pred, y_true = (pred_cloud_reduced, torch.zeros(pred_cloud_reduced.shape[0])), (torch.from_numpy(gt_cloud_occupied), torch.zeros(gt_cloud_occupied.shape[0]))
    # merged_cloud = merge_pred_gt(y_pred, y_true)[0]
    # test_distance = 0.05
    # merged_cloud_reduced = data_transforms.collapse_close_points(merged_cloud, d=test_distance)

    # mm.data_buffer._same_point_distance_limit = test_distance
    # mm.data_buffer.create_masks(y_pred, y_true)
    # recall = recall_metric._calc(y_pred, y_true, data_buffer=mm.data_buffer)
    # precision = precision_metric._calc(y_pred, y_true, data_buffer=mm.data_buffer)
    # f1 = f1_metric._calc(y_pred, y_true, data_buffer=mm.data_buffer)

    # report = []
    # for eps in np.arange(start=0.001, stop=0.1, step=0.0001):
    #     mm.data_buffer._same_point_distance_limit = eps
    #     mm.data_buffer.create_masks(y_pred, y_true)
    #     recall = recall_metric._calc(y_pred, y_true, data_buffer=mm.data_buffer)
    #     precision = precision_metric._calc(y_pred, y_true, data_buffer=mm.data_buffer)
    #     f1 = f1_metric._calc(y_pred, y_true, data_buffer=mm.data_buffer)
    #     report.append({'eps': eps, 'recall': recall, 'precision': precision, 'f1': f1, 'n_clusters': mm.data_buffer.cluster_counts()[0].shape[0]})
    # ok_records = [r for r in report if r['n_clusters'] >= 1000]
    # max_f1, max_recall = max(r['f1'] for r in ok_records), max(r['recall'] for r in ok_records)
    # max_f1_records = [r for r in ok_records if r['f1'] == max_f1]
    # max_recall_records = [r for r in ok_records if r['recall'] == max_recall]

    # probs = data_transforms.collapse_close_points(probs, d=0.007)
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
