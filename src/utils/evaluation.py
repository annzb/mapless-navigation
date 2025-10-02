import os
import pickle

from collections import defaultdict
from tqdm import tqdm

import numpy as np



def save_predictions(model_manager, save_file_path):
    radar_clouds, gt_clouds, predicted_clouds, poses, metrics = [], [], [], [], defaultdict(list)

    for sample_idx in tqdm(range(len(model_manager.train_loader.dataset))):
        input_cloud_np, true_cloud_np, pose = model_manager.train_loader.dataset[sample_idx]
        y_pred_tensor, report = model_manager.evaluate_batch(input_cloud_np, true_cloud_np)
        pred_cloud_np = y_pred_tensor[0].cpu().numpy()


        radar_clouds.append(input_cloud_np)
        gt_clouds.append(true_cloud_np)
        predicted_clouds.append(pred_cloud_np)
        poses.append(pose)

        for metric_name, metric_value in report.items():
            metrics[metric_name].append(metric_value)

    for metric_name, metric_values in metrics.items():
        metrics[metric_name] = np.array(metric_values)

    data = {
        'radar_clouds': radar_clouds,
        'gt_clouds': gt_clouds,
        'predicted_clouds': predicted_clouds,
        'poses': poses,
        'metrics': metrics
    }

    with open(save_file_path, 'wb') as f:
        pickle.dump(data, f)


def read_predictions(save_file_path):
    with open(save_file_path, 'rb') as f:
        data = pickle.load(f)
    return (
        data['radar_clouds'],
        data['gt_clouds'], 
        data['predicted_clouds'],
        data['poses'],
        data['metrics']
    )


# def merge_pred_gt(y, y_other):
#     (y_cloud, y_batch_indices), (y_cloud_other, y_batch_indices_other) = y, y_other
#     merged_points = []

#     for b in y_batch_indices.unique():
#         y_mask, y_other_mask = y_batch_indices == b, y_batch_indices_other == b
#         y_batch_cloud, y_other_batch_cloud = y_cloud[y_mask], y_cloud_other[y_other_mask]
#         if len(y_other_batch_cloud) == 0: 
#             batch_merged_points = y_batch_cloud
#         else:
#             batch_merged_points = torch.cat([y_batch_cloud, y_other_batch_cloud])
#         merged_points.append(batch_merged_points)

#     return merged_points


# def count_empty_gt_clouds(mm, threshold=0.5):
#     n_empty = 0
#     for sample_idx in tqdm(range(len(mm.train_loader.dataset))):
#         _, gt_cloud, _ = mm.train_loader.dataset[sample_idx]
#         gt_cloud_occupied = gt_cloud[gt_cloud[:, 3] >= threshold]
#         if gt_cloud_occupied.shape[0] == 0:
#             # print(f'Empty gt cloud at sample {sample_idx}')
#             n_empty += 1
#     return n_empty


def find_best_sample(mm, metric_name, loss=False):
    best_sample_idx = None
    best_metric_value = np.inf if loss else -np.inf

    for sample_idx in tqdm(range(len(mm.train_loader.dataset))):
        x_tensor, y_true_tensor, outputs_tensor = run_inference(mm, sample_idx)['tensor']
        embeddings_tensor, y_pred_tensor = unpack_model_output(outputs_tensor)
        report = report_metrics(mm, y_pred=y_pred_tensor, y_true=y_true_tensor)
        if metric_name not in report:
            raise ValueError(f'Metric {metric_name} not found, available metrics: {report.keys()}')
        
        if (report[metric_name] < best_metric_value) if loss else (report[metric_name] > best_metric_value):
            best_metric_value = report[metric_name]
            best_sample_idx = sample_idx

    return best_sample_idx, best_metric_value
