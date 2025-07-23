import numpy as np
import os
import torch
from sklearn.cluster import DBSCAN
from tqdm import tqdm

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


def count_empty_gt_clouds(mm, threshold=0.5):
    n_empty = 0
    for sample_idx in tqdm(range(len(mm.train_loader.dataset))):
        _, gt_cloud, _ = mm.train_loader.dataset[sample_idx]
        gt_cloud_occupied = gt_cloud[gt_cloud[:, 3] >= threshold]
        if gt_cloud_occupied.shape[0] == 0:
            print(f'Empty gt cloud at sample {sample_idx}')
            n_empty += 1
    return n_empty


def run_inference(mm, sample_idx):
    input_cloud, gt_cloud, _ = mm.train_loader.dataset[sample_idx]
    input_tensor = torch.from_numpy(input_cloud)
    input_batch_idx = np.zeros(input_cloud.shape[0], dtype=np.int64)
    input_batch_idx_tensor = torch.as_tensor(input_batch_idx, dtype=torch.int64)
    gt_batch_idx = np.zeros(gt_cloud.shape[0], dtype=np.int64)
    gt_batch_idx_tensor = torch.as_tensor(gt_batch_idx, dtype=torch.int64)

    with torch.no_grad():
        outputs = mm.model((input_tensor.to(mm.device), input_batch_idx_tensor.to(mm.device)), debug=True)
    outputs_tensor = [o.cpu() for o in outputs]
    outputs_numpy = [o.numpy() for o in outputs_tensor]
    
    return {
        'numpy': ((input_cloud, input_batch_idx), (gt_cloud, gt_batch_idx), outputs_numpy), 
        'tensor': ((input_tensor, input_batch_idx_tensor), (torch.as_tensor(gt_cloud, dtype=torch.float32), gt_batch_idx_tensor), outputs_tensor)
    }


def unpack_model_output(outputs):
    decoder_output, decoder_output_flat, pred_clouds_flat, pred_batch_idx = outputs
    y_pred = (pred_clouds_flat, pred_batch_idx)
    return decoder_output[0], y_pred


def report_metrics(mm, y_pred, y_true):
    mm.data_buffer.create_masks(y=y_pred, y_other=y_true)

    mm.reset_metrics(mode='test')
    mm.apply_metrics(y_pred, y_true, data_buffer=mm.data_buffer, mode='test')
    report = mm.report_metrics(mode='test')

    loss = mm.loss_fn(y_pred=y_pred, y_true=y_true, data_buffer=mm.data_buffer)
    report['loss'] = loss.item()
    return report


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


def run():
    model_path = '/Users/anna/data/rmodels/sweep1/29_best_train_loss.pth'

    params = get_params()
    params['device_name'] = 'cpu'
    params['dataset_params']['partial'] = 1
    params['loss_params']['occupancy_threshold'] = 0.8

    mm = PointModelManager(**params)
    mm.init_model(model_path=model_path)

    n_empty_gt_clouds = count_empty_gt_clouds(mm, threshold=0.8) # threshold=params['loss_params']['occupancy_threshold'])
    print(f'Number of samples: {len(mm.train_loader.dataset)}, number of empty gt clouds: {n_empty_gt_clouds}')

    # Model 29, best for 0.8 - 25, best for 0.7 - 78
    best_sample_idx, best_metric_value = find_best_sample(mm, metric_name='loss', loss=True)
    print(f'Best sample idx: {best_sample_idx}, best loss: {best_metric_value}')
    sample_idx = best_sample_idx

    '''
    Expected array shapes:
    - input_idx: (N_radar_points, )  7748
    - input: (N_radar_points, 4)
    - gt: (N_lidar_points, 4)  4112
    ''' 
    inference_results = run_inference(mm, sample_idx)
    x_np, y_true_np, outputs_np = inference_results['numpy']
    x_tensor, y_true_tensor, outputs_tensor = inference_results['tensor']

    '''
    Expected array shapes:
    - embeddings: (N_predicted_points, 4) 2048
    - y_pred[0]: (N_predicted_points, 4)  2048
    - y_pred[1]: (N_predicted_points, )   2048
    - y_true[0]: (N_lidar_points, 4)
    - y_true[1]: (N_lidar_points, )
    ''' 
    embeddings_np, y_pred_np = unpack_model_output(outputs_np)
    embeddings_tensor, y_pred_tensor = unpack_model_output(outputs_tensor)
    report = report_metrics(mm, y_pred=y_pred_tensor, y_true=y_true_tensor)

    for metric_name, metric_value in report.items():
        metric_name_short = metric_name.split('_')[-1]
        print(f'{metric_name_short}: {metric_value}')

    y_pred_cloud_occupied = y_pred_np[0][y_pred_np[0][:, 3] >= params['loss_params']['occupancy_threshold']]
    y_true_cloud_occupied = y_true_np[0][y_true_np[0][:, 3] >= params['loss_params']['occupancy_threshold']]

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
        clouds=[y_pred_np[0], y_true_np[0], y_pred_cloud_occupied, y_true_cloud_occupied],
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
