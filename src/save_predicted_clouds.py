
import numpy as np
import open3d as o3d
import os
import torch

from collections import defaultdict
from tqdm import tqdm

from utils.params import get_params
from train_points_generative import PointModelManager
from show_points import run_inference, unpack_model_output, report_metrics



def save_predictions(mm, save_file):
    radar_clouds, gt_clouds, predicted_clouds, poses, metrics = [], [], [], [], defaultdict(list)

    for sample_idx in tqdm(range(len(mm.train_loader.dataset))):  # for every sample in the dataset
        inference_results = run_inference(mm, sample_idx)         # run model
        x_tensor, y_true_tensor, outputs_tensor = inference_results['tensor']
        embeddings_tensor, y_pred_tensor = unpack_model_output(outputs_tensor)
        report = report_metrics(mm, y_pred=y_pred_tensor, y_true=y_true_tensor)

        x_np, y_true_np, outputs_np = inference_results['numpy']
        embeddings_np, y_pred_np = unpack_model_output(outputs_np)

        radar_clouds.append(x_np)
        gt_clouds.append(y_true_np[0])
        predicted_clouds.append(y_pred_np[0])
        poses.append(mm.train_loader.dataset.poses[sample_idx])

        for metric_name, metric_value in report.items():
            metrics[metric_name].append(metric_value)

    for metric_name, metric_values in metrics.items():
        metrics[metric_name] = np.array(metric_values)

    np.savez(
        save_file,
        radar_clouds=radar_clouds,
        gt_clouds=gt_clouds,
        predicted_clouds=predicted_clouds,
        poses=poses,
        metrics=metrics
    )


def read_predictions(save_file):
    data = np.load(save_file, allow_pickle=True)
    return (
        list(data['radar_clouds']),
        list(data['gt_clouds']),
        list(data['predicted_clouds']),
        list(data['poses']),
        data['metrics'].item()
    )


def transform_cloud(cloud, pose):
    pts = cloud[:, :3]
    ones = np.ones((pts.shape[0], 1))
    pts_h = np.hstack([pts, ones])
    transformed = (pose @ pts_h.T).T[:, :3]
    new_cloud = np.copy(cloud)
    new_cloud[:, :3] = transformed
    return new_cloud


def get_map(lidar_clouds, poses, t):
    clouds = [
        transform_cloud(lidar_clouds[i], poses[i])
        for i in range(t + 1)
    ]
    return np.concatenate(clouds, axis=0)


def create_o3d_cloud(cloud, base_color=(1, 1, 1)):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(cloud[:, :3])
    intensities = cloud[:, 3]
    if np.max(intensities) > 1e-3:
        norm = np.clip(intensities / np.max(intensities), 0.0, 1.0)
    else:
        norm = np.zeros_like(intensities)
    colors = np.tile(norm[:, None], (1, 3)) * np.array(base_color)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd


def render_step(vis, map_cloud_np, radar_cloud_np, map_geom, radar_geom):
    map_pcd = create_o3d_cloud(map_cloud_np, (1, 1, 1))
    radar_pcd = create_o3d_cloud(radar_cloud_np, (1, 0, 0))

    map_geom.points = map_pcd.points
    map_geom.colors = map_pcd.colors
    radar_geom.points = radar_pcd.points
    radar_geom.colors = radar_pcd.colors

    vis.update_geometry(map_geom)
    vis.update_geometry(radar_geom)
    vis.poll_events()
    vis.update_renderer()


def animate_map(lidar_clouds, radar_clouds, poses):
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window("Point Cloud Viewer", width=1280, height=720)
    vis.get_render_option().point_size = 2.0

    current_step = [0]
    map_geom = o3d.geometry.PointCloud()
    radar_geom = o3d.geometry.PointCloud()
    vis.add_geometry(map_geom)
    vis.add_geometry(radar_geom)

    def update():
        t = current_step[0]
        map_cloud = get_map(lidar_clouds, poses, t)
        radar_cloud = transform_cloud(radar_clouds[t], poses[t])
        render_step(vis, map_cloud, radar_cloud, map_geom, radar_geom)

    def step_forward(vis):
        if current_step[0] < len(lidar_clouds) - 1:
            current_step[0] += 1
            update()

    def step_backward(vis):
        if current_step[0] > 0:
            current_step[0] -= 1
            update()

    vis.register_key_callback(ord('D'), step_forward)
    vis.register_key_callback(ord('A'), step_backward)

    update()
    vis.run()
    vis.destroy_window()


def run():
    model_path = '/home/arpg/projects/mapless-navigation/trained_models/28jul25_w3q0/best_train_loss.pth'
    save_file = 'predictions.npz'

    params = get_params()
    params['device_name'] = 'cpu'
    params['dataset_params']['partial'] = 1.0
    # params['loss_params']['occupancy_threshold'] = 0.6

    mm = PointModelManager(**params)
    mm.init_model(model_path=model_path)

    if not os.path.exists(save_file):
        save_predictions(mm, save_file)
    radar_clouds, gt_clouds, predicted_clouds, poses, metrics = read_predictions(save_file)

    animate_map(radar_clouds, gt_clouds, poses)


if __name__ == '__main__':
    run()
