import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d
import torch.nn as nn
from dataset import get_dataset, clouds_to_grids
from model_polar import PolarToCartesian
from model_unet import Unet1C3DPolar, CloudsToGrids
from loss_spatial_prob import SoftMatchingLossScaled
import metrics as metric_defs
from train_points import get_model
from visualize_heatmap_clouds import show_radar_pcl


def show_radar_grid(grid, voxel_size, point_range, intensity_threshold_percent=0.0):
    """
    Visualizes a 3D grid (voxel grid) as a point cloud.

    Args:
        grid (np.ndarray): Voxel grid of shape [X, Y, Z, 1] with intensity values.
        voxel_size (float): The size of each voxel.
        point_range (tuple): Tuple (xmin, xmax, ymin, ymax, zmin, zmax) specifying the range of the grid.
        intensity_threshold_percent (float): Minimum intensity threshold as a percentage for filtering.
    """
    xmin, xmax, ymin, ymax, zmin, zmax = point_range
    X, Y, Z, _ = grid.shape
    grid_flat = grid.reshape(-1)
    indices = np.argwhere(grid_flat > 0).flatten()

    # min_intensity = grid_flat[indices].min()
    # max_intensity = grid_flat[indices].max()
    # normalized_intensities = (grid_flat[indices] - min_intensity) / (max_intensity - min_intensity)
    # valid_idx = normalized_intensities >= intensity_threshold_percent / 100
    #
    # indices = indices[valid_idx]
    # normalized_intensities = normalized_intensities[valid_idx]
    x_idx = indices // (Y * Z)
    y_idx = (indices % (Y * Z)) // Z
    z_idx = indices % Z
    x_coords = xmin + x_idx * voxel_size
    y_coords = ymin + y_idx * voxel_size
    z_coords = zmin + z_idx * voxel_size

    coords = np.stack([x_coords, y_coords, z_coords], axis=1)
    cmap = plt.get_cmap("plasma")
    colors = cmap(grid_flat[indices])[:, :3]
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(coords)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=3.0)
    o3d.visualization.draw_geometries([pcd, axes], "Radar Grid")


def show_occupancy_grid(grid, voxel_size, point_range, prob_threshold=0):
    xmin, xmax, ymin, ymax, zmin, zmax = point_range
    X, Y, Z, _ = grid.shape
    probabilities = 1 - 1 / (1 + np.exp(grid[..., 0]))
    valid_mask = (grid[..., 0] != 0) & (probabilities >= prob_threshold)

    x_idx, y_idx, z_idx = np.where(valid_mask)
    x_coords = xmin + x_idx * voxel_size
    y_coords = ymin + y_idx * voxel_size
    z_coords = zmin + z_idx * voxel_size
    coords = np.stack([x_coords, y_coords, z_coords], axis=1)

    cmap = plt.get_cmap("plasma")
    colors = cmap(probabilities[valid_mask])[:, :3]
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(coords)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=3.0)
    o3d.visualization.draw_geometries([pcd, axes], "Occupancy Grid")


def apply_layers(model, polar_frames, radar_config):
    ptc = PolarToCartesian(radar_config)
    cartesian_points = ptc(polar_frames)
    # cartesian_points = model.polar_to_cartesian(polar_frames)
    # downsampled_points = model.down(cartesian_points)
    # points = downsampled_points[..., :3]
    # features = downsampled_points[..., 3:]
    # log_odds = self.pointnet(points, features)
    # probabilities = self.apply_sigmoid(log_odds)
    return cartesian_points

    # reshaped_frames = polar_frames.view(batch_size, model.radar_config.num_azimuth_bins * model.radar_config.num_range_bins, model.radar_config.num_elevation_bins)
    # transformed_frames = model.transformer(reshaped_frames)
    # transformed_frames = transformed_frames.view(batch_size, model.radar_config.num_azimuth_bins, model.radar_config.num_range_bins, model.radar_config.num_elevation_bins)
    # cartesian_points = model.polar_to_cartesian(transformed_frames)
    # less_points = model.down(cartesian_points)
    # log_odds = model.pointnet(less_points)
    # print('less_points', less_points.shape)
    # probabilities = model.apply_sigmoid(less_points)


def main():
    OCCUPANCY_THRESHOLD = 0.6
    POINT_MATCH_RADIUS = 0.2
    BATCH_SIZE = 4
    DATASET_PART = 0.05
    loss_spatial_weight = 1.0
    loss_probability_weight = 1.0
    loss_matching_temperature = 0.2
    octomap_voxel_size = 0.25
    model_path = "/home/arpg/projects/mapping-ros/src/mapless-navigation/best_grid_model_jan16.pth"

    if os.path.isdir('/media/giantdrive'):
        dataset_path = '/media/giantdrive/coloradar/dataset2.h5'
        device_name = 'cuda:0'
    else:
        dataset_path = '/home/arpg/projects/coloradar_plus_processing_tools/dataset2.h5'
        device_name = 'cpu'
    train_loader, val_loader, test_loader, radar_config = get_dataset(dataset_path, batch_size=BATCH_SIZE,  partial=DATASET_PART, occupancy_threshold=OCCUPANCY_THRESHOLD, grid=True, grid_voxel_size=octomap_voxel_size)

    model = get_model(radar_config, occupancy_threshold=OCCUPANCY_THRESHOLD, grid=True)
    device = torch.device(device_name if torch.cuda.is_available() else "cpu")
    print('\ndevice', device)
    # model.load_state_dict(torch.load(model_path, map_location=device))
    # model.to(device)
    # model.eval()

    ptc = PolarToCartesian(radar_config)
    ctg = CloudsToGrids(voxel_size=octomap_voxel_size, point_range=radar_config.point_range)

    loss_fn = SoftMatchingLossScaled(alpha=loss_spatial_weight, beta=loss_probability_weight, matching_temperature=loss_matching_temperature, distance_threshold=POINT_MATCH_RADIUS)
    iou = metric_defs.IoU(max_point_distance=POINT_MATCH_RADIUS, probability_threshold=OCCUPANCY_THRESHOLD)
    chamfer = metric_defs.WeightedChamfer()

    with torch.no_grad():
        for radar_frames, lidar_frames, _ in train_loader:
            radar_frames = radar_frames.to(device)
            radar_cartesian_points = ptc(radar_frames)
            radar_grids = clouds_to_grids(radar_cartesian_points.cpu().numpy(), voxel_size=octomap_voxel_size, point_range=radar_config.point_range)
            radar_grids_trained = ctg(radar_cartesian_points)
            lidar_frames = lidar_frames.to(device)
            # pred_frames = model(radar_frames)

            # pred_frames = apply_layers(model, radar_frames, radar_config)
            # print('azimuth_scale', model.polar_to_cartesian.azimuth_scale)
            # print('azimuth_bias', model.polar_to_cartesian.azimuth_bias)
            # print('azimuth_cos_weight', model.polar_to_cartesian.azimuth_cos_weight)
            # print('azimuth_sin_weight', model.polar_to_cartesian.azimuth_sin_weight)

            # radar_clouds = image_to_pcl(radar_frames, radar_config)
            # first_radar_cloud = radar_clouds[0].cpu().numpy()
            # predicted_clouds = apply_layers(model, radar_frames)  # model(radar_frames)
            # first_predicted_cloud = pred_frames[0].cpu()
            # first_lidar_cloud = lidar_frames[0].cpu().numpy()

            # print('first_radar_cloud', first_radar_cloud.min(), first_radar_cloud.max(), first_radar_cloud.mean())
            # print('batch', radar_clouds.min(), radar_clouds.max(), radar_clouds.mean())
            # visualize_polar_image(radar_frames[0], radar_config)
            # show_occupancy_grid(lidar_frames[0].cpu().numpy(), voxel_size=octomap_voxel_size, point_range=radar_config.point_range)
            # show_radar_pcl(radar_cartesian_points[0].cpu().numpy())
            show_radar_grid(radar_grids[0], voxel_size=octomap_voxel_size, point_range=radar_config.point_range)
            show_radar_grid(radar_grids_trained[0], voxel_size=octomap_voxel_size, point_range=radar_config.point_range)
            # show_occupancy_pcl(first_predicted_cloud.numpy())

            # print(first_predicted_cloud[0])
            # print(first_predicted_cloud[1])
            # print(first_predicted_cloud[2])
            # print(first_predicted_cloud[3])
            # print(first_predicted_cloud[:, 3].max())

            # gt_cloud = model.filter_probs(lidar_frames[0])
            # gt_loss = loss_fn(gt_cloud, gt_cloud).item()
            # print('gt_loss', gt_loss)
            # print('iou', iou(gt_cloud, first_predicted_cloud))
            # print('gt chamfer', chamfer(gt_cloud, gt_cloud))
            break


if __name__ == '__main__':
    main()
