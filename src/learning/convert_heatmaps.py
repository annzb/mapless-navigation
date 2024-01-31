#!/usr/bin/env python
import os
import pickle
import numpy as np

from scipy.spatial.transform import Rotation as R
from tqdm import tqdm

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from src.coloradar_tools import get_heatmap, get_single_chip_params


def associate_radar_with_pose(radar_timestamps, true_timestamps):
    indices = []
    for rt in radar_timestamps:
        diff = np.abs(true_timestamps - rt)
        indices.append(np.argmin(diff))
    return indices


def get_localized_pointcloud(pose, true_map, azimuth_angle=90., elevation_angle=90., x_max=5., y_max=10., z_max=5.):
    orientation = R.from_quat(pose[3:])
    local_points = orientation.inv().apply(true_map[:, :3] - pose[:3])
    box_mask = (
        (local_points[:, 0] >= -x_max) & (local_points[:, 0] <= x_max) &
        (local_points[:, 1] >= 0) & (local_points[:, 1] <= y_max) &
        (local_points[:, 2] >= -z_max) & (local_points[:, 2] <= z_max)
    )
    tan_elevation = np.tan(np.radians(elevation_angle))
    elevation_mask = (
        local_points[:, 1] ** 2 >=
        (tan_elevation ** 2) * (local_points[:, 0] ** 2 + local_points[:, 2] ** 2)
    )
    tan_azimuth = np.tan(np.radians(azimuth_angle / 2))
    azimuth_mask = (
        np.abs(local_points[:, 0] / np.maximum(local_points[:, 1], np.finfo(float).eps)) <= tan_azimuth
    )
    fov_mask = box_mask & elevation_mask & azimuth_mask
    points_in_fov = true_map[fov_mask]
    transformed_points = np.hstack((orientation.inv().apply(points_in_fov[:, :3] - pose[:3]), points_in_fov[:, 3].reshape(-1, 1)))
    return transformed_points


def points_to_grid(points, x_min=-5, x_max=5, y_min=0, y_max=10, z_min=-5, z_max=5, resolution=0.25):
    grid_size_x = int((x_max - x_min) / resolution)
    grid_size_y = int((y_max - y_min) / resolution)
    grid_size_z = int((z_max - z_min) / resolution)
    grid = np.zeros((grid_size_x, grid_size_y, grid_size_z), dtype=float) + 0.5
    coordinates = points[:, :3]
    occupancy_odds = points[:, 3]

    point_indices = np.floor((coordinates - np.array([x_min, y_min, z_min])) / resolution).astype(int)
    in_bounds_mask = (
            (point_indices[:, 0] >= 0) & (point_indices[:, 0] < grid_size_x) &
            (point_indices[:, 1] >= 0) & (point_indices[:, 1] < grid_size_y) &
            (point_indices[:, 2] >= 0) & (point_indices[:, 2] < grid_size_z)
    )
    filtered_indices = point_indices[in_bounds_mask]
    filtered_odds = occupancy_odds[in_bounds_mask]
    for idx, odds in zip(filtered_indices, filtered_odds):
        grid[tuple(idx)] = 1.0 / (1 + np.exp(-odds))
    return grid


def main(run_folder_name='ec_hallways_run0', dataset_filename='dataset.pkl'):
    print('Processing', run_folder_name)
    map_resolution = 0.25
    x_min, x_max = -9, 9
    y_min, y_max = 0, 10
    z_min, z_max = -3, 3
    coloradar_dir = '/home/ann/mapping/coloradar'
    calib_folder_name = 'calib'
    map_file_path = os.path.join(coloradar_dir, run_folder_name + '_lidar_octomap_points.csv')

    run_dir = os.path.join(coloradar_dir, run_folder_name)
    params = get_single_chip_params(calib_dir=os.path.join(coloradar_dir, calib_folder_name))
    poses = np.loadtxt(os.path.join(run_dir, 'groundtruth/groundtruth_poses.txt'))
    pose_timestamps = np.loadtxt(os.path.join(run_dir, 'groundtruth/timestamps.txt'))
    heatmap_timestamps = np.loadtxt(os.path.join(run_dir, 'single_chip/heatmaps/timestamps.txt'))
    pose_indices = associate_radar_with_pose(heatmap_timestamps, pose_timestamps)
    map_points = np.loadtxt(map_file_path, delimiter=',', skiprows=1)
    # print(map_points.shape)
    # print(map_points[0])
    # print('Saving total map')
    # save_total_map(map_points, poses)

    print('Calculating frames')
    map_frames = []
    poses_matched, pose_timestamps_matched = [], []
    frame_grids = []
    heatmaps = []
    for heatmap_idx, pose_idx in tqdm(enumerate(pose_indices)):
        poses_matched.append(poses[pose_idx])
        pose_timestamps_matched.append(pose_timestamps[pose_idx])

        heatmap = get_heatmap(
            filename=os.path.join(run_dir, 'single_chip/heatmaps/data/heatmap_' + str(heatmap_idx) + '.bin'),
            params=params['heatmap']
        )
        # print(calculate_heatmap_stats(heatmap))
        heatmaps.append(heatmap)
        # if heatmap_idx % 10 == 0:
        #     save_heatmap_image(heatmap, idx=heatmap_idx)

        # true_points = select_points_from_pose(map_points, x_max=x_max, y_max=y_max, z_max=z_max)
        localized_points = get_localized_pointcloud(
            poses[pose_idx], map_points,
            azimuth_angle=152.6,    # 76.3 degrees
            elevation_angle=135.4,  # 67.7 degrees
            x_max=x_max, y_max=y_max, z_max=z_max
        )
        frame_grid = points_to_grid(
            localized_points, resolution=map_resolution,
            x_max=x_max, y_max=y_max, z_max=z_max,
            x_min=-x_max, y_min=0, z_min=-z_max
        )
        map_frames.append(localized_points)
        frame_grids.append(frame_grid)

    print('Heatmap shape', heatmaps[0].shape)
    print('GT frame shape', localized_points.shape)
    print('GT grid shape', frame_grid.shape)

    if os.path.isfile(dataset_filename):
        with open(dataset_filename, 'rb') as f:
            data = pickle.load(f)
    else:
        data = {}
    data[run_folder_name] = {
        'heatmaps': heatmaps,
        'gt_grids': frame_grids,
        'poses': poses_matched,
        'pose_timestamps': pose_timestamps_matched
        # 'gt_points': map_frames
    }
    print('Writing runs:', data.keys())
    with open(dataset_filename, 'wb') as f:
        pickle.dump(data, f)

    visualize_true_frames(map_frames, x_max=x_max, y_max=y_max, z_max=z_max)


def save_total_map(total_map, poses, filename='total_map'):
    print('shape', total_map.shape)
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    # colors = (total_map[:, 2] - total_map[:, 2].min()) / (total_map[:, 2].max() - total_map[:, 2].min())
    norm = plt.Normalize(vmin=0, vmax=1)
    colors = plt.cm.jet(norm(total_map[:, 3]))
    ax.scatter(total_map[:, 0], total_map[:, 1], total_map[:, 2], c=colors, s=1)

    # Draw the trajectory from poses as a line
    trajectory = np.array([pose[:3] for pose in poses])
    ax.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2], color='green', linewidth=2)

    ax.view_init(elev=ax.elev - 5)
    plt.savefig(filename + '_jan.png', dpi=600)

    ax.view_init(azim=ax.azim + 45, elev=ax.elev - 5)
    plt.savefig(filename + '_2_jan.png', dpi=600)

    ax.view_init(azim=ax.azim + 90, elev=ax.elev - 15)
    plt.savefig(filename + '_3_jan.png', dpi=600)

    plt.close()


def visualize_true_frames(frames, x_max=5., y_max=10., z_max=10.):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    point_size = 3
    scatter = ax.scatter([], [], [], c=[], s=point_size)
    colorbar = fig.colorbar(scatter, ax=ax, label='Occupancy Log Likelihood')
    init_azim, init_elev = ax.azim, ax.elev

    for i, frame in enumerate(frames):
        # if i in (110, 140, 190):
        ax.clear()
        scatter = ax.scatter(frame[:, 0], frame[:, 1], frame[:, 2], c=frame[:, 3], s=point_size)
        ax.scatter([0], [0], [0], c='black', s=point_size * 2)
        colorbar.update_normal(scatter)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_xlim([-x_max, x_max])
        ax.set_ylim([0, y_max])
        ax.set_zlim([-z_max, z_max])
        plt.draw()
        plt.pause(0.2)
            # filename = f'gt_{i}'
            # ax.view_init(elev=ax.elev - 5)
            # plt.savefig(filename + '.png', dpi=600)
            # ax.view_init(azim=ax.azim + 15, elev=ax.elev + 5)
            # plt.savefig(filename + '_2.png', dpi=600)
            # ax.view_init(azim=ax.azim + 45, elev=ax.elev + 10)
            # plt.savefig(filename + '_3.png', dpi=600)
            # ax.view_init(azim=init_azim, elev=init_elev)
    plt.close()


if __name__ == '__main__':
    ds_file = '/home/ann/mapping/mn_ws/src/mapless-navigation/dataset_5runs_widefov.pkl'
    # assert os.path.isfile(ds_file)
    # for f in ('ec_hallways_run0', 'arpg_lab_run0', 'longboard_run0', 'outdoors_run0'):
    for f in ('arpg_lab_run0', ):
        main(run_folder_name=f, dataset_filename=ds_file)
