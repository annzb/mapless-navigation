#!/usr/bin/env python
import os
import pickle
import numpy as np

from PIL import Image
from tqdm import tqdm

import rosbag
from sensor_msgs import point_cloud2
from scipy.spatial.transform import Rotation as R

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

# import octomap
# from octomap_msgs.msg import Octomap
# export LD_LIBRARY_PATH=/home/ann/mapping/venv/lib:$LD_LIBRARY_PATH

from coloradar_tools import get_heatmap, get_single_chip_params


def associate_radar_with_pose(radar_timestamps, true_timestamps):
    indices = []
    for rt in radar_timestamps:
        diff = np.abs(true_timestamps - rt)
        indices.append(np.argmin(diff))
    return indices


def get_localized_pointcloud(pose, true_map, x_max=5., y_max=10., z_max=5.):
    orientation = R.from_quat(pose[3:])
    local_points = orientation.inv().apply(true_map[:, :3] - pose[:3])

    box_mask = (
            (local_points[:, 0] >= -x_max) & (local_points[:, 0] <= x_max) &
            (local_points[:, 1] >= 0) & (local_points[:, 1] <= y_max) &
            (local_points[:, 2] >= -z_max) & (local_points[:, 2] <= z_max)
    )
    elevation_mask = (
        local_points[:, 1] ** 2 >=
        3 * (local_points[:, 0] ** 2 + local_points[:, 2] ** 2)
    )
    fov_mask = box_mask & elevation_mask
    points_in_fov = true_map[fov_mask]
    transformed_points = np.hstack((orientation.inv().apply(points_in_fov[:, :3] - pose[:3]), points_in_fov[:, 3].reshape(-1, 1)))
    return transformed_points


def points_to_grid(points, x_min=-5, x_max=5, y_min=0, y_max=10, z_min=-5, z_max=5, resolution=0.25):
    grid_size_x = int((x_max - x_min) / resolution)
    grid_size_y = int((y_max - y_min) / resolution)
    grid_size_z = int((z_max - z_min) / resolution)
    grid = np.zeros((grid_size_x, grid_size_y, grid_size_z), dtype=float)
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


def frame_grid_to_points(frame_grid, pose, timestamp, resolution=0.25, prob_threshold=0):
    assert 0 <= prob_threshold <= 1
    grid_size_x, grid_size_y, grid_size_z = frame_grid.shape
    x_min, y_min, z_min = -grid_size_x * resolution / 2, 0, -grid_size_z * resolution / 2

    # Find indices of non-zero occupancy probabilities
    occupied_indices = np.argwhere(frame_grid > 0) if prob_threshold == 0 else np.argwhere(frame_grid >= prob_threshold)
    world_coordinates = occupied_indices * resolution + np.array([x_min, y_min, z_min])
    orientation = R.from_quat(pose[3:])
    world_coordinates = orientation.apply(world_coordinates) + pose[:3]

    probabilities = frame_grid[occupied_indices[:, 0], occupied_indices[:, 1], occupied_indices[:, 2]]
    points_map = {
        (world_coordinates[i, 0], world_coordinates[i, 1], world_coordinates[i, 2]
         ): (probabilities[i], timestamp)
        for i in range(len(occupied_indices))
    }
    return points_map


def main(run_folder_name='ec_hallways_run0'):
    map_resolution = 0.25
    x_min, x_max = -5, 5
    y_min, y_max = 0, 10
    z_min, z_max = -3, 3
    coloradar_dir = '/home/arpg/mapping/coloradar'
    calib_folder_name = 'calib'
    map_file_path = os.path.join(coloradar_dir, 'ec_hallways_run0_lidar_octomap_points.csv')

    # topic_name = "/lidar_filtered/octomap_point_cloud_centers"
    # octomap_topic_name = "/lidar_filtered/octomap_full"
    # last_octomap_msg = None
    # with rosbag.Bag(bag_file_path, 'r') as bag:
    #     for topic, msg, t in bag.read_messages(topics=[octomap_topic_name]):
    #         last_octomap_msg = msg
    # if last_octomap_msg is not None:
    #     octree = octomap.OcTree(last_octomap_msg.resolution)
    #     octree.readBinary(last_octomap_msg.data)
    #     for it in octree.begin_tree():
    #         if octree.isNodeOccupied(it):
    #             coord = it.getCoordinate()
    #             prob = it.getOccupancy()
    #             print("Leaf Node Coordinate:", coord)
    #             print("Occupancy Probability:", prob)
    # else:
    #     print("No Octomap messages found on the topic.")
    # return
    # x_min, x_max = x_min / map_resolution, x_max / map_resolution
    # y_min, y_max = y_min / map_resolution, y_max / map_resolution
    # z_min, z_max = z_min / map_resolution, z_max / map_resolution
    # last_msg = None
    # with rosbag.Bag(bag_file_path, "r") as bag:
    #     for _, msg, _ in bag.read_messages(topics=[topic_name]):
    #         last_msg = msg
    # if last_msg:
    #     gen = point_cloud2.read_points(last_msg, field_names=("x", "y", "z"), skip_nans=True)
    #     true_map = np.array(list(gen))
    # else:
    #     raise ValueError('True map not initialized.')

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
    save_total_map(map_points[:3], poses)

    print('Calculating frames')
    map_frames = []
    reversed_map_frames = {}
    frame_grids = []
    heatmaps = []
    for heatmap_idx, pose_idx in tqdm(enumerate(pose_indices)):
        heatmap = get_heatmap(
            filename=os.path.join(run_dir, 'single_chip/heatmaps/data/heatmap_' + str(heatmap_idx) + '.bin'),
            params=params['heatmap']
        )
        # print(calculate_heatmap_stats(heatmap))
        heatmaps.append(heatmap)
        # if heatmap_idx in set(range(10, 20)):
        #     save_heatmap_image(heatmap, idx=heatmap_idx)

        localized_points = get_localized_pointcloud(
            poses[pose_idx], map_points,
            x_max=x_max, y_max=y_max, z_max=z_max
        )
        frame_grid = points_to_grid(
            localized_points, resolution=map_resolution,
            x_max=x_max, y_max=y_max, z_max=z_max,
            x_min=-x_max, y_min=0, z_min=-z_max
        )
        map_frames.append(localized_points)
        frame_grids.append(frame_grid)

        reversed_points = frame_grid_to_points(
            frame_grid=frame_grid,
            pose=poses[pose_idx],
            timestamp=pose_timestamps[pose_idx]
        )
        reversed_map_frames.update(reversed_points)
        # print(f'Non-zero probability points in grid out of 32768: {np.count_nonzero(frame_grid)}')
        # print('Heatmap', heatmap_idx, heatmap.shape)
        # print('Pointcloud from pose', pose_idx, localized_points.shape)

    print('Heatmap shape', heatmaps[0].shape)
    print('GT frame shape', localized_points.shape)
    print('GT grid shape', frame_grid.shape)

    save_total_map(np.array([np.array(point) for point in reversed_map_frames]), poses, filename='total_map_reversed')

    if os.path.isfile('dataset.pkl'):
        with open('dataset.pkl', 'wb') as f:
            data = pickle.load(f)
    else:
        data = {}
    data[run_folder_name] = {
        'heatmaps': heatmaps,
        'gt_grids': frame_grids,
        'poses': poses,
        'pose_timestamps': pose_timestamps
        # 'gt_points': map_frames
    }
    pickle.dump(data, f)

    visualize_true_frames(map_frames, x_max=x_max, y_max=y_max, z_max=z_max)


def save_total_map(total_map, poses, filename='total_map'):
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    # colors = (total_map[:, 2] - total_map[:, 2].min()) / (total_map[:, 2].max() - total_map[:, 2].min())
    norm = plt.Normalize(vmin=total_map[:, 2].min(), vmax=total_map[:, 2].max())
    colors = plt.cm.jet(norm(total_map[:, 2]))
    ax.scatter(total_map[:, 0], total_map[:, 1], total_map[:, 2], c=colors, s=1)

    # Draw the trajectory from poses as a line
    trajectory = np.array([pose[:3] for pose in poses])
    ax.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2], color='green', linewidth=2)

    ax.view_init(elev=ax.elev - 5)
    plt.savefig(filename + '.png', dpi=600)

    ax.view_init(azim=ax.azim + 45, elev=ax.elev - 5)
    plt.savefig(filename + '_2.png', dpi=600)

    ax.view_init(azim=ax.azim + 90, elev=ax.elev - 15)
    plt.savefig(filename + '_3.png', dpi=600)

    plt.close()


def save_heatmap_image(heatmap, filename='heatmap', idx=0):
    normalized_image_3d = np.zeros_like(heatmap)
    for i in range(heatmap.shape[-1]):
        channel = heatmap[:, :, :, i]
        min_val = channel.min()
        max_val = channel.max()
        if max_val > min_val:
            normalized_image_3d[:, :, :, i] = (channel - min_val) / (max_val - min_val)

    reshaped_image = normalized_image_3d.transpose(1, 2, 0, 3).reshape(64, 64, -1)
    image_2d_rgb = np.zeros((*reshaped_image.shape[:2], 3), dtype=np.uint8)
    image_2d_rgb[..., 0] = reshaped_image[..., 0] * 255  # Red channel
    image_2d_rgb[..., 1] = reshaped_image[..., 1] * 255  # Green channel
    image_2d_rgb = image_2d_rgb.astype(np.uint8)
    Image.fromarray(image_2d_rgb).save(filename + str(idx) + '.png')


def calculate_heatmap_stats(heatmap):
    stats = {}
    if heatmap.shape[-1] != 2:
        raise ValueError("Array must have shape (X, Y, Z, 2)")

    # Mean and Standard Deviation for each channel
    stats['mean_channel_1'] = np.mean(heatmap[:, :, :, 0])
    stats['std_dev_channel_1'] = np.std(heatmap[:, :, :, 0])
    stats['mean_channel_2'] = np.mean(heatmap[:, :, :, 1])
    stats['std_dev_channel_2'] = np.std(heatmap[:, :, :, 1])

    # Minimum and Maximum Values for each channel
    stats['min_channel_1'] = np.min(heatmap[:, :, :, 0])
    stats['max_channel_1'] = np.max(heatmap[:, :, :, 0])
    stats['min_channel_2'] = np.min(heatmap[:, :, :, 1])
    stats['max_channel_2'] = np.max(heatmap[:, :, :, 1])

    # Quantiles and Percentiles for each channel
    for percentile in [25, 50, 75]:
        stats[f'percentile_{percentile}_channel_1'] = np.percentile(heatmap[:, :, :, 0], percentile)
        stats[f'percentile_{percentile}_channel_2'] = np.percentile(heatmap[:, :, :, 1], percentile)

    # Data Type and Scale
    stats['data_type'] = heatmap.dtype

    return stats


def visualize_true_frames(frames, x_max=5., y_max=10., z_max=10.):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    point_size = 3
    scatter = ax.scatter([], [], [], c=[], s=point_size)
    colorbar = fig.colorbar(scatter, ax=ax, label='Probability')

    for frame in frames:
        # print(f'Total points in frame: {len(frame)}, non-zero probability points: {sum(frame[:, 3] > 0)}')
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

    plt.close()


if __name__ == '__main__':
    main()
