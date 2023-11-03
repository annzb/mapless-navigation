import os
import numpy as np

import rosbag
from sensor_msgs import point_cloud2

from .coloradar_tools import get_heatmap, get_single_chip_params


def associate_radar_with_pose(radar_timestamps, true_timestamps):
    indices = []
    for rt in radar_timestamps:
        diff = np.abs(true_timestamps - rt)
        indices.append(np.argmin(diff))
    return indices


def get_localized_pointcloud(pose, true_map, x_range, y_range, z_range):
    # Pose is [x, y, z, qx, qy, qz, qw]
    x, y, z = pose[:3]
    mask = (
        (true_map[:, 0] > x - x_range) & (true_map[:, 0] < x + x_range) &
        (true_map[:, 1] > y - y_range) & (true_map[:, 1] < y + y_range) &
        (true_map[:, 2] > z - z_range) & (true_map[:, 2] < z + z_range)
    )
    return true_map[mask]


def main():
    x_range, y_range, z_range = 10.0, 10.0, 10.0
    coloradar_dir = '/home/ann/mapping/coloradar'
    run_folder_name = 'hallways_run0'
    calib_folder_name = 'calib'
    bag_file_path = os.path.join(coloradar_dir, 'ec_hallways_run0_lidar_octomap.bag')
    topic_name = "/lidar_filtered/octomap_point_cloud_centers"

    last_msg = None
    with rosbag.Bag(bag_file_path, "r") as bag:
        for _, msg, _ in bag.read_messages(topics=[topic_name]):
            last_msg = msg
    if last_msg:
        gen = point_cloud2.read_points(last_msg, field_names=("x", "y", "z"), skip_nans=True)
        true_map = np.array(list(gen))
    else:
        raise ValueError('True map not initialized.')

    run_dir = os.path.join(coloradar_dir, run_folder_name)
    params = get_single_chip_params(calib_dir=os.path.join(coloradar_dir, calib_folder_name))
    poses = np.loadtxt(os.path.join(run_dir, 'single_chip/groundtruth/poses.txt'))
    pose_timestamps = np.loadtxt(os.path.join(run_dir, 'single_chip/groundtruth/timestamps.txt'))
    heatmap_timestamps = np.loadtxt(os.path.join(run_dir, 'single_chip/heatmaps/timestamps.txt'))
    pose_indices = associate_radar_with_pose(heatmap_timestamps, pose_timestamps)

    # num_files = len(os.listdir(os.path.join(run_dir, 'single_chip/heatmaps/data')))
    for heatmap_idx, pose_idx in enumerate(pose_indices):
        heatmap = get_heatmap(
            filename=os.path.join(run_dir, 'single_chip/heatmaps/data/heatmap_' + str(heatmap_idx) + '.bin'),
            params=params['heatmap']
        )
        localized_points = get_localized_pointcloud(poses[pose_idx], true_map, x_range, y_range, z_range)
        print('Heatmap', heatmap_idx, heatmap.shape)
        print('Pointcloud from pose', pose_idx, true_map.shape)


if __name__ == '__main__':
    main()
