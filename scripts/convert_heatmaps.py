#!/usr/bin/env python
import os
import numpy as np

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


def get_localized_pointcloud(pose, true_map, x_max=5., y_max=10., z_max=5., fov_angle=45):
    orientation = R.from_quat(pose[3:])
    local_points = orientation.inv().apply(true_map - pose[:3])

    box_mask = (
            (local_points[:, 0] >= -x_max) & (local_points[:, 0] <= x_max) &
            (local_points[:, 1] >= 0) & (local_points[:, 1] <= y_max) &
            (local_points[:, 2] >= -z_max) & (local_points[:, 2] <= z_max)
    )
    cone_mask = (
            local_points[:, 0] ** 2 + local_points[:, 2] ** 2 <=
            (local_points[:, 1] * np.tan(np.radians(fov_angle / 2))) ** 2
    )
    fov_mask = box_mask & cone_mask

    return local_points[fov_mask]


def main():
    map_resolution = 0.25
    x_min, x_max = -4, 4
    y_min, y_max = 0, 8
    z_min, z_max = -4, 4
    coloradar_dir = '/home/ann/mapping/coloradar'
    run_folder_name = 'hallways_run0'
    calib_folder_name = 'calib'
    bag_file_path = os.path.join(coloradar_dir, 'ec_hallways_run0_lidar_octomap.bag')
    topic_name = "/lidar_filtered/octomap_point_cloud_centers"

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
    poses = np.loadtxt(os.path.join(run_dir, 'groundtruth/groundtruth_poses.txt'))
    # poses = np.hstack((poses_raw[:, :3] / map_resolution, poses_raw[:, 3:]))
    pose_timestamps = np.loadtxt(os.path.join(run_dir, 'groundtruth/timestamps.txt'))
    heatmap_timestamps = np.loadtxt(os.path.join(run_dir, 'single_chip/heatmaps/timestamps.txt'))
    pose_indices = associate_radar_with_pose(heatmap_timestamps, pose_timestamps)
    print(pose_indices)
    print('Saving total map')
    save_total_map(true_map, poses)

    print('Calculating frames')
    # num_files = len(os.listdir(os.path.join(run_dir, 'single_chip/heatmaps/data')))
    map_frames = []
    for heatmap_idx, pose_idx in enumerate(pose_indices):
        heatmap = get_heatmap(
            filename=os.path.join(run_dir, 'single_chip/heatmaps/data/heatmap_' + str(heatmap_idx) + '.bin'),
            params=params['heatmap']
        )
        localized_points = get_localized_pointcloud(poses[pose_idx], true_map, x_max=x_max, y_max=y_max, z_max=z_max)
        map_frames.append(localized_points)
        # print('Heatmap', heatmap_idx, heatmap.shape)
        # print('Pointcloud from pose', pose_idx, localized_points.shape)
    print('Heatmap shape', heatmap.shape)
    print('GT shape', localized_points.shape)
    print('GT point', localized_points[0])

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


def visualize_true_frames(frames, x_max=5., y_max=10., z_max=10.):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    # ax.view_init(elev=40, azim=-60)

    for frame in frames:
        ax.scatter(frame[:, 0], frame[:, 1], frame[:, 2], s=1)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_xlim([-x_max, x_max])
        ax.set_ylim([0, y_max])
        ax.set_zlim([-z_max, z_max])
        plt.draw()
        plt.pause(0.2)
        ax.clear()
    plt.close()

    # for frame, pose in zip(frames, poses):
    #     fig = plt.figure(figsize=(12, 10))
    #     ax = fig.add_subplot(111, projection='3d')
    #
    #     # Draw frame points with coloration based on z-coordinate
    #     norm = plt.Normalize(vmin=frame[:, 2].min(), vmax=frame[:, 2].max())
    #     colors = plt.cm.jet(norm(frame[:, 2]))
    #     ax.scatter(frame[:, 0], frame[:, 1], frame[:, 2], c=colors, s=1)
    #
    #     # Draw the current location as a separate object (red point)
    #     ax.scatter(pose[0], pose[1], pose[2], color='red', s=100)
    #
    #     # Draw the direction as a line (green line)
    #     direction = R.from_quat(pose[3:]).apply([1, 0, 0])
    #     line_length = 1.0  # Adjust as needed
    #     end_point = pose[:3] + direction * line_length
    #     ax.plot([pose[0], end_point[0]], [pose[1], end_point[1]], [pose[2], end_point[2]], color='green')
    #
    #     plt.draw()
    #     plt.pause(0.5)  # Display each frame for 0.5 seconds
    # plt.close()


if __name__ == '__main__':
    main()
