import os
import struct
import subprocess

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d

from scipy.spatial.transform import Rotation


def read_tf_file(filename):
    if not os.path.exists(filename):
        print('File ' + filename + ' not found')
        return

    with open(filename, mode='r') as file:
        lines = file.readlines()

    t = [float(s) for s in lines[0].split()]
    r = [float(s) for s in lines[1].split()]

    return t, r


def read_waveform_cfg(wave_filename, wave_params):
    wave_params['data_type'] = 'adc_samples'

    with open(wave_filename, mode='r') as file:
        lines = file.readlines()

    int_vals = ['num_rx', 'num_tx', 'num_adc_samples_per_chirp', 'num_chirps_per_frame']
    for line in lines:
        vals = line.split()
        if vals[0] in int_vals:
            wave_params[vals[0]] = int(vals[1])
        else:
            wave_params[vals[0]] = float(vals[1])

    return wave_params


def read_antenna_cfg(filename):
    antenna_cfg = {}

    with open(filename, mode='r') as file:
        lines = file.readlines()

    for line in lines:
        vals = line.split()

        if vals[0] != '#':
            if vals[0] == 'num_rx':
                antenna_cfg['num_rx'] = int(vals[1])
                antenna_cfg['rx_locations'] = [0] * antenna_cfg['num_rx']
            elif vals[0] == 'num_tx':
                antenna_cfg['num_tx'] = int(vals[1])
                antenna_cfg['tx_locations'] = [0] * antenna_cfg['num_tx']
            elif vals[0] == 'rx':
                antenna_cfg['rx_locations'][int(vals[1])] = (int(vals[2]), int(vals[3]))
            elif vals[0] == 'tx':
                antenna_cfg['tx_locations'][int(vals[1])] = (int(vals[2]), int(vals[3]))
            elif vals[0] == 'F_design':
                antenna_cfg['F_design'] = float(vals[1])

    return antenna_cfg


def read_coupling_cfg(coupling_filename):
    coupling_calib = {}

    with open(coupling_filename, mode='r') as file:
        lines = file.readlines()

    num_tx = int(lines[0].split(':')[1])
    num_rx = int(lines[1].split(':')[1])
    num_range_bins = int(lines[2].split(':')[1])
    coupling_calib['num_tx'] = num_tx
    coupling_calib['num_rx'] = num_rx
    coupling_calib['num_range_bins'] = num_range_bins
    coupling_calib['num_doppler_bins'] = int(lines[3].split(':')[1])
    data_str = lines[4].split(':')[1]
    data_arr = np.array(data_str.split(',')).astype('float')
    data_arr = data_arr[:-1:2] + 1j * data_arr[1::2]
    coupling_calib['data'] = data_arr.reshape(num_tx, num_rx, num_range_bins)

    return coupling_calib


def get_single_chip_params(calib_dir):
    wave_params = {'sensor_type': 'single_chip'}
    hm_params = {'sensor_type': 'single_chip'}
    pc_params = {'sensor_type': 'single_chip', 'data_type': 'pointcloud'}

    tf_filename = calib_dir + '/transforms/base_to_single_chip.txt'

    t, r = read_tf_file(tf_filename)

    wave_params['translation'] = t
    wave_params['rotation'] = r
    hm_params['translation'] = t
    hm_params['rotation'] = r
    pc_params['translation'] = t
    pc_params['rotation'] = r

    wave_filename = calib_dir + '/single_chip/waveform_cfg.txt'
    wave_params = read_waveform_cfg(wave_filename, wave_params)

    antenna_filename = calib_dir + '/single_chip/antenna_cfg.txt'
    antenna_cfg = read_antenna_cfg(antenna_filename)

    coupling_filename = calib_dir + '/single_chip/coupling_calib.txt'
    coupling_calib = read_coupling_cfg(coupling_filename)

    return {'waveform': wave_params,
            'pointcloud': pc_params,
            'antenna': antenna_cfg,
            'coupling': coupling_calib}


def get_heatmap(filename, num_elevation_bins, num_azimuth_bins, num_range_bins):
    if not os.path.exists(filename):
        raise ValueError(f'File {filename} not found')
    with open(filename, mode='rb') as file:
        frame_bytes = file.read()

    frame_vals = struct.unpack(str(len(frame_bytes) // 4) + 'f', frame_bytes)
    frame_vals = np.array(frame_vals)
    frame = frame_vals.reshape((
        num_elevation_bins,
        num_azimuth_bins,
        num_range_bins,
        2
    ))  # 2 vals for each bin (doppler peak intensity and peak location)
    return frame



class RadarParameters:
    def __init__(self, coloradar_path):
        self.num_range_bins = None
        self.num_elevation_bins = None
        self.num_azimuth_bins = None
        self.range_bin_width = None
        self.azimuth_bins = []
        self.elevation_bins = []

        hm_cfg_file_path = self._find_cfg(coloradar_path)
        if not os.path.isfile(hm_cfg_file_path):
            raise ValueError(f'Heatmap config {hm_cfg_file_path} not found')
        self._parse_config_file(hm_cfg_file_path)

        self.test_output_dir = os.path.join(coloradar_path, 'test_output')
        if not os.path.isdir(self.test_output_dir):
            os.mkdir(self.test_output_dir)

    def _find_cfg(self, coloradar_path):
        return os.path.join(coloradar_path, 'calib', 'single_chip', 'heatmap_cfg.txt')

    def _parse_config_file(self, hm_cfg_file_path):
        with open(hm_cfg_file_path, 'r') as f:
            for line in f:
                if line.startswith("num_range_bins"):
                    self.num_range_bins = int(line.split()[1])
                elif line.startswith("num_elevation_bins"):
                    self.num_elevation_bins = int(line.split()[1])
                elif line.startswith("num_azimuth_bins"):
                    self.num_azimuth_bins = int(line.split()[1])
                elif line.startswith("range_bin_width"):
                    self.range_bin_width = float(line.split()[1])
                elif line.startswith("azimuth_bins"):
                    self.azimuth_bins = list(map(float, line.split()[1:]))
                elif line.startswith("elevation_bins"):
                    self.elevation_bins = list(map(float, line.split()[1:]))
        if not self.num_range_bins or not self.range_bin_width:
            raise ValueError(f'Missing range parameters: num_range_bins, range_bin_width in {hm_cfg_file_path}')
        self.max_range = np.round(self.range_bin_width * self.num_range_bins, 1)

    def display_azimuth_fov_options(self):
        self._draw_fov(self.azimuth_bins)

    def display_elevation_fov_options(self):
        self._draw_fov(self.elevation_bins)

    def _draw_fov(self, bins):
        fig, ax = plt.subplots(subplot_kw={'projection': 'polar'}, figsize=(35, 21))
        num_sections = len(bins) // 2
        if num_sections < 16:
            color_map = [cm.get_cmap('inferno', num_sections)(i) for i in range(num_sections)]
        else:
            plasma_colors = cm.get_cmap('plasma', num_sections // 2)
            magma_colors = cm.get_cmap('inferno', num_sections // 2)
            color_map = [
                color
                for pair in zip(plasma_colors(range(num_sections // 2)), magma_colors(range(num_sections // 2)))
                for color in pair
            ]
        ax.plot([0, 0], [0, 1], color='red', lw=2)

        for bin_idx, left_bin_start in enumerate(bins[:num_sections]):
            left_bin_end = bins[bin_idx + 1] if bin_idx < num_sections else 0
            ax.fill_between([left_bin_start, left_bin_end], 0, 1, color=color_map[bin_idx], alpha=0.9)
            ax.fill_between([-left_bin_end, -left_bin_start], 0, 1, color=color_map[bin_idx], alpha=0.9)
            bin_center = (left_bin_start + left_bin_end) / 2
            bin_number = num_sections - 1 - bin_idx
            ax.text(bin_center, 0.5, f'{bin_number}', fontsize=9, ha='center', va='center', color='white')
            ax.text(-bin_center, 0.5, f'{bin_number}', fontsize=9, ha='center', va='center', color='white')

        ax.set_ylim(0, 1)
        ax.set_theta_zero_location("N")
        ax.set_theta_direction(-1)
        ax.set_thetamin(np.degrees(bins[0]))
        ax.set_thetamax(np.degrees(bins[-1]))
        ax.grid(False)
        ax.set_yticklabels([])
        ax.xaxis.set_ticks([])
        bin_labels = np.degrees(bins)
        for bin_value, label in zip(bins, bin_labels):
            ax.text(bin_value, 1.02, f'{label:.1f}°', fontsize=7, ha='center')
        plt.show()

    def get_fov_parameters(self, azimuth_fov_idx, elevation_fov_idx, max_range_meters):
        if not 0 <= azimuth_fov_idx <= self.num_azimuth_bins // 2 - 1:
            raise ValueError(f'Select azimuth FOV from 0 to {self.num_azimuth_bins // 2 - 1}')
        if not 0 <= elevation_fov_idx <= self.num_elevation_bins // 2 - 1:
            raise ValueError(f'Select azimuth FOV from 0 to {self.num_elevation_bins // 2 - 1}')
        if not 0 < max_range_meters <= self.max_range:
            raise ValueError(f'Select max range from 0 to {self.max_range}')
        azimuth_fov_degrees = np.round(np.degrees(-self.azimuth_bins[self.num_azimuth_bins // 2 - 1 - azimuth_fov_idx]), 1)
        elevation_fov_degrees = np.round(np.degrees(-self.elevation_bins[self.num_elevation_bins // 2 - 1 - elevation_fov_idx]), 1)
        return {
            'horizontalFov': azimuth_fov_degrees * 2,
            'verticalFov': elevation_fov_degrees * 2,
            'range': max_range_meters
        }

class CascadeRadarParameters(RadarParameters):
    def _find_cfg(self, coloradar_path):
        return os.path.join(coloradar_path, 'calib', 'cascade', 'heatmap_cfg.txt')


def show_pcl(pcd_file_path):
    pcd = o3d.io.read_point_cloud(pcd_file_path)
    axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=10.0)
    o3d.visualization.draw_geometries([pcd, axes])


def show_pcl_prob(pcd_file_path, prob_threshold=0):
    with open(pcd_file_path, 'r') as f:
        lines = f.readlines()
    data_start_idx = next(i for i, line in enumerate(lines) if line.strip().startswith("DATA"))
    data = np.loadtxt(lines[data_start_idx+1:], delimiter=' ')

    points = data[:, :3]
    log_odds = data[:, 3]
    probabilities = 1 - 1 / (1 + np.exp(log_odds))

    mask = probabilities >= prob_threshold
    filtered_points = points[mask]
    filtered_probs = probabilities[mask]

    cmap = plt.get_cmap("plasma")
    colors = cmap(filtered_probs)[:, :3]

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(filtered_points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=10.0)
    o3d.visualization.draw_geometries([pcd, axes])


class ColoradarDataset:
    def __init__(self, dataset_path):
        if not os.path.isdir(dataset_path):
            raise ValueError(f'Dataset path {dataset_path} does not exist')
        self.coloradar_path = dataset_path
        self.runs_path = os.path.join(self.coloradar_path, 'kitti')
        if not os.path.isdir(self.runs_path):
            raise ValueError(f'Data path {self.runs_path} does not exist')
        self.test_output_dir = os.path.join(self.coloradar_path, 'test_output')
        if not os.path.isdir(self.test_output_dir):
            os.mkdir(self.test_output_dir)

    def list_runs(self):
        return os.listdir(self.runs_path)

    def interpolate_poses_for_lidar(self, run_name):
        command = [
            './build/interpolate_poses_for_lidar',
            self.coloradar_path, run_name,
            f'outputFilePath={os.path.join(self.test_output_dir, run_name + "_lidar_poses_interpolated.txt")}'
        ]
        _run_command(command)

    def interpolate_poses_for_cascade(self, run_name):
        command = [
            './build/interpolate_poses_for_cascade',
            self.coloradar_path, run_name,
            f'outputFilePath={os.path.join(self.test_output_dir, run_name + "_cascade_poses_interpolated.txt")}'
        ]
        _run_command(command)

    def show_poses(self, run_name):
        self.interpolate_poses_for_lidar(run_name)
        self.interpolate_poses_for_cascade(run_name)

        gt_timestamps_path = os.path.join(self.runs_path, run_name, 'groundtruth', 'timestamps.txt')
        lidar_timestamps_path = os.path.join(self.runs_path, run_name, 'lidar', 'timestamps.txt')
        cascade_timestamps_path = os.path.join(self.runs_path, run_name, 'cascade', 'heatmaps', 'timestamps.txt')

        gt_poses_path = os.path.join(self.runs_path, run_name, 'groundtruth', 'groundtruth_poses.txt')
        # gt_poses_path = os.path.join(self.test_output_dir, run_name + "_orig_poses.txt")  # os.path.join(self.test_output_dir, run_name + "_gt_poses.txt")
        lidar_poses_path = os.path.join(self.test_output_dir, run_name + "_lidar_poses_interpolated.txt")
        cascade_poses_path = os.path.join(self.test_output_dir, run_name + "_cascade_poses_interpolated.txt")
        for fp in gt_timestamps_path, lidar_timestamps_path, cascade_timestamps_path, gt_poses_path, lidar_poses_path, cascade_poses_path:
            if not os.path.isfile(fp):
                raise ValueError(f'File {fp} does not exist')

        gt_timestamps, lidar_timestamps, cascade_timestamps = np.loadtxt(gt_timestamps_path), np.loadtxt(lidar_timestamps_path), np.loadtxt(cascade_timestamps_path)
        gt_poses, lidar_poses, cascade_poses = np.loadtxt(gt_poses_path), np.loadtxt(lidar_poses_path), np.loadtxt(cascade_poses_path)
        gt_translations, gt_rotations = gt_poses[:, :3], gt_poses[:, 3:]
        lidar_translations, lidar_rotations = lidar_poses[:, :3], lidar_poses[:, 3:]
        cascade_translations, cascade_rotations = cascade_poses[:, :3], cascade_poses[:, 3:]

        # original_poses = np.loadtxt(original_gt_poses_path)
        # original_translations, original_rotations = original_poses[:, :3], original_poses[:, 3:]
        # idx = len(original_poses) // 4 * 3
        # print("Orig pose before save", original_poses[idx, 3], original_poses[idx, 4], original_poses[idx, 5], original_poses[idx, 6])
        # print("Orig pose after save", gt_poses[idx, 3], gt_poses[idx, 4], gt_poses[idx, 5], gt_poses[idx, 6])

        # plot_translations(original_translations, gt_translations, gt_timestamps, gt_timestamps, 'Original Poses vs Processed', 'processed')
        # plot_rotations(original_rotations, gt_rotations, gt_timestamps, gt_timestamps, 'Original Poses vs Processed', 'processed')

        plot_translations(gt_translations, lidar_translations, gt_timestamps, lidar_timestamps, 'Ground Truth vs Lidar', 'lidar')
        plot_rotations(gt_rotations, lidar_rotations, gt_timestamps, lidar_timestamps, 'Ground Truth vs Lidar', 'lidar')
        plot_translations(gt_translations, cascade_translations, gt_timestamps, cascade_timestamps, 'Ground Truth vs Cascade', 'cascade')
        plot_rotations(gt_rotations, cascade_rotations, gt_timestamps, cascade_timestamps, 'Ground Truth vs Cascade', 'cascade')

        # plot_rotations_euler(gt_rotations, lidar_rotations, gt_timestamps, lidar_timestamps, 'Ground Truth vs Lidar', 'lidar')

        # plt.plot(cascade_timestamps, cascade_rotations[:, 3], label='cascade_w', color='b')

    def build_octomap(
            self, run_name, map_resolution=0.1,
            horizontal_fov=360, vertical_fov=180, max_range=0
    ):
        print('Building map for', run_name, '...')
        command = [
            './build/build_octomap', self.coloradar_path, run_name,
            f'map_resolution={map_resolution}',
            f'horizontalFov={horizontal_fov}',
            f'verticalFov={vertical_fov}',
            f'range={max_range}'
        ]
        _run_command(command)

    def show_octomap(self, run_name):
        show_pcl(os.path.join(self.runs_path, run_name, 'lidar_maps', 'map.pcd'))

    def show_octomap_prob(self, run_name, prob_threshold=0):
        show_pcl_prob(os.path.join(self.runs_path, run_name, 'lidar_maps', 'map.pcd'), prob_threshold=prob_threshold)


def _run_command(command):
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1)
    for line in process.stdout:
        print(line, end='')
    for line in process.stderr:
        print(line, end='')
    process.stdout.close()
    process.stderr.close()
    process.wait()


def filter_cloud(
        pcd_file_path, output_dir=None,
        random_pcl_radius=10, random_pcl_step=0.5, random_pcl_empty_portion=0.5,
        horizontal_fov=360, vertical_fov=33.2, max_range=20
):
    command = [
        './build/filter_cloud',
        f'pcdFilePath={pcd_file_path}',
        f'randomPclRadius={random_pcl_radius}',
        f'randomPclStep={random_pcl_step}',
        f'randomPclEmptyPortion={random_pcl_empty_portion}',
        f'horizontalFov={horizontal_fov}',
        f'verticalFov={vertical_fov}',
        f'range={max_range}'
    ]
    if output_dir:
        command.append(f'outputDir={output_dir}')
    _run_command(command)

def plot_translations(gt, other, gt_ts, other_ts, title, label):
    plt.figure(figsize=(10, 6))
    plt.subplot(3, 1, 1)
    plt.plot(gt_ts, gt[:, 0], label='gt_x', color='r')
    plt.plot(other_ts, other[:, 0], label=f'{label}_x', linestyle='--', color='b')
    plt.title(f'{title} - Translation X')
    plt.legend()

    plt.subplot(3, 1, 2)
    plt.plot(gt_ts, gt[:, 1], label='gt_y', color='r')
    plt.plot(other_ts, other[:, 1], label=f'{label}_y', linestyle='--', color='b')
    plt.title(f'{title} - Translation Y')
    plt.legend()

    plt.subplot(3, 1, 3)
    plt.plot(gt_ts, gt[:, 2], label='gt_z', color='r')
    plt.plot(other_ts, other[:, 2], label=f'{label}_z', linestyle='--', color='b')
    plt.title(f'{title} - Translation Z')
    plt.legend()

    plt.tight_layout()
    plt.show()

def plot_rotations(gt, other, gt_ts, other_ts, title, label):
    plt.figure(figsize=(10, 8))
    plt.subplot(4, 1, 1)
    plt.plot(gt_ts, gt[:, 0], label='gt_rx', color='r')
    plt.plot(other_ts, other[:, 0], label=f'{label}_rx', linestyle='--', color='b')
    plt.title(f'{title} - Rotation X')
    plt.legend()

    plt.subplot(4, 1, 2)
    plt.plot(gt_ts, gt[:, 1], label='gt_ry', color='r')
    plt.plot(other_ts, other[:, 1], label=f'{label}_ry', linestyle='--', color='b')
    plt.title(f'{title} - Rotation Y')
    plt.legend()

    plt.subplot(4, 1, 3)
    plt.plot(gt_ts, gt[:, 2], label='gt_rz', color='r')
    plt.plot(other_ts, other[:, 2], label=f'{label}_rz', linestyle='--', color='b')
    plt.title(f'{title} - Rotation Z')
    plt.legend()

    plt.subplot(4, 1, 4)
    plt.plot(gt_ts, gt[:, 3], label='gt_w', color='r')
    plt.plot(other_ts, other[:, 3], label=f'{label}_w', linestyle='--', color='b')
    plt.title(f'{title} - Rotation W')
    plt.legend()

    plt.tight_layout()
    plt.show()


def plot_rotations_euler(gt, other, gt_ts, other_ts, title, label):
    roll, pitch, yaw = quaternion_to_euler(gt)
    roll_other, pitch_other, yaw_other = quaternion_to_euler(other)

    plt.figure(figsize=(10, 8))
    plt.subplot(3, 1, 1)
    plt.plot(gt_ts, np.degrees(roll), label='gt_roll', color='r')
    plt.plot(other_ts, np.degrees(roll_other), label=f'{label}_roll', linestyle='--', color='b')
    plt.title(f'{title} - Roll')
    plt.legend()

    plt.subplot(3, 1, 2)
    plt.plot(gt_ts, np.degrees(pitch), label='gt_pitch', color='r')
    plt.plot(other_ts, np.degrees(pitch_other), label=f'{label}_pitch', linestyle='--', color='b')
    plt.title(f'{title} - Pitch')
    plt.legend()

    plt.subplot(3, 1, 3)
    plt.plot(gt_ts, np.degrees(yaw), label='gt_yaw', color='r')
    plt.plot(other_ts, np.degrees(yaw_other), label=f'{label}_yaw', linestyle='--', color='b')
    plt.title(f'{title} - Yaw')
    plt.legend()

    plt.tight_layout()
    plt.show()

    backward_quats = euler_to_quaternion(roll, pitch, yaw)
    backward_quats_other = euler_to_quaternion(roll_other, pitch_other, yaw_other)
    plot_rotations(backward_quats, backward_quats_other, gt_ts, other_ts, title, label)


def quaternion_to_euler(quaternions):
    x, y, z, w = quaternions[..., 0], quaternions[..., 1], quaternions[..., 2], quaternions[..., 3]
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)
    sinp = 2 * (w * y - z * x)
    pitch = np.where(np.abs(sinp) >= 1, np.sign(sinp) * (np.pi / 2), np.arcsin(sinp))
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)
    return roll, pitch, yaw


def euler_to_quaternion(roll, pitch, yaw):
    half_roll = roll / 2.0
    half_pitch = pitch / 2.0
    half_yaw = yaw / 2.0
    cy = np.cos(half_yaw)
    sy = np.sin(half_yaw)
    cp = np.cos(half_pitch)
    sp = np.sin(half_pitch)
    cr = np.cos(half_roll)
    sr = np.sin(half_roll)
    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    quaternions = np.stack([x, y, z, w], axis=-1)
    return quaternions
