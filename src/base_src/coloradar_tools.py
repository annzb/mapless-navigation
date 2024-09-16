import os
import struct
import subprocess

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d


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
        hm_cfg_file_path = os.path.join(coloradar_path, 'calib', 'single_chip', 'heatmap_cfg.txt')
        if not os.path.isfile(hm_cfg_file_path):
            raise ValueError(f'Heatmap config {hm_cfg_file_path} not found')
        self.num_range_bins = None
        self.num_elevation_bins = None
        self.num_azimuth_bins = None
        self.range_bin_width = None
        self.azimuth_bins = []
        self.elevation_bins = []
        self._parse_config_file(hm_cfg_file_path)

        self.test_output_dir = os.path.join(coloradar_path, 'test_output')
        if not os.path.isdir(self.test_output_dir):
            os.mkdir(self.test_output_dir)

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
        azimuth_fov_degrees = np.round(np.degrees(-self.azimuth_bins[self.num_azimuth_bins // 2 - 1 - azimuth_fov_idx]),
                                       1)
        elevation_fov_degrees = np.round(
            np.degrees(-self.elevation_bins[self.num_elevation_bins // 2 - 1 - elevation_fov_idx]), 1)
        return {
            'horizontal_fov': azimuth_fov_degrees,
            'vertical_fov': elevation_fov_degrees,
            'range': max_range_meters
        }


def show_pcl(pcd_file_path):
    pcd = o3d.io.read_point_cloud(pcd_file_path)
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

    def list_runs(self):
        return os.listdir(self.runs_path)

    def build_octomap(
            self, run_name, map_resolution=0.1,
            horizontal_fov=360, vertical_fov=180, max_range=0
    ):
        print('Building map for ', run_name, '...')
        command = [
            './build/build_octomap', self.coloradar_path, run_name,
            f'map_resolution={map_resolution}',
            f'horizontalFov={horizontal_fov}',
            f'verticalFov={vertical_fov}',
            f'range={max_range}'
        ]
        _run_command(command)

    def show_octomap(self, run_name):
        show_pcl(os.path.join(self.coloradar_path, 'lidar_maps', run_name, 'map.pcd'))


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
        './build/filter_cloud', pcd_file_path,
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
