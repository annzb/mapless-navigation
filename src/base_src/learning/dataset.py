import h5py
import json
import math
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

from radar_config import RadarConfig


def read_h5_dataset(file_path):
    data_dict = {}
    with h5py.File(file_path, 'r') as f:
        config = json.loads(f['config'][()])
        data_content = config.get('data_content', [])
        runs = config.get('runs', [])
        for content in data_content:
            data_dict[content] = {}
            for run in runs:
                dataset_name = f"{content}_{run}"
                sizes_dataset_name = f"{dataset_name}_sizes"
                if dataset_name in f:
                    if sizes_dataset_name in f:
                        flat_data = f[dataset_name][:]
                        sizes = f[sizes_dataset_name][:]
                        offsets = np.cumsum(sizes)
                        pointclouds = np.split(flat_data, offsets[:-1])

                        data_dict[content][run] = pointclouds
                    else:
                        data_dict[content][run] = f[dataset_name][:]
                else:
                    print(f"Dataset {dataset_name} not found in the file.")
    return data_dict, RadarConfig.from_dict(config.get('radar_config', {}))


# def clouds_to_grids(clouds, voxel_size, point_range):
#     """
#     Converts a point cloud to a voxel grid, aggregating intensities by their maximum value.
#
#     Args:
#         clouds (np.ndarray): Point cloud of shape [N_frames, N_points, 4] (X, Y, Z, intensity).
#         voxel_size (float): The size of each cubical voxel.
#         point_range (tuple): Tuple (xmin, xmax, ymin, ymax, zmin, zmax) specifying the range of the grid.
#
#     Returns:
#         np.ndarray: Voxel grid of shape [N_frames, X, Y, Z, 1].
#     """
#     N_frames = len(clouds)
#     xmin, xmax, ymin, ymax, zmin, zmax = point_range
#     grid_size = (
#         math.ceil((xmax - xmin) / voxel_size),
#         math.ceil((ymax - ymin) / voxel_size),
#         math.ceil((zmax - zmin) / voxel_size)
#     )
#     voxel_grid = np.full((N_frames, *grid_size, 1), fill_value=-np.inf, dtype=np.float32)
#     for frame_idx in range(N_frames):
#         frame_points = clouds[frame_idx]
#         coords = frame_points[:, :3]
#         intensity = frame_points[:, 3]
#         voxel_indices = np.floor((coords - np.array([xmin, ymin, zmin])) / voxel_size).astype(int)
#         valid_mask = np.all( (voxel_indices >= 0) & (voxel_indices < np.array(grid_size)), axis=1)
#         voxel_indices = voxel_indices[valid_mask]
#         intensity = intensity[valid_mask]
#         for idx, val in zip(voxel_indices, intensity):
#             x, y, z = idx
#             voxel_grid[frame_idx, x, y, z, 0] = max(voxel_grid[frame_idx, x, y, z, 0], val)
#     voxel_grid[voxel_grid == -np.inf] = 0
#     return voxel_grid


def clouds_to_grids(clouds, radar_config):
    """
    Converts a point cloud to a voxel grid, aggregating probabilities by their maximum value.

    Args:
        clouds (np.ndarray): Point cloud of shape [N_frames, N_points, 4] (X, Y, Z, intensity).
        radar_config (RadarConfig): Config.

    Returns:
        np.ndarray: Voxel grid of shape [N_frames, X, Y, Z, 1].
    """
    N_frames = len(clouds)
    xmin, _, ymin, _, zmin, _ = radar_config.point_range
    voxel_grid = np.full((N_frames, *radar_config.grid_size, 1), fill_value=-np.inf, dtype=np.float32)

    for frame_idx in range(N_frames):
        frame_points = clouds[frame_idx]
        coords = frame_points[:, :3]
        intensity = frame_points[:, 3]
        voxel_indices = np.floor((coords - np.array([xmin, ymin, zmin])) / radar_config.grid_resolution).astype(int)
        valid_mask = np.all((voxel_indices >= 0) & (voxel_indices < np.array(radar_config.grid_size)), axis=1)
        voxel_indices = voxel_indices[valid_mask]
        intensity = intensity[valid_mask]
        flat_indices = (
            voxel_indices[:, 0] * radar_config.grid_size[1] * radar_config.grid_size[2] +
            voxel_indices[:, 1] * radar_config.grid_size[2] +
            voxel_indices[:, 2]
        )
        flattened_grid = voxel_grid[frame_idx, ..., 0].flatten()
        unique_indices, inverse_indices = np.unique(flat_indices, return_inverse=True)
        max_values = np.zeros_like(unique_indices, dtype=np.float32)
        for i, idx in enumerate(unique_indices):
            max_values[i] = np.max(intensity[inverse_indices == i])
        flattened_grid[unique_indices] = np.maximum(flattened_grid[unique_indices], max_values)
        voxel_grid[frame_idx, ..., 0] = flattened_grid.reshape(radar_config.grid_size)
    # WARNING
    voxel_grid[voxel_grid == -np.inf] = 0.001
    return voxel_grid


def process_radar_frames(radar_frames, intensity_mean=None, intensity_std=None):
    if (intensity_mean is None) != (intensity_std is None):
        raise ValueError("Both intensity_mean and intensity_std must be provided or neither.")
    if intensity_mean is None:
        intensity_mean = np.mean(radar_frames)
        intensity_std = np.std(radar_frames)
    radar_frames = (radar_frames - intensity_mean) / intensity_std
    # radar_frames -= radar_frames.min()  # For positive values only
    return radar_frames, intensity_mean, intensity_std


def process_lidar_frames(lidar_frames):
    for i in range(len(lidar_frames)):
        lidar_frames[i][..., 3] = 1 / (1 + np.exp(-lidar_frames[i][..., 3]))
    return lidar_frames


class RadarDataset(Dataset):
    def __init__(self, radar_frames, lidar_frames, poses, *args, intensity_mean=None, intensity_std=None, name='dataset', **kwargs):
        self.X, self.intensity_mean, self.intensity_std = process_radar_frames(radar_frames, intensity_mean, intensity_std)
        self.Y = process_lidar_frames(lidar_frames)
        self.poses = poses
        self.name = name.capitalize()

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        radar_frame = self.X[index]
        lidar_frame = self.Y[index]
        pose = self.poses[index]
        return radar_frame, lidar_frame, pose

    @staticmethod
    def custom_collate_fn(batch):
        radar_frames, lidar_frames, poses = zip(*batch)
        radar_frames = torch.tensor(radar_frames)
        lidar_frames = [torch.tensor(frame) for frame in lidar_frames]
        poses = torch.tensor(poses)
        return radar_frames, lidar_frames, poses

    def print_log(self):
        print(f'{self.name} input shape:', self.X.shape)
        print(f'{self.name} output shape:', len(self.Y), 'frames,', len(self.Y[0][0]), 'dims.')


class RadarDatasetGrid(RadarDataset):
    def __init__(self, *args, radar_config, voxel_size=0.1, **kwargs):
        super().__init__(*args, **kwargs)
        # self.X = clouds_to_grids(self.X, voxel_size, radar_config.point_range)
        # self.Y = clouds_to_grids(self.Y, radar_config)

    # @staticmethod
    # def custom_collate_fn(batch):
    #     radar_frames, lidar_frames, poses = zip(*batch)
    #     radar_frames = torch.tensor(radar_frames)
    #     lidar_frames = torch.tensor(lidar_frames)
    #     poses = torch.tensor(poses)
    #     return radar_frames, lidar_frames, poses

    def print_log(self):
        print(f'{self.name} input shape:', self.X.shape)
        # print(f'{self.name} output shape:', self.Y.shape)


def get_dataset(dataset_file_path, partial=1.0, batch_size=16, shuffle_runs=True, random_state=42, occupancy_threshold=0.0, grid=False, grid_voxel_size=0.1):
    data_dict, radar_config = read_h5_dataset(dataset_file_path)
    radar_frames = data_dict['cascade_heatmaps']
    lidar_frames = data_dict['lidar_map_frames']
    poses = data_dict['cascade_poses']

    if shuffle_runs:
        radar_frames = np.concatenate(list(radar_frames.values()), axis=0)
        lidar_frames = [np.array(frame) for run_frames in lidar_frames.values() for frame in run_frames]
        poses = np.concatenate(list(poses.values()), axis=0)
    else:
        raise NotImplementedError("Non-shuffled runs are not implemented.")
    _, num_azimuth_bins, num_range_bins, num_elevation_bins = radar_frames.shape
    radar_config.set_radar_frame_params(num_azimuth_bins=num_azimuth_bins, num_range_bins=num_range_bins, num_elevation_bins=num_elevation_bins, grid_voxel_size=grid_voxel_size)
    # print('point range:', radar_config.point_range)

    # filter empty clouds
    filtered_indices = [i for i, frame in enumerate(lidar_frames) if len(frame) > 0 and any(frame[:, 3] >= occupancy_threshold)]
    print(f'Filtered {len(radar_frames) - len(filtered_indices)} empty frames out of {len(radar_frames)}.')
    radar_frames = np.array(radar_frames[filtered_indices])
    lidar_frames = [lidar_frames[i] for i in filtered_indices]
    poses = poses[filtered_indices]

    # reduce dataset
    num_samples = int(len(radar_frames) * partial)
    radar_frames = radar_frames[:num_samples]
    lidar_frames = lidar_frames[:num_samples]
    poses = poses[:num_samples]
    print(num_samples, 'samples total.')

    radar_train, radar_temp, lidar_train, lidar_temp, poses_train, poses_temp = train_test_split(radar_frames, lidar_frames, poses, test_size=0.5, random_state=random_state)
    radar_val, radar_test, lidar_val, lidar_test, poses_val, poses_test = train_test_split(radar_temp, lidar_temp, poses_temp, test_size=0.6, random_state=random_state)

    dataset_class = RadarDatasetGrid if grid else RadarDataset
    train_dataset = dataset_class(radar_train, lidar_train, poses_train, radar_config=radar_config, voxel_size=grid_voxel_size, name='train')
    val_dataset = dataset_class(radar_val, lidar_val, poses_val, radar_config=radar_config, voxel_size=grid_voxel_size, intensity_mean=train_dataset.intensity_mean, intensity_std=train_dataset.intensity_std, name='valid')
    test_dataset = dataset_class(radar_test, lidar_test, poses_test, radar_config=radar_config, voxel_size=grid_voxel_size, intensity_mean=train_dataset.intensity_mean, intensity_std=train_dataset.intensity_std, name='test')

    train_dataset.print_log()
    val_dataset.print_log()
    test_dataset.print_log()
    print()

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=dataset_class.custom_collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=dataset_class.custom_collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=dataset_class.custom_collate_fn)

    return train_loader, val_loader, test_loader, radar_config
