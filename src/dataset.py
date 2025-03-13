import h5py
import json
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from sklearn.model_selection import train_test_split

from utils.radar_config import RadarConfig


def read_h5_dataset(file_path):
    print('Reading dataset from', file_path)
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
        radar_frames = torch.stack([torch.tensor(frame) for frame in radar_frames])  # [B, ...]
        poses = torch.stack([torch.tensor(pose) for pose in poses])  # [B, ...]
        lidar_tensors = [torch.tensor(frame) for frame in lidar_frames]
        lidar_frames_padded = pad_sequence(lidar_tensors, batch_first=True, padding_value=float('nan'))  # [B, max_N, 4]
        return radar_frames, lidar_frames_padded, poses

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


def get_dataset(
        dataset_file_path, dataset_type,
        partial=1.0, batch_size=16, shuffle_runs=True, random_state=42, grid_voxel_size=1.0,
        occupied_only=False, occupancy_threshold=0.5
):
    data_dict, radar_config = read_h5_dataset(dataset_file_path)
    radar_frames = data_dict['cascade_heatmaps']
    lidar_frames = data_dict['lidar_map_samples']
    poses = data_dict['cascade_poses']

    if shuffle_runs:
        radar_frames = np.concatenate(list(radar_frames.values()), axis=0)
        lidar_frames = [np.array(frame) for run_frames in lidar_frames.values() for frame in run_frames]
        poses = np.concatenate(list(poses.values()), axis=0)
    else:
        raise NotImplementedError("Non-shuffled runs are not implemented.")
    # print('radar_frames.shape', radar_frames.shape)
    _, num_elevation_bins, num_azimuth_bins, num_range_bins = radar_frames.shape
    radar_config.set_radar_frame_params(num_azimuth_bins=num_azimuth_bins, num_range_bins=num_range_bins, num_elevation_bins=num_elevation_bins, grid_voxel_size=grid_voxel_size)
    # print('point range:', radar_config.point_range)

    # filter empty clouds
    filtered_indices = [i for i, frame in enumerate(lidar_frames) if len(frame) > 0 and (any(frame[:, 3] >= occupancy_threshold) if occupied_only else True)]  # TODO: fix for grid
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

    # for i, sample in enumerate(lidar_frames):
    #     valid, resolution = validate_octomap_pointcloud(sample, tolerance=1e-2)
    #     if valid:
    #         if i == 0:
    #             print('valid sample', i, 'resolution:', resolution)
    #     else:
    #         print('invalid sample', i, 'resolution:', resolution)

    radar_train, radar_temp, lidar_train, lidar_temp, poses_train, poses_temp = train_test_split(radar_frames, lidar_frames, poses, test_size=0.5, random_state=random_state)
    radar_val, radar_test, lidar_val, lidar_test, poses_val, poses_test = train_test_split(radar_temp, lidar_temp, poses_temp, test_size=0.6, random_state=random_state)

    # dataset_class = RadarDatasetGrid if grid else RadarDataset
    train_dataset = dataset_type(radar_train, lidar_train, poses_train, radar_config=radar_config, voxel_size=grid_voxel_size, name='train')
    val_dataset = dataset_type(radar_val, lidar_val, poses_val, radar_config=radar_config, voxel_size=grid_voxel_size, intensity_mean=train_dataset.intensity_mean, intensity_std=train_dataset.intensity_std, name='valid')
    test_dataset = dataset_type(radar_test, lidar_test, poses_test, radar_config=radar_config, voxel_size=grid_voxel_size, intensity_mean=train_dataset.intensity_mean, intensity_std=train_dataset.intensity_std, name='test')

    train_dataset.print_log()
    val_dataset.print_log()
    test_dataset.print_log()
    print()

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=dataset_type.custom_collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=dataset_type.custom_collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=dataset_type.custom_collate_fn)

    return train_loader, val_loader, test_loader, radar_config
