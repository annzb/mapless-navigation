import h5py
import json
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


def process_radar_frames(radar_frames, intensity_mean=None, intensity_std=None):
    print('intensity_mean, intensity_std', intensity_mean, intensity_std)
    if (intensity_mean is None) != (intensity_std is None):
        raise ValueError("Both intensity_mean and intensity_std must be provided or neither.")
    if intensity_mean is None:
        intensity_mean = np.mean(radar_frames)
        intensity_std = np.std(radar_frames)
    radar_frames = (radar_frames - intensity_mean) / intensity_std
    return radar_frames, intensity_mean, intensity_std


def process_lidar_frames(lidar_frames):
    for i in range(len(lidar_frames)):
        if len(lidar_frames[i] < 1):
            print('WARNING: lidar frame', i, 'has no points')
        # print(f'lidar_frames[{i}]', lidar_frames[i].shape, lidar_frames[i].min(), lidar_frames[i].mean(), lidar_frames[i].max())
        # print(f'lidar_frames[{i}][..., 3]', lidar_frames[i][..., 3].shape, lidar_frames[i][..., 3].min(), lidar_frames[i][..., 3].mean(), lidar_frames[i][..., 3].max())
        lidar_frames[i][..., 3] = 1 / (1 + np.exp(-lidar_frames[i][..., 3]))
        # print(f'lidar_frames[{i}][..., 3]', lidar_frames[i][..., 3].shape, lidar_frames[i][..., 3].min(), lidar_frames[i][..., 3].mean(), lidar_frames[i][..., 3].max())
    return lidar_frames


class RadarDataset(Dataset):
    def __init__(self, radar_frames, lidar_frames, poses, is_3d=True, intensity_mean=None, intensity_std=None):
        self.X, self.intensity_mean, self.intensity_std = process_radar_frames(radar_frames, intensity_mean, intensity_std)
        self.Y = process_lidar_frames(lidar_frames)
        self.poses = poses

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        radar_frame = self.X[index]
        lidar_frame = self.Y[index]
        pose = self.poses[index]
        return radar_frame, lidar_frame, pose


def custom_collate_fn(batch):
    radar_frames, lidar_frames, poses = zip(*batch)
    radar_frames = torch.tensor(radar_frames)
    lidar_frames = [torch.tensor(frame) for frame in lidar_frames]
    poses = torch.tensor(poses)
    return radar_frames, lidar_frames, poses


def get_dataset(dataset_file_path, partial=1.0, batch_size=16, shuffle_runs=True, random_state=42):
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
    radar_config.set_radar_frame_params(num_azimuth_bins=num_azimuth_bins, num_range_bins=num_range_bins, num_elevation_bins=num_elevation_bins)

    # filter empty clouds
    filtered_indices = [i for i, frame in enumerate(lidar_frames) if len(frame) > 0]
    radar_frames = np.array(radar_frames[filtered_indices])
    lidar_frames = [np.array(lidar_frames[i]) for i in filtered_indices]
    poses = poses[filtered_indices]

    # reduce dataset
    num_samples = int(len(radar_frames) * partial)
    radar_frames = radar_frames[:num_samples]
    lidar_frames = lidar_frames[:num_samples]
    poses = poses[:num_samples]
    print(num_samples, ' samples total.')

    radar_train, radar_temp, lidar_train, lidar_temp, poses_train, poses_temp = train_test_split(radar_frames, lidar_frames, poses, test_size=0.5, random_state=random_state)
    radar_val, radar_test, lidar_val, lidar_test, poses_val, poses_test = train_test_split(radar_temp, lidar_temp, poses_temp, test_size=0.6, random_state=random_state)

    train_dataset = RadarDataset(radar_train, lidar_train, poses_train)
    val_dataset = RadarDataset(radar_val, lidar_val, poses_val, intensity_mean=train_dataset.intensity_mean, intensity_std=train_dataset.intensity_std)
    test_dataset = RadarDataset(radar_test, lidar_test, poses_test, intensity_mean=train_dataset.intensity_mean, intensity_std=train_dataset.intensity_std)

    print('Train input shape:', train_dataset.X.shape)
    print('Train output shape:', len(train_dataset.Y), 'frames,', len(train_dataset.Y[0][0]), 'dims.')
    print('Valid input shape:', val_dataset.X.shape)
    print('Valid output shape:', len(val_dataset.Y), 'frames,', len(val_dataset.Y[0][0]), 'dims.')
    print('Test input shape:', test_dataset.X.shape)
    print('Test output shape:', len(test_dataset.Y), 'frames,', len(test_dataset.Y[0][0]), 'dims.')
    print()

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=custom_collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=custom_collate_fn)

    return train_loader, val_loader, test_loader, radar_config

