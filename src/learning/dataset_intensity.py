#!/usr/bin/env python3

import numpy as np
import pickle

from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset


def normalize_heatmap(heatmap, mean_channel_1, std_channel_1, is_3d=False):
    heatmap_1c = (heatmap[:, :, :, 0] - mean_channel_1) / std_channel_1
    if is_3d:
        return np.expand_dims(heatmap_1c, axis=0)
    return heatmap_1c


def process_heatmaps(heatmaps, mean_channel_1=None, std_channel_1=None, is_3d=False):
    if mean_channel_1 is None or std_channel_1 is None:
        heatmaps_flat = np.concatenate(heatmaps, axis=0)
        mean_channel_1 = np.mean(heatmaps_flat[:, :, :, 0])
        std_channel_1 = np.std(heatmaps_flat[:, :, :, 0])

    normalized_heatmaps = np.array([normalize_heatmap(
        hm, mean_channel_1, std_channel_1, is_3d=is_3d
    ) for hm in heatmaps])
    return normalized_heatmaps, mean_channel_1, std_channel_1


def process_grids(grids):
    grids = grids.transpose(0, 3, 2, 1)
    return grids


class HeatmapTrainingDataset(Dataset):
    def __init__(self, heatmaps, occupancy_grids, true_poses=np.array([]), is_3d=False, **kwargs):
        self.true_poses = tuple([None] * len(heatmaps)) if true_poses.size == 0 else true_poses
        self.X, self.mean_channel_1, self.std_channel_1 = process_heatmaps(heatmaps, is_3d=is_3d, **kwargs)
        self.Y = process_grids(occupancy_grids)

    def __getitem__(self, index):
        return self.X[index], self.Y[index], self.true_poses[index]

    def __len__(self):
        return len(self.X)


def split_dataset(n_samples):
    total_idx = np.arange(n_samples)
    train_indices, remain_indices = train_test_split(total_idx, test_size=0.4, random_state=42)
    validate_indices, test_indices = train_test_split(remain_indices, test_size=0.5, random_state=42)
    return train_indices, validate_indices, test_indices


def get_dataset(dataset_filepath, visualize=False, is_3d=False, batch_size=32):
    with open(dataset_filepath, 'rb') as f:
        data = pickle.load(f)
    print('Runs in dataset:', ', '.join(data.keys()))

    heatmaps, gt_grids, poses = [], [], []
    for run_name in data:
        heatmaps.extend(data[run_name]['heatmaps'])
        gt_grids.extend(data[run_name]['gt_grids'])
        poses.extend(data[run_name]['poses'])
    heatmaps, gt_grids, poses = np.array(heatmaps), np.array(gt_grids), np.array(poses)
    print('Raw input shape:', heatmaps.shape)
    print('Raw output shape:', gt_grids.shape)
    # Input shape: (1042, 16, 64, 64)
    # Output shape: (1042, 32, 32, 32)

    train_indices, validate_indices, test_indices = split_dataset(len(heatmaps))
    x_train, x_valid, x_test = heatmaps[train_indices], heatmaps[validate_indices], heatmaps[test_indices]
    y_train, y_valid, y_test = gt_grids[train_indices], gt_grids[validate_indices], gt_grids[test_indices]
    poses_train, poses_valid, poses_test = poses[train_indices], poses[validate_indices], poses[test_indices]

    # x_train, x_valid, x_test, y_train, y_valid, y_test = split_dataset(x_total=heatmaps, y_total=gt_grids)
    # print(len(y_train[0] > 0))
    # print('X train sample:', x_train[0])
    # print('Y train sample:', y_train[0])
    # print('X valid sample:', x_valid[0])
    # print('Y valid sample:', y_valid[0])

    train_dataset = HeatmapTrainingDataset(x_train, y_train, is_3d=is_3d, true_poses=poses_train)
    valid_dataset = HeatmapTrainingDataset(
        x_valid, y_valid, is_3d=is_3d, true_poses=poses_valid,
        mean_channel_1=train_dataset.mean_channel_1, std_channel_1=train_dataset.std_channel_1
    )
    test_dataset = HeatmapTrainingDataset(
        x_test, y_test, is_3d=is_3d, true_poses=poses_test,
        mean_channel_1=train_dataset.mean_channel_1, std_channel_1=train_dataset.std_channel_1
    )
    print('Train input shape:', train_dataset.X.shape)
    print('Validate input shape:', valid_dataset.X.shape)
    print('Test input shape:', test_dataset.X.shape)
    print('Output shape:', train_dataset.Y.shape)
    print('Test poses shape:', poses_test.shape)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, valid_loader, test_loader
