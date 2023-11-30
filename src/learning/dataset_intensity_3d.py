#!/usr/bin/env python3

import numpy as np
import pickle

from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset


def normalize_heatmap(heatmap, mean_channel_1, std_channel_1):
    heatmap_1c = (heatmap[:, :, :, 0] - mean_channel_1) / std_channel_1
    return heatmap_1c.reshape(1, 64, 64, 16)


def process_heatmaps(heatmaps, mean_channel_1=None, std_channel_1=None):
    if mean_channel_1 is None or std_channel_1 is None:
        heatmaps_flat = np.concatenate(heatmaps, axis=0)
        mean_channel_1 = np.mean(heatmaps_flat[:, :, :, 0])
        std_channel_1 = np.std(heatmaps_flat[:, :, :, 0])

    normalized_heatmaps = np.array([normalize_heatmap(hm, mean_channel_1, std_channel_1) for hm in heatmaps])
    return normalized_heatmaps, mean_channel_1, std_channel_1


class HeatmapTrainingDataset(Dataset):
    def __init__(self, heatmaps, occupancy_grids):
        self.X, self.mean_channel_1, self.std_channel_1 = process_heatmaps(heatmaps)
        self.Y = occupancy_grids

    def __getitem__(self, index):
        return self.X[index], self.Y[index]

    def __len__(self):
        return len(self.X)


class HeatmapTestingDataset(Dataset):
    def __init__(self, heatmaps, occupancy_grids, mean_channel_1, std_channel_1):
        self.X, _, _ = process_heatmaps(heatmaps, mean_channel_1, std_channel_1)
        self.Y = occupancy_grids

    def __getitem__(self, index):
        return self.X[index], self.Y[index]

    def __len__(self):
        return len(self.X)


def split_dataset(x_total, y_total):
    N = len(x_total)  # total number of samples
    total_idx = np.arange(N)
    train_indices, remain_indices = train_test_split(total_idx, test_size=0.4, random_state=42)
    validate_indices, test_indices = train_test_split(remain_indices, test_size=0.5, random_state=42)

    x_train, x_valid, x_test = x_total[train_indices], x_total[validate_indices], x_total[test_indices]
    y_train, y_valid, y_test = y_total[train_indices], y_total[validate_indices], y_total[test_indices]

    return x_train, x_valid, x_test, y_train, y_valid, y_test


def get_dataset(dataset_filepath, visualize=False):
    with open(dataset_filepath, 'rb') as f:
        data = pickle.load(f)
    heatmaps, gt_grids = np.array(data['heatmaps']), np.array(data['gt_grids'])

    print('Raw input shape:', heatmaps.shape)
    print('Raw output shape:', gt_grids.shape)
    # Input shape: (1042, 16, 64, 64, 2)
    # Output shape: (1042, 32, 32, 32)

    x_train, x_valid, x_test, y_train, y_valid, y_test = split_dataset(x_total=heatmaps, y_total=gt_grids)
    print(len(y_train[0] > 0))
    print('X train sample:', x_train[0])
    # print('Y train sample:', y_train[0])
    # print('X valid sample:', x_valid[0])
    # print('Y valid sample:', y_valid[0])

    train_dataset = HeatmapTrainingDataset(x_train, y_train)
    valid_dataset = HeatmapTestingDataset(
        x_valid, y_valid,
        mean_channel_1=train_dataset.mean_channel_1, std_channel_1=train_dataset.std_channel_1
    )
    test_dataset = HeatmapTestingDataset(
        x_test, y_test,
        mean_channel_1=train_dataset.mean_channel_1, std_channel_1=train_dataset.std_channel_1
    )
    print('Train input shape:', train_dataset.X.shape)
    print('Validate input shape:', valid_dataset.X.shape)
    print('Test input shape:', test_dataset.X.shape)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    return train_loader, valid_loader, test_loader
