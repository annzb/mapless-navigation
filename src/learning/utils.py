import pickle

import numpy as np

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize


def select_points_from_pose(map_pcl, x_max=5., y_max=10., z_max=5.):
    box_mask = (
        (map_pcl[:, 0] >= -x_max) & (map_pcl[:, 0] <= x_max) &
        (map_pcl[:, 1] >= 0) & (map_pcl[:, 1] <= y_max) &
        (map_pcl[:, 2] >= -z_max) & (map_pcl[:, 2] <= z_max)
    )
    points_in_fov = map_pcl[box_mask]
    points_dict = {(point[0], point[1], point[2]): 1.0 / (1 + np.exp(-point[3])) for point in points_in_fov}
    return points_dict


def save_heatmap_image(heatmap, filename='heatmap.png'):
    heatmap_slice = heatmap[2, :, :, 0]
    normalized_slice = Normalize()(heatmap_slice)
    colored_image = np.zeros((48, 64, 3))
    colored_image[:, :, 0] = normalized_slice  # Red channel
    colored_image[:, :, 1] = 0.5  # Constant value for Green channel
    colored_image[:, :, 2] = 0.5  # Constant value for Blue channel
    plt.imsave(filename, colored_image)


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


if __name__ == "__main__":
    with open('dataset.pkl', 'rb') as f:
        data = pickle.load(f)
    heatmaps = np.array(data['heatmaps'])
    for i in (0, 50, 100, 150, 200):
        save_heatmap_image(heatmaps[i], filename=f'heatmap_intensity_{i}.png')
