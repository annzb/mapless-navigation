import pickle

import numpy as np
from tqdm import tqdm

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize


def cartesian_to_polar_grid(
        grids,
        x_min, x_max, y_max, z_min, z_max,
        azimuth_bins, elevation_bins, range_bins,
        resolution=0.25, range_bin_width=0.125
):
    N, X, Y, Z = grids.shape
    polar_grid_shape = (N, len(elevation_bins), len(azimuth_bins), len(range_bins))
    polar_grid = np.full(polar_grid_shape, -1, dtype=np.float32)
    x, y, z = np.mgrid[x_min:x_max:resolution, 0:y_max:resolution, z_min:z_max:resolution]

    r_from = np.sqrt(x ** 2 + y ** 2 + z ** 2)
    r_to = np.sqrt((x + resolution) ** 2 + (y + resolution) ** 2 + (z + resolution) ** 2)
    azimuth_from = np.arctan2(y, x) - np.pi / 2
    azimuth_to = np.arctan2(y + resolution, x + resolution) - np.pi / 2
    elevation_from = np.arcsin(z / np.maximum(r_from, 1e-9))
    elevation_to = np.arcsin((z + resolution) / np.maximum(r_to, 1e-9))
    r_from, r_to = np.minimum(r_from, r_to), np.maximum(r_from, r_to)
    azimuth_from, azimuth_to = np.minimum(azimuth_from, azimuth_to), np.maximum(azimuth_from, azimuth_to)
    elevation_from, elevation_to = np.minimum(elevation_from, elevation_to), np.maximum(elevation_from, elevation_to)

    azimuth_mask = (azimuth_bins[:, 0][..., None, None, None] <= azimuth_to) & (azimuth_bins[:, 1][..., None, None, None] >= azimuth_from)
    elevation_mask = (elevation_bins[:, 0][..., None, None, None] <= elevation_to) & (elevation_bins[:, 1][..., None, None, None] >= elevation_from)
    range_mask = (range_bins[:, None, None, None] <= r_to) & (range_bins[:, None, None, None] + range_bin_width >= r_from)

    for n in tqdm(range(N)):
        for i in range(X):
            for j in range(Y):
                for k in range(Z):
                    if grids[n, i, j, k] == -1:
                        continue
                    for az_idx in np.where(azimuth_mask[:, i, j, k])[0]:
                        for el_idx in np.where(elevation_mask[:, i, j, k])[0]:
                            for r_idx in np.where(range_mask[:, i, j, k])[0]:
                                polar_grid[n, el_idx, az_idx, r_idx] = max(polar_grid[n, el_idx, az_idx, r_idx], grids[n, i, j, k])
    return polar_grid


def parse_polar_bins(center_coords):
    num_bins = len(center_coords)
    if num_bins % 2:
        raise ValueError('Uneven bins')
    edge_coords = []

    for right_center_idx in range(int(num_bins / 2), num_bins):
        start = 0 if not edge_coords else edge_coords[-1][1]
        bin_width = (center_coords[right_center_idx] - start) * 2
        end = start + bin_width
        edge_coords.append(np.array([start, end]))

    edge_coords = np.array(edge_coords)
    return np.vstack([-np.flip(edge_coords), edge_coords])


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
    heatmap_slice = heatmap[4, :, :, 0] if len(heatmap.shape) == 4 else heatmap[2, :, :]
    normalized_slice = Normalize()(heatmap_slice)
    colored_image = np.zeros((56, 64, 3))
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
    with open('dataset_7runs_smallfov.pkl', 'rb') as f:
        data = pickle.load(f)

    params = data.pop('params')['heatmap']
    azimuth_bins, elevation_bins = params['azimuth_bins'], params['elevation_bins']
    # print(parse_polar_bins(elevation_bins))
    azimuth_bins = parse_polar_bins(azimuth_bins)
    # print(np.degrees(azimuth_bins[0][0]), np.degrees(azimuth_bins[-1][1]))

    heatmaps = np.array(data['ec_hallways_run1']['heatmaps'])
    for i in (0, 50, 100, 150, 200):
        save_heatmap_image(heatmaps[i], filename=f'heatmap_intensity_{i}.png')
