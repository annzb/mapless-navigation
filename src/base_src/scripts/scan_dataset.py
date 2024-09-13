import os
import pickle
import numpy as np

from src.learning.utils import save_heatmap_image, parse_polar_bins


def cartesian_to_polar_grid(grids, x_min, x_max, y_max, z_min, z_max, azimuth_bins, elevation_bins, range_bins, resolution=0.25, range_bin_width=0.125):
    N, X, Y, Z = grids.shape
    polar_grid_shape = (N, len(elevation_bins), len(azimuth_bins), len(range_bins))
    polar_grid = np.full(polar_grid_shape, -1, dtype=np.float32)

    # Use np.mgrid to create a full grid
    x, y, z = np.mgrid[x_min:x_max:resolution, 0:y_max:resolution, z_min:z_max:resolution]
    # x, y, z = x[:X, :Y, :Z], y[:X, :Y, :Z], z[:X, :Y, :Z]  # Ensure dimensions match

    # Calculate spherical coordinates
    r_from = np.sqrt(x ** 2 + y ** 2 + z ** 2)
    r_to = np.sqrt((x + resolution) ** 2 + (y + resolution) ** 2 + (z + resolution) ** 2)
    azimuth_from = np.arctan2(y, x) - np.pi / 2
    azimuth_to = np.arctan2(y + resolution, x + resolution) - np.pi / 2
    elevation_from = np.arcsin(z / np.maximum(r_from, 1e-9))
    elevation_to = np.arcsin((z + resolution) / np.maximum(r_to, 1e-9))

    # Correct ordering if necessary
    r_from, r_to = np.minimum(r_from, r_to), np.maximum(r_from, r_to)
    azimuth_from, azimuth_to = np.minimum(azimuth_from, azimuth_to), np.maximum(azimuth_from, azimuth_to)
    elevation_from, elevation_to = np.minimum(elevation_from, elevation_to), np.maximum(elevation_from, elevation_to)

    # Compute masks for each dimension
    azimuth_mask = (azimuth_bins[:, 0][..., None, None, None] <= azimuth_to) & (azimuth_bins[:, 1][..., None, None, None] >= azimuth_from)
    elevation_mask = (elevation_bins[:, 0][..., None, None, None] <= elevation_to) & (elevation_bins[:, 1][..., None, None, None] >= elevation_from)
    range_mask = (range_bins[:, None, None, None] <= r_to) & (range_bins[:, None, None, None] + range_bin_width >= r_from)

    for n in range(N):
        for i in range(X):
            for j in range(Y):
                for k in range(Z):
                    if grids[n, i, j, k] == -1:  # Skip invalid points
                        continue

                    # Aggregate values into polar grid for valid bins
                    try:
                        for az_idx in np.where(azimuth_mask[:, i, j, k])[0]:
                            for el_idx in np.where(elevation_mask[:, i, j, k])[0]:
                                for r_idx in np.where(range_mask[:, i, j, k])[0]:
                                    polar_grid[n, el_idx, az_idx, r_idx] = max(polar_grid[n, el_idx, az_idx, r_idx], grids[n, i, j, k])
                    except:
                        # print(N, X, Y, Z)
                        # print(i, j, k)
                        raise

    return polar_grid


def main():
    occupied_threshold = 0.75
    empty_threshold = 0.25
    x_max, y_max, z_max = 13.25, 8, 4.5
    azimuth_from, azimuth_to = 4, 59  # from -59.68° to 59.68°
    elevation_from, elevation_to = 4, 11  # from -38° to 38°

    ds_file = '/home/ann/mapping/mn_ws/src/mapless-navigation/dataset_7runs_rangelimit.pkl'
    if os.path.isfile(ds_file):
        with open(ds_file, 'rb') as f:
            data = pickle.load(f)
    print(data['ec_hallways_run1'].keys())

    params = data.pop('params')['heatmap']
    range_bin_width = round(params['range_bin_width'], 3)
    range_bins = np.arange(0, y_max, range_bin_width)
    print(range_bins)
    azimuth_bins = parse_polar_bins(params['azimuth_bins'])[azimuth_from:azimuth_to + 1]
    elevation_bins = parse_polar_bins(params['elevation_bins'])[elevation_from:elevation_to + 1]

    for idx in (49, 99, 199):
        # print('frame', idx + 1)
        # grid = data['ec_hallways_run1']['gt_grids'][idx]
        # grid_known = grid[grid != -1]
        # print(grid_known.shape, grid.shape)
        # print('not empty', grid[(grid > empty_threshold)].size)
        # print('not occupied', grid[(grid < occupied_threshold)].size)
        # print('known occupied', grid_known[(grid_known >= occupied_threshold)].size)
        # print('known empty', grid_known[(grid_known <= empty_threshold)].size)
        #
        # polar_grids = cartesian_to_polar_grid(
        #         np.array([grid]), x_min=-x_max, x_max=x_max, y_max=y_max, z_min=-z_max, z_max=z_max,
        #         azimuth_bins=azimuth_bins, elevation_bins=elevation_bins, range_bins=range_bins
        #     )
        # print(polar_grids.shape)
        # print(polar_grids[polar_grids != -np.inf].shape)
        polar_grid = data['ec_hallways_run1']['polar_grids'][idx]
        # grid[(grid == -1) | (grid == -np.inf)] = 0.5
        print('polar unknown', polar_grid[polar_grid < 0].size / polar_grid.size, '%')
        save_heatmap_image(heatmap=polar_grid, filename=f'grid_polar{idx + 1}_raw.png')

        polar_grid[(polar_grid == -1) | (polar_grid == -np.inf)] = 0.5
        # print('polar not empty', polar_grid[(polar_grid > empty_threshold)].size)
        # print('polar not occupied', polar_grid[(polar_grid < occupied_threshold)].size)
        uncertain_count = polar_grid[(polar_grid > empty_threshold) & (polar_grid < occupied_threshold)].size
        occupied_count = polar_grid[polar_grid >= occupied_threshold].size
        print(uncertain_count / polar_grid.size, '% uncertain')
        print(occupied_count / polar_grid.size, '% occupied')
        print(polar_grid[polar_grid <= empty_threshold].size / polar_grid.size, '% empty')
        save_heatmap_image(heatmap=polar_grid, filename=f'grid_polar{idx + 1}.png')
        print()
    return

    totals, occupied_num, empty_num = [], [], []

    for run_name, run_data in data.items()[:1]:
        grids = np.array(run_data['gt_grids'])
        polar_grids = cartesian_to_polar_grid(
            grids, x_min=-x_max, x_max=x_max, y_max=y_max, z_min=-z_max, z_max=z_max,
            azimuth_bins=azimuth_bins, elevation_bins=elevation_bins, range_bins=range_bins
        )

        # grids[grids == -1] = 0.5
        # grids_in_fov = select_fov(grids, azimuth_angle=94, elevation_angle=36)
        grids_in_fov = grids[grids != -1.0]
        print(run_name, 'shape', grids_in_fov.shape, 'total shape', grids.shape, 'ratio', grids_in_fov.size / grids.size)

        occupied_num.append(grids_in_fov[grids_in_fov >= occupied_threshold].size)
        empty_num.append(grids_in_fov[grids_in_fov <= empty_threshold].size)
        print(
            round(occupied_num[-1] / grids_in_fov.size, 4),
            round(empty_num[-1] / grids_in_fov.size, 4),
            round((grids_in_fov.size - empty_num[-1] - occupied_num[-1]) / grids_in_fov.size, 4)
        )

    print(totals, occupied_num, empty_num)


if __name__ == '__main__':
    main()
