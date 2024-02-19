import os
import pickle
import numpy as np

from src.learning.utils import save_heatmap_image


def select_fov(grids, azimuth_angle, elevation_angle):
    tan_azimuth = np.tan(np.radians(90 - azimuth_angle / 2))
    tan_elevation = np.tan(np.radians(90 - elevation_angle / 2))

    x, y, z = np.meshgrid(np.arange(grids.shape[1]), np.arange(grids.shape[2]), np.arange(grids.shape[3]), indexing='ij')
    azimuth_mask = y ** 2 >= (tan_azimuth * x) ** 2
    elevation_mask = y ** 2 >= (tan_elevation * z) ** 2
    fov_mask = azimuth_mask & elevation_mask
    selected_elements = grids[:, fov_mask]
    return selected_elements


def cartesian_to_polar_grid(
        grids, x_min, x_max, y_max, z_min, z_max,
        azimuth_bins, elevation_bins, range_bins
):
    N, X, Y, Z = grids.shape
    # Calculate range bins based on the range step and the maximum range
    # max_range = np.sqrt(x_max ** 2 + y_max ** 2 + z_max ** 2)
    # range_bins = np.arange(0, max_range, range_step)

    # Initialize the polar grid
    polar_grid_shape = (N, len(azimuth_bins) - 1, len(elevation_bins) - 1, len(range_bins) - 1)
    polar_grid = np.full(polar_grid_shape, -np.inf)  # Use -inf for proper max aggregation
    # Calculate the center coordinates of each voxel
    x_coords = np.linspace(x_min, x_max, X)
    y_coords = np.linspace(0, y_max, Y)
    z_coords = np.linspace(z_min, z_max, Z)

    for n in range(N):
        for i, x in enumerate(x_coords):
            for j, y in enumerate(y_coords):
                for k, z in enumerate(z_coords):
                    # Cartesian to polar conversion
                    r = np.sqrt(x ** 2 + y ** 2 + z ** 2)
                    azimuth = np.arctan2(y, x)
                    elevation = np.arcsin(z / r)
                    # Find the corresponding bins
                    azimuth_idx = np.digitize(azimuth, azimuth_bins) - 1
                    elevation_idx = np.digitize(elevation, elevation_bins) - 1
                    range_idx = np.digitize(r, range_bins) - 1

                    # Aggregate values into polar grid
                    if 0 <= azimuth_idx < polar_grid.shape[1] and 0 <= elevation_idx < polar_grid.shape[
                        2] and 0 <= range_idx < polar_grid.shape[3]:
                        polar_grid[n, azimuth_idx, elevation_idx, range_idx] = max(
                            polar_grid[n, azimuth_idx, elevation_idx, range_idx], grids[n, i, j, k])

    return polar_grid


def main():
    occupied_threshold = 0.75
    empty_threshold = 0.25
    x_max, y_max, z_max = 8.5, 8, 2.5
    azimuth_from, azimuth_to = 8, 55  # from -46.46° to 46.46°
    elevation_from, elevation_to = 5, 10  # from -17.97 to 17.97

    ds_file = '/home/ann/mapping/mn_ws/src/mapless-navigation/dataset_7runs_smallfov.pkl'
    if os.path.isfile(ds_file):
        with open(ds_file, 'rb') as f:
            data = pickle.load(f)

    params = data.pop('params')['heatmap']
    range_bins = np.arange(0, y_max, round(params['range_bin_width'], 3))
    azimuth_bins = params['azimuth_bins'][azimuth_from:azimuth_to + 1]
    elevation_bins = params['elevation_bins'][elevation_from:elevation_to + 1]

    grid = data['ec_hallways_run0']['gt_grids'][9]
    polar_grid = cartesian_to_polar_grid(
            np.array([grid]), x_min=-x_max, x_max=x_max, y_max=y_max, z_min=-z_max, z_max=z_max,
            azimuth_bins=azimuth_bins, elevation_bins=elevation_bins, range_bins=range_bins
        )
    print(polar_grid.shape)
    return
    save_heatmap_image(heatmap=polar_grid, filename=f'grid_polar10.png')
    raise

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
