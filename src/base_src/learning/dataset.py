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


# def validate_octomap_pointcloud(point_cloud, tolerance=1e-3):
#     if point_cloud.shape[0] < 2:
#         raise ValueError(f'Bad cloud shape {point_cloud.shape}')
#     coords = point_cloud[:, :3]
#     tree = cKDTree(coords)
#     distances, _ = tree.query(coords, k=2)
#     estimated_resolution = np.round(np.min(distances[:, 1]), 3)
#     remainder = np.mod(coords, estimated_resolution / 2)
#     aligned = np.all((remainder < tolerance) | (remainder > ((estimated_resolution / 2) - tolerance)))
#     return aligned, estimated_resolution


def validate_octomap_pointcloud(point_cloud, tolerance=1e-3):
    if point_cloud.shape[0] < 2:
        raise ValueError(f'Bad cloud shape {point_cloud.shape}')

    # Extract only the spatial coordinates (X, Y, Z) from the point cloud.
    coords = point_cloud[:, :3]

    # Build a KD-tree for efficient nearest neighbor lookup.
    tree = cKDTree(coords)
    # Query for each point's two closest neighbors (the first is the point itself).
    distances, indices = tree.query(coords, k=2)

    # The estimated resolution is taken as the smallest non-zero distance among nearest neighbors.
    estimated_resolution = np.round(np.min(distances[:, 1]), 3)

    # Compute the deviation of each point's nearest neighbor distance from the estimated resolution.
    deviations = np.abs(distances[:, 1] - estimated_resolution)
    inconsistent_points = np.where(deviations > tolerance)[0]

    if inconsistent_points.size > 0:
        print(f"Found {inconsistent_points.size} points with inconsistent neighbor distances:")
        for idx in inconsistent_points:
            actual_distance = distances[idx, 1]
            deviation = deviations[idx]
            point_coords = coords[idx]
            # Retrieve the neighbor's index and then its coordinates.
            neighbor_idx = indices[idx, 1]
            neighbor_coords = coords[neighbor_idx]
            print(
                f"Point {idx} at coordinates {point_coords} has neighbor {neighbor_idx} at coordinates {neighbor_coords} "
                f"with distance = {actual_distance:.6f} (deviation = {deviation:.6f} from expected {estimated_resolution:.6f})."
            )
            input()
        valid = False
    else:
        valid = True

    return valid, estimated_resolution


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


def get_dataset(dataset_file_path, dataset_type, partial=1.0, batch_size=16, shuffle_runs=True, random_state=42, grid_voxel_size=1.0):
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
    filtered_indices = [i for i, frame in enumerate(lidar_frames) if len(frame) > 0]  # and any(frame[:, 3] >= occupancy_threshold)]
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


def get_aligned_indices(point_cloud, grid_resolution, grid_limits, tolerance=1e-2):
    """
    Return the indices of points in the point cloud that are aligned with the grid.

    A point is considered aligned if, for each axis (x, y, z), its coordinate (after subtracting
    the grid minimum for that axis) is an integer multiple of grid_resolution, within tolerance.
    Only points within the grid limits are considered.

    Parameters:
        point_cloud (np.ndarray): An array of shape [N, 4] where the first three columns are X, Y, Z.
        grid_resolution (float): The spacing between grid nodes.
        grid_limits (tuple or list): A sequence of six numbers (x_min, x_max, y_min, y_max, z_min, z_max).
        tolerance (float): Allowed numerical tolerance for checking alignment (default 1e-3).

    Returns:
        np.ndarray: A 1D array of indices of points that are aligned with the grid.
    """
    # Unpack the grid limits
    x_min, x_max, y_min, y_max, z_min, z_max = grid_limits

    # Extract spatial coordinates from the point cloud.
    coords = point_cloud[:, :3]

    # Ensure the points are within the grid limits.
    within_limits = (
        (coords[:, 0] >= x_min) & (coords[:, 0] <= x_max) &
        (coords[:, 1] >= y_min) & (coords[:, 1] <= y_max) &
        (coords[:, 2] >= z_min) & (coords[:, 2] <= z_max)
    )

    # Compute the offset from the grid minimum for each axis and normalize by the resolution.
    diff = (coords - np.array([x_min, y_min, z_min])) / grid_resolution

    # For each axis, the coordinate is aligned if the normalized offset is nearly an integer.
    aligned_x = np.abs(diff[:, 0] - np.round(diff[:, 0])) < tolerance
    aligned_y = np.abs(diff[:, 1] - np.round(diff[:, 1])) < tolerance
    aligned_z = np.abs(diff[:, 2] - np.round(diff[:, 2])) < tolerance

    # Combine the conditions.
    aligned_mask = within_limits & aligned_x & aligned_y & aligned_z

    # Return the indices of points that are aligned.
    return np.nonzero(aligned_mask)[0]


from scipy.spatial import cKDTree


def validate_octomap_pointcloud(point_cloud, tolerance=1e-3):
    if point_cloud.shape[0] < 2:
        raise ValueError(f"Bad cloud shape {point_cloud.shape}")

    # Use all columns for misaligned point output, but only use the first three for checking grid alignment.
    coords = point_cloud[:, :3]

    # Build a KD-tree for efficient nearest neighbor lookup.
    tree = cKDTree(coords)
    # Query each point's two closest neighbors (the first is the point itself).
    distances, _ = tree.query(coords, k=2)

    # Estimated resolution is taken as the smallest nonzero nearest neighbor distance.
    estimated_resolution = np.round(np.min(distances[:, 1]), 3)

    # Use the first point as the reference ("origin").
    origin = coords[0]

    misaligned_points_list, aligned_points_list = [], []

    # Check each point's coordinates relative to the origin.
    for idx, point in enumerate(coords):
        diff = point - origin
        # For this point, check each axis.
        for axis in range(3):
            # Compute the ratio of the difference to the resolution.
            ratio = diff[axis] / estimated_resolution
            # If the ratio isn't nearly an integer, mark the point as misaligned.
            if abs(ratio - round(ratio)) > tolerance:
                misaligned_points_list.append(point_cloud[idx])
                # Once one axis fails, we don't need to check further axes for this point.
                break
            else:
                aligned_points_list.append(point_cloud[idx])

    # Convert the list to a NumPy array with shape [M, 4].
    misaligned_points = np.array(misaligned_points_list) if misaligned_points_list else np.empty((0, 4))
    aligned_points = np.array(aligned_points_list) if aligned_points_list else np.empty((0, 4))

    return aligned_points, misaligned_points, estimated_resolution