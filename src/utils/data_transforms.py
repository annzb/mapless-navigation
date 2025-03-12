import numpy as np

from utils.radar_config import RadarConfig


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


# def get_aligned_indices(point_cloud, grid_resolution, grid_limits, tolerance=1e-2):
#     """
#     Return the indices of points in the point cloud that are aligned with the grid.
#
#     A point is considered aligned if, for each axis (x, y, z), its coordinate (after subtracting
#     the grid minimum for that axis) is an integer multiple of grid_resolution, within tolerance.
#     Only points within the grid limits are considered.
#
#     Parameters:
#         point_cloud (np.ndarray): An array of shape [N, 4] where the first three columns are X, Y, Z.
#         grid_resolution (float): The spacing between grid nodes.
#         grid_limits (tuple or list): A sequence of six numbers (x_min, x_max, y_min, y_max, z_min, z_max).
#         tolerance (float): Allowed numerical tolerance for checking alignment (default 1e-3).
#
#     Returns:
#         np.ndarray: A 1D array of indices of points that are aligned with the grid.
#     """
#     # Unpack the grid limits
#     x_min, x_max, y_min, y_max, z_min, z_max = grid_limits
#
#     # Extract spatial coordinates from the point cloud.
#     coords = point_cloud[:, :3]
#
#     # Ensure the points are within the grid limits.
#     within_limits = (
#         (coords[:, 0] >= x_min) & (coords[:, 0] <= x_max) &
#         (coords[:, 1] >= y_min) & (coords[:, 1] <= y_max) &
#         (coords[:, 2] >= z_min) & (coords[:, 2] <= z_max)
#     )
#
#     # Compute the offset from the grid minimum for each axis and normalize by the resolution.
#     diff = (coords - np.array([x_min, y_min, z_min])) / grid_resolution
#
#     # For each axis, the coordinate is aligned if the normalized offset is nearly an integer.
#     aligned_x = np.abs(diff[:, 0] - np.round(diff[:, 0])) < tolerance
#     aligned_y = np.abs(diff[:, 1] - np.round(diff[:, 1])) < tolerance
#     aligned_z = np.abs(diff[:, 2] - np.round(diff[:, 2])) < tolerance
#
#     # Combine the conditions.
#     aligned_mask = within_limits & aligned_x & aligned_y & aligned_z
#
#     # Return the indices of points that are aligned.
#     return np.nonzero(aligned_mask)[0]
#
#
# from scipy.spatial import cKDTree
#
#
# def validate_octomap_pointcloud(point_cloud, tolerance=1e-3):
#     if point_cloud.shape[0] < 2:
#         raise ValueError(f"Bad cloud shape {point_cloud.shape}")
#
#     # Use all columns for misaligned point output, but only use the first three for checking grid alignment.
#     coords = point_cloud[:, :3]
#
#     # Build a KD-tree for efficient nearest neighbor lookup.
#     tree = cKDTree(coords)
#     # Query each point's two closest neighbors (the first is the point itself).
#     distances, _ = tree.query(coords, k=2)
#
#     # Estimated resolution is taken as the smallest nonzero nearest neighbor distance.
#     estimated_resolution = np.round(np.min(distances[:, 1]), 3)
#
#     # Use the first point as the reference ("origin").
#     origin = coords[0]
#
#     misaligned_points_list, aligned_points_list = [], []
#
#     # Check each point's coordinates relative to the origin.
#     for idx, point in enumerate(coords):
#         diff = point - origin
#         # For this point, check each axis.
#         for axis in range(3):
#             # Compute the ratio of the difference to the resolution.
#             ratio = diff[axis] / estimated_resolution
#             # If the ratio isn't nearly an integer, mark the point as misaligned.
#             if abs(ratio - round(ratio)) > tolerance:
#                 misaligned_points_list.append(point_cloud[idx])
#                 # Once one axis fails, we don't need to check further axes for this point.
#                 break
#             else:
#                 aligned_points_list.append(point_cloud[idx])
#
#     # Convert the list to a NumPy array with shape [M, 4].
#     misaligned_points = np.array(misaligned_points_list) if misaligned_points_list else np.empty((0, 4))
#     aligned_points = np.array(aligned_points_list) if aligned_points_list else np.empty((0, 4))
#
#     return aligned_points, misaligned_points, estimated_resolution
