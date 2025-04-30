import numpy as np
import torch

from utils.radar_config import RadarConfig


class NumpyDataTransform:
    def __init__(self, radar_config: RadarConfig):
        self._storage_type = np.ndarray
        self.radar_config = radar_config
        el_bins = np.array(radar_config.clipped_elevation_bins, dtype=float)
        az_bins = np.array(radar_config.clipped_azimuth_bins, dtype=float)
        r_bins  = np.linspace(
            0,
            (radar_config.num_range_bins - 1) * radar_config.range_bin_width,
            radar_config.num_range_bins,
            dtype=float
        )
        el_grid, az_grid, r_grid = np.meshgrid(el_bins, az_bins, r_bins, indexing="ij")
        x = r_grid * np.cos(el_grid) * np.sin(az_grid)
        y = r_grid * np.cos(el_grid) * np.cos(az_grid)
        z = r_grid * np.sin(el_grid)
        self.cartesian_coords = np.stack((x, y, z), axis=-1).reshape(-1, 3)

    def process_grid_input(self, samples, **kwargs):
        if len(samples) < 1:
            raise ValueError("Empty samples")
        if not isinstance(samples, self._storage_type):
            raise TypeError(f"samples must be of type {self._storage_type}")
        
        multiple = False
        if len(samples.shape) == 3:
            samples = np.expand_dims(samples, 0)
        elif len(samples.shape) == 4:
            multiple = True
        else:
            raise ValueError("Heatmap must be of shape (El, Az, R) or (B, El, Az, R)")
        N, El, Az, R = samples.shape
        if El != self.radar_config.num_elevation_bins or R != self.radar_config.num_range_bins or Az != self.radar_config.num_azimuth_bins:
            raise ValueError("Heatmap shape does not match radar config's clipped bin dimensions.")

        return samples, multiple
    
    def process_point_input(self, samples, **kwargs):
        if len(samples) < 1:
            raise ValueError("Empty samples")
        
        multiple = False
        if isinstance(samples, self._storage_type):  # input is an np.array if all clouds are the same size or there's only one cloud
            if len(samples.shape) == 2:
                samples = [samples]                  # convert to collection of one cloud
            elif len(samples.shape) == 3:
                multiple = True
            else:
                raise ValueError("samples must be a list or a numpy array of shape (B, N, 4) or (N, 4)")
            if samples[0].shape[1] != 4:
                raise ValueError("samples must be a list or a numpy array of shape (B, N, 4) or (N, 4)")
        else:
            if isinstance(samples, list):  # input is a list if clouds may have different sizes
                multiple = True
            for sample in samples:
                if not isinstance(sample, self._storage_type):
                    raise TypeError(f"Each element in samples must be of type {self._storage_type}")
                if len(sample.shape) != 2 or sample.shape[1] != 4:
                    raise ValueError("Each element in samples must be of shape (N,4).")
                
        # always return a list of clouds to support different sizes
        return samples, multiple
        

    def polar_grid_to_cartesian_points(self, grids, **kwargs):
        samples, multiple = self.process_grid_input(grids, **kwargs)
        B = samples.shape[0]
        flat_int = samples.reshape(B, -1)
        coords_batch = np.tile(self.cartesian_coords[None, ...],(B, 1, 1))
        intensities = flat_int[..., None]
        points_all = np.concatenate((coords_batch, intensities), axis=2)
        return points_all if multiple else points_all[0]
    
    def filter_point_intensity(self, points, threshold=0.0, **kwargs):
        cloud_list, input_has_multiple_samples = self.process_point_input(points, **kwargs)
        filtered_clouds, nonempty_cloud_idx = [], []
        for i in range(len(cloud_list)):
            filtered_cloud = cloud_list[i][cloud_list[i][:, 3] > threshold]
            if filtered_cloud.shape[0] != 0:
                filtered_clouds.append(filtered_cloud)
                nonempty_cloud_idx.append(i)
        return (filtered_clouds if input_has_multiple_samples else filtered_clouds[0]), np.array(nonempty_cloud_idx)
    
    def scale_point_intensity(self, points, intensity_mean=None, intensity_std=None, **kwargs):
        if None in (intensity_mean, intensity_std) and intensity_mean != intensity_std:
            raise ValueError("Both intensity_mean and intensity_std must be provided, or neither.")
        cloud_list, input_has_multiple_samples = self.process_point_input(points, **kwargs)
        if intensity_mean is None:
            all_points = np.vstack(cloud_list)
            intensity_mean = float(np.mean(all_points[:, 3]))
            intensity_std  = float(np.std(all_points[:, 3]))
        scaled_clouds = []
        for cloud in cloud_list:
            cloud_scaled = cloud.astype(np.float32, copy=True)
            cloud_scaled[:, 3] = (cloud_scaled[:, 3] - intensity_mean) / intensity_std
            scaled_clouds.append(cloud_scaled)
        return (scaled_clouds if input_has_multiple_samples else scaled_clouds[0]), intensity_mean, intensity_std

    def scale_point_coords(self, points, coord_means=None, coord_stds=None, **kwargs):
        if (coord_means is None) != (coord_stds is None):
            raise ValueError("Both coord_means and coord_stds must be provided, or neither.")
        cloud_list, input_has_multiple_samples = self.process_point_input(points, **kwargs)
        if coord_means is None:
            all_points = np.vstack(cloud_list)
            coord_means = np.mean(all_points[:, :3], axis=0)  # (3,)
            coord_stds = np.std(all_points[:, :3], axis=0) + 1e-6  # (3,) avoid div-by-zero
        scaled_clouds = []
        for cloud in cloud_list:
            cloud_scaled = cloud.astype(np.float32, copy=True)
            cloud_scaled[:, :3] = (cloud_scaled[:, :3] - coord_means) / coord_stds
            scaled_clouds.append(cloud_scaled)
        return (scaled_clouds if input_has_multiple_samples else scaled_clouds[0]), coord_means, coord_stds


# class TorchDataTransform:
#     def __init__(self, radar_config: RadarConfig):
#         self._storage_type = np.ndarray
#         self.radar_config = radar_config
#         el_bins = torch.tensor(radar_config.clipped_elevation_bins, dtype=torch.float32)
#         az_bins = torch.tensor(radar_config.clipped_azimuth_bins, dtype=torch.float32)
#         r_bins = torch.linspace(0, (radar_config.num_range_bins - 1) * radar_config.range_bin_width,
#                                 radar_config.num_range_bins)
#         el_grid, az_grid, r_grid = torch.meshgrid(el_bins, az_bins, r_bins, indexing="ij")
#         x = r_grid * torch.cos(el_grid) * torch.sin(az_grid)
#         y = r_grid * torch.cos(el_grid) * torch.cos(az_grid)
#         z = r_grid * torch.sin(el_grid)
#         self.cartesian_coords = torch.stack([x, y, z], dim=-1).reshape(-1, 3)
#
#     def process_grid_input(self, samples, **kwargs):
#         if not samples.ndim or not samples.size:
#             raise ValueError("Empty samples")
#         if not isinstance(samples, self._storage_type):
#             raise TypeError(f"samples must be of type {self._storage_type}")
#
#         multiple = False
#         if len(samples.shape) == 3:
#             samples = np.expand_dims(samples, 0)
#         elif len(samples.shape) == 4:
#             multiple = True
#         else:
#             raise ValueError("Heatmap must be of shape (El, Az, R) or (B, El, Az, R)")
#         N, El, Az, R = samples.shape
#         if El != self.radar_config.num_elevation_bins or R != self.radar_config.num_range_bins or Az != self.radar_config.num_azimuth_bins:
#             raise ValueError("Heatmap shape does not match radar config's clipped bin dimensions.")
#
#         return samples, multiple
#
#     def process_point_input(self, samples, **kwargs):
#         if not samples.ndim or not samples.size:
#             raise ValueError("Empty samples")
#
#         multiple = False
#         if isinstance(samples,
#                       self._storage_type):  # input is an np.array if all clouds are the same size or there's only one cloud
#             if len(samples.shape) == 2:
#                 samples = [samples]  # convert to collection of one cloud
#             elif len(samples.shape) == 3:
#                 multiple = True
#             else:
#                 raise ValueError("samples must be a list or a numpy array of shape (B, N, 4) or (N, 4)")
#             if samples[0].shape[1] != 4:
#                 raise ValueError("samples must be a list or a numpy array of shape (B, N, 4) or (N, 4)")
#         else:
#             if isinstance(samples, list):  # input is a list if clouds may have different sizes
#                 multiple = True
#             for sample in samples:
#                 if not isinstance(sample, self._storage_type):
#                     raise TypeError(f"Each element in samples must be of type {self._storage_type}")
#                 if len(sample.shape) != 2 or sample.shape[1] != 4:
#                     raise ValueError("Each element in samples must be of shape (N,4).")
#
#         # always return a list of clouds to support different sizes
#         return samples, multiple
#
#     def polar_grid_to_cartesian_points(self, grids, **kwargs):
#         grids, input_has_multiple_samples = self.process_grid_input(grids, **kwargs)
#         num_samples = grids.shape[0]
#         grids_flat = grids.reshape(num_samples, -1)
#         coords_batch = self.cartesian_coords.unsqueeze(0).expand(num_samples, -1, -1)
#         intensities = grids_flat.unsqueeze(-1)
#         points_all = torch.cat((coords_batch, intensities), dim=2)
#         points = points_all if input_has_multiple_samples else points_all.squeeze(0)
#         return points


def polar_to_cartesian_grid(heatmap, radar_config, device=None):
    if isinstance(heatmap, np.ndarray):
        heatmap = torch.from_numpy(heatmap)
    if device is not None:
        heatmap = heatmap.to(device)

    if heatmap.dim() == 3:
        heatmap = heatmap.unsqueeze(0)  # (1, El, R, Az)
        batched = False
    elif heatmap.dim() == 4:
        batched = True
    else:
        raise ValueError("Heatmap must be of shape (El, R, Az) or (B, El, R, Az)")
    B, El, R, Az = heatmap.shape
    if El != radar_config.num_elevation_bins or R != radar_config.num_range_bins or Az != radar_config.num_azimuth_bins:
        raise ValueError("Heatmap shape does not match radar config's clipped bin dimensions.")

    el_bins = torch.tensor(radar_config.clipped_elevation_bins, dtype=torch.float32, device=heatmap.device)
    az_bins = torch.tensor(radar_config.clipped_azimuth_bins, dtype=torch.float32, device=heatmap.device)
    r_bins = torch.linspace(0, (R - 1) * radar_config.range_bin_width, R, device=heatmap.device)
    el_grid, r_grid, az_grid = torch.meshgrid(el_bins, r_bins, az_bins, indexing="ij")  # (El, R, Az)

    x = r_grid * torch.cos(el_grid) * torch.sin(az_grid)
    y = r_grid * torch.cos(el_grid) * torch.cos(az_grid)
    z = r_grid * torch.sin(el_grid)
    coords = torch.stack([x, y, z], dim=-1).reshape(-1, 3)  # shape: (N_vox, 3)

    # Voxel grid config
    voxel_size = radar_config.grid_resolution
    min_x, max_x, min_y, max_y, min_z, max_z = radar_config.point_range
    grid_size_xyz = radar_config.grid_size  # (X, Y, Z)
    grid_size_zyx = (grid_size_xyz[2], grid_size_xyz[1], grid_size_xyz[0])  # (Z, Y, X)

    # Convert real coords to voxel indices
    voxel_origin = torch.tensor([min_x, min_y, min_z], dtype=torch.float32, device=coords.device)
    voxel_indices = ((coords - voxel_origin) / voxel_size).long()  # (N_vox, 3)

    # Keep only valid indices
    valid_mask = (
        (voxel_indices[:, 0] >= 0) & (voxel_indices[:, 0] < grid_size_xyz[0]) &
        (voxel_indices[:, 1] >= 0) & (voxel_indices[:, 1] < grid_size_xyz[1]) &
        (voxel_indices[:, 2] >= 0) & (voxel_indices[:, 2] < grid_size_xyz[2])
    )
    voxel_indices = voxel_indices[valid_mask]
    voxel_indices_zyx = voxel_indices[:, [2, 1, 0]]  # convert (X,Y,Z) → (Z,Y,X)

    # Compute flat indices for efficient accumulation
    flat_indices = (
        voxel_indices_zyx[:, 0] * (grid_size_zyx[1] * grid_size_zyx[2]) +
        voxel_indices_zyx[:, 1] * grid_size_zyx[2] +
        voxel_indices_zyx[:, 2]
    )
    # Prepare output grid
    output = torch.zeros((B, *grid_size_zyx), dtype=heatmap.dtype, device=heatmap.device)

    # Flatten and index valid polar heatmap values
    heatmap_flat = heatmap.reshape(B, -1)[:, valid_mask]  # (B, valid_voxels)

    # Aggregate intensity values into cartesian grid
    for b in range(B):
        output[b].reshape(-1).index_add_(0, flat_indices, heatmap_flat[b])
    return output if batched else output[0]


def polar_grid_to_cartesian_points(heatmap, radar_config, device=None):
    if isinstance(heatmap, np.ndarray):
        heatmap = torch.from_numpy(heatmap)
    if device is not None:
        heatmap = heatmap.to(device)

    if heatmap.dim() == 3:
        heatmap = heatmap.unsqueeze(0)  # (1, El, R, Az)
        batched = False
    elif heatmap.dim() == 4:
        batched = True
    else:
        raise ValueError("Heatmap must be of shape (El, R, Az) or (B, El, R, Az)")
    B, El, R, Az = heatmap.shape
    if El != radar_config.num_elevation_bins or R != radar_config.num_range_bins or Az != radar_config.num_azimuth_bins:
        raise ValueError("Heatmap shape does not match radar config's clipped bin dimensions.")

    el_bins = torch.tensor(radar_config.clipped_elevation_bins, dtype=torch.float32, device=heatmap.device)
    az_bins = torch.tensor(radar_config.clipped_azimuth_bins, dtype=torch.float32, device=heatmap.device)
    r_bins = torch.linspace(0, (R - 1) * radar_config.range_bin_width, R, device=heatmap.device)
    el_grid, r_grid, az_grid = torch.meshgrid(el_bins, r_bins, az_bins, indexing="ij")

    x = r_grid * torch.cos(el_grid) * torch.sin(az_grid)
    y = r_grid * torch.cos(el_grid) * torch.cos(az_grid)
    z = r_grid * torch.sin(el_grid)
    coords = torch.stack((x, y, z), dim=-1).reshape(-1, 3)

    heat_flat = heatmap.reshape(B, -1)
    coords_batch = coords.unsqueeze(0).expand(B, -1, -1)
    intensities = heat_flat.unsqueeze(-1)               
    points_all = torch.cat((coords_batch, intensities), dim=2)
    points = points_all if batched else points_all.squeeze(0)
    return points


def filter_point_intensity(
        points: torch.Tensor | np.ndarray, 
        batch_indices: torch.Tensor | np.ndarray = None, 
        device: torch.device | str = None, 
        threshold: float = 0.0
    ) -> tuple[torch.Tensor, torch.Tensor]:
    if isinstance(points, np.ndarray):
        points = torch.from_numpy(points)
    if device is not None:
        points = points.to(device)
    points = points.to(torch.float32)

    # Build flat_points (P,4) and batch_idx (P,)
    if batch_indices is None:
        if points.dim() == 2 and points.shape[1] == 4:
            flat_points = points
            batch_idx = torch.zeros(points.shape[0], dtype=torch.long, device=points.device)
        elif points.dim() == 3 and points.shape[2] == 4:
            B, N, _ = points.shape
            flat_points = points.reshape(-1, 4)
            batch_idx = torch.arange(B, device=points.device, dtype=torch.long).unsqueeze(1).expand(B, N).reshape(-1)
        else:
            raise ValueError("When batch_indices is None, points must be shape (N,4) or (B,N,4).")
    else:
        if isinstance(batch_indices, np.ndarray):
            batch_idx = torch.from_numpy(batch_indices)
        else:
            batch_idx = batch_indices.clone()
        if device is not None:
            batch_idx = batch_idx.to(device)
        flat_points = points
        if flat_points.dim() != 2 or flat_points.shape[1] != 4:
            raise ValueError("With batch_indices, points must be flat of shape (P,4).")
        if batch_idx.dim() != 1 or batch_idx.shape[0] != flat_points.shape[0]:
            raise ValueError("batch_indices must have shape (P,) matching points.")

    mask = flat_points[:, 3] > threshold
    filtered_points = flat_points[mask]
    filtered_batch_idx = batch_idx[mask]
    return filtered_points, filtered_batch_idx


def scale_point_intensity(
        points: torch.Tensor | np.ndarray,
        intensity_mean: float = None,
        intensity_std: float = None,
        device: torch.device | str = None,
        batch_indices: torch.Tensor | np.ndarray = None
) -> tuple[torch.Tensor, float, float]:
    if (intensity_mean is None) ^ (intensity_std is None):
        raise ValueError("Both intensity_mean and intensity_std must be provided, or neither.")

    if isinstance(points, np.ndarray):
        points = torch.from_numpy(points)
    if device is not None:
        points = points.to(device)

    points = points.to(torch.float32)

    if batch_indices is not None:
        if isinstance(batch_indices, np.ndarray):
            batch_idx = torch.from_numpy(batch_indices)
        else:
            batch_idx = batch_indices.clone()
        if device is not None:
            batch_idx = batch_idx.to(device)
        if points.dim() != 2 or points.shape[1] != 4:
            raise ValueError("With batch_indices, points must be flat of shape (P,4).")
        intensities = points[:, 3]
    else:
        if points.dim() == 2 and points.shape[1] == 4:
            intensities = points[:, 3]
        elif points.dim() == 3 and points.shape[2] == 4:
            intensities = points[..., 3].reshape(-1)
        else:
            raise ValueError("Without batch_indices, points must be of shape (N,4) or (B,N,4).")

    if intensity_mean is None:
        vals = intensities.detach().cpu().numpy()
        intensity_mean = float(np.mean(vals))
        intensity_std  = float(np.std(vals))
    scaled = points.clone()
    if batch_indices is not None:
        scaled[:, 3] = (intensities - intensity_mean) / intensity_std
    else:
        scaled[..., 3] = (intensities - intensity_mean) / intensity_std
    return scaled, intensity_mean, intensity_std


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
