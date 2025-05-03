import numpy as np

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
        

    def polar_grid_to_cartesian_points(self, samples, **kwargs):
        samples, multiple = self.process_grid_input(samples, **kwargs)
        B = samples.shape[0]
        num_points = self.cartesian_coords.shape[0]
        points_all = np.empty((B, num_points, 4), dtype=samples.dtype)
        points_all[:, :, :3] = self.cartesian_coords
        points_all[:, :, 3:4] = samples.reshape(B, -1)[..., None]
        return points_all if multiple else points_all[0]
    
    def filter_point_intensity(self, points, threshold=0.0, **kwargs):
        points, input_has_multiple_samples = self.process_point_input(points, **kwargs)
        filtered_clouds, nonempty_cloud_idx = [], []
        for i in range(len(points)):
            filtered_cloud = points[i][points[i][:, 3] > threshold]
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
