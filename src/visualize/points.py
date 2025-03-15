import torch
import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d


def show_occupancy_pcl(cloud, prob_threshold=0):
    probabilities = 1 - 1 / (1 + np.exp(cloud[:, 3]))
    mask = probabilities >= prob_threshold
    filtered_points = cloud[:, :3][mask]
    filtered_probs = probabilities[mask]
    # fp_max, fp_min = filtered_probs.max(), filtered_probs.min()
    # normalized_probs = (filtered_probs - fp_min) / (fp_max - fp_min)

    cmap = plt.get_cmap("plasma")
    colors = cmap(filtered_probs)[:, :3]

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(filtered_points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=5.0)
    o3d.visualization.draw_geometries([pcd, axes])

def show_radar_pcl(cloud, intensity_threshold_percent=0.0):
    min_intensity = np.min(cloud[:, 3])
    max_intensity = np.max(cloud[:, 3])
    normalized_intensities = (cloud[:, 3] - min_intensity) / (max_intensity - min_intensity)
    filtered_idx = normalized_intensities >= intensity_threshold_percent / 100
    cmap = plt.get_cmap("plasma")
    colors = cmap(normalized_intensities[filtered_idx])[:, :3]
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(cloud[:, :3][filtered_idx])
    pcd.colors = o3d.utility.Vector3dVector(colors)
    axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=3.0)
    o3d.visualization.draw_geometries([pcd, axes], "Radar Point Cloud Visualization")


def image_to_pcl(images, radar_config, batch_size=1):
    azimuths = torch.tensor(radar_config.clipped_azimuth_bins)
    ranges = torch.linspace(0, radar_config.num_range_bins * radar_config.range_bin_width, radar_config.num_range_bins)
    elevations = torch.tensor(radar_config.clipped_elevation_bins)

    azimuths_grid, ranges_grid, elevations_grid = torch.meshgrid(azimuths, ranges, elevations, indexing="ij")
    azimuths_grid = azimuths_grid.unsqueeze(0).expand(batch_size, -1, -1, -1)
    ranges_grid = ranges_grid.unsqueeze(0).expand(batch_size, -1, -1, -1)
    elevations_grid = elevations_grid.unsqueeze(0).expand(batch_size, -1, -1, -1)

    cos_azimuths = torch.cos(azimuths_grid)
    sin_azimuths = torch.sin(azimuths_grid)
    cos_elevations = torch.cos(elevations_grid)
    sin_elevations = torch.sin(elevations_grid)

    x = ranges_grid * cos_elevations * sin_azimuths  # Right
    y = ranges_grid * cos_elevations * cos_azimuths  # Forward
    z = ranges_grid * sin_elevations                 # Up

    x = x.flatten(start_dim=1).unsqueeze(-1)
    y = y.flatten(start_dim=1).unsqueeze(-1)
    z = z.flatten(start_dim=1).unsqueeze(-1)

    intensity = images.flatten(start_dim=1, end_dim=3).unsqueeze(-1)
    cartesian_points = torch.cat((x, y, z, intensity), dim=-1)

    return cartesian_points  # [B, N, 4]


def apply_layers(model, polar_frames, radar_config):
    # ptc = PolarToCartesian(radar_config)
    # cartesian_points = ptc(polar_frames)
    cartesian_points = model.polar_to_cartesian(polar_frames)
    # downsampled_points = model.down(cartesian_points)
    # points = downsampled_points[..., :3]
    # features = downsampled_points[..., 3:]
    # log_odds = self.pointnet(points, features)
    # probabilities = self.apply_sigmoid(log_odds)
    return cartesian_points

    # reshaped_frames = polar_frames.view(batch_size, model.radar_config.num_azimuth_bins * model.radar_config.num_range_bins, model.radar_config.num_elevation_bins)
    # transformed_frames = model.transformer(reshaped_frames)
    # transformed_frames = transformed_frames.view(batch_size, model.radar_config.num_azimuth_bins, model.radar_config.num_range_bins, model.radar_config.num_elevation_bins)
    # cartesian_points = model.polar_to_cartesian(transformed_frames)
    # less_points = model.down(cartesian_points)
    # log_odds = model.pointnet(less_points)
    # print('less_points', less_points.shape)
    # probabilities = model.apply_sigmoid(less_points)
