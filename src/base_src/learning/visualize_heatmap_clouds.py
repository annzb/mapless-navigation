import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d
from dataset import get_dataset
from scipy.spatial.transform import Rotation as R


def show_occupancy_pcl(cloud, prob_threshold=0):
    probabilities = 1 - 1 / (1 + np.exp(cloud[:, 3]))
    mask = probabilities >= prob_threshold
    filtered_points = cloud[:, :3][mask]
    filtered_probs = probabilities[mask]
    fp_max, fp_min = filtered_probs.max(), filtered_probs.min()
    normalized_probs = (filtered_probs - fp_min) / (fp_max - fp_min)

    cmap = plt.get_cmap("plasma")
    colors = cmap(normalized_probs)[:, :3]

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


# def image_to_pcl(images, radar_config):
#     azimuth_scale = torch.ones(radar_config.num_azimuth_bins)
#     azimuth_bias = torch.zeros(radar_config.num_azimuth_bins)
#     azimuth_cos_weight = torch.ones(radar_config.num_azimuth_bins)
#     azimuth_sin_weight = torch.ones(radar_config.num_azimuth_bins)
#     elevation_scale = torch.ones(radar_config.num_elevation_bins)
#     elevation_bias = torch.zeros(radar_config.num_elevation_bins)
#     elevation_cos_weight = torch.ones(radar_config.num_elevation_bins)
#     elevation_sin_weight = torch.ones(radar_config.num_elevation_bins)
#     range_scale = torch.ones(radar_config.num_range_bins)
#     range_bias = torch.zeros(radar_config.num_range_bins)
#     batch_size = images.shape[0]
#
#     azimuths = torch.tensor(radar_config.clipped_azimuth_bins)
#     azimuths = azimuths * azimuth_scale + azimuth_bias
#     ranges = torch.linspace(0, radar_config.num_range_bins * radar_config.range_bin_width, radar_config.num_range_bins)
#     ranges = ranges * range_scale + range_bias
#     elevations = torch.tensor(radar_config.clipped_elevation_bins)
#     elevations = elevations * elevation_scale + elevation_bias
#
#     azimuths_grid, ranges_grid, elevations_grid = torch.meshgrid(azimuths, ranges, elevations, indexing="ij")
#     azimuths_grid = azimuths_grid.unsqueeze(0).expand(batch_size, -1, -1, -1)
#     ranges_grid = ranges_grid.unsqueeze(0).expand(batch_size, -1, -1, -1)
#     elevations_grid = elevations_grid.unsqueeze(0).expand(batch_size, -1, -1, -1)
#
#     cos_azimuths = azimuth_cos_weight.view(1, -1, 1, 1) * torch.cos(azimuths_grid)
#     sin_azimuths = azimuth_sin_weight.view(1, -1, 1, 1) * torch.sin(azimuths_grid)
#     cos_elevations = elevation_cos_weight.view(1, 1, 1, -1) * torch.cos(elevations_grid)
#     sin_elevations = elevation_sin_weight.view(1, 1, 1, -1) * torch.sin(elevations_grid)
#
#     x = ranges_grid * cos_elevations * cos_azimuths
#     y = ranges_grid * cos_elevations * sin_azimuths
#     z = ranges_grid * sin_elevations
#     x = x.flatten(start_dim=1).unsqueeze(-1)
#     y = y.flatten(start_dim=1).unsqueeze(-1)
#     z = z.flatten(start_dim=1).unsqueeze(-1)
#
#     intensity = images.flatten(start_dim=1, end_dim=3).unsqueeze(-1)
#     cartesian_points = torch.cat((x, y, z, intensity), dim=-1)
#
#     return cartesian_points  # [B, N, 4]

def image_to_pcl(images, radar_config):
    azimuth_scale = torch.ones(radar_config.num_azimuth_bins)
    azimuth_bias = torch.zeros(radar_config.num_azimuth_bins)
    azimuth_cos_weight = torch.ones(radar_config.num_azimuth_bins)
    azimuth_sin_weight = torch.ones(radar_config.num_azimuth_bins)
    elevation_scale = torch.ones(radar_config.num_elevation_bins)
    elevation_bias = torch.zeros(radar_config.num_elevation_bins)
    elevation_cos_weight = torch.ones(radar_config.num_elevation_bins)
    elevation_sin_weight = torch.ones(radar_config.num_elevation_bins)
    range_scale = torch.ones(radar_config.num_range_bins)
    range_bias = torch.zeros(radar_config.num_range_bins)
    batch_size = images.shape[0]

    azimuths = torch.tensor(radar_config.clipped_azimuth_bins)
    azimuths = azimuths * azimuth_scale + azimuth_bias
    ranges = torch.linspace(0, radar_config.num_range_bins * radar_config.range_bin_width, radar_config.num_range_bins)
    ranges = ranges * range_scale + range_bias
    elevations = torch.tensor(radar_config.clipped_elevation_bins)
    elevations = elevations * elevation_scale + elevation_bias

    azimuths_grid, ranges_grid, elevations_grid = torch.meshgrid(azimuths, ranges, elevations, indexing="ij")
    azimuths_grid = azimuths_grid.unsqueeze(0).expand(batch_size, -1, -1, -1)
    ranges_grid = ranges_grid.unsqueeze(0).expand(batch_size, -1, -1, -1)
    elevations_grid = elevations_grid.unsqueeze(0).expand(batch_size, -1, -1, -1)

    cos_azimuths = azimuth_cos_weight.view(1, -1, 1, 1) * torch.cos(azimuths_grid)
    sin_azimuths = azimuth_sin_weight.view(1, -1, 1, 1) * torch.sin(azimuths_grid)
    cos_elevations = elevation_cos_weight.view(1, 1, 1, -1) * torch.cos(elevations_grid)
    sin_elevations = elevation_sin_weight.view(1, 1, 1, -1) * torch.sin(elevations_grid)

    x = ranges_grid * cos_elevations * sin_azimuths  # Right
    y = ranges_grid * cos_elevations * cos_azimuths  # Forward
    z = ranges_grid * sin_elevations                # Up

    x = x.flatten(start_dim=1).unsqueeze(-1)
    y = y.flatten(start_dim=1).unsqueeze(-1)
    z = z.flatten(start_dim=1).unsqueeze(-1)

    intensity = images.flatten(start_dim=1, end_dim=3).unsqueeze(-1)
    cartesian_points = torch.cat((x, y, z, intensity), dim=-1)

    return cartesian_points  # [B, N, 4]


def visualize_polar_image(image, radar_config):
    """
    Visualize the polar image as a 2D intensity map.

    Args:
        image (torch.Tensor): Input radar image of shape [B, A, R, E] (Batch, Azimuth, Range, Elevation).
        radar_config: Configuration containing azimuth and elevation bins.
    """
    aggregated_image = image.sum(dim=-1).cpu().numpy()  # [A, R]
    plt.figure(figsize=(8, 6))
    plt.imshow(aggregated_image, extent=[0, radar_config.num_range_bins, radar_config.num_azimuth_bins, 0], cmap='plasma', aspect='auto')
    plt.colorbar(label="Intensity")
    plt.xlabel("Range Bins")
    plt.ylabel("Azimuth Bins")
    plt.title("Polar Image Visualization")
    plt.show()


def main():
    BATCH_SIZE = 4
    DATASET_PART = 0.1

    if os.path.isdir('/media/giantdrive'):
        dataset_path = '/media/giantdrive/coloradar/dataset1.h5'
    else:
        dataset_path = '/home/arpg/projects/coloradar_plus_processing_tools/coloradar_plus_processing_tools/dataset1.h5'
    train_loader, val_loader, test_loader, radar_config = get_dataset(dataset_path, batch_size=BATCH_SIZE, partial=DATASET_PART)

    for radar_frames, lidar_frames, _ in train_loader:
        radar_clouds = image_to_pcl(radar_frames, radar_config)
        first_radar_cloud = radar_clouds[0].cpu().numpy()
        first_lidar_cloud = lidar_frames[0].cpu().numpy()
        # visualize_polar_image(radar_frames[0], radar_config)
        show_radar_pcl(first_radar_cloud)
        show_occupancy_pcl(first_lidar_cloud)
        break


if __name__ == '__main__':
    main()