import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import Normalize
from matplotlib import cm


def voxelize_points(
    points: torch.Tensor,
    grid_size: torch.Tensor,
    grid_bounds_min: torch.Tensor,
    grid_bounds_max: torch.Tensor,
    **kwargs
) -> torch.Tensor:
    """
    Performs differentiable voxelization using trilinear interpolation.
    Args:
        points: A tensor of points with features, shape (N, 4) where [:, :3] is XYZ and [:, 3] is the feature (e.g., occupancy).
        grid_size: A tensor with the grid dimensions, e.g., torch.tensor([128, 128, 64]).
        grid_bounds_min: A tensor with the min XYZ bounds, e.g., torch.tensor([-10, 0, -5]).
        grid_bounds_max: A tensor with the max XYZ bounds, e.g., torch.tensor([10, 20, 5]).
    Returns:
        A flattened voxel grid tensor with averaged features.
    """
    nx, ny, nz = grid_size.long()
    num_cells = nx * ny * nz
    if points.shape[0] == 0:
       return torch.zeros(num_cells, device=points.device, dtype=points.dtype)
       # raise ValueError("No points to voxelize")

    coords = (points[:, :3] - grid_bounds_min) / (grid_bounds_max - grid_bounds_min) * grid_size
    c0 = coords.floor().long()
    x0, y0, z0 = c0[:, 0], c0[:, 1], c0[:, 2]
    x1, y1, z1 = x0 + 1, y0 + 1, z0 + 1
    wd = (coords - c0.float())
    wx, wy, wz = wd[:, 0], wd[:, 1], wd[:, 2]
    w000 = (1 - wx) * (1 - wy) * (1 - wz)
    w100 =      wx  * (1 - wy) * (1 - wz)
    w010 = (1 - wx) * wy  * (1 - wz)
    w001 = (1 - wx) * (1 - wy) * wz
    w110 =      wx  * wy  * (1 - wz)
    w101 =      wx  * (1 - wy) * wz
    w011 = (1 - wx) * wy  * wz
    w111 =      wx  * wy  * wz
    features = points[:, 3]

    scattered_features = torch.zeros(num_cells, device=points.device, dtype=points.dtype)
    scattered_weights = torch.zeros(num_cells, device=points.device, dtype=points.dtype)
    def scatter(x_idx, y_idx, z_idx, weights):
        x_idx = x_idx.clip(0, nx - 1)
        y_idx = y_idx.clip(0, ny - 1)
        z_idx = z_idx.clip(0, nz - 1)
        flat_indices = x_idx * (ny * nz) + y_idx * nz + z_idx
        scattered_features.scatter_add_(0, flat_indices, features * weights)
        scattered_weights.scatter_add_(0, flat_indices, weights)

    scatter(x0, y0, z0, w000)
    scatter(x1, y0, z0, w100)
    scatter(x0, y1, z0, w010)
    scatter(x0, y0, z1, w001)
    scatter(x1, y1, z0, w110)
    scatter(x1, y0, z1, w101)
    scatter(x0, y1, z1, w011)
    scatter(x1, y1, z1, w111)
    final_grid = scattered_features / scattered_weights.clamp(min=1e-6)
    return final_grid
