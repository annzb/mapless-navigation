import torch
import pytest

from src.metrics.diff_utils import voxelize_points


@pytest.fixture
def common_setup():
    """Provides a common setup for all voxelization tests."""
    return {
        "grid_size": torch.tensor([2, 2, 2], dtype=torch.float32),
        "grid_bounds_min": torch.tensor([0, 0, 0], dtype=torch.float32),
        "grid_bounds_max": torch.tensor([2, 2, 2], dtype=torch.float32),
        "feature_val": 1.0,
    }

def test_empty_input(common_setup):
    """Test if empty point cloud returns a zero grid."""
    points = torch.empty((0, 4), dtype=torch.float32)
    voxel_grid = voxelize_points(points, **common_setup)
    expected_grid = torch.zeros(8)
    torch.testing.assert_close(voxel_grid, expected_grid)

def test_point_at_voxel_corner(common_setup):
    """Test a point lying exactly on a voxel corner (0,0,0)."""
    feature_val = common_setup.pop("feature_val")
    points = torch.tensor([[0., 0., 0., feature_val]])
    voxel_grid = voxelize_points(points, **common_setup)
    
    expected_grid = torch.zeros(8)
    expected_grid[0] = feature_val
    torch.testing.assert_close(voxel_grid, expected_grid)

def test_point_at_grid_center_vertex(common_setup):
    """Test a point at the geometric center of the grid, which is a vertex."""
    points = torch.tensor([[1., 1., 1., 1.0]])
    voxel_grid = voxelize_points(points, **common_setup)
    expected_grid = torch.zeros(8)
    expected_grid[7] = 1.0
    torch.testing.assert_close(voxel_grid, expected_grid)

def test_point_on_face_center(common_setup):
    """Test a point on the center of a face shared by 4 voxels."""
    feature_val = common_setup.pop("feature_val")
    points = torch.tensor([[1.0, 0.5, 0.5, feature_val]])
    voxel_grid = voxelize_points(points, **common_setup)

    expected_grid = torch.zeros(8)
    indices = [4, 5, 6, 7] # Voxels (1,0,0), (1,0,1), (1,1,0), (1,1,1)
    expected_grid[indices] = feature_val
    torch.testing.assert_close(voxel_grid, expected_grid)

def test_point_on_edge_center(common_setup):
    """Test a point on the center of an edge shared by 2 voxels."""
    feature_val = common_setup.pop("feature_val")
    points = torch.tensor([[1., 1., 0.5, feature_val]])
    voxel_grid = voxelize_points(points, **common_setup)
    
    expected_grid = torch.zeros(8)
    indices = [6, 7] # Voxels (1,1,0) and (1,1,1)
    expected_grid[indices] = feature_val
    torch.testing.assert_close(voxel_grid, expected_grid)
    
def test_point_outside_bounds_positive(common_setup):
    """Test a point far outside the positive grid bounds."""
    feature_val = common_setup.pop("feature_val")
    points = torch.tensor([[10., 10., 10., feature_val]])
    voxel_grid = voxelize_points(points, **common_setup)
    
    expected_grid = torch.zeros(8)
    expected_grid[7] = feature_val # Index for max corner (1,1,1)
    torch.testing.assert_close(voxel_grid, expected_grid)

def test_point_outside_bounds_negative(common_setup):
    """Test a point far outside the negative grid bounds."""
    feature_val = common_setup.pop("feature_val")
    points = torch.tensor([[-10., -10., -10., feature_val]])
    voxel_grid = voxelize_points(points, **common_setup)
    
    expected_grid = torch.zeros(8)
    expected_grid[0] = feature_val # Index for min corner (0,0,0)
    torch.testing.assert_close(voxel_grid, expected_grid)

def test_multiple_points_accumulation(common_setup):
    """Test correct accumulation from two points affecting the same voxels."""
    common_setup.pop("feature_val")
    points = torch.tensor([
        [0.0, 0.0, 0.0, 1.0],  # Feature 1.0 to voxel (0,0,0)
        [0.0, 0.0, 0.5, 3.0]   # Feature 3.0 split between (0,0,0) and (0,0,1)
    ])
    voxel_grid = voxelize_points(points, **common_setup)
    
    expected_grid = torch.zeros(8)
    
    # Voxel (0,0,0): (1.0*1.0 + 3.0*0.5) / (1.0 + 0.5)
    expected_grid[0] = 2.5 / 1.5
    # Voxel (0,0,1): (3.0*0.5) / 0.5
    expected_grid[1] = 3.0
    
    torch.testing.assert_close(voxel_grid, expected_grid)

def test_point_equidistant_to_8_voxels(common_setup):
    """Test a point truly equidistant from 8 voxel cells' boundaries."""
    # This point normalizes to grid coordinates (0.5, 0.5, 0.5), which
    # correctly distributes its feature to all 8 surrounding voxels.
    points = torch.tensor([[0.5, 0.5, 0.5, 1.0]])
    voxel_grid = voxelize_points(points, **common_setup)
    
    # After averaging, each of the 8 voxels should have the original feature value.
    expected_grid = torch.full((8,), 1.0)
    torch.testing.assert_close(voxel_grid, expected_grid)


def create_sphere_point_cloud(num_points=4000, radius=4.0, center=[0., 0., 0.]):
    """
    Generates a point cloud on a sphere's surface.
    MODIFIED: Feature values now vary from 0 to 1 based on the z-coordinate.
    """
    phi = np.random.uniform(0, 2 * np.pi, num_points)
    costheta = np.random.uniform(-1, 1, num_points)
    theta = np.arccos(costheta)

    x = radius * np.sin(theta) * np.cos(phi) + center[0]
    y = radius * np.sin(theta) * np.sin(phi) + center[1]
    z = radius * np.cos(theta) + center[2]

    # Feature is based on height (z-value), normalized from 0 to 1
    features = (z - center[2] + radius) / (2 * radius)
    features = features.reshape(-1, 1)

    points_np = np.hstack((np.vstack((x, y, z)).T, features))
    return torch.from_numpy(points_np).float()

def visualize_voxelization():
    """
    Generates a point cloud, voxelizes it, and displays the result
    with colors and transparency mapped to feature values.
    """
    # 1. Define Parameters
    grid_resolution = 16
    grid_size = torch.tensor([grid_resolution] * 3, dtype=torch.float32)
    bounds = 5.0 
    grid_bounds_min = torch.tensor([-bounds] * 3, dtype=torch.float32)
    grid_bounds_max = torch.tensor([bounds] * 3, dtype=torch.float32)
    
    # 2. Create Point Cloud
    points = create_sphere_point_cloud(num_points=5000, radius=4.0)

    # 3. Voxelize
    voxel_grid_flat = voxelize_points(points, grid_size, grid_bounds_min, grid_bounds_max)

    # 4. Prepare Data for Plotting
    voxel_grid_3d = voxel_grid_flat.reshape(tuple(grid_size.long().tolist())).cpu().numpy()
    p_np = points.numpy()

    # 5. Create the Visualization
    fig = plt.figure(figsize=(16, 8))
    plt.style.use('dark_background')
    cmap = cm.get_cmap('viridis')

    # --- Subplot 1: Original Point Cloud ---
    ax1 = fig.add_subplot(121, projection='3d')
    # MODIFIED: Color points based on their feature value (4th column)
    sc = ax1.scatter(p_np[:, 0], p_np[:, 1], p_np[:, 2], s=2, c=p_np[:, 3], cmap=cmap, alpha=0.8)
    fig.colorbar(sc, ax=ax1, label='Point Feature Value', shrink=0.6)
    
    ax1.set_title("Original Point Cloud", fontsize=16)
    ax1.set_xlabel("X"); ax1.set_ylabel("Y"); ax1.set_zlabel("Z")
    ax1.set_xlim(-bounds, bounds); ax1.set_ylim(-bounds, bounds); ax1.set_zlim(-bounds, bounds)
    ax1.view_init(elev=30, azim=45)

    # --- Subplot 2: Voxelized Representation ---
    ax2 = fig.add_subplot(122, projection='3d')
    
    # MODIFICATION START: Create RGBA color array for voxels
    voxel_mask = voxel_grid_3d > 0.05
    
    # Normalize the grid values to [0, 1] to map to the colormap
    norm = Normalize(vmin=voxel_grid_3d[voxel_mask].min(), vmax=voxel_grid_3d[voxel_mask].max())
    
    # Apply colormap to get RGB, then create an empty RGBA array
    colors_rgb = cmap(norm(voxel_grid_3d))
    colors_rgba = np.zeros(voxel_grid_3d.shape + (4,))
    colors_rgba[..., :3] = colors_rgb[..., :3]
    
    # Set alpha channel based on feature value (higher value = more opaque)
    # Clipping ensures alpha is between 0 and 1
    alpha_values = np.clip(voxel_grid_3d * 2.5, 0, 1)
    colors_rgba[..., 3] = alpha_values
    # MODIFICATION END

    ax2.voxels(voxel_mask, facecolors=colors_rgba, edgecolor='gray', linewidth=0.2)
    
    # Add a matching color bar
    mappable = cm.ScalarMappable(cmap=cmap, norm=norm)
    fig.colorbar(mappable, ax=ax2, label='Voxel Feature Value', shrink=0.6)

    ax2.set_title("Voxelized Representation", fontsize=16)
    ax2.set_xlabel("X"); ax2.set_ylabel("Y"); ax2.set_zlabel("Z")
    ax2.set_xlim(0, grid_resolution); ax2.set_ylim(0, grid_resolution); ax2.set_zlim(0, grid_resolution)
    ax2.view_init(elev=30, azim=45)

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    visualize_voxelization()
