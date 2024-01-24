import matplotlib.pyplot as plt
import numpy as np
import os
import time

import torch
from scipy.spatial.transform import Rotation as R

# export PYTHONPATH=$PYTHONPATH:src/octomap_radar_analysis/src

from dataset_intensity import get_dataset
from model_intensity import Unet1C2D
from model_intensity_3d import Unet1C3D


def get_device():
    if torch.cuda.is_available():
        print('GPU is available.')
        device = torch.device("cuda")
    else:
        print('GPU is not available, using CPU.')
        device = torch.device("cpu")
    return device


def visualize_grids(
        true_grids, predicted_grids,
        odds_threshold=0.0, resolution=0.25,
        x_min_meters=-6, x_max_meters=6,
        y_min_meters=0, y_max_meters=8,
        z_min_meters=0, z_max_meters=4
):
    fig = plt.figure(figsize=(12, 6))

    # first subplot for true grids
    ax_true, ax_predicted = fig.add_subplot(121, projection='3d'), fig.add_subplot(122, projection='3d')

    for true_grid, predicted_grid in zip(true_grids, predicted_grids):
        try:
            for grid, ax in ((true_grid, ax_true), (predicted_grid, ax_predicted)):
                threshold_indices = np.where(grid > odds_threshold)
                odds = grid[threshold_indices]
                xs, ys, zs = threshold_indices
                xs, ys, zs = xs * resolution + x_min_meters, ys * resolution + y_min_meters, zs * resolution + z_min_meters
                ax.scatter(xs, ys, zs, c=odds, s=1)
                ax.set_xlabel('X')
                ax.set_ylabel('Y')
                ax.set_zlabel('Z')
                ax.set_xlim([x_min_meters, x_max_meters])
                ax.set_ylim([y_min_meters, y_max_meters])
                ax.set_zlim([z_min_meters, z_max_meters])

            ax_true.set_title('True Grids')
            ax_predicted.set_title('Predicted Grids')
            plt.draw()
            plt.pause(1.5)
            ax_true.clear(), ax_predicted.clear()
        except KeyboardInterrupt:
            plt.close()
            break

    plt.close()


def predict_frame(device, model, input_frame):
    model.eval()
    with torch.no_grad():
        input_frame = input_frame.to(device)
        output = model(input_frame)
        output_probs = torch.sigmoid(output)
    return output_probs


def frame_grid_to_points(frame_grid, pose, timestamp, resolution=0.25, prob_threshold=0.):
    assert 0 <= prob_threshold <= 1
    grid_size_x, grid_size_y, grid_size_z = frame_grid.shape
    x_min, y_min, z_min = -grid_size_x * resolution / 2, 0, -grid_size_z * resolution / 2

    # Find indices of non-zero occupancy probabilities
    occupied_indices = np.argwhere(frame_grid > 0) if prob_threshold == 0 else np.argwhere(frame_grid >= prob_threshold)
    world_coordinates = occupied_indices * resolution + np.array([x_min, y_min, z_min])
    orientation = R.from_quat(pose[3:])
    world_coordinates = orientation.apply(world_coordinates) + pose[:3]

    probabilities = frame_grid[occupied_indices[:, 0], occupied_indices[:, 1], occupied_indices[:, 2]]
    points_map = {
        (world_coordinates[i, 0], world_coordinates[i, 1], world_coordinates[i, 2]
         ): (probabilities[i], timestamp)
        for i in range(len(occupied_indices))
    }
    return points_map


def test(
        loss_alpha, loss_gamma, occupancy_threshold=0.5, is_3d=False,
        visualize=False, dataset_filepath='dataset.pkl', model_folder='models', resolution_meters=0.25
):
    model_path = os.path.join(model_folder, f'model_1C{3 if is_3d else 2}D_a{int(loss_alpha * 100)}g{loss_gamma}.pt')
    _, _, test_loader = get_dataset(dataset_filepath=dataset_filepath, is_3d=is_3d)

    device = get_device()
    if is_3d:
        model = Unet1C3D().double().to(device)
    else:
        model = Unet1C2D().double().to(device)
    model.load_state_dict(torch.load(model_path))

    total_map = {}

    model.eval()
    with torch.no_grad():
        for input_frame, _ in test_loader:
            timestamp = time.time()
            input_frame = input_frame.to(device)
            output = model(input_frame)
            output_grid = torch.sigmoid(output)
            predicted_points = frame_grid_to_points(
                frame_grid=output_grid, pose=None, timestamp=timestamp,
                resolution=resolution_meters, prob_threshold=occupancy_threshold
            )
            total_map.update(predicted_points)

    # x_min_meters = -5
    # x_max_meters = 5
    # y_min_meters = 0
    # y_max_meters = 10
    # z_min_meters = -5
    # z_max_meters = 5
    # if visualize:
    #     visualize_grids(
    #         true_grids=test_loader.dataset.Y, predicted_grids=predicted_output,
    #         odds_threshold=occupancy_threshold, resolution=resolution_meters,
    #         x_min_meters=x_min_meters, x_max_meters=x_max_meters,
    #         y_min_meters=y_min_meters, y_max_meters=y_max_meters,
    #         z_min_meters=z_min_meters, z_max_meters=z_max_meters
    #     )


if __name__ == "__main__":
    a = 0.7
    g = 1
    thresholds = (0.4, 0.5, 0.6)
    test(loss_alpha=a, loss_gamma=g, occupancy_threshold=.6, is_3d=True, visualize=False)
