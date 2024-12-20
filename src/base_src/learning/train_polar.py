import os.path

import torch
import torch.nn as nn
import torch.nn.functional as F

from dataset import get_dataset
from model_polar import RadarOccupancyModel


def check_memory(model, data_loader, device):
    # 1. Model Parameter Memory
    total_params = sum(p.numel() for p in model.parameters())
    param_memory = total_params * 4 / (1024 ** 2)  # Convert bytes to MB (float32)
    print(f"Model parameter memory: {param_memory:.2f} MB")

    data_iter = iter(data_loader)
    radar_frames, _, _ = next(data_iter)
    radar_frames = radar_frames.to(device)

    with torch.no_grad():
        cartesian_point_clouds = model.polar_to_cartesian(radar_frames)
        log_odds = model.pointnet(cartesian_point_clouds)

    activation_memory = (cartesian_point_clouds.numel() + log_odds.numel()) * 4 / (1024 ** 2)  # Convert bytes to MB
    print(f"Activation memory: {activation_memory:.2f} MB")

    # 3. Input Memory
    input_memory = radar_frames.numel() * 4 / (1024 ** 2)  # Convert bytes to MB
    print(f"Input memory: {input_memory:.2f} MB")

    # Total Memory
    total_memory = param_memory + activation_memory + input_memory
    print(f"Total estimated GPU memory: {total_memory:.2f} MB")


def spatial_prob_loss(pred_cloud, true_cloud, occupancy_threshold=0.5, point_match_radius=1.0):
    pred_occupied = pred_cloud[pred_cloud[:, -1] >= occupancy_threshold]
    true_occupied = true_cloud[true_cloud[:, -1] >= occupancy_threshold]
    pred_xyz, true_xyz = pred_occupied[:, :3], true_occupied[:, :3]
    pred_probs, true_probs = pred_occupied[:, 3], true_occupied[:, 3]
    print('pred_xyz', pred_xyz.shape, 'true_xyz', true_xyz.shape)
    if pred_xyz.shape[0] == 0 or true_xyz.shape[0] == 0:
        # No matching points available, return a high penalty
        return torch.tensor(1.0, device=pred_cloud.device)

    matched_true_xyz, matched_pred_xyz, matched_true_idx, matched_pred_idx = match_pointclouds(true_xyz, pred_xyz, max_distance=point_match_radius)
    print('matched_true_xyz', matched_true_xyz.shape, 'matched_true_idx', matched_true_idx.shape)
    print('matched_pred_xyz', matched_pred_xyz.shape, 'matched_pred_idx', matched_pred_idx.shape)

    unmatched_mask = torch.ones(true_xyz.size(0), device=true_xyz.device, dtype=torch.bool)
    unmatched_mask[matched_true_idx] = False
    num_unmatched_points = unmatched_mask.sum()
    matched_distances = torch.norm(matched_true_xyz - matched_pred_xyz, dim=-1)
    spatial_error = matched_distances.mean() + point_match_radius * num_unmatched_points
    prob_error = F.mse_loss(true_probs[matched_true_idx], pred_probs[matched_pred_idx]) + num_unmatched_points

    loss = spatial_error + prob_error
    return loss


def match_pointclouds(true_xyz, pred_xyz, max_distance=float('inf')):
    """
    Matches true points to predicted points with a maximum distance threshold.

    Args:
        true_xyz (torch.Tensor): Ground truth points, shape [N_true, 3].
        pred_xyz (torch.Tensor): Predicted points, shape [N_pred, 3].
        max_distance (float): Maximum allowable distance for matching points.

    Returns:
        matched_true_xyz (torch.Tensor): Matched true points, shape [M, 3].
        matched_pred_xyz (torch.Tensor): Matched predicted points, shape [M, 3].
        matched_true_idx (torch.Tensor): Indices of matched true points, shape [M].
        matched_pred_idx (torch.Tensor): Indices of matched predicted points, shape [M].
    """
    # Calculate pairwise distances between true and predicted points
    dists = torch.cdist(true_xyz, pred_xyz)  # Shape: [N_true, N_pred]

    # Apply maximum distance threshold
    valid_mask = dists <= max_distance
    dists[~valid_mask] = float('inf')  # Set distances exceeding threshold to infinity

    # Perform matching
    matched_true_idx = []
    matched_pred_idx = []

    for i in range(dists.size(0)):  # Iterate over true points
        # Get the minimum distance for the current true point
        min_dist, min_idx = dists[i].min(dim=0)
        if min_dist != float('inf'):  # Check if a valid match exists within the threshold
            matched_true_idx.append(i)
            matched_pred_idx.append(min_idx.item())
            dists[:, min_idx] = float('inf')  # Invalidate the matched predicted point

    # Gather matched points
    matched_true_idx = torch.tensor(matched_true_idx, dtype=torch.long, device=true_xyz.device)
    matched_pred_idx = torch.tensor(matched_pred_idx, dtype=torch.long, device=pred_xyz.device)
    matched_true_xyz = true_xyz[matched_true_idx]
    matched_pred_xyz = pred_xyz[matched_pred_idx]

    return matched_true_xyz, matched_pred_xyz, matched_true_idx, matched_pred_idx


def train(model, optimizer, loss_fn, train_loader, val_loader, device, num_epochs=10, save_path="best_model.pth", occupancy_threshold=0.5):
    best_val_loss = float('inf')

    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")

        model.train()
        train_loss = 0.0
        for radar_frames, lidar_frames, poses in train_loader:
            radar_frames = radar_frames.to(device)
            lidar_frames = [lidar_cloud.to(device) for lidar_cloud in lidar_frames]
            pred_probabilities = model(radar_frames)
            # print('true shape', lidar_frames[0].shape, lidar_frames[1].shape, ', output shape:', pred_probabilities.shape)

            batch_loss = 0
            for pred_cloud, true_cloud in zip(pred_probabilities, lidar_frames):
                batch_loss += spatial_prob_loss(pred_cloud, true_cloud, occupancy_threshold=0.5)
            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()
            train_loss += batch_loss.item()
            print('batch_loss', batch_loss)
            raise

        train_loss /= len(train_loader)

        # validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for radar_frames, packed_lidar_frames, poses in val_loader:
                radar_frames = radar_frames.to(device)
                packed_lidar_frames = packed_lidar_frames.to(device)

                outputs = model(radar_frames)
                lidar_frames, lidar_lengths = pad_packed_sequence(packed_lidar_frames, batch_first=True)
                outputs = outputs[:lidar_frames.size(0), :lidar_frames.size(1)]
                loss = loss_fn(outputs, lidar_frames[..., 3])
                val_loss += loss.item()

        val_loss /= len(val_loader)

        # save model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), save_path)

        print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Best Val Loss: {best_val_loss:.4f}")


def evaluate(model, test_loader, device, loss_fn):
    model.eval()
    test_loss = 0.0
    with torch.no_grad():
        for radar_frames, packed_lidar_frames, poses in test_loader:
            radar_frames = radar_frames.to(device)
            packed_lidar_frames = packed_lidar_frames.to(device)

            outputs = model(radar_frames)
            # lidar_frames, lidar_lengths = pad_packed_sequence(packed_lidar_frames, batch_first=True)
            # outputs = outputs[:lidar_frames.size(0), :lidar_frames.size(1)]
            # loss = loss_fn(outputs, lidar_frames[..., 3])

            # test_loss += loss.item()

    test_loss /= len(test_loader)
    print(f"Test Loss: {test_loss:.4f}")


def main():
    if os.path.isdir('/media/giantdrive'):
        dataset_path = '/media/giantdrive/coloradar/dataset1.h5'
        device_name = 'cuda:1'
    else:
        dataset_path = '/home/arpg/projects/coloradar_plus_processing_tools/coloradar_plus_processing_tools/dataset1.h5'
        device_name = 'cuda'
    train_loader, val_loader, test_loader, radar_config = get_dataset(dataset_path, batch_size=4,  partial=0.1)
    model = RadarOccupancyModel(radar_config)
    device = torch.device(device_name if torch.cuda.is_available() else "cpu")
    print('\ndevice', device)
    model.to(device)

    # check_memory(model, train_loader, device)
    # return

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()  # BCEWithLogitsLoss

    train(model, optimizer, loss_fn, train_loader, val_loader, device, num_epochs=10, save_path="best_model.pth")
    model.load_state_dict(torch.load("best_model.pth"))
    evaluate(model, test_loader, device, loss_fn)


if __name__ == '__main__':
    main()
    # point xyz loss
