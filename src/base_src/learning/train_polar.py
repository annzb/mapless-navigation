import os.path

import torch

from dataset import get_dataset
from loss_spatial_prob import SpatialProbLoss
from model_polar import RadarOccupancyModel


def train(model, optimizer, loss_fn, train_loader, val_loader, device, num_epochs=10, save_path="best_model.pth"):
    best_val_loss = float('inf')

    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")

        model.train()
        train_loss = 0.0
        for radar_frames, lidar_frames, poses in train_loader:
            radar_frames = radar_frames.to(device)
            lidar_frames = [lidar_cloud.to(device) for lidar_cloud in lidar_frames]
            pred_probabilities = model(radar_frames)

            batch_loss = 0.0
            for pred_cloud, true_cloud in zip(pred_probabilities, lidar_frames):
                batch_loss += loss_fn(pred_cloud, true_cloud)
            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()
            train_loss += batch_loss.item()

        train_loss /= len(train_loader)

        # validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for radar_frames, lidar_frames, poses in val_loader:
                radar_frames = radar_frames.to(device)
                lidar_frames = [lidar_cloud.to(device) for lidar_cloud in lidar_frames]
                pred_probabilities = model(radar_frames)

                batch_loss = 0
                for pred_cloud, true_cloud in zip(pred_probabilities, lidar_frames):
                    batch_loss += loss_fn(pred_cloud, true_cloud)
                val_loss += batch_loss.item()

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
        for radar_frames, lidar_frames, poses in test_loader:
            radar_frames = radar_frames.to(device)
            lidar_frames = [lidar_cloud.to(device) for lidar_cloud in lidar_frames]
            pred_probabilities = model(radar_frames)

            batch_loss = 0
            for pred_cloud, true_cloud in zip(pred_probabilities, lidar_frames):
                batch_loss += loss_fn(pred_cloud, true_cloud)
            test_loss += batch_loss.item()


    test_loss /= len(test_loader)
    print(f"Test Loss: {test_loss:.4f}")


def main():
    OCCUPANCY_THRESHOLD = 0.6
    POINT_MATCH_RADIUS = 1.0
    BATCH_SIZE = 4
    N_EPOCHS = 50
    DATASET_PART = 1.0
    LEARNING_RATE = 1e-3

    if os.path.isdir('/media/giantdrive'):
        dataset_path = '/media/giantdrive/coloradar/dataset1.h5'
        device_name = 'cuda:1'
    else:
        dataset_path = '/home/arpg/projects/coloradar_plus_processing_tools/coloradar_plus_processing_tools/dataset2.h5'
        device_name = 'cuda'
    train_loader, val_loader, test_loader, radar_config = get_dataset(dataset_path, batch_size=BATCH_SIZE,  partial=DATASET_PART)
    model = RadarOccupancyModel(radar_config)
    device = torch.device(device_name if torch.cuda.is_available() else "cpu")
    print('\ndevice', device)
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_fn = SpatialProbLoss(occupancy_threshold=OCCUPANCY_THRESHOLD, point_match_radius=POINT_MATCH_RADIUS)
    train(model, optimizer, loss_fn, train_loader, val_loader, device, num_epochs=N_EPOCHS, save_path="best_model.pth")
    model.load_state_dict(torch.load("best_model.pth"))
    evaluate(model, test_loader, device, loss_fn)


if __name__ == '__main__':
    main()
