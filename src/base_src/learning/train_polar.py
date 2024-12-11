import os.path

import torch
import torch.nn as nn

from dataset import get_dataset
from model_polar import RadarOccupancyModel
from torch.nn.utils.rnn import pad_packed_sequence


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


def train(model, optimizer, loss_fn, train_loader, val_loader, device, num_epochs=10, save_path="best_model.pth"):
    best_val_loss = float('inf')

    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")

        # training
        model.train()
        train_loss = 0.0
        for radar_frames, packed_lidar_frames, poses in train_loader:
            radar_frames = radar_frames.to(device)
            packed_lidar_frames = packed_lidar_frames.to(device)

            outputs = model(radar_frames)
            lidar_frames, lidar_lengths = pad_packed_sequence(packed_lidar_frames, batch_first=True)
            outputs = outputs[:lidar_frames.size(0), :lidar_frames.size(1)]
            loss = loss_fn(outputs, lidar_frames[..., 3])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

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
            lidar_frames, lidar_lengths = pad_packed_sequence(packed_lidar_frames, batch_first=True)
            outputs = outputs[:lidar_frames.size(0), :lidar_frames.size(1)]
            loss = loss_fn(outputs, lidar_frames[..., 3])

            test_loss += loss.item()

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
    loss_fn = nn.BCEWithLogitsLoss()

    train(model, optimizer, loss_fn, train_loader, val_loader, device, num_epochs=10, save_path="best_model.pth")
    model.load_state_dict(torch.load("best_model.pth"))
    evaluate(model, test_loader, device, loss_fn)


if __name__ == '__main__':
    main()
