import os.path

import torch
import wandb

import metrics
from dataset import get_dataset
from loss_spatial_prob import SpatialProbLoss, SoftMatchingLoss
from model_polar import RadarOccupancyModel


def train(model, optimizer, loss_fn, train_loader, val_loader, device, num_epochs=10, save_path="best_model.pth"):
    best_val_loss = float('inf')

    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")

        model.train()
        train_loss, train_iou, train_chamfer = 0.0, 0.0, 0.0

        for radar_frames, lidar_frames, poses in train_loader:
            radar_frames = radar_frames.to(device)
            lidar_frames = [lidar_cloud.to(device) for lidar_cloud in lidar_frames]
            pred_probabilities = model(radar_frames)

            batch_loss = 0.0
            for pred_cloud, true_cloud in zip(pred_probabilities, lidar_frames):
                batch_loss += loss_fn(pred_cloud, true_cloud)
                train_iou += metrics.iou(pred_cloud, true_cloud)
                train_chamfer += metrics.weighted_chamfer(pred_cloud, true_cloud)
            train_loss += batch_loss.item()
            batch_loss /= len(radar_frames)

            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()

        train_loss /= len(train_loader)
        train_iou /= len(train_loader)
        train_chamfer /= len(train_loader)

        # validation
        model.eval()
        val_loss, val_iou, val_chamfer = 0.0, 0.0, 0.0
        with torch.no_grad():
            for radar_frames, lidar_frames, poses in val_loader:
                radar_frames = radar_frames.to(device)
                lidar_frames = [lidar_cloud.to(device) for lidar_cloud in lidar_frames]
                pred_probabilities = model(radar_frames)

                for pred_cloud, true_cloud in zip(pred_probabilities, lidar_frames):
                    val_loss += loss_fn(pred_cloud, true_cloud).item()
                    val_iou += metrics.iou(pred_cloud, true_cloud)
                    val_chamfer += metrics.weighted_chamfer(pred_cloud, true_cloud)

        val_loss /= len(val_loader)
        val_iou /= len(train_loader)
        val_chamfer /= len(train_loader)

        # save model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), save_path)

        wandb.log({
            "epoch": epoch, "train_loss": train_loss, "train_iou": train_iou, "train_chamfer": train_chamfer,
            "valid_loss": val_loss, "valid_iou": val_iou, "valid_chamfer": val_chamfer, "best_valid_loss": best_val_loss
        })
        print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Best Val Loss: {best_val_loss:.4f}")


def evaluate(model, test_loader, device, loss_fn):
    model.eval()
    test_loss, test_iou, test_chamfer = 0.0, 0.0, 0.0
    with torch.no_grad():
        for radar_frames, lidar_frames, poses in test_loader:
            radar_frames = radar_frames.to(device)
            lidar_frames = [lidar_cloud.to(device) for lidar_cloud in lidar_frames]
            pred_probabilities = model(radar_frames)

            for pred_cloud, true_cloud in zip(pred_probabilities, lidar_frames):
                test_loss += loss_fn(pred_cloud, true_cloud).item()
                test_iou += metrics.iou(pred_cloud, true_cloud)
                test_chamfer += metrics.weighted_chamfer(pred_cloud, true_cloud)

    test_loss /= len(test_loader)
    test_iou /= len(test_loader)
    test_chamfer /= len(test_loader)
    wandb.log({"test_loss": test_loss, "test_iou": test_iou, "test_chamfer": test_chamfer})
    print(f"Test Loss: {test_loss:.4f}")


def main():
    # OCCUPANCY_THRESHOLD = 0.6
    # POINT_MATCH_RADIUS = 1.0
    BATCH_SIZE = 4
    N_EPOCHS = 100
    DATASET_PART = 1.0
    LEARNING_RATE = 0.01
    loss_spatial_weight = 1.0
    loss_probability_weight = 1.0
    loss_smoothness_weight = 0.1
    model_save_path = "best_model.pth"

    if os.path.isdir('/media/giantdrive'):
        dataset_path = '/media/giantdrive/coloradar/dataset2.h5'
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
    loss_fn = SoftMatchingLoss()
    # loss_fn = SpatialProbLoss(occupancy_threshold=OCCUPANCY_THRESHOLD, point_match_radius=POINT_MATCH_RADIUS)

    wandb.init(
        project="radar-occupancy",
        config={
            "dataset": os.path.basename(dataset_path),
            "model": model.name,
            "learning_rate": LEARNING_RATE,
            "epochs": N_EPOCHS,
            "dataset_part": DATASET_PART,
            "batch_size": BATCH_SIZE,
            "loss": {
                "name": loss_fn.__class__.__name__,
                "spatial_weight": loss_spatial_weight,
                "probability_weight": loss_probability_weight,
                "smoothness_weight": loss_smoothness_weight
            }
        }
    )
    train(model, optimizer, loss_fn, train_loader, val_loader, device, num_epochs=N_EPOCHS, save_path=model_save_path)
    model.load_state_dict(torch.load(model_save_path))
    evaluate(model, test_loader, device, loss_fn)
    wandb.log({"best_model_path":  os.path.abspath(model_save_path)})
    wandb.finish()


if __name__ == '__main__':
    main()
