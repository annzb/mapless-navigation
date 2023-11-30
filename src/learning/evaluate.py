import matplotlib.pyplot as plt
import numpy as np

import torch
from torch import nn

# export PYTHONPATH=$PYTHONPATH:src/octomap_radar_analysis/src

from dataset import get_dataset
from dataset_intensity import get_dataset as get_dataset_1C2D
from loss_focal import FocalLoss
from model import BaseTransform
from model_intensity import Unet1C2D


def get_device():
    if torch.cuda.is_available():
        print('GPU is available.')
        device = torch.device("cuda")
    else:
        print('GPU is not available, using CPU.')
        device = torch.device("cpu")
    return device


# def dice_coefficient(predictions, targets, smooth=1.0):
#     intersection = (predictions & targets).float().sum((1, 2, 3))
#     union = predictions.float().sum((1, 2, 3)) + targets.float().sum((1, 2, 3))
#     dice = (2. * intersection + smooth) / (union + smooth)
#     return dice.mean()


def f1_score(TP, FP, TN, FN):
    precision = TP / (TP + FP + 1e-12)
    recall = TP / (TP + FN + 1e-12)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-12)
    return f1


def accuracy_score(TP, FP, TN, FN):
    accuracy = (TP + TN) / (TP + FP + TN + FN)
    return accuracy


def calc_tfpn(predictions, targets):
    true_pos = (predictions & targets).sum().float()
    false_pos = (predictions & ~targets).sum().float()
    true_neg = (~predictions & ~targets).sum().float()
    false_neg = (~predictions & targets).sum().float()
    return true_pos, false_pos, true_neg, false_neg


def test_model(test_loader, model, criterion, device, occupancy_threshold=0.5):
    metrics = (nn.L1Loss(), nn.BCELoss(), nn.MSELoss())
    binary_metrics = (f1_score, accuracy_score)

    model.eval()
    test_loss = 0
    predicted_output = []
    TP, FP, TN, FN = 0, 0, 0, 0
    num_samples, num_batches = 0, 0
    metric_values = [0 for _ in metrics]

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            output_probs = torch.sigmoid(output)
            output_binary, target_binary = output_probs > occupancy_threshold, target > occupancy_threshold

            loss = criterion(output, target)
            test_loss += loss
            predicted_output.append(output_probs.cpu().numpy())

            num_samples += data.size(0)
            num_batches += 1
            tp_batch, fp_batch, tn_batch, fn_batch = calc_tfpn(output_binary, target_binary)
            TP += tp_batch
            FP += fp_batch
            TN += tn_batch
            FN += fn_batch
            for i, metric in enumerate(metrics):
                metric_values[i] += metric(output_probs, target)

    print(f'Total testing loss: {test_loss}')
    for i, metric in enumerate(metrics):
        loss_per_sample = metric_values[i] / num_samples
        loss_per_batch = metric_values[i] / num_batches
        print(f'Avg {metric.__class__.__name__} loss: {loss_per_sample} per sample, {loss_per_batch} per batch')
    for metric in binary_metrics:
        loss = metric(TP, FP, TN, FN)
        print(f'{metric.__name__}: {loss}')

    return np.concatenate(predicted_output, axis=0)


def visualize_grids(
        true_grids, predicted_grids,
        odds_threshold=0.0, resolution=0.25,
        x_min_meters=-6, x_max_meters=6,
        y_min_meters=0, y_max_meters=8,
        z_min_meters=0, z_max_meters=4
):
    fig = plt.figure(figsize=(12, 6))
    cmap = plt.get_cmap('jet')

    # first subplot for true grids
    ax_true, ax_predicted = fig.add_subplot(121, projection='3d'), fig.add_subplot(122, projection='3d')

    for true_grid, predicted_grid in zip(true_grids, predicted_grids):
        try:
            for grid, ax in ((true_grid, ax_true), (predicted_grid, ax_predicted)):
                threshold_indices = np.where(grid > odds_threshold)
                odds = grid[threshold_indices]
                xs, ys, zs = threshold_indices
                xs, ys, zs = xs * resolution + x_min_meters, ys * resolution + y_min_meters, zs * resolution + z_min_meters
                ax.scatter(xs, ys, zs, c=odds, s=1, cmap=cmap)
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


def test_1c2d(loss_alpha, loss_gamma, occupancy_threshold=0.5):
    train_loader, valid_loader, test_loader = get_dataset_1C2D(dataset_filepath='dataset.pkl')
    device = get_device()
    model = Unet1C2D().double().to(device)
    model.load_state_dict(torch.load(f'model_1C2D_a{int(loss_alpha * 100)}g{loss_gamma}.pt'))
    criterion = FocalLoss(alpha=loss_alpha, gamma=loss_gamma)
    predicted_output = test_model(test_loader, model, criterion, device, occupancy_threshold=occupancy_threshold)

    resolution_meters = 0.25
    x_min_meters = -4
    x_max_meters = 4
    y_min_meters = 0
    y_max_meters = 8
    z_min_meters = -4
    z_max_meters = 4
    visualize_grids(
        true_grids=test_loader.dataset.Y, predicted_grids=predicted_output,
        odds_threshold=occupancy_threshold, resolution=resolution_meters,
        x_min_meters=x_min_meters, x_max_meters=x_max_meters,
        y_min_meters=y_min_meters, y_max_meters=y_max_meters,
        z_min_meters=z_min_meters, z_max_meters=z_max_meters
    )


if __name__ == "__main__":
    target_occupancy_threshold = 0.4
    # test_1c2d(target_occupancy_threshold)
    # train_2C()
