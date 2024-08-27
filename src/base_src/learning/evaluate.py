import matplotlib.pyplot as plt
import numpy as np
import os

import torch
from torch import nn

# export PYTHONPATH=$PYTHONPATH:src/octomap_radar_analysis/src

from dataset_intensity import get_dataset
from loss_focal import FocalLoss
from model_intensity import Unet1C2D
from model_intensity_3d import Unet1C3D


def get_device():
    if torch.cuda.is_available():
        print('GPU is available.')
        device = torch.device("cuda:1")
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


def iou_score(TP, FP, TN, FN):
    iou = TP / (TP + FP + FN)
    return iou


def calc_tfpn(predictions, targets):
    true_pos = (predictions & targets).sum().float().item()
    false_pos = (predictions & ~targets).sum().float().item()
    true_neg = (~predictions & ~targets).sum().float().item()
    false_neg = (~predictions & targets).sum().float().item()
    return true_pos, false_pos, true_neg, false_neg


def test_model(
        test_loader, model, criterion, device,
        occupied_threshold=0.5, empty_threshold=0.49,
        outfile=None, output_is_prob=False, monitor=None
):
    metrics = (nn.L1Loss(), nn.MSELoss())
    binary_metrics = (f1_score, accuracy_score, iou_score)

    model.eval()
    test_loss = 0
    predicted_output = []
    stats = {
        'occupied': {'tp': 0, 'tn': 0, 'fp': 0, 'fn': 0, 'to_binary': lambda vals: vals == 2},
        'empty': {'tp': 0, 'tn': 0, 'fp': 0, 'fn': 0, 'to_binary': lambda vals: vals == 1},
        'uncertain': {'tp': 0, 'tn': 0, 'fp': 0, 'fn': 0, 'to_binary': lambda vals: vals == 0}
    }
    num_samples, num_batches = 0, 0
    metric_values = [0 for _ in metrics]
    num_points, num_accurate = 0, 0

    with torch.no_grad():
        for data, target, _ in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            test_loss += loss

            if not output_is_prob:
                output = torch.sigmoid(output)
            predicted_output.append(output.cpu().numpy())

            output_cat, target_cat = torch.zeros_like(output, dtype=torch.int), torch.zeros_like(target, dtype=torch.int)
            output_cat[output <= empty_threshold], output_cat[output >= occupied_threshold] = 1, 2
            target_cat[target <= empty_threshold], target_cat[target >= occupied_threshold] = 1, 2

            num_samples += data.size(0)
            num_points += num_samples * data.size(1) * data.size(2) * data.size(3) * data.size(4)
            num_batches += 1
            num_accurate += (output_cat == target_cat).sum().int().item()
            # print('num_accurate', num_accurate)
            # print('num_points', num_points)
            # print('data size', data.size())
            # print('output_cat size', output_cat.size())
            # print('output_cat', output_cat)
            # print('target_cat', target_cat)
            for class_name, class_stats in stats.items():
                output_binary, target_binary = class_stats['to_binary'](output_cat), class_stats['to_binary'](target_cat)
                # print(class_name, 'output_binary', output_binary)
                # print('target_binary', target_binary)
                # print()
                tp_batch, fp_batch, tn_batch, fn_batch = calc_tfpn(output_binary, target_binary)
                stats[class_name]['tp'] += tp_batch
                stats[class_name]['fp'] += fp_batch
                stats[class_name]['tn'] += tn_batch
                stats[class_name]['fn'] += fn_batch
            for i, metric in enumerate(metrics):
                metric_values[i] += metric(output, target)

    print(f'Total testing loss: {test_loss}, total accuracy {round(num_accurate / num_points, 4)}')
    print(f'Occupancy threshold: {occupied_threshold}')
    if monitor is not None:
        monitor.log({
            'evaluation_loss_total': test_loss, 
            'occupied_threshold': occupied_threshold, 
            'empty_threshold': empty_threshold
        })
    if outfile:
        with open(outfile, 'a') as f:
            f.write(f'{test_loss};')
    for i, metric in enumerate(metrics):
        loss_per_sample = metric_values[i] / num_samples
        loss_per_batch = metric_values[i] / num_batches
        if monitor is not None:
            monitor.log({
                f'{metric.__class__.__name__}_loss_per_sample': loss_per_sample, 
                f'{metric.__class__.__name__}_loss_per_batch': loss_per_batch     
            })
        print(f'Avg {metric.__class__.__name__} loss: {loss_per_sample} per sample, {loss_per_batch} per batch')
    for metric in binary_metrics:
        for class_name, class_stats in stats.items():
            loss = metric(class_stats['tp'], class_stats['fp'], class_stats['tn'], class_stats['fn'])
            print(f'{metric.__name__} for class {class_name}: {loss}')
            if outfile:
                with open(outfile, 'a') as f:
                    f.write(f'{class_name}.{loss};')
            if monitor is not None:
                monitor.log({f'{class_name}_{metric.__name__}': loss})

    if outfile:
        with open(outfile, 'a') as f:
            f.write('\n')

    return np.concatenate(predicted_output, axis=0)


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


def test(
        loss_alpha, loss_gamma, occupancy_threshold=0.5, is_3d=False,
        visualize=False, outfile=None, dataset_filepath='dataset.pkl', model_folder='models'
):
    model_path = os.path.join(model_folder, f'model_1C{3 if is_3d else 2}D_a{int(loss_alpha * 100)}g{loss_gamma}.pt')
    train_loader, valid_loader, test_loader = get_dataset(dataset_filepath=dataset_filepath, is_3d=is_3d)

    device = get_device()
    if is_3d:
        model = Unet1C3D().double().to(device)
    else:
        model = Unet1C2D().double().to(device)
    model.load_state_dict(torch.load(model_path))
    criterion = FocalLoss(alpha=loss_alpha, gamma=loss_gamma)

    predicted_output = test_model(
        test_loader, model, criterion, device,
        occupancy_threshold=occupancy_threshold, outfile=outfile
    )

    resolution_meters = 0.25
    x_min_meters = -5
    x_max_meters = 5
    y_min_meters = 0
    y_max_meters = 10
    z_min_meters = -5
    z_max_meters = 5
    if visualize:
        visualize_grids(
            true_grids=test_loader.dataset.Y, predicted_grids=predicted_output,
            odds_threshold=occupancy_threshold, resolution=resolution_meters,
            x_min_meters=x_min_meters, x_max_meters=x_max_meters,
            y_min_meters=y_min_meters, y_max_meters=y_max_meters,
            z_min_meters=z_min_meters, z_max_meters=z_max_meters
        )


if __name__ == "__main__":
    is_3d = True
    alphas = (0.8, 0.85, 0.78)
    gammas = (1, )
    thresholds = (0.4, 0.5, 0.6, 0.7)

    colab_root, local_root, brute_root = '/content/drive/My Drive', '/home/ann/mapping/mn_ws/src/mapless-navigation', '/home/annz/mapping/mn_ws/src/mapless-navigation'
    if os.path.isdir(colab_root):
        root = colab_root
    elif os.path.isdir(local_root):
        root = local_root
    else:
        root = brute_root
    score_file = os.path.join(root, 'test_scores_3d_jan_dropout.csv')
    dataset_file = '/media/giantdrive/coloradar/dataset_5runs.pkl' if root == brute_root else os.path.join(root, 'dataset_5runs.pkl')
    train_loader, valid_loader, test_loader = get_dataset(dataset_filepath=dataset_file, is_3d=is_3d)
    device = get_device()

    for a in alphas:
        for g in gammas:
            model_path = os.path.join('models', f'model_1C{3 if is_3d else 2}D_a{int(a * 100)}g{g}.pt')
            model = Unet1C3D().double().to(device)
            model.load_state_dict(torch.load(model_path))
            criterion = FocalLoss(alpha=a, gamma=g)

            for t in thresholds:
                print('||======')
                print(f'Alpha {a}, Gamma {g}, Threshold {t}, Evaluation:')
                # with open(outfile_name, 'a') as f:
                #     f.write(f'{a};{g};{t};')
                test_model(
                    test_loader, model, criterion, device,
                    occupancy_threshold=t, outfile=score_file
                )
                print()
