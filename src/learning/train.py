#!/home/ann/mapping/venv/bin/python3
import os

import torch
import torch.optim as optim
from torch import nn
import wandb

# export PYTHONPATH=$PYTHONPATH:src/octomap_radar_analysis/src

from dataset_intensity import get_dataset
from loss_focal import FocalLoss
from loss_polar import WeightedBceLoss
from model_intensity import Unet1C2D
from model_intensity_3d import Unet1C3D
from model_intensity_3d_polar import Unet1C3DPolar
from evaluate import test_model


def get_device():
    if torch.cuda.is_available():
        print('GPU is available.')
        device = torch.device("cuda")
    else:
        print('GPU is not available, using CPU.')
        device = torch.device("cpu")
    return device


def train_model(train_loader, model, criterion, optimizer, device):
    model.train()
    train_loss = 0
    for batch_idx, (data, target, _) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        output = model(data)
        loss = criterion(output, target)
        train_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return train_loss / len(train_loader)


def validate_model(valid_loader, model, criterion, device):
    model.eval()
    valid_loss = 0
    with torch.no_grad():
        for data, target, _ in valid_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            valid_loss += loss.item()
    return valid_loss / len(valid_loader)


def train(
        train_loader, valid_loader, test_loader,
        loss_alpha=None, loss_gamma=None, loss_w=None,
        num_epochs=10, is_3d=False, occupied_threshold=0.5, empty_threshold=0.49,
        model_folder='models', learning_rate=0.01, monitor=None, use_polar=False
):
    if not (isinstance(occupied_threshold, float) and
            isinstance(empty_threshold, float) and
            0 < occupied_threshold < 1 and
            0 < empty_threshold < 1 and
            occupied_threshold > empty_threshold
    ):
        raise ValueError('Invalid thresholds')

    best_valid_loss = float('inf')
    os.makedirs(model_folder, exist_ok=True)
    model_path = os.path.join(model_folder, f'model_march24_1C')

    device = get_device()
    if use_polar:
        if not isinstance(loss_w, int) and not isinstance(loss_w, float):
            raise ValueError('Invalid loss_w, expected a number')
        model_path += f'3D_w{loss_w}.pt'
        model = Unet1C3DPolar().double().to(device)
        criterion = WeightedBceLoss(w=loss_w)
        output_is_prob = True
    else:
        output_is_prob = False
        if not isinstance(loss_alpha, int) or not isinstance(loss_alpha, float):
            raise ValueError('Invalid loss_alpha, expected a number')
        if not isinstance(loss_gamma, int) or not isinstance(loss_gamma, float):
            raise ValueError('Invalid loss_gamma, expected a number')
        param_str = f'a{int(loss_alpha * 100)}g{loss_gamma}'
        if is_3d:
            model_path += f'3D_{param_str}.pt'
            model = Unet1C3D().double().to(device)
        else:
            model_path += f'2D_{param_str}.pt'
            model = Unet1C2D().double().to(device)
        criterion = FocalLoss(alpha=loss_alpha, gamma=loss_gamma)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        print(f'Epoch {epoch + 1}/{num_epochs}:')
        train_loss = train_model(train_loader, model, criterion, optimizer, device)
        valid_loss = validate_model(valid_loader, model, criterion, device)
        print(f'Validation loss: {valid_loss:.4f}')

        # Save the model if the validation loss decreased
        if valid_loss < best_valid_loss:
            print(f'Validation loss decreased ({best_valid_loss:.6f} --> {valid_loss:.6f}). Saving model ...')
            torch.save(model.state_dict(), model_path)
            best_valid_loss = valid_loss
        if monitor is not None:
            monitor.log({'epoch': epoch, 'train_loss': train_loss, 'valid_loss': valid_loss})

    # test(loss_alpha, loss_gamma, occupancy_threshold=occupancy_threshold - 0.1, is_3d=is_3d, visualize=False, outfile=None)
    test_model(
        test_loader, model, criterion, device,
        occupied_threshold=occupied_threshold, empty_threshold=empty_threshold,
        outfile=None, output_is_prob=output_is_prob
    )
    # test(loss_alpha, loss_gamma, occupancy_threshold=occupancy_threshold + 0.1, is_3d=is_3d, visualize=False, outfile=None)
    return test_loader, model, criterion, device


def run():
    # train(loss_alpha=0.1, loss_gamma=1, num_epochs=10, is_3d=True, occupancy_threshold=0.5)
    # raise ValueError('Finish')
    use_polar = True
    occupied_threshold, empty_threshold = 0.8, 0.2
    learning_rate = 0.01
    batch_size = 32
    alphas = (0.8, 0.85, 0.7, 0.9)
    gammas = (1, )
    thresholds = (0.4, 0.5, 0.6, 0.7)
    n_epochs = (100, 100, 100, 100)
    config = {
        'gamma': gammas[0],
        'learning_rate': learning_rate,
        'architecture': 'UNet-3D-polar',
        'dropout': 0.1,
        'dataset': '7runs_polar',
        'epochs': n_epochs[0],
        'batch_size': batch_size,
        "optimizer": "adam",
        'use_polar': use_polar
    }

    colab_root, local_root, brute_root = '/content/drive/My Drive', '/home/ann/mapping/mn_ws/src/mapless-navigation', '/home/annz/mapping/mn_ws/src/mapless-navigation'
    if os.path.isdir(colab_root):
        root = colab_root
    elif os.path.isdir(local_root):
        root = local_root
    else:
        root = brute_root
    score_file = os.path.join(root, 'test_scores_3d.csv')
    dataset_filename = 'dataset_7runs_rangelimit.pkl'
    dataset_file = f'/media/giantdrive/coloradar/{dataset_filename}' if root == brute_root else os.path.join(root, dataset_filename)
    train_loader, valid_loader, test_loader = get_dataset(dataset_filepath=dataset_file, is_3d=True, use_polar=use_polar, batch_size=batch_size)

    # for g in gammas:
    #     for i, a in enumerate(alphas):
    #         config.update({'alpha': a})
    wandb.init(project='radar-occupancy', entity='annazabn', config=config)

    # print(f'Alpha {a}, Gamma {g}, Training:')
    test_loader, model, criterion, device = train(
        train_loader, valid_loader, test_loader,
        loss_w=5, learning_rate=learning_rate,
        num_epochs=30, is_3d=True, use_polar=use_polar,
        occupied_threshold=occupied_threshold, empty_threshold=empty_threshold,
        model_folder=root
    )
    wandb.watch(model, log="all")
    wandb.finish()
    # for t in thresholds:
    #     print('||======')
    #     print(f'Threshold {t}, Evaluation:')
    #     with open(score_file, 'a') as f:
    #         f.write(f'{a};{g};{t};')
    #     test_model(
    #         test_loader, model, criterion, device,
    #         occupancy_threshold=t, outfile=score_file
    #     )
    #     print()


if __name__ == "__main__":
    run()
