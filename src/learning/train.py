#!/home/ann/mapping/venv/bin/python3
import os

import torch
import torch.optim as optim
from torch import nn
import wandb

# export PYTHONPATH=$PYTHONPATH:src/octomap_radar_analysis/src

from dataset_intensity import get_dataset
from loss_focal import FocalLoss
from model_intensity import Unet1C2D
from model_intensity_3d import Unet1C3D
from evaluate import test_model


def get_device():
    if torch.cuda.is_available():
        print('GPU is available.')
        device = torch.device("cuda:1")
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
        loss_alpha, loss_gamma, num_epochs=10, is_3d=False, occupancy_threshold=0.5,
        model_folder='models', learning_rate=0.01, monitor=None
):
    best_valid_loss = float('inf')
    os.makedirs(model_folder, exist_ok=True)
    model_path = os.path.join(model_folder, f'model_jan29_1C{3 if is_3d else 2}D_a{int(loss_alpha * 100)}g{loss_gamma}.pt')

    device = get_device()
    if is_3d:
        model = Unet1C3D().double().to(device)
    else:
        model = Unet1C2D().double().to(device)
    criterion = FocalLoss(alpha=loss_alpha, gamma=loss_gamma)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)

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
        occupancy_threshold=occupancy_threshold, outfile=None
    )
    # test(loss_alpha, loss_gamma, occupancy_threshold=occupancy_threshold + 0.1, is_3d=is_3d, visualize=False, outfile=None)
    return test_loader, model, criterion, device


def run():
    # train(loss_alpha=0.1, loss_gamma=1, num_epochs=10, is_3d=True, occupancy_threshold=0.5)
    # raise ValueError('Finish')
    learning_rate = 0.01
    batch_size = 32
    alphas = (0.8, 0.85, 0.7, 0.9)
    gammas = (1, )
    thresholds = (0.4, 0.5, 0.6, 0.7)
    n_epochs = (100, 100, 100, 100)
    config = {
        'gamma': gammas[0],
        'learning_rate': learning_rate,
        'architecture': 'UNet-3D-dropout',
        'dataset': '5runs',
        'epochs': n_epochs[0],
        'batch_size': batch_size,
        "optimizer": "adam"
    }

    colab_root, local_root, brute_root = '/content/drive/My Drive', '/home/ann/mapping/mn_ws/src/mapless-navigation', '/home/annz/mapping/mn_ws/src/mapless-navigation'
    if os.path.isdir(colab_root):
        root = colab_root
    elif os.path.isdir(local_root):
        root = local_root
    else:
        root = brute_root
    score_file = os.path.join(root, 'test_scores_3d.csv')
    dataset_file = '/media/giantdrive/coloradar/dataset_5runs.pkl' if root == brute_root else os.path.join(root, 'dataset_5runs.pkl')
    train_loader, valid_loader, test_loader = get_dataset(dataset_filepath=dataset_file, is_3d=True, batch_size=batch_size)

    for g in gammas:
        for i, a in enumerate(alphas):
            config.update({'alpha': a})
            wandb.init(project='radar-occupancy', entity='annazabn', config=config)

            print(f'Alpha {a}, Gamma {g}, Training:')
            test_loader, model, criterion, device = train(
                train_loader, valid_loader, test_loader,
                loss_alpha=a, loss_gamma=g, learning_rate=learning_rate,
                num_epochs=n_epochs[i], is_3d=True, occupancy_threshold=0.4, model_folder=root
            )
            wandb.watch(model, log="all")

            for t in thresholds:
                print('||======')
                print(f'Threshold {t}, Evaluation:')
                with open(score_file, 'a') as f:
                    f.write(f'{a};{g};{t};')
                test_model(
                    test_loader, model, criterion, device,
                    occupancy_threshold=t, outfile=score_file
                )
                print()
            wandb.finish()


if __name__ == "__main__":
    run()
