#!/home/ann/mapping/venv/bin/python3
import os

import torch
import torch.optim as optim
from torch import nn

# export PYTHONPATH=$PYTHONPATH:src/octomap_radar_analysis/src

from dataset_intensity import get_dataset
from loss_focal import FocalLoss
from model_intensity import Unet1C2D
from model_intensity_3d import Unet1C3D
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
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        output = model(data)
        loss = criterion(output, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def validate_model(valid_loader, model, criterion, device):
    model.eval()
    valid_loss = 0

    with torch.no_grad():
        for data, target in valid_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            valid_loss += loss.item()

    return valid_loss / len(valid_loader)


def train(
        train_loader, valid_loader, test_loader,
        loss_alpha, loss_gamma, num_epochs=10, is_3d=False, occupancy_threshold=0.5,
         model_folder='models'
):
    learning_rate = 0.05
    best_valid_loss = float('inf')
    os.makedirs(model_folder, exist_ok=True)
    model_path = os.path.join(model_folder, f'model_1C{3 if is_3d else 2}D_a{int(loss_alpha * 100)}g{loss_gamma}.pt')

    device = get_device()
    if is_3d:
        model = Unet1C3D().double().to(device)
    else:
        model = Unet1C2D().double().to(device)
    criterion = FocalLoss(alpha=loss_alpha, gamma=loss_gamma)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        print(f'Epoch {epoch + 1}/{num_epochs}:')
        train_model(train_loader, model, criterion, optimizer, device)
        valid_loss = validate_model(valid_loader, model, criterion, device)
        print(f'Validation loss: {valid_loss:.4f}')

        # Save the model if the validation loss decreased
        if valid_loss < best_valid_loss:
            print(f'Validation loss decreased ({best_valid_loss:.6f} --> {valid_loss:.6f}). Saving model ...')
            torch.save(model.state_dict(), model_path)
            best_valid_loss = valid_loss

    # test(loss_alpha, loss_gamma, occupancy_threshold=occupancy_threshold - 0.1, is_3d=is_3d, visualize=False, outfile=None)
    test_model(
        test_loader, model, criterion, device,
        occupancy_threshold=occupancy_threshold, outfile=None
    )
    # test(loss_alpha, loss_gamma, occupancy_threshold=occupancy_threshold + 0.1, is_3d=is_3d, visualize=False, outfile=None)
    return test_loader, model, criterion, device


if __name__ == "__main__":
    # train(loss_alpha=0.1, loss_gamma=1, num_epochs=10, is_3d=True, occupancy_threshold=0.5)
    # raise ValueError('Finish')

    alphas = (0.8, 0.92, 0.85)
    gammas = (1, )
    thresholds = (0.4, 0.5, 0.6)
    n_epochs = (50, 50, 50)

    colab_root, local_root = '/content/drive/My Drive', '/home/ann/mapping/mn_ws/src/mapless-navigation'
    root = colab_root if os.path.isdir(colab_root) else local_root
    score_file = os.path.join(root, 'test_scores_3d.csv')
    dataset_file = os.path.join(root, 'dataset.pkl')
    train_loader, valid_loader, test_loader = get_dataset(dataset_filepath=dataset_file, is_3d=True)

    for g in gammas:
        for i, a in enumerate(alphas):
            print(f'Alpha {a}, Gamma {g}, Training:')
            test_loader, model, criterion, device = train(
                train_loader, valid_loader, test_loader,
                loss_alpha=a, loss_gamma=g,
                num_epochs=n_epochs[i], is_3d=True, occupancy_threshold=0.4, model_folder=root
            )
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
