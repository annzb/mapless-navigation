#!/home/ann/mapping/venv/bin/python3

import torch
import torch.optim as optim
from torch import nn

# export PYTHONPATH=$PYTHONPATH:src/octomap_radar_analysis/src

from dataset_intensity import get_dataset
from loss_focal import FocalLoss
from model_intensity import Unet1C2D
from model_intensity_3d import Unet1C3D
from evaluate import test


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


def train(loss_alpha, loss_gamma, num_epochs=10, is_3d=False, occupancy_threshold=0.5):
    learning_rate = 0.05
    best_valid_loss = float('inf')

    train_loader, valid_loader, test_loader = get_dataset(dataset_filepath='dataset.pkl', is_3d=is_3d)
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
            torch.save(model.state_dict(), f'model_1C{3 if is_3d else 2}D_a{int(loss_alpha * 100)}g{loss_gamma}.pt')
            best_valid_loss = valid_loss

    # test(loss_alpha, loss_gamma, occupancy_threshold=occupancy_threshold - 0.1, is_3d=is_3d, visualize=False, outfile=None)
    test(loss_alpha, loss_gamma, occupancy_threshold=occupancy_threshold, is_3d=is_3d, visualize=False, outfile=None)
    # test(loss_alpha, loss_gamma, occupancy_threshold=occupancy_threshold + 0.1, is_3d=is_3d, visualize=False, outfile=None)


if __name__ == "__main__":
    # train(loss_alpha=0.1, loss_gamma=1, num_epochs=10, is_3d=True, occupancy_threshold=0.5)
    # raise ValueError('Finish')

    alphas = (0.3, 0.5, 0.7, 0.9, 0.95)
    gammas = (1, 2, 3, 4, 5)
    thresholds = (0.4, 0.5, 0.6)
    n_epochs = 30
    outfile = 'test_scores.csv'

    for a in alphas:
        for g in gammas:
            print(f'Alpha {a}, Gamma {g}, Threshold {t}, Training:')
            train(loss_alpha=a, loss_gamma=g, num_epochs=n_epochs, is_3d=True, occupancy_threshold=0.4)
            for t in thresholds:
                print('||======')
                print(f'Threshold {t}, Evaluation:')
                with open(outfile, 'a') as f:
                    f.write(f'{a};{g};{t};')
                test(loss_alpha=a, loss_gamma=g, occupancy_threshold=t, outfile=outfile)
                print()
