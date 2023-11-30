#!/home/ann/mapping/venv/bin/python3

import torch
import torch.optim as optim
from torch import nn

# export PYTHONPATH=$PYTHONPATH:src/octomap_radar_analysis/src

from dataset import get_dataset
from dataset_intensity import get_dataset as get_dataset_1C2D
from loss_focal import FocalLoss
from model import BaseTransform
from model_intensity import Unet1C2D
from evaluate import test_1c2d


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


def train_2C():
    num_epochs = 10
    learning_rate = 0.01
    best_valid_loss = float('inf')

    train_loader, valid_loader, test_loader = get_dataset(dataset_filepath='dataset.pkl')
    device = get_device()
    model = BaseTransform().double().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        print(f'Epoch {epoch + 1}/{num_epochs}:')
        train_model(train_loader, model, criterion, optimizer, device)
        valid_loss = validate_model(valid_loader, model, criterion, device)
        print(f'Validation loss: {valid_loss:.4f}')

        # Save the model if the validation loss decreased
        if valid_loss < best_valid_loss:
            print(f'Validation loss decreased ({best_valid_loss:.6f} --> {valid_loss:.6f}). Saving model ...')
            torch.save(model.state_dict(), 'model.pt')
            best_valid_loss = valid_loss

    # print('Starting testing:')
    # test_model(test_loader, model, device)
    # print('Testing finished.')


def train_1c2d(loss_alpha, loss_gamma, num_epochs=10):
    learning_rate = 0.05
    best_valid_loss = float('inf')

    train_loader, valid_loader, test_loader = get_dataset_1C2D(dataset_filepath='dataset.pkl')
    device = get_device()
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
            torch.save(model.state_dict(), f'model_1C2D_a{int(loss_alpha * 100)}g{loss_gamma}.pt')
            best_valid_loss = valid_loss


if __name__ == "__main__":
    alphas = (0.05, 0.1, 0.3, 0.5, 0.8, 0.9, 0.95, 0.98)
    gammas = (1, 2, 3, 4, 5, 6, 7, 8)
    thresholds = (0.4, 0.5, 0.6)
    n_epochs = 30

    for a in alphas:
        for g in gammas:
            train_1c2d(loss_alpha=a, loss_gamma=g, num_epochs=n_epochs)
            for t in thresholds:
                print('||======')
                print(f'Alpha {a}, Gamma {g}, Threshold {t}, Evaluation:')
                test_1c2d(loss_alpha=a, loss_gamma=g, occupancy_threshold=t)
                print()
