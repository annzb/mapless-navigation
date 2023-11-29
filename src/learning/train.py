#!/home/ann/mapping/venv/bin/python3

import torch
import torch.optim as optim
from torch import nn

# export PYTHONPATH=$PYTHONPATH:src/octomap_radar_analysis/src

from dataset import get_dataset
from model import BaseTransform


def get_device():
    if torch.cuda.is_available():
        print('GPU is available.')
        device = torch.device("cuda")
    else:
        print('GPU is not available, using CPU.')
        device = torch.device("cpu")
    return device


def train_model(train_loader, model, criterion, optimizer, device):
    # Set model to training mode
    model.train()

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        # Forward pass
        output = model(data)
        loss = criterion(output, target)

        # Backward pass and optimization
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


def main():
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


if __name__ == "__main__":
    main()
