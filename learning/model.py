import torch
import torch.nn as nn


class BaseTransform(nn.Module):
    def __init__(self):
        super().__init__()
        # [N, 2, 64, 64, 16]
        self.down1 = nn.Sequential(
            nn.Conv3d(2, 16, kernel_size=2, stride=2, padding=0),   # [N, 16, 32, 32, 8]
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, stride=2))  # [N, 16, 16, 16, 4]

        self.down2 = nn.Sequential(
            nn.Conv3d(16, 128, kernel_size=2, stride=2, padding=0),  # [N, 128, 8, 8, 2]
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, stride=2))  # [N, 128, 4, 4, 1]
        # squeeze
        # [N, 128, 4, 4]

        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2, padding=0, output_padding=0),
            nn.ReLU()
        )
        # [N, 64, 8, 8]
        self.conv1 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU()
        )
        # [N, 64, 8, 8]

        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2, padding=0, output_padding=0),
            nn.ReLU()
        )
        # [N, 32, 16, 16]
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU()
        )
        # [N, 32, 16, 16]

    def forward(self, x):
        # [N, 2, 64, 64, 16]

        out = self.down1(x)
        # [N, 16, 16, 16, 4]
        # print("Shape after down1: ", out.shape)

        out = self.down2(out)
        # [N, 64, 4, 4, 1]
        # print("Shape after down2: ", out.shape)

        out = out.squeeze(4)
        # [N, 64, 4, 4]
        # print("Shape after squeeze: ", out.shape)

        out = self.up1(out)
        # [N, 64, 8, 8]
        # print("Shape after up1: ", out.shape)

        out = self.conv1(out)
        # [N, 64, 8, 8]
        # print("Shape after conv1: ", out.shape)

        out = self.up2(out)
        # [N, 32, 16, 16]
        # print("Shape after up2: ", out.shape)

        out = self.conv2(out)
        # [N, 32, 16, 16]
        # print("Shape after conv2: ", out.shape)

        out = torch.sigmoid(out)
        # [N, 32, 32, 32]
        return out
