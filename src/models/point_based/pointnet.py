from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from Pointnet_Pointnet2_pytorch.models.pointnet2_utils import PointNetSetAbstraction, PointNetFeaturePropagation


# class PointNet(nn.Module):
#     def __init__(self):
#         super(PointNet, self).__init__()
#         # PointNet++ MSG backbone
#         self.sa1 = PointNetSetAbstraction(3776, 0.2, 16, 1 + 3, [32, 32, 64], False)
#         self.sa2 = PointNetSetAbstraction(944, 0.4, 16, 64 + 3, [64, 64, 128], False)
#         self.sa3 = PointNetSetAbstraction(236, 0.6, 16, 128 + 3, [128, 128, 256], False)
#         self.fp3 = PointNetFeaturePropagation(384, [256, 256])
#         self.fp2 = PointNetFeaturePropagation(320, [256, 128])
#         self.fp1 = PointNetFeaturePropagation(128, [128, 128, 128])
#
#         # Global context pooling
#         self.global_pool = nn.AdaptiveAvgPool1d(1)
#
#         # Prediction layers
#         self.conv1 = nn.Conv1d(256, 128, 1)
#         self.bn1 = nn.BatchNorm1d(128)
#         self.drop1 = nn.Dropout(0.5)
#         self.conv2 = nn.Conv1d(128, 1, 1)  # Single channel for log odds
#
#     def forward(self, points):
#         coords, features = points[..., :3], points[..., 3:]
#         xyz = coords.permute(0, 1)         # [3, N_points]
#         features = features.permute(0, 1)  # [n_features, N_points]
#
#         l0_xyz, l0_points = xyz, features
#         l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)
#         l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
#         l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
#
#         l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)
#         l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)
#         l0_points = self.fp1(l0_xyz, l1_xyz, None, l1_points)
#
#         # Incorporate global context
#         global_context = self.global_pool(l0_points).expand_as(l0_points)
#         combined_features = torch.cat((l0_points, global_context), dim=1)
#
#         x = self.drop1(self.bn1(self.conv1(combined_features)))
#         log_odds = self.conv2(x)           # [1, N_points]
#         log_odds = log_odds.permute(0, 1)  # [N_points, 1]
#         return torch.cat((coords[..., :3], log_odds), dim=-1)


class PointNet(nn.Module):
    def __init__(self):
        super(PointNet, self).__init__()
        # PointNet++ MSG backbone
        self.sa1 = PointNetSetAbstraction(2048, 0.4, 16, 1 + 3, [32, 32, 64], False)
        self.sa2 = PointNetSetAbstraction(512, 0.8, 16, 64 + 3, [64, 64, 128], False)
        self.sa3 = PointNetSetAbstraction(128, 1.2, 16, 128 + 3, [128, 128, 256], False)
        self.fp3 = PointNetFeaturePropagation(384, [256, 256])
        self.fp2 = PointNetFeaturePropagation(320, [256, 128])
        self.fp1 = PointNetFeaturePropagation(128, [128, 128, 128])

        # Global context pooling
        self.global_pool = nn.AdaptiveAvgPool1d(1)

        # Prediction layers
        self.conv1 = nn.Conv1d(256, 128, 1)
        self.bn1 = nn.BatchNorm1d(128)
        self.drop1 = nn.Dropout(0.5)
        self.conv2 = nn.Conv1d(128, 1, 1)  # Single channel for log odds

    def forward(self, coords, features):
        xyz = coords.permute(0, 2, 1)  # [B, 3, N_points]
        features = features.permute(0, 2, 1)  # [B, n_features, N_points]

        l0_xyz, l0_points = xyz, features
        # print('xyz, features, l0_xyz, l0_points', xyz.shape, features.shape, l0_xyz.shape, l0_points.shape)
        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)

        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)
        l0_points = self.fp1(l0_xyz, l1_xyz, None, l1_points)

        # Incorporate global context
        global_context = self.global_pool(l0_points).expand_as(l0_points)
        combined_features = torch.cat((l0_points, global_context), dim=1)

        x = F.relu(self.drop1(self.bn1(self.conv1(combined_features))))
        log_odds = self.conv2(x)              # [B, 1, N_points]
        log_odds = log_odds.permute(0, 2, 1)  # [B, N_points, 1]
        return torch.cat((coords[..., :3], log_odds), dim=-1)
    

class PointNet2(nn.Module):
    def __init__(self, num_features=4):
        super(PointNet2, self).__init__()
        self.sa1 = PointNetSetAbstraction(
            npoint=1024, radius=0.5, nsample=32, 
            in_channel=num_features + 3, mlp=[32, 32, 64], group_all=False
        )
        self.sa2 = PointNetSetAbstraction(
            npoint=256, radius=1.0, nsample=32, 
            in_channel=64 + 3, mlp=[64, 64, 128], group_all=False
        )
        self.fp2 = PointNetFeaturePropagation(
            in_channel=128 + 64, mlp=[128, 128]
        )
        self.fp1 = PointNetFeaturePropagation(
            in_channel=128 + num_features, mlp=[128, 128, 64]
        )
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.conv1 = nn.Conv1d(128, 64, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.drop1 = nn.Dropout(0.5)
        self.conv2 = nn.Conv1d(64, 1, 1)

    def forward(self, coords, features):
        xyz = coords.permute(0, 2, 1)            # [B, 3, N_points]
        features = features.permute(0, 2, 1)     # [B, num_features, N_points]

        l0_xyz, l0_points = xyz, features
        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)

        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)
        l0_points = self.fp1(l0_xyz, l1_xyz, l0_points, l1_points)

        global_context = self.global_pool(l0_points).expand_as(l0_points)
        combined_features = torch.cat((l0_points, global_context), dim=1)

        x = F.relu(self.drop1(self.bn1(self.conv1(combined_features))))
        log_odds = self.conv2(x)              # [B, 1, N_points]
        log_odds = log_odds.permute(0, 2, 1)  # [B, N_points, 1]

        return torch.cat((coords[..., :3], log_odds), dim=-1)


class PointNet2Spatial(nn.Module):
    def __init__(self, num_features=4, dropout=0.5):
        super(PointNet2Spatial, self).__init__()
        self.sa1 = PointNetSetAbstraction(
            npoint=1024, radius=0.5, nsample=32, 
            in_channel=num_features + 3, mlp=[32, 32, 64], group_all=False
        )
        self.sa2 = PointNetSetAbstraction(
            npoint=256, radius=1.0, nsample=32, 
            in_channel=64 + 3, mlp=[64, 64, 128], group_all=False
        )
        self.fp2 = PointNetFeaturePropagation(
            in_channel=128 + 64, mlp=[128, 128]
        )
        self.fp1 = PointNetFeaturePropagation(
            in_channel=128 + num_features, mlp=[128, 128, 64]
        )
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.conv1 = nn.Conv1d(128, 64, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.drop1 = nn.Dropout(dropout)
        self.conv2 = nn.Conv1d(64, 4, 1)

    def forward(self, coords, features):
        xyz = coords.permute(0, 2, 1)            # [B, 3, N]
        features = features.permute(0, 2, 1)     # [B, F, N]

        l0_xyz, l0_points = xyz, features
        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)

        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)
        l0_points = self.fp1(l0_xyz, l1_xyz, l0_points, l1_points)

        global_context = self.global_pool(l0_points).expand_as(l0_points)
        combined_features = torch.cat((l0_points, global_context), dim=1)

        x = F.relu(self.drop1(self.bn1(self.conv1(combined_features))))
        output = self.conv2(x)                   # [B, 4, N]
        output = output.permute(0, 2, 1)         # [B, N, 4]
        return output


class SinglePointEncoder(nn.Module):
    def __init__(self, output_size=1024, output_dim=16, batch_norm: bool = True, dropout: Optional[float] = None, **kwargs):
        super().__init__()
        if not isinstance(output_size, int):
            raise ValueError('output_size must be an integer')
        if output_size <= 0:
            raise ValueError('output_size must be positive')
        if not isinstance(output_dim, int):
            raise ValueError('output_dim must be an integer')
        if output_dim <= 0:
            raise ValueError('output_dim must be positive')
        if not isinstance(batch_norm, bool):
            raise ValueError('batch_norm must be a boolean')
        if dropout is not None and not isinstance(dropout, float):
            raise ValueError('dropout must be a float')
        if dropout is not None and (dropout < 0.0 or dropout > 1.0):
            raise ValueError('dropout must be between 0.0 and 1.0')

        self.output_size = output_size
        self.output_dim = output_dim

        self.sa1 = PointNetSetAbstraction(
            npoint=output_size * 2, radius=0.4, nsample=32,
            in_channel=4 + 3, mlp=[16, 16, 64], group_all=False
        )
        self.sa2 = PointNetSetAbstraction(
            npoint=output_size, radius=0.8, nsample=32,
            in_channel=64 + 3, mlp=[64, 64, 256], group_all=False
        )

        layers = [nn.Conv1d(256, 128, 1)]
        if batch_norm:
            layers.append(nn.BatchNorm1d(128))
        layers.append(nn.ReLU())
        if dropout is not None:
            layers.append(nn.Dropout(p=dropout))
        layers.append(nn.Conv1d(128, output_dim, 1))
        self.project = nn.Sequential(*layers)

    def forward(self, cloud):
        cloud = cloud.unsqueeze(0).permute(0, 2, 1)    # (N, 4) -> (1, 4, N)
        coords = cloud[:, :3, :]

        l1_xyz, l1_points = self.sa1(coords, cloud)    
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)

        projected = self.project(l2_points)     # → (1, output_dim, output_size)
        projected = projected.squeeze(0).permute(1, 0)  # → (output_size, output_dim)

        return projected
