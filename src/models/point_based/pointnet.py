import torch
import torch.nn as nn

from Pointnet_Pointnet2_pytorch.models.pointnet2_utils import PointNetSetAbstraction, PointNetFeaturePropagation


class PointNet(nn.Module):
    def __init__(self):
        super(PointNet, self).__init__()
        # PointNet++ MSG backbone
        self.sa1 = PointNetSetAbstraction(3776, 0.2, 16, 1 + 3, [32, 32, 64], False)
        self.sa2 = PointNetSetAbstraction(944, 0.4, 16, 64 + 3, [64, 64, 128], False)
        self.sa3 = PointNetSetAbstraction(236, 0.6, 16, 128 + 3, [128, 128, 256], False)
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

    def forward(self, points, features):
        xyz = points.permute(0, 2, 1)  # [B, 3, N_points]
        features = features.permute(0, 2, 1)  # [B, n_features, N_points]

        l0_xyz, l0_points = xyz, features
        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)

        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)
        l0_points = self.fp1(l0_xyz, l1_xyz, None, l1_points)

        # Incorporate global context
        global_context = self.global_pool(l0_points).expand_as(l0_points)
        combined_features = torch.cat((l0_points, global_context), dim=1)

        x = self.drop1(self.bn1(self.conv1(combined_features)))
        log_odds = self.conv2(x)  # [B, 1, N_points]
        log_odds = log_odds.permute(0, 2, 1)  # [B, N_points, 1]
        return torch.cat((points[..., :3], log_odds), dim=-1)
