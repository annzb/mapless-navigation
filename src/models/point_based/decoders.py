from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from Pointnet_Pointnet2_pytorch.models.pointnet2_utils import PointNetFeaturePropagation, PointNetSetAbstraction


class MlpDecoder(nn.Module):
    def __init__(self, latent_dim=4, output_size=1024, output_dim=4, layer_norm: bool = True, dropout: Optional[float] = None, **kwargs):
        super().__init__()
        if not isinstance(output_size, int):
            raise ValueError('output_size must be an integer')
        if output_size <= 0:
            raise ValueError('output_size must be positive')
        if not isinstance(output_dim, int):
            raise ValueError('output_dim must be an integer')
        if output_dim <= 0:
            raise ValueError('output_dim must be positive')
        if not isinstance(latent_dim, int):
            raise ValueError('latent_dim must be an integer')
        if latent_dim <= 0:
            raise ValueError('latent_dim must be positive')
        if not isinstance(layer_norm, bool):
            raise ValueError('layer_norm must be a boolean')
        if dropout is not None and not isinstance(dropout, float):
            raise ValueError('dropout must be a float')
        if dropout is not None and (dropout < 0.0 or dropout > 1.0):
            raise ValueError('dropout must be between 0.0 and 1.0')

        self.output_size = output_size
        self.output_dim = output_dim

        layers = []
        layer_dims = [latent_dim, 128, 256, 512]
        for in_dim, out_dim in zip(layer_dims[:-1], layer_dims[1:]):
            layers.append(nn.Linear(in_dim, out_dim))
            if layer_norm:
                layers.append(nn.LayerNorm(out_dim))
            layers.append(nn.ReLU())
            if dropout is not None:
                layers.append(nn.Dropout(p=dropout))
        layers.append(nn.Linear(layer_dims[-1], output_size * output_dim))
        self.mlp = nn.Sequential(*layers)

    def forward(self, z):
        B = z.shape[0]
        out = self.mlp(z)  # (B, output_size * output_dim)
        return out.view(B, self.output_size, self.output_dim)


class FoldingDecoder(nn.Module):
    """
    Decodes a latent vector into a point cloud by "folding" a 2D grid.
    """
    def __init__(
        self, 
        latent_dim: int, 
        output_dim: int = 4, 
        layer_norm: bool = True, 
        dropout: Optional[float] = None, 
        **kwargs
    ):
        super().__init__()
        # The MLP's input is the latent vector size + 2 for the 2D grid coordinates (u, v)
        input_dim = latent_dim + 2
        
        # A common architecture for folding MLPs
        layers = []
        layer_dims = [input_dim, 512, 512] 
        for in_dim, out_dim in zip(layer_dims[:-1], layer_dims[1:]):
            layers.append(nn.Linear(in_dim, out_dim))
            if layer_norm:
                layers.append(nn.LayerNorm(out_dim))
            layers.append(nn.ReLU())
            if dropout is not None:
                layers.append(nn.Dropout(p=dropout))
        
        layers.append(nn.Linear(layer_dims[-1], output_dim))
        self.mlp = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor, grid: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z (torch.Tensor): Latent vectors, shape (B, latent_dim).
            grid (torch.Tensor): The 2D grid, shape (N, 2).

        Returns:
            torch.Tensor: The predicted point cloud, shape (B, N, output_dim).
        """
        B = z.shape[0]  # Batch size
        N = grid.shape[0]  # Number of points to predict

        # Tile the latent vector and grid to prepare for concatenation
        z_tiled = z.unsqueeze(1).expand(B, N, -1)
        grid_tiled = grid.unsqueeze(0).expand(B, -1, -1)

        # Concatenate each grid point with its corresponding latent vector
        combined = torch.cat([z_tiled, grid_tiled], dim=2)
        
        # Pass through the MLP to get the final folded point cloud
        point_cloud = self.mlp(combined)
        
        return point_cloud


class UpsamplingBlock(nn.Module):
    """A block for upsampling points and refining features."""
    def __init__(self, in_channels, out_channels, mlp_dims):
        super().__init__()
        self.mlp = nn.Sequential()
        for i, dim in enumerate(mlp_dims):
            self.mlp.add_module(f'conv_{i}', nn.Conv1d(in_channels, dim, 1))
            self.mlp.add_module(f'bn_{i}', nn.BatchNorm1d(dim))
            self.mlp.add_module(f'relu_{i}', nn.ReLU())
            in_channels = dim
        self.mlp.add_module('final_conv', nn.Conv1d(in_channels, out_channels, 1))

    def forward(self, xyz, features):
        """
        Args:
            xyz (Tensor): (B, 3, N) point coordinates
            features (Tensor): (B, C, N) point features
        """
        # Upsample features using nearest neighbor interpolation
        # This doubles the number of points, creating a denser feature grid
        features_upsampled = F.interpolate(features, scale_factor=2, mode='nearest')
        xyz_upsampled = F.interpolate(xyz, scale_factor=2, mode='nearest')
        refined_features = self.mlp(features_upsampled)
        return xyz_upsampled, refined_features


class UpsamplingDecoder(nn.Module):
    def __init__(self, latent_dim=32, output_size=1024, output_dim=4, num_coarse_points=128):
        super().__init__()
        self.output_size = output_size
        self.output_dim = output_dim
        self.num_coarse_points = num_coarse_points

        coarse_feature_dim = 256
        self.fc_coarse = nn.Linear(latent_dim, num_coarse_points * (3 + coarse_feature_dim))
        
        # 2. Upsampling Blocks
        # 128 points -> 256 points
        self.upsample1 = UpsamplingBlock(coarse_feature_dim, 128, mlp_dims=[256, 128])
        # 256 points -> 512 points
        self.upsample2 = UpsamplingBlock(128, 128, mlp_dims=[128, 128])
        # 512 points -> 1024 points
        self.upsample3 = UpsamplingBlock(128, 128, mlp_dims=[128, 128])

        # 3. Final Prediction Head
        self.final_head = nn.Sequential(
            nn.Conv1d(128, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, output_dim, 1)
        )

    def forward(self, z):
        B = z.shape[0]
        
        # Generate initial coarse cloud from the latent vector
        coarse_cloud = self.fc_coarse(z).view(B, self.num_coarse_points, 3 + 256)
        
        # Separate coordinates and features, and permute to (B, C, N)
        xyz = coarse_cloud[..., :3].permute(0, 2, 1)
        features = coarse_cloud[..., 3:].permute(0, 2, 1)

        # Sequentially upsample and refine
        xyz, features = self.upsample1(xyz, features)
        xyz, features = self.upsample2(xyz, features)
        xyz, features = self.upsample3(xyz, features)
        
        # Predict final point attributes (e.g., x, y, z offsets + occupancy)
        final_attrs = self.final_head(features) # (B, output_dim, N)
        
        # Combine original upsampled coordinates with predicted attributes
        # Here we assume the final output includes the coordinates
        # If output_dim is 4 (x,y,z,occ), we can treat the first 3 as final coordinates
        # Or, as a more stable alternative, predict *offsets* from the upsampled xyz
        
        final_cloud = final_attrs.permute(0, 2, 1) # (B, N, output_dim)
        
        return final_cloud


class UpsamplingDecoderV2(nn.Module):
    """
    A more robust upsampling decoder inspired by generator networks like PointNet++'s generator.
    """
    def __init__(self, latent_dim=32, output_size=1024, output_dim=4, num_coarse_points=128):
        super().__init__()
        self.output_size = output_size
        self.output_dim = output_dim
        self.num_coarse_points = num_coarse_points
        feature_dim = 128

        self.fc1 = nn.Linear(latent_dim, self.num_coarse_points * feature_dim)
        self.bn1 = nn.BatchNorm1d(feature_dim) 
        self.upsample1 = nn.Linear(feature_dim, 256 * 2)
        self.upsample2 = nn.Linear(256, 256 * 2)
        self.upsample3 = nn.Linear(256, 256 * 2)
        self.final_mlp = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )

    def forward(self, z):
        B = z.shape[0]
        x = self.fc1(z).view(B, self.num_coarse_points, 128)
        x = x.permute(0, 2, 1)
        x = self.bn1(x)
        x = x.permute(0, 2, 1)
        x = self.upsample1(x).view(B, self.num_coarse_points * 2, 256)
        x = self.upsample2(x).view(B, self.num_coarse_points * 4, 256)
        x = self.upsample3(x).view(B, self.num_coarse_points * 8, 256)
        point_cloud = self.final_mlp(x)
        return point_cloud


# class UpsamplingDecoderV2(nn.Module):
#     """
#     A more robust upsampling decoder inspired by generator networks like PointNet++'s generator.
#     """
#     def __init__(self, latent_dim=32, output_size=1024, output_dim=4, num_coarse_points=128):
#         super().__init__()
#         self.output_size = output_size
#         self.output_dim = output_dim
#         self.num_coarse_points = num_coarse_points

#         self.fc1 = nn.Linear(latent_dim, self.num_coarse_points * 128)
#         self.bn1 = nn.BatchNorm1d(self.num_coarse_points)
#         self.upsample1 = nn.Linear(128, 256 * 2)
#         self.upsample2 = nn.Linear(256, 256 * 2)
#         self.upsample3 = nn.Linear(256, 256 * 2)
#         self.final_mlp = nn.Sequential(
#             nn.Linear(256, 128),
#             nn.ReLU(),
#             nn.Linear(128, output_dim)
#         )

#     def forward(self, z):
#         B = z.shape[0]
#         for name, param in self.named_parameters(): print(name, param)
#         x = self.fc1(z).view(B, self.num_coarse_points, 128)
#         x = self.upsample1(x).view(B, self.num_coarse_points * 2, 256)
#         x = self.upsample2(x).view(B, self.num_coarse_points * 4, 256)
#         x = self.upsample3(x).view(B, self.num_coarse_points * 8, 256)
#         point_cloud = self.final_mlp(x)
#         for name, param in self.named_parameters(): print(name, param)
#         return point_cloud
