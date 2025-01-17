import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from model_polar import PolarToCartesian


class CloudsToGrids(nn.Module):
    def __init__(self, radar_config, num_hidden_dim=8):
        """
        Initializes the CloudsToGrids layer with an IntensityAggregation layer.

        Args:
            radar_config: Configuration object containing voxel size, grid size, and other parameters.
        """
        super().__init__()
        self.voxel_size = radar_config.grid_resolution
        self.grid_size = radar_config.grid_size  # (X, Y, Z)
        self.point_range = radar_config.point_range  # (xmin, xmax, ymin, ymax, zmin, zmax)
        self.aggregate_intensity = nn.Sequential(
            nn.Linear(radar_config.num_radar_points, num_hidden_dim),
            nn.ReLU(),
            nn.Linear(num_hidden_dim, radar_config.num_grid_voxels)
        )

    def forward(self, clouds):
        """
        Converts a batch of point clouds to voxel grids.

        Args:
            clouds (torch.Tensor): Point cloud of shape [B, N_points, 4] (X, Y, Z, intensity).

        Returns:
            torch.Tensor: Voxel grid of shape [B, X, Y, Z, 1].
        """
        batch_size = clouds.shape[0]
        intensity = clouds[..., 3]
        flat_voxel_values = self.aggregate_intensity(intensity)
        voxel_grid = flat_voxel_values.view(batch_size, self.grid_size[0], self.grid_size[1], self.grid_size[2], -1)
        return voxel_grid


class GridReshape(nn.Module):
    def __init__(self, channels):
        """
        A convolutional module to transform a tensor of shape
        [B, C, 16, 28, 270] into [B, C, 16, 32, 256].

        Args:
            in_channels (int): Number of input channels (C).
            out_channels (int): Number of output channels (C).
        """
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(
                in_channels=channels, out_channels=channels,
                kernel_size=(1, 1, 8), dilation=(1, 1, 2)
            ),
            nn.BatchNorm3d(channels),
            nn.ConvTranspose3d(channels, channels, kernel_size=(1, 5, 1)),
            nn.BatchNorm3d(channels),
            nn.ReLU()
        )

    def forward(self, x):
        return self.conv(x)


# no dim change
class ConvBlock(nn.Module):
    def __init__(self, in_c, out_c):
            super().__init__()
            self.conv1 = nn.Conv3d(in_c, out_c, kernel_size=3, padding=1)
            self.bn1 = nn.BatchNorm3d(out_c)
            self.conv2 = nn.Conv3d(out_c, out_c, kernel_size=3, padding=1)
            self.bn2 = nn.BatchNorm3d(out_c)
            self.relu = nn.ReLU()

    def forward(self, inputs):
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        return x


class EncoderBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv = ConvBlock(in_c, out_c)
        self.pool = nn.MaxPool3d(2)

    def forward(self, inputs):
        # print(f'Shape before conv {inputs.shape}')
        x = self.conv(inputs)
        # print(f'Shape after conv {x.shape}')
        p = self.pool(x)
        # print(f'Shape after pool {p.shape}')
        # print('----')
        return x, p


class DecoderBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.up = nn.ConvTranspose3d(in_c, out_c, kernel_size=2, stride=2, padding=0)
        self.conv = ConvBlock(out_c+out_c, out_c)

    def forward(self, inputs, skip):
        x = self.up(inputs)
        x = torch.cat((x, skip), dim=1)
        x = self.conv(x)
        # print('----')
        return x


class Unet1C3DPolar(nn.Module):
    def __init__(self, radar_config):
        super().__init__()
        self.name = 'cart+grid+unet_v1.0'
        # self.increase_depth = nn.Conv3d(
        #     1, 1,
        #     kernel_size=(2, 1, 1), dilation=(2, 1, 1),
        #     padding=(5, 0, 0), padding_mode='circular'
        # )
        # self.dropout = nn.Dropout3d(dropout_rate)
        self.polar_to_cartesian = PolarToCartesian(radar_config)
        self.clouds_to_grids = CloudsToGrids(radar_config)
        # self.reshape = InputReshape(1, 1)
        self.e1 = EncoderBlock(1, 32)
        self.e2 = EncoderBlock(32, 64)
        self.e3 = EncoderBlock(64, 128)
        # self.e4 = EncoderBlock(128, 256)
        self.b = ConvBlock(128, 256)
        # self.d1 = DecoderBlock(512, 256)
        self.d2 = DecoderBlock(256, 128)
        self.d3 = DecoderBlock(128, 64)
        self.d4 = DecoderBlock(64, 1)
        # self.output = nn.Conv3d(32,  1, kernel_size=1)
        self.reshape = GridReshape(1)

    def forward(self, polar_frames):
        cartesian_points = self.polar_to_cartesian(polar_frames)  # [B, 151040, 4]
        print('cartesian_points.shape', cartesian_points.shape)
        grids = self.clouds_to_grids(cartesian_points)  # [B, 270, 28, 16, 1]
        print('grids.shape', grids.shape)
        grids = grids.permute(0, 4, 3, 2, 1)
        print('grids.shape', grids.shape)
        grids = self.reshape(grids)
        print('reshaped grids.shape', grids.shape)
        s1, p1 = self.e1(grids)
        print('shape after e1', p1.shape)
        s2, p2 = self.e2(p1)
        print('shape after e2', p2.shape)
        s_final, p_final = self.e3(p2)
        print('shape after e3', p_final.shape)
        # s4, p4 = self.e4(p3)
        b = self.b(p_final)
        # b = self.dropout(b)
        # print('shape after bottleneck', b.shape)
        # d1 = self.d1(b, s4)
        d2 = self.d2(b, s_final)
        # print('shape after d1', b.shape)
        d3 = self.d3(d2, s2)
        # print('shape after d2', d3.shape)
        d4 = self.d4(d3, s1)
        print('shape d4', d4.shape)
        # outputs = self.output(d4)
        # print('shape after output', outputs.shape)
        # if outputs.size(1) == 1:  # convert [N, 1, D, H, W] to [N, D, H, W]
        #     outputs = outputs.squeeze(1)
            # print('shape after squeeze', outputs.shape)
        # print('---------')
        outputs = torch.sigmoid(d4)
        return outputs
