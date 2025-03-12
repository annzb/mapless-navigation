import torch
import torch.nn as nn

from models.base import RadarOccupancyModel
from models.grid_based.polar_to_cartesian import PolarToCartesianGrids


class AttentionAggregator(nn.Module):
    def __init__(self, input_dim, output_dim, num_heads=1):
        """
        Attention-based aggregator for mapping N inputs to M outputs.

        Args:
            input_dim (int): Dimensionality of input features (per point).
            output_dim (int): Dimensionality of output features (per point).
            num_heads (int): Number of attention heads (default=1).
        """
        super().__init__()
        self.num_heads = num_heads
        self.query = nn.Parameter(torch.randn(output_dim, input_dim))  # [M, Input_Dim]
        self.attention = nn.MultiheadAttention(embed_dim=input_dim, num_heads=num_heads, batch_first=True)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape [Batch, N_points, Input_Dim].

        Returns:
            torch.Tensor: Output tensor of shape [Batch, M, Input_Dim].
        """
        batch_size = x.shape[0]
        query = self.query.unsqueeze(0).expand(batch_size, -1, -1)
        output, _ = self.attention(query, x, x)
        return output


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
            nn.BatchNorm3d(channels)
            # nn.ReLU()
        )

    def forward(self, x):
        return self.conv(x)


# no size change
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
        # x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        # x = self.relu(x)
        return x


class EncoderBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv = ConvBlock(in_c, out_c)
        self.pool = nn.MaxPool3d(2)

    def forward(self, inputs):
        x = self.conv(inputs)
        p = self.pool(x)
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
        return x


class GridOccupancyModel(RadarOccupancyModel):
    def __init__(self, radar_config, device, *args, **kwargs):
        super().__init__(radar_config, *args, **kwargs)
        self.name = 'grid+unet+logits+pad_v1.0'

        self.polar_to_cartesian = PolarToCartesianGrids(radar_config, device)
        # self.reshape = GridReshape(1)


    def forward(self, polar_frames):                                      # [B, Az, R, El]
        polar_frames = polar_frames.unsqueeze(-1).permute(0, 4, 3, 2, 1)  # [B, 1, El, R, Az]
        cartesian_grids = self.polar_to_cartesian(polar_frames)           # [B, 1, 16, 28, 270]
        grids = self.add_padding(cartesian_grids, padding=(0, 4, 2))      # [B, 1, 16, 32, 272]
        # cartesian_grids = cartesian_grids.permute(0, 4, 3, 2, 1)
        # grids = self.reshape(cartesian_grids)
        s1, p1 = self.e1(grids)                                           # [B, 32, 16, 32, 272], [B, 32, 8, 16, 136]
        s2, p2 = self.e2(p1)                                              # [B, 64, 8, 16, 136],  [B, 64, 4, 8, 68]
        s3, p3 = self.e3(p2)                                              # [B, 128, 4, 8, 68],   [B, 128, 2, 4, 34]
        b = self.bottleneck(p3)                                           # [B, 256, 2, 4, 34]
        o1 = self.d1(b, s3)                                               # [B, 128, 4, 8, 68]
        o2 = self.d2(o1, s2)                                              # [B, 64, 8, 16, 136]
        o3 = self.d3(o2, s1)                                              # [B, 32, 16, 32, 272]
        output = self.output(o3)                                          # [B, 1, 16, 32, 272]
        # print('output with padding shape:', output.shape)
        output = self.remove_padding(output, padding=(0, 4, 2)).squeeze(1).permute(0, 3, 2, 1)  # [B, 270, 28, 16]
        # print('output final shape:', output.shape)
        return output
