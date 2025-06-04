import torch
import torch.nn as nn


# class GridReshape(nn.Module):
#     def __init__(self, in_channels: int, in_shape: tuple, out_shape: tuple, batch_norm: bool = True):
#         """
#         Reshape a tensor from in_shape → out_shape using convolutions.

#         Args:
#             channels (int): Input/output channel count (assumed unchanged)
#             in_shape (tuple): (D_in, H_in, W_in)
#             out_shape (tuple): (D_out, H_out, W_out)
#         """
#         super().__init__()
#         D_in, H_in, W_in = in_shape
#         D_out, H_out, W_out = out_shape

#         layers = []

#         if D_in != D_out:
#             layers.append(nn.ConvTranspose3d(in_channels, in_channels, kernel_size=(3, 1, 1), stride=(D_out // D_in, 1, 1)))
#             if batch_norm:
#                 layers.append(nn.BatchNorm3d(in_channels))

#         if H_in != H_out:
#             layers.append(nn.ConvTranspose3d(in_channels, in_channels, kernel_size=(1, 3, 1), stride=(1, H_out // H_in, 1)))
#             if batch_norm:
#                 layers.append(nn.BatchNorm3d(in_channels))

#         if W_in != W_out:
#             factor = W_out // W_in if W_out > W_in else 1
#             kernel = 8 if factor > 1 else 3
#             stride = factor
#             layers.append(nn.Conv3d(
#                 in_channels, in_channels,
#                 kernel_size=(1, 1, kernel),
#                 stride=(1, 1, stride),
#                 dilation=(1, 1, 2 if kernel == 8 else 1)
#             ))
#             if batch_norm:
#                 layers.append(nn.BatchNorm3d(in_channels))

#         self.reshape = nn.Sequential(*layers)

#     def forward(self, x):
#         return self.reshape(x)


class GridReshape(nn.Module):
    def __init__(self, in_channels: int, out_shape: tuple, batch_norm: bool = True):
        super().__init__()
        upsample = nn.Upsample(size=out_shape, mode='trilinear', align_corners=False)
        conv = nn.Conv3d(in_channels, in_channels, kernel_size=1)
        layers = [upsample, conv]
        if batch_norm:
            layers.append(nn.BatchNorm3d(in_channels))
        self.reshape = nn.Sequential(*layers)

    def forward(self, x):
        return self.reshape(x)
    

class GridUnreshape(nn.Module):
    def __init__(self, in_channels: int, out_shape: tuple, batch_norm: bool = True):
        super().__init__()
        downsample = nn.Sequential(
            nn.Conv3d(in_channels, in_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels, in_channels, kernel_size=1)
        )
        resize = nn.Upsample(size=out_shape, mode='trilinear', align_corners=False)
        layers = [downsample, resize]
        if batch_norm:
            layers.append(nn.BatchNorm3d(in_channels))
        self.reshape = nn.Sequential(*layers)

    def forward(self, x):
        return self.reshape(x)



if __name__ == '__main__':
    out_shape = (16, 32, 256)
    grid_reshape = GridReshape(in_channels=1, out_shape=out_shape)
    X = torch.randn(2, 1, 16, 29, 274)
    Y = grid_reshape(X)
    print(Y.shape)
    assert Y.shape == (2, 1, *out_shape)

    grid_unreshape = GridUnreshape(in_channels=1, out_shape=(16, 29, 274))
    Z = grid_unreshape(Y)
    print(Z.shape)
    assert Z.shape == (2, 1, 16, 29, 274)
