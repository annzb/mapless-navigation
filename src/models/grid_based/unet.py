import torch
import torch.nn as nn
import torch.nn.functional as F


# no size change
# class ConvBlock(nn.Module):
#     def __init__(self, in_c, out_c):
#             super().__init__()
#             self.conv1 = nn.Conv3d(in_c, out_c, kernel_size=3, padding=1)
#             self.bn1 = nn.BatchNorm3d(out_c)
#             self.conv2 = nn.Conv3d(out_c, out_c, kernel_size=3, padding=1)
#             self.bn2 = nn.BatchNorm3d(out_c)
#             self.relu = nn.ReLU()

#     def forward(self, inputs):
#         x = self.conv1(inputs)
#         x = self.bn1(x)
#         # x = self.relu(x)
#         x = self.conv2(x)
#         x = self.bn2(x)
#         # x = self.relu(x)
#         return x


# class EncoderBlock(nn.Module):
#     def __init__(self, in_c, out_c):
#         super().__init__()
#         self.conv = ConvBlock(in_c, out_c)
#         self.pool = nn.MaxPool3d(2)

#     def forward(self, inputs):
#         x = self.conv(inputs)
#         p = self.pool(x)
#         return x, p


# class DecoderBlock(nn.Module):
#     def __init__(self, in_c, out_c):
#         super().__init__()
#         self.up = nn.ConvTranspose3d(in_c, out_c, kernel_size=2, stride=2, padding=0)
#         self.conv = ConvBlock(out_c+out_c, out_c)

#     def forward(self, inputs, skip):
#         x = self.up(inputs)
#         x = torch.cat((x, skip), dim=1)
#         x = self.conv(x)
#         return x


# class Unet1C3D(nn.Module):
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         self.e1 = EncoderBlock(1, 32)
#         self.e2 = EncoderBlock(32, 64)
#         self.e3 = EncoderBlock(64, 128)

#         self.bottleneck = ConvBlock(128, 256)

#         self.d1 = DecoderBlock(256, 128)
#         self.d2 = DecoderBlock(128, 64)
#         self.d3 = DecoderBlock(64, 32)

#         self.output = nn.Sequential(
#             nn.Conv3d(32, 64, kernel_size=1),
#             nn.BatchNorm3d(64),
#             # nn.ReLU(),
#             nn.Conv3d(64, 1, kernel_size=1),
#             nn.Sigmoid()
#         )

#     def add_padding(self, grids, padding=(0, 0, 0)):
#         """
#         Adds padding to the input grids.

#         Args:
#             grids (torch.Tensor): Input tensor of shape [B, C, A, B, C].
#             padding (tuple): Padding sizes for dimensions A, B, C in the format (a, b, c).

#         Returns:
#             torch.Tensor: Padded tensor.
#         """
#         pad_a, pad_b, pad_c = padding
#         padded_grids = F.pad(grids, (0, pad_c, 0, pad_b, 0, pad_a))
#         return padded_grids

#     def remove_padding(self, grids, padding=(0, 0, 0)):
#         """
#         Removes padding from the input grids.

#         Args:
#             grids (torch.Tensor): Padded tensor of shape [B, C, A+p, B+q, C+r].
#             padding (tuple): Padding sizes for dimensions A, B, C in the format (a, b, c).

#         Returns:
#             torch.Tensor: Tensor with padding removed.
#         """
#         pad_a, pad_b, pad_c = padding
#         cropped_grids = grids[..., pad_a:, pad_b:, pad_c:]
#         return cropped_grids


#     def forward(self, cartesian_grids):                                   # [B, 1, 16, 28, 270]
#         grids = self.add_padding(cartesian_grids, padding=(0, 4, 2))      # [B, 1, 16, 32, 272]
#         # cartesian_grids = cartesian_grids.permute(0, 4, 3, 2, 1)
#         # grids = self.reshape(cartesian_grids)
#         s1, p1 = self.e1(grids)                                           # [B, 32, 16, 32, 272], [B, 32, 8, 16, 136]
#         s2, p2 = self.e2(p1)                                              # [B, 64, 8, 16, 136],  [B, 64, 4, 8, 68]
#         s3, p3 = self.e3(p2)                                              # [B, 128, 4, 8, 68],   [B, 128, 2, 4, 34]
#         b = self.bottleneck(p3)                                           # [B, 256, 2, 4, 34]
#         o1 = self.d1(b, s3)                                               # [B, 128, 4, 8, 68]
#         o2 = self.d2(o1, s2)                                              # [B, 64, 8, 16, 136]
#         o3 = self.d3(o2, s1)                                              # [B, 32, 16, 32, 272]
#         output = self.output(o3)                                          # [B, 1, 16, 32, 272]
#         # print('output with padding shape:', output.shape)
#         output = self.remove_padding(output, padding=(0, 4, 2)).squeeze(1).permute(0, 3, 2, 1)  # [B, 270, 28, 16]
#         # print('output final shape:', output.shape)
#         return output


class ConvBlock(nn.Module):
    def __init__(self, in_c, out_c, block_size=2, use_batch_norm=True):
        super().__init__()
        layers = []
        for i in range(block_size):
            conv_in = in_c if i == 0 else out_c
            layers.append(nn.Conv3d(conv_in, out_c, kernel_size=3, padding=1))
            if use_batch_norm:
                layers.append(nn.BatchNorm3d(out_c))
            layers.append(nn.ReLU(inplace=True))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class EncoderBlock(nn.Module):
    def __init__(self, in_c, out_c, block_size, use_batch_norm):
        super().__init__()
        self.conv = ConvBlock(in_c, out_c, block_size, use_batch_norm)
        self.pool = nn.MaxPool3d(2)

    def forward(self, x):
        x = self.conv(x)
        p = self.pool(x)
        return x, p


class DecoderBlock(nn.Module):
    def __init__(self, in_c, out_c, block_size, use_batch_norm):
        super().__init__()
        self.up = nn.ConvTranspose3d(in_c, out_c, kernel_size=2, stride=2)
        self.conv = ConvBlock(out_c * 2, out_c, block_size, use_batch_norm)

    def forward(self, x, skip):
        x = self.up(x)
        x = torch.cat((x, skip), dim=1)
        return self.conv(x)


class UNet3D(nn.Module):
    def __init__(self, in_channels, in_shape, depth, conv_block_size, conv_block_batch_norm, base_channels=32):
        super().__init__()

        assert in_channels > 0
        assert depth > 0
        assert conv_block_size > 0
        assert all(d > 0 for d in in_shape)

        self.in_shape = in_shape
        self.depth = depth
        encoder_channels = [in_channels] + [base_channels * (2 ** i) for i in range(depth)]

        self.encoders = nn.ModuleList()
        for i in range(depth):
            self.encoders.append(EncoderBlock(encoder_channels[i], encoder_channels[i + 1], conv_block_size, conv_block_batch_norm))

        self.bottleneck = ConvBlock(encoder_channels[-1], encoder_channels[-1] * 2, conv_block_size, conv_block_batch_norm)

        decoder_channels = encoder_channels[::-1]
        self.decoders = nn.ModuleList()
        for i in range(depth):
            self.decoders.append(DecoderBlock(decoder_channels[i] * 2, decoder_channels[i], conv_block_size, conv_block_batch_norm))

        output_layers = [nn.Conv3d(base_channels, base_channels // 2, kernel_size=1)]
        if conv_block_batch_norm:
            output_layers.append(nn.BatchNorm3d(base_channels // 2))
        output_layers.append(nn.ReLU(inplace=True))
        output_layers.append(nn.Conv3d(base_channels // 2, 1, kernel_size=1))
        output_layers.append(nn.Sigmoid())
        self.output_layer = nn.Sequential(*output_layers)

    def forward(self, x):
        skips = []
        for encoder in self.encoders:
            s, x = encoder(x)
            skips.append(s)

        x = self.bottleneck(x)

        for decoder, skip in zip(self.decoders, reversed(skips)):
            x = decoder(x, skip)

        return self.output_layer(x)
    

if __name__ == '__main__':
    in_channels = 1
    in_shape = (16, 32, 256)
    depth = 3
    conv_block_size = 2
    conv_block_batch_norm = True
    unet = UNet3D(in_channels, in_shape, depth, conv_block_size, conv_block_batch_norm)
    x = torch.randn(2, 1, *in_shape)
    y = unet(x)
    print(y.shape)
