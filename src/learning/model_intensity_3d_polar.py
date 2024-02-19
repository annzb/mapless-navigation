import torch
import torch.nn as nn
# Heatmap shape (2, 16, 64, 2)
# GT grid shape (72, 32, 16)


class InputReshape(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.down = nn.Conv3d(in_c, out_c, kernel_size=(1, 5, 5))
        self.bn1 = nn.BatchNorm3d(out_c)
        self.up = nn.ConvTranspose3d(out_c, out_c, kernel_size=(3, 2, 2), stride=(3, 2, 2))
        self.bn2 = nn.BatchNorm3d(out_c)
        self.pool = nn.MaxPool3d((2, 3, 3))
        self.relu = nn.ReLU()

    def forward(self, inputs):
        x = self.down(inputs)
        # print(f'Shape after down {x.shape}')
        x = self.bn1(x)
        x = self.relu(x)
        x = self.up(x)
        # print(f'Shape after up {x.shape}')
        x = self.bn2(x)
        x = self.relu(x)
        x = self.pool(x)
        # print(f'Shape after pool {x.shape}')
        # print('----')
        return x


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
        # print(f'Shape after conv1 {x.shape}')
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        # print(f'Shape after conv2 {x.shape}')
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


class Unet1C3D(nn.Module):
    def __init__(self, dropout_rate=0.1):
        super().__init__()
        # self.increase_depth = nn.Conv3d(
        #     1, 1,
        #     kernel_size=(2, 1, 1), dilation=(2, 1, 1),
        #     padding=(5, 0, 0), padding_mode='circular'
        # )
        self.dropout = nn.Dropout3d(dropout_rate)
        self.reshape = InputReshape(1, 1)
        self.e1 = EncoderBlock(1, 32)
        self.e2 = EncoderBlock(32, 64)
        self.e3 = EncoderBlock(64, 128)
        # self.e4 = EncoderBlock(128, 256)
        self.b = ConvBlock(128, 256)
        # self.d1 = DecoderBlock(512, 256)
        self.d2 = DecoderBlock(256, 128)
        self.d3 = DecoderBlock(128, 64)
        self.d4 = DecoderBlock(64, 32)
        self.output = nn.Conv3d(32,  1, kernel_size=1)

    def forward(self, inputs):
        inputs = self.reshape(inputs)
        # print('shape after reshape', inputs.shape)
        s1, p1 = self.e1(inputs)
        # print('shape after e1', p1.shape)
        s2, p2 = self.e2(p1)
        # print('shape after e2', p2.shape)
        s_final, p_final = self.e3(p2)
        # print('shape after e3', p_final.shape)
        # s4, p4 = self.e4(p3)
        b = self.b(p_final)
        b = self.dropout(b)
        # print('shape after bottleneck', b.shape)
        # d1 = self.d1(b, s4)
        d2 = self.d2(b, s_final)
        # print('shape after d1', b.shape)
        d3 = self.d3(d2, s2)
        # print('shape after d2', d3.shape)
        d4 = self.d4(d3, s1)
        # print('shape after d3', d4.shape)
        outputs = self.output(d4)
        # print('shape after output', outputs.shape)
        if outputs.size(1) == 1:  # convert [N, 1, D, H, W] to [N, D, H, W]
            outputs = outputs.squeeze(1)
            # print('shape after squeeze', outputs.shape)
        # print('---------')
        return outputs
