import torch
import torch.nn as nn

from models.common import Conv

__all__ = [
    'D3Inception',
    'Stem',
    'InceptionA',
    'DepthwiseSeparableConv',
    'InceptionB',
    'InceptionC',
    'RedutionA',
    'RedutionB',
    'Avarage_Pooling',
]


class Stem(nn.Module):
    def __init__(self, c1, scale=1):
        super().__init__()
        ch1 = [32, 32, 64, 96] * scale
        ch2_1 = [64, 96] * scale
        ch2_2 = [64, 64, 64, 96] * scale
        ch3 = [64] * scale

        self.dehaze = Conv(c1, c1, 3, 1, act=nn.ReLU())
        self.conv1 = nn.Sequential(Conv(c1, ch1[0], 3, s=2), Conv(ch1[0], ch1[1], 3), Conv(ch1[1], ch1[2], 3, s=1))
        self.conv1_1 = nn.MaxPool2d(3, 2, 1)
        self.conv1_2 = Conv(ch1[2], ch1[3], 3, s=2)
        self.conv2_1 = nn.Sequential(Conv(ch1[3] + ch1[2], ch2_1[0], 1), Conv(ch2_1[0], ch2_1[1], 3))
        self.conv2_2 = nn.Sequential(Conv(ch1[3] + ch1[2], ch2_2[0], 1), Conv(ch2_2[0], ch2_2[1], (7, 1)),
                                     Conv(ch2_2[1], ch2_2[2], (1, 7)), Conv(ch2_2[2], ch2_2[3], 3))
        self.conv3_1 = Conv(ch2_2[3] + ch2_1[1], ch3[0], 3, 1)
        self.shortcut = nn.Sequential(
            Conv(c1, ch3[0], 1, s=4, act=False),
            nn.BatchNorm2d(ch3[0]),
        )

    def forward(self, x):
        x_dehazed = self.dehaze(x)
        x_conv1 = self.conv1(x_dehazed)
        x1 = torch.cat([self.conv1_1(x_conv1), self.conv1_2(x_conv1)], dim=1)
        x2 = torch.cat([self.conv2_1(x1), self.conv2_2(x1)], dim=1)
        x3 = self.conv3_1(x2)
        return x3 + self.shortcut(x_dehazed)


class InceptionA(nn.Module):
    def __init__(self, c1, c2, s=1, act=True):
        super().__init__()
        c_ = c2 // 4
        c_out = c2 - 3 * c_
        self.conv1 = Conv(c1, c_out, 1, s=s, act=act)
        self.conv2 = nn.Sequential(Conv(c1, c_, 1, s=1, act=act), Conv(c_, c_, 3, s=s, p=4, d=4, act=act))
        self.conv3 = nn.Sequential(Conv(c1, c_, 1, s=1, act=act), Conv(c_, c_, 3, s=s, p=6, d=6, act=act),
                                   Conv(c_, c_, 3, s=1, act=act))
        self.pool = nn.Sequential(nn.AvgPool2d(kernel_size=3, stride=s, padding=1), Conv(c1, c_, 1, act=act))

    def forward(self, x):
        return torch.cat([self.conv1(x), self.conv2(x), self.conv3(x), self.pool(x)], dim=1)


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, act=True):
        super().__init__()
        padding = kernel_size // 2 if isinstance(kernel_size, int) else (kernel_size[0] // 2, kernel_size[1] // 2)
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, groups=in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, 1, 0)
        self.act = nn.ReLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        return self.act(self.pointwise(self.depthwise(x)))


class InceptionB(nn.Module):
    def __init__(self, c1, c2, s=1, act=True):
        super().__init__()
        c_ = c2 // 4
        c_out = c2 - 3 * c_
        self.conv1 = Conv(c1, c_out, 1, s=s, act=act)
        self.conv2 = nn.Sequential(Conv(c1, c_, 1, s=s, act=act),
                                   DepthwiseSeparableConv(c_, c_, 5, stride=1, act=act),
                                   DepthwiseSeparableConv(c_, c_, 5, stride=s, act=act))
        self.conv3 = nn.Sequential(Conv(c1, c_, 1, s=s, act=act),
                                   DepthwiseSeparableConv(c_, c_, 5, stride=1, act=act),
                                   DepthwiseSeparableConv(c_, c_, 5, stride=s, act=act),
                                   DepthwiseSeparableConv(c_, c_, 5, stride=1, act=act),
                                   DepthwiseSeparableConv(c_, c_, 5, stride=s, act=act))
        self.pool = nn.Sequential(nn.MaxPool2d(kernel_size=3, stride=s, padding=1), Conv(c1, c_, 1, s=1, act=act))

    def forward(self, x):
        return torch.cat([self.conv1(x), self.conv2(x), self.conv3(x), self.pool(x)], dim=1)


class InceptionC(nn.Module):
    def __init__(self, c1, c2, s=1, act=True):
        super().__init__()
        c_ = c2 // 6
        c_out = c2 - 5 * c_
        self.conv1 = Conv(c1, c_out, 1, s=s, act=act)
        self.conv2 = Conv(c1, c_, 1, s=s, act=act)
        self.conv2_1 = Conv(c_, c_, (3, 1), s=1, act=act)
        self.conv2_2 = Conv(c_, c_, (1, 3), s=1, act=act)
        self.conv3 = nn.Sequential(Conv(c1, c_, 1, s=s, act=act), Conv(c_, c_, (1, 3), s=1, act=act),
                                   Conv(c_, c_, (3, 1), s=1, act=act))
        self.conv3_1 = Conv(c_, c_, (3, 1), s=1, act=act)
        self.conv3_2 = Conv(c_, c_, (1, 3), s=1, act=act)
        self.pool = nn.Sequential(nn.MaxPool2d(kernel_size=3, stride=s, padding=1), Conv(c1, c_, 1, s=1, act=act))

    def forward(self, x):
        x2 = self.conv2(x)
        x3 = self.conv3(x)
        return torch.cat([self.conv1(x), self.conv2_1(x2), self.conv2_2(x2), self.conv3_1(x3), self.conv3_2(x3), self.pool(x)], dim=1)


class RedutionA(nn.Module):
    def __init__(self, c1, c2, s=2, act=True):
        super().__init__()
        c_ = c2 // 3
        c_out = c2 - 2 * c_
        self.conv1 = Conv(c1, c_out, 3, s=s, act=act)
        self.conv2 = nn.Sequential(Conv(c1, c_, 1, s=1, act=act), Conv(c_, c_, 3, s=1, act=act),
                                   Conv(c_, c_, 3, s=s, act=act))
        self.pool = nn.Sequential(Conv(c1, c_, 1, s=1, act=act), nn.MaxPool2d(3, s, 1))

    def forward(self, x):
        return torch.cat([self.conv1(x), self.conv2(x), self.pool(x)], dim=1)


class RedutionB(nn.Module):
    def __init__(self, c1, c2, s=2, act=True):
        super().__init__()
        c_ = c2 // 3
        c_out = c2 - 2 * c_
        self.conv1 = nn.Sequential(Conv(c1, c_out, 1, s=1, act=act), Conv(c_out, c_out, 3, s=s, act=act))
        self.conv2 = nn.Sequential(Conv(c1, c_, 1, s=1, act=act), Conv(c_, c_, (1, 7), s=1, act=act),
                                   Conv(c_, c_, (7, 1), s=1, act=act), Conv(c_, c_, k=3, s=s, act=act))
        self.pool = nn.Sequential(Conv(c1, c_, 1, s=1, act=act), nn.MaxPool2d(3, s, 1))

    def forward(self, x):
        return torch.cat([self.conv1(x), self.conv2(x), self.pool(x)], dim=1)


class Avarage_Pooling(nn.Module):
    def __init__(self):
        super().__init__()
        self.pool = nn.AvgPool2d(3, 2, 1)

    def forward(self, x):
        return self.pool(x)


class D3Inception(nn.Module):
    def __init__(self, c1=3, scale=1):
        super().__init__()
        stem_channels = 64 * scale
        c2 = 128 * scale
        c3 = 192 * scale

        self.stem = Stem(c1, scale)
        self.inception_a = InceptionA(stem_channels, stem_channels)
        self.redution_a = RedutionA(stem_channels, c2)
        self.inception_b = InceptionB(c2, c2)
        self.redution_b = RedutionB(c2, c3)
        self.inception_c = InceptionC(c3, c3)
        self.avpool = Avarage_Pooling()

    def forward(self, x):
        x0 = self.stem(x)
        x1 = self.inception_a(x0)
        x2 = self.redution_a(x1)
        x3 = self.inception_b(x2)
        x4 = self.redution_b(x3)
        x5 = self.inception_c(x4)
        x6 = self.avpool(x5)
        return x2, x3, x4, x5, x6
