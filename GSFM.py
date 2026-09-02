import torch
import torch.nn as nn
import torch.nn.functional as F

from models.common import Conv

__all__ = ['GSFM', 'Ghost_SDI', 'SDI']


class GhostConv(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, g=1, act=True):
        super().__init__()
        c_ = c2 // 2
        self.cv1 = Conv(c1, c_, k, s, None, g, act=act)
        self.cv2 = Conv(c_, c_, 3, 1, None, c_, act=act)

    def forward(self, x):
        y = self.cv1(x)
        return torch.cat((y, self.cv2(y)), 1)


class GSFM(nn.Module):
    def __init__(self, channel):
        super().__init__()
        self.convs = nn.ModuleList([GhostConv(i, channel[-1], k=3, s=1) for i in channel])

    def forward(self, xs):
        ans = torch.ones_like(xs[-1])
        target_size = xs[-1].shape[2:]

        for i, x in enumerate(xs):
            if x.shape[-1] > target_size[-1]:
                x = F.adaptive_avg_pool2d(x, target_size)
            elif x.shape[-1] < target_size[-1]:
                x = F.interpolate(x, size=target_size, mode='bilinear', align_corners=True)
            ans = ans * self.convs[i](x)

        return ans


Ghost_SDI = GSFM


class SDI(nn.Module):
    def __init__(self, channel):
        super().__init__()
        self.convs = nn.ModuleList([nn.Conv2d(i, channel[-1], kernel_size=3, stride=1, padding=1) for i in channel])

    def forward(self, xs):
        ans = torch.ones_like(xs[-1])
        target_size = xs[-1].shape[2:]

        for i, x in enumerate(xs):
            if x.shape[-1] > target_size[-1]:
                x = F.adaptive_avg_pool2d(x, target_size)
            elif x.shape[-1] < target_size[-1]:
                x = F.interpolate(x, size=target_size, mode='bilinear', align_corners=True)
            ans = ans * self.convs[i](x)

        return ans
