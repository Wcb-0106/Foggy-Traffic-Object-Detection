import torch.nn as nn

from models.common import AConv, Concat, RepNCSPELAN4

__all__ = ['FUSEM']


class FUSEM(nn.Module):
    def __init__(self, channels, c2, c3, c4, c5=1, bridge=False):
        super().__init__()
        if len(channels) != 2:
            raise ValueError('FUSEM expects two input feature maps')

        self.bridge = bridge
        self.bridge_layer = nn.Sequential(
            AConv(channels[0], c2),
            nn.Upsample(scale_factor=2, mode='nearest'),
        ) if bridge else nn.Identity()

        fused_channels = c2 + channels[1] if bridge else sum(channels)
        self.concat = Concat(1)
        self.block = RepNCSPELAN4(fused_channels, c2, c3, c4, c5)

    def forward(self, xs):
        source, target = xs
        source = self.bridge_layer(source)
        return self.block(self.concat([source, target]))
