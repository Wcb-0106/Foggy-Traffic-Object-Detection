import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from models.common import Conv

try:
    from timm.models.layers import DropPath
except Exception:
    class DropPath(nn.Identity):
        pass

__all__ = ['iRMB', 'IRMB']


class SE(nn.Module):
    def __init__(self, channel=512, reduction=16):
        super().__init__()
        hidden = max(channel // reduction, 1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, hidden, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channel, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class iRMB(nn.Module):
    def __init__(
        self,
        dim_in,
        dim_out,
        dim_head=32,
        norm_in=True,
        has_skip=True,
        exp_ratio=1.0,
        act_layer='relu',
        v_proj=True,
        dw_ks=3,
        stride=1,
        dilation=1,
        se_ratio=0.0,
        window_size=7,
        attn_s=True,
        attn_drop=0.,
        drop=0.,
        drop_path=0.,
        v_group=False,
        attn_pre=False,
    ):
        super().__init__()
        self.norm = nn.BatchNorm2d(dim_in) if norm_in else nn.Identity()
        dim_mid = int(dim_in * exp_ratio)
        self.has_skip = (dim_in == dim_out and stride == 1) and has_skip
        self.attn_s = attn_s

        if self.attn_s:
            assert dim_in % dim_head == 0, 'dim should be divisible by num_heads'
            self.dim_head = dim_head
            self.window_size = window_size
            self.num_head = dim_in // dim_head
            self.scale = self.dim_head ** -0.5
            self.attn_pre = attn_pre
            self.qk = Conv(dim_in, int(dim_in * 2), k=1, act=False)
            self.v = Conv(dim_in, dim_mid, k=1, g=self.num_head if v_group else 1, act=False)
            self.attn_drop = nn.Dropout(attn_drop)
        else:
            self.v = Conv(dim_in, dim_mid, k=1, act=act_layer) if v_proj else nn.Identity()

        self.conv_local = Conv(dim_mid, dim_mid, k=dw_ks, s=stride, d=dilation, g=dim_mid)
        self.se = SE(dim_mid, reduction=se_ratio) if se_ratio > 0.0 else nn.Identity()
        self.proj_drop = nn.Dropout(drop)
        self.proj = Conv(dim_mid, dim_out, k=1, act=False)
        self.drop_path = DropPath(drop_path) if drop_path else nn.Identity()

        self.attn_map = None
        self.attn_hw = None

    def forward(self, x):
        shortcut = x
        x = self.norm(x)
        B, C, H, W = x.shape
        self.attn_map = None
        self.attn_hw = None

        if self.attn_s:
            if self.window_size <= 0:
                window_size_W = W
                window_size_H = H
            else:
                window_size_W = self.window_size
                window_size_H = self.window_size

            pad_r = (window_size_W - W % window_size_W) % window_size_W
            pad_b = (window_size_H - H % window_size_H) % window_size_H
            x = F.pad(x, (0, pad_r, 0, pad_b, 0, 0))

            n1 = (H + pad_b) // window_size_H
            n2 = (W + pad_r) // window_size_W
            x = rearrange(
                x,
                'b c (h1 n1) (w1 n2) -> (b n1 n2) c h1 w1',
                n1=n1,
                n2=n2,
            ).contiguous()
            b, c, h, w = x.shape

            qk = self.qk(x)
            qk = rearrange(
                qk,
                'b (qk heads dim_head) h w -> qk b heads (h w) dim_head',
                qk=2,
                heads=self.num_head,
                dim_head=self.dim_head,
            ).contiguous()
            q = qk[0]
            k = qk[1]
            attn_spa = (q @ k.transpose(-2, -1)) * self.scale
            attn_spa = attn_spa.softmax(dim=-1)

            if not self.training:
                with torch.no_grad():
                    attn_vis = attn_spa.mean(dim=1).mean(dim=1)
                    attn_vis = rearrange(
                        attn_vis,
                        '(b n1 n2) (h1 w1) -> b (h1 n1) (w1 n2)',
                        b=B,
                        n1=n1,
                        n2=n2,
                        h1=h,
                        w1=w,
                    ).contiguous()
                    self.attn_map = attn_vis[:, :H, :W].contiguous().detach().float().cpu()
                    self.attn_hw = (H, W)

            attn_spa = self.attn_drop(attn_spa)

            if self.attn_pre:
                x = rearrange(
                    x,
                    'b (heads dim_head) h w -> b heads (h w) dim_head',
                    heads=self.num_head,
                ).contiguous()
                x_spa = attn_spa @ x
                x_spa = rearrange(
                    x_spa,
                    'b heads (h w) dim_head -> b (heads dim_head) h w',
                    heads=self.num_head,
                    h=h,
                    w=w,
                ).contiguous()
                x_spa = self.v(x_spa)
            else:
                v = self.v(x)
                v = rearrange(
                    v,
                    'b (heads dim_head) h w -> b heads (h w) dim_head',
                    heads=self.num_head,
                ).contiguous()
                x_spa = attn_spa @ v
                x_spa = rearrange(
                    x_spa,
                    'b heads (h w) dim_head -> b (heads dim_head) h w',
                    heads=self.num_head,
                    h=h,
                    w=w,
                ).contiguous()

            x = rearrange(
                x_spa,
                '(b n1 n2) c h1 w1 -> b c (h1 n1) (w1 n2)',
                n1=n1,
                n2=n2,
            ).contiguous()

            if pad_r > 0 or pad_b > 0:
                x = x[:, :, :H, :W].contiguous()
        else:
            x = self.v(x)

        if self.has_skip:
            x = x + self.se(self.conv_local(x))
        else:
            x = self.se(self.conv_local(x))

        x = self.proj_drop(x)
        x = self.proj(x)

        if self.has_skip:
            x = shortcut + self.drop_path(x)

        return x


IRMB = iRMB
