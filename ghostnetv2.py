import torch
import torch.nn as nn
import math


# =========================
# 基础模块
# =========================
def _make_divisible(v, divisor=4):
    return int(math.ceil(v / divisor) * divisor)


# =========================
# Ghost Module (V1)
# =========================
class GhostModule(nn.Module):
    def __init__(self, inp, oup, kernel_size=1, ratio=2, dw_size=3, stride=1, relu=True):
        super().__init__()
        init_channels = math.ceil(oup / ratio)
        new_channels = init_channels * (ratio - 1)

        self.primary_conv = nn.Sequential(
            nn.Conv2d(inp, init_channels, kernel_size, stride, kernel_size // 2, bias=False),
            nn.BatchNorm2d(init_channels),
            nn.ReLU(inplace=True) if relu else nn.Identity(),
        )

        self.cheap_operation = nn.Sequential(
            nn.Conv2d(init_channels, new_channels, dw_size, 1, dw_size // 2, groups=init_channels, bias=False),
            nn.BatchNorm2d(new_channels),
            nn.ReLU(inplace=True) if relu else nn.Identity(),
        )

        self.out_channels = oup

    def forward(self, x):
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)
        out = torch.cat([x1, x2], dim=1)
        return out[:, :self.out_channels, :, :]


# =========================
# DFC Attention（GhostNetV2核心）
# =========================
class DFCAttention(nn.Module):
    def __init__(self, channels):
        super().__init__()

        self.pool = nn.AvgPool2d(2)
        self.conv_h = nn.Conv2d(channels, channels, (1, 3), padding=(0, 1), groups=channels)
        self.conv_w = nn.Conv2d(channels, channels, (3, 1), padding=(1, 0), groups=channels)

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        identity = x

        x = self.pool(x)
        x = self.conv_h(x)
        x = self.conv_w(x)
        x = self.upsample(x)

        return identity * self.sigmoid(x)


# =========================
# Ghost Bottleneck V2
# =========================
class GhostBottleneckV2(nn.Module):
    def __init__(self, inp, hidden_dim, oup, stride):
        super().__init__()

        self.stride = stride

        # expansion
        self.ghost1 = GhostModule(inp, hidden_dim, relu=True)

        # DFC Attention（只在stride=1时用）
        self.attn = DFCAttention(hidden_dim) if stride == 1 else None

        # depthwise
        if stride > 1:
            self.dwconv = nn.Sequential(
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
            )
        else:
            self.dwconv = nn.Identity()

        # projection
        self.ghost2 = GhostModule(hidden_dim, oup, relu=False)

        # shortcut
        if inp == oup and stride == 1:
            self.shortcut = nn.Identity()
        else:
            self.shortcut = nn.Sequential(
                nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
                nn.BatchNorm2d(inp),
                nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )

    def forward(self, x):
        residual = x

        x = self.ghost1(x)

        if self.attn is not None:
            x = self.attn(x)

        x = self.dwconv(x)
        x = self.ghost2(x)

        return x + self.shortcut(residual)


# =========================
# GhostNetV2 主模型
# =========================
class GhostNetV2(nn.Module):
    def __init__(self, num_classes=1000, width=1.0):
        super().__init__()

        # 初始卷积
        output_channel = _make_divisible(16 * width)
        self.conv_stem = nn.Sequential(
            nn.Conv2d(3, output_channel, 3, 2, 1, bias=False),
            nn.BatchNorm2d(output_channel),
            nn.ReLU(inplace=True),
        )

        input_channel = output_channel

        # 网络配置（简化版）
        cfgs = [
            # k, t, c, s
            [3, 16, 16, 1],
            [3, 48, 24, 2],
            [3, 72, 24, 1],
            [5, 72, 40, 2],
            [5, 120, 40, 1],
            [3, 240, 80, 2],
            [3, 200, 80, 1],
            [3, 184, 80, 1],
            [3, 480, 112, 1],
            [5, 672, 160, 2],
        ]

        layers = []
        for k, exp_size, c, s in cfgs:
            output_channel = _make_divisible(c * width)
            hidden_channel = _make_divisible(exp_size * width)

            layers.append(
                GhostBottleneckV2(input_channel, hidden_channel, output_channel, s)
            )
            input_channel = output_channel

        self.blocks = nn.Sequential(*layers)

        # 最后层
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(input_channel, num_classes)

    def forward(self, x):
        x = self.conv_stem(x)
        x = self.blocks(x)
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x