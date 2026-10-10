import torch
import torch.nn as nn
import math


# -----------------------------
# Ghost Module
# -----------------------------
class GhostModule(nn.Module):
    def __init__(self, inp, oup, kernel_size=1, ratio=2, dw_size=3, stride=1, relu=True):
        super(GhostModule, self).__init__()
        self.oup = oup

        init_channels = math.ceil(oup / ratio)
        new_channels = init_channels * (ratio - 1)

        # 主卷积（真实特征）
        self.primary_conv = nn.Sequential(
            nn.Conv2d(inp, init_channels, kernel_size, stride, kernel_size // 2, bias=False),
            nn.BatchNorm2d(init_channels),
            nn.ReLU(inplace=True) if relu else nn.Identity(),
        )

        # 生成 ghost 特征
        self.cheap_operation = nn.Sequential(
            nn.Conv2d(init_channels, new_channels, dw_size, 1, dw_size // 2,
                      groups=init_channels, bias=False),
            nn.BatchNorm2d(new_channels),
            nn.ReLU(inplace=True) if relu else nn.Identity(),
        )

    def forward(self, x):
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)
        out = torch.cat([x1, x2], dim=1)
        return out[:, :self.oup, :, :]


# -----------------------------
# Ghost Bottleneck
# -----------------------------
class GhostBottleneck(nn.Module):
    def __init__(self, inp, hidden_dim, oup, stride):
        super(GhostBottleneck, self).__init__()

        self.stride = stride

        self.ghost1 = GhostModule(inp, hidden_dim, relu=True)

        if stride > 1:
            self.dwconv = nn.Conv2d(hidden_dim, hidden_dim, 3, stride=stride,
                                    padding=1, groups=hidden_dim, bias=False)
            self.bn_dw = nn.BatchNorm2d(hidden_dim)

        self.ghost2 = GhostModule(hidden_dim, oup, relu=False)

        if stride == 1 and inp == oup:
            self.shortcut = nn.Identity()
        else:
            self.shortcut = nn.Sequential(
                nn.Conv2d(inp, inp, 3, stride=stride, padding=1,
                          groups=inp, bias=False),
                nn.BatchNorm2d(inp),
                nn.Conv2d(inp, oup, 1, bias=False),
                nn.BatchNorm2d(oup),
            )

    def forward(self, x):
        out = self.ghost1(x)

        if self.stride > 1:
            out = self.bn_dw(self.dwconv(out))

        out = self.ghost2(out)

        return out + self.shortcut(x)


# -----------------------------
# GhostNet 主体
# -----------------------------
class GhostNet(nn.Module):
    def __init__(self, num_classes=2):
        super(GhostNet, self).__init__()

        # Stem
        self.conv_stem = nn.Sequential(
            nn.Conv2d(3, 16, 3, 2, 1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True)
        )

        # 主干网络
        self.blocks = nn.Sequential(
            GhostBottleneck(16, 16, 16, 1),

            GhostBottleneck(16, 48, 24, 2),
            GhostBottleneck(24, 72, 24, 1),

            GhostBottleneck(24, 72, 40, 2),
            GhostBottleneck(40, 120, 40, 1),

            GhostBottleneck(40, 240, 80, 2),
            GhostBottleneck(80, 200, 80, 1),
        )

        # Head
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(80, num_classes)

    def forward(self, x):
        x = self.conv_stem(x)
        x = self.blocks(x)
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x