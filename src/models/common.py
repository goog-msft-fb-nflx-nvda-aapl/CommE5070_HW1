"""Shared building blocks, ported verbatim from
minzwon/sota-music-tagging-models (training/modules.py), MIT-licensed.
Cited in README.md / MATERIALS.md.
"""
import torch.nn as nn


class Conv_1d(nn.Module):
    def __init__(self, input_channels, output_channels, shape=3, stride=1, pooling=2):
        super().__init__()
        self.conv = nn.Conv1d(input_channels, output_channels, shape, stride=stride, padding=shape // 2)
        self.bn = nn.BatchNorm1d(output_channels)
        self.relu = nn.ReLU()
        self.mp = nn.MaxPool1d(pooling)

    def forward(self, x):
        return self.mp(self.relu(self.bn(self.conv(x))))


class Conv_2d(nn.Module):
    def __init__(self, input_channels, output_channels, shape=3, stride=1, pooling=2):
        super().__init__()
        self.conv = nn.Conv2d(input_channels, output_channels, shape, stride=stride, padding=shape // 2)
        self.bn = nn.BatchNorm2d(output_channels)
        self.relu = nn.ReLU()
        self.mp = nn.MaxPool2d(pooling)

    def forward(self, x):
        return self.mp(self.relu(self.bn(self.conv(x))))


class SqueezeExcite2d(nn.Module):
    """Squeeze-and-Excitation (Hu et al., CVPR 2018) — channel-wise gating.
    Per the user's prior run's SE-ResNet model (task2_se_cnn.py) and
    lecture-adjacent literature; not in minzwon's repo, added directly."""

    def __init__(self, channels, reduction=16):
        super().__init__()
        hidden = max(1, channels // reduction)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(channels, hidden)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden, channels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        n, c, _, _ = x.shape
        w = self.pool(x).view(n, c)
        w = self.sigmoid(self.fc2(self.relu(self.fc1(w)))).view(n, c, 1, 1)
        return x * w


class Res_2d_SE(nn.Module):
    """Res_2d with a Squeeze-Excitation gate after the residual add."""

    def __init__(self, input_channels, output_channels, shape=3, stride=2, se_reduction=16):
        super().__init__()
        self.conv_1 = nn.Conv2d(input_channels, output_channels, shape, stride=stride, padding=shape // 2)
        self.bn_1 = nn.BatchNorm2d(output_channels)
        self.conv_2 = nn.Conv2d(output_channels, output_channels, shape, padding=shape // 2)
        self.bn_2 = nn.BatchNorm2d(output_channels)
        self.se = SqueezeExcite2d(output_channels, reduction=se_reduction)

        self.diff = False
        if (stride != 1) or (input_channels != output_channels):
            self.conv_3 = nn.Conv2d(input_channels, output_channels, shape, stride=stride, padding=shape // 2)
            self.bn_3 = nn.BatchNorm2d(output_channels)
            self.diff = True
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.bn_2(self.conv_2(self.relu(self.bn_1(self.conv_1(x)))))
        out = self.se(out)
        if self.diff:
            x = self.bn_3(self.conv_3(x))
        out = x + out
        return self.relu(out)


class Res_2d(nn.Module):
    def __init__(self, input_channels, output_channels, shape=3, stride=2):
        super().__init__()
        self.conv_1 = nn.Conv2d(input_channels, output_channels, shape, stride=stride, padding=shape // 2)
        self.bn_1 = nn.BatchNorm2d(output_channels)
        self.conv_2 = nn.Conv2d(output_channels, output_channels, shape, padding=shape // 2)
        self.bn_2 = nn.BatchNorm2d(output_channels)

        self.diff = False
        if (stride != 1) or (input_channels != output_channels):
            self.conv_3 = nn.Conv2d(input_channels, output_channels, shape, stride=stride, padding=shape // 2)
            self.bn_3 = nn.BatchNorm2d(output_channels)
            self.diff = True
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.bn_2(self.conv_2(self.relu(self.bn_1(self.conv_1(x)))))
        if self.diff:
            x = self.bn_3(self.conv_3(x))
        out = x + out
        return self.relu(out)
