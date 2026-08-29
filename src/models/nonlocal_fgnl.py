"""CRNN + Fully Generalized Non-Local (FGNL) block, ported from Keras/TF to
PyTorch from ian-k-1217/Fully-Generalized-Non-Local-Network
(CRNN_FGNL/src/models.py). Method 2-3 in the assignment doc:
  positions/channels/layers generalized non-local network for singer ID,
  AAAI 2021.

Input: precomputed log-mel spectrogram (batch, n_mels=128, T), same
convention as confound_crnn.py / crnn_zain.py.

The FGNL block computes non-local self-attention over positions pooled from
*two* feature scales at once (cb4, the coarsest conv output, and cb3, one
scale finer) — the "generalized" part — then further diversifies the
attention via `channel_denominator` (=32) reduced-channel "rolled" affinity
maps mixed by a small gating MLP (the paper's Mixture-of-Softmax-Experts /
MoSE reweighting), before projecting back and adding a residual connection
to the theta-scale (cb4) features. This is a direct, shape-for-shape port
of the original's `theta`/`phi`/`g` branches, `tf.roll`-based diversification,
Gaussian pre-smoothing, and MoSE gate — the only simplifications are
PyTorch idioms in place of the original's raw `cv2`-generated Gaussian
kernel (built here with `torch` directly) and Keras `Conv2D`s reformulated
as their 1x1-conv-on-flattened-positions equivalent (pointwise `Linear`).
`Softmax` output is dropped in favor of raw logits + `nn.CrossEntropyLoss`,
and the original's L1/L2 activity regularizers are omitted (no direct
PyTorch equivalent; can be added back via weight_decay / an explicit
penalty term if desired).
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _gaussian_kernel2d(kernel_size=5, sigma=0.707):
    coords = torch.arange(kernel_size, dtype=torch.float32) - (kernel_size - 1) / 2
    g1d = torch.exp(-(coords**2) / (2 * sigma**2))
    g1d /= g1d.sum()
    g2d = torch.outer(g1d, g1d)
    return g2d / g2d.sum()


class GaussianSmooth(nn.Module):
    """Fixed, non-trainable depthwise Gaussian blur, matching the original's
    `Gaussian_Filter_Block` (cv2-built kernel -> DepthwiseConv2D)."""

    def __init__(self, channels, kernel_size=5, sigma=0.707):
        super().__init__()
        kernel = _gaussian_kernel2d(kernel_size, sigma)
        weight = kernel.expand(channels, 1, kernel_size, kernel_size).clone()
        self.register_buffer("weight", weight)
        self.channels = channels
        self.padding = kernel_size // 2

    def forward(self, x):
        return F.conv2d(x, self.weight, padding=self.padding, groups=self.channels)


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, pooling=True):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
        self.act = nn.ELU()
        self.bn = nn.BatchNorm2d(out_ch)
        self.pool = nn.MaxPool2d((2, 2), stride=(2, 2)) if pooling else nn.Identity()
        self.drop = nn.Dropout2d(0.1)

    def forward(self, x):
        return self.drop(self.pool(self.bn(self.act(self.conv(x)))))


class FGNLPhiGBranch(nn.Module):
    """1x1 channel reduction + Gaussian smoothing + flatten to (N, H*W, C')."""

    def __init__(self, in_ch, channel_denominator=32):
        super().__init__()
        self.reduced = max(1, in_ch // channel_denominator)
        self.reduce = nn.Conv2d(in_ch, self.reduced, kernel_size=1)
        self.smooth = GaussianSmooth(self.reduced)

    def forward(self, x):
        x = self.smooth(self.reduce(x))  # (N, C', H, W)
        n, c, h, w = x.shape
        return x.flatten(2).transpose(1, 2)  # (N, H*W, C')


class FullyGeneralizedNonLocal(nn.Module):
    def __init__(self, theta_channels, multiscale_channels, channel_denominator=32):
        super().__init__()
        self.reduced = max(1, theta_channels // channel_denominator)
        self.theta_reduce = nn.Conv2d(theta_channels, self.reduced, kernel_size=1)
        self.theta_smooth = GaussianSmooth(self.reduced)

        self.phi_branches = nn.ModuleList(
            FGNLPhiGBranch(c, channel_denominator) for c in multiscale_channels
        )
        self.g_branches = nn.ModuleList(
            FGNLPhiGBranch(c, channel_denominator) for c in multiscale_channels
        )

        # MoSE gate: shared pointwise MLP over each roll's globally-pooled scalar
        self.mose = nn.Sequential(nn.Linear(1, 16), nn.ReLU(), nn.Linear(16, 1))
        self.project = nn.Conv2d(self.reduced * self.reduced, theta_channels, kernel_size=1)

    def forward(self, theta_input, multiscale_inputs):
        n, c, h, w = theta_input.shape
        theta = self.theta_smooth(self.theta_reduce(theta_input))
        theta = theta.flatten(2).transpose(1, 2)  # (N, H*W, C')

        phi = torch.cat([b(x) for b, x in zip(self.phi_branches, multiscale_inputs)], dim=1)
        g = torch.cat([b(x) for b, x in zip(self.g_branches, multiscale_inputs)], dim=1)

        rolls = []
        for k in range(self.reduced):
            rolled_phi = torch.roll(phi, shifts=k, dims=2)
            affinity = torch.einsum("bpc,bqc->bpq", theta, rolled_phi)
            affinity = affinity.softmax(dim=1)
            yk = torch.einsum("bpq,bqc->bpc", affinity, g)  # (N, H*W, C')
            rolls.append(yk)
        R = torch.stack(rolls, dim=-1)  # (N, H*W, C', rolls)

        w_scalar = R.mean(dim=(1, 2))  # (N, rolls)
        gate_scores = self.mose(w_scalar.unsqueeze(-1)).squeeze(-1)  # (N, rolls)
        gate = gate_scores.softmax(dim=1).view(n, 1, 1, -1)
        reweighted = R * gate  # (N, H*W, C', rolls)

        y = reweighted.reshape(n, h, w, self.reduced * self.reduced).permute(0, 3, 1, 2)
        y = self.project(y)  # (N, C, H, W)
        return theta_input + y


class CRNN_FGNL(nn.Module):
    def __init__(self, n_class=20, n_mels=128, channel_denominator=32):
        super().__init__()
        self.bn0 = nn.BatchNorm1d(n_mels)

        self.cb1 = ConvBlock(1, 64, pooling=False)
        self.cb2 = ConvBlock(64, 128, pooling=True)
        self.cb3 = ConvBlock(128, 128, pooling=True)
        self.cb4 = ConvBlock(128, 128, pooling=True)

        self.fgnl = FullyGeneralizedNonLocal(
            theta_channels=128, multiscale_channels=[128, 128], channel_denominator=channel_denominator
        )

        self.gru1 = None  # lazily built once we know freq*channel at runtime
        self._gru2 = nn.GRU(32, 32, batch_first=True)
        self.dropout = nn.Dropout(0.3)
        self.dense = nn.Linear(32, n_class)
        self._n_class = n_class

    def _ensure_gru1(self, input_size, device):
        if self.gru1 is None:
            self.gru1 = nn.GRU(input_size, 32, batch_first=True).to(device)

    def embed(self, x):
        x = self.bn0(x)
        x = x.unsqueeze(1)

        cb1 = self.cb1(x)
        cb2 = self.cb2(cb1)
        cb3 = self.cb3(cb2)
        cb4 = self.cb4(cb3)

        z = self.fgnl(cb4, [cb4, cb3])  # (N, 128, f4, t4)

        # (N, C, freq, time) -> (N, time, freq, C) -> (N, time, freq*C)
        z = z.permute(0, 3, 2, 1)
        z = z.reshape(z.size(0), z.size(1), -1)

        self._ensure_gru1(z.size(-1), z.device)
        z, _ = self.gru1(z)
        z, _ = self._gru2(z)
        return z[:, -1, :]

    def forward(self, x):
        emb = self.embed(x)
        return self.dense(self.dropout(emb))
