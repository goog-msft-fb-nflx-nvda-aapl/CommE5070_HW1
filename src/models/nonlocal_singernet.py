"""Faithful port of our own prior-year HW1 submission's non-local ResNet
(`NonLocalSingerNet`, task2_nonlocal.py — a classic Wang et al.-style
non-local block, not the same architecture as this project's `fgnl`, whose
`CRNN_FGNL` (src/models/nonlocal_fgnl.py) ports the "Fully Generalized
Non-Local Network for singer identification" (Kuo et al., AAAI 2021)
instead). From
github.com/goog-msft-fb-nflx-nvda-aapl/NTU (CommE5070 hw1_submission).

See src/models/singer_senet.py's docstring for the round-5 deep-research
context (Gemini vs. Qwen disagreement on whether this channel-ramped
32->512 style backbone is worth porting for this dataset scale) — ported
faithfully here to settle it by measurement rather than by either engine's
unsourced claim.

Same mel frontend/training-recipe constants as the original script:
n_mels=128, n_fft=2048, hop_length=256, f_min=20/f_max=8000, top_db=80,
per-sample mel normalization, 10s training crops.

Usage: python -m src.train --model nonlocal_singernet \
    --data_index_dir data/index --out_dir results/nonlocal_singernet \
    --optimizer adamw --label_smoothing 0.1 --lr 3e-4 --weight_decay 1e-4
"""
import torch
import torch.nn as nn
import torchaudio


class NonLocalBlock(nn.Module):
    """Classic Wang et al.-style non-local block (theta/phi/g/out 1x1 convs,
    softmax attention over all spatial positions) plus an SE-style channel
    gate on the block's residual contribution, exactly as in the original
    script — not to be confused with `src/models/nonlocal_fgnl.py`'s
    FGNL block, a different, later (AAAI 2021) formulation."""

    def __init__(self, in_channels, reduction=2):
        super().__init__()
        inter_ch = max(in_channels // reduction, 32)

        self.theta = nn.Conv2d(in_channels, inter_ch, 1, bias=False)
        self.phi = nn.Conv2d(in_channels, inter_ch, 1, bias=False)
        self.g = nn.Conv2d(in_channels, inter_ch, 1, bias=False)
        self.out = nn.Conv2d(inter_ch, in_channels, 1, bias=False)
        self.bn = nn.BatchNorm2d(in_channels)

        self.ch_attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(in_channels, in_channels // 4),
            nn.ReLU(),
            nn.Linear(in_channels // 4, in_channels),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, h, w = x.shape
        n = h * w

        theta = self.theta(x).view(b, -1, n).permute(0, 2, 1)  # (B, N, C')
        phi = self.phi(x).view(b, -1, n)  # (B, C', N)
        g = self.g(x).view(b, -1, n).permute(0, 2, 1)  # (B, N, C')

        attn = torch.softmax(torch.bmm(theta, phi) / (theta.shape[-1] ** 0.5), dim=-1)
        out = torch.bmm(attn, g).permute(0, 2, 1).view(b, -1, h, w)
        out = self.bn(self.out(out))

        ch_w = self.ch_attn(x).view(b, c, 1, 1)
        return x + out * ch_w


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1, use_nonlocal=False):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ELU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ELU(),
        )
        self.shortcut = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
            nn.BatchNorm2d(out_ch),
        ) if in_ch != out_ch or stride != 1 else nn.Identity()
        self.nl = NonLocalBlock(out_ch) if use_nonlocal else nn.Identity()
        self.drop = nn.Dropout2d(0.1)

    def forward(self, x):
        out = self.conv(x)
        out = out + self.shortcut(x)
        out = self.nl(out)
        return self.drop(out)


class NonLocalSingerNet(nn.Module):
    def __init__(self, sample_rate=16000, n_fft=2048, hop_length=256, n_mels=128,
                 f_min=20.0, f_max=8000.0, n_class=20):
        super().__init__()
        self.spec = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate, n_fft=n_fft, hop_length=hop_length,
            n_mels=n_mels, f_min=f_min, f_max=f_max,
        )
        self.to_db = torchaudio.transforms.AmplitudeToDB(top_db=80)

        self.stem = nn.Sequential(
            nn.Conv2d(1, 32, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(32), nn.ELU(),
            nn.MaxPool2d(3, stride=2, padding=1),
        )
        self.layer1 = ConvBlock(32, 64, stride=1, use_nonlocal=False)
        self.layer2 = ConvBlock(64, 128, stride=2, use_nonlocal=True)
        self.layer3 = ConvBlock(128, 256, stride=2, use_nonlocal=True)
        self.layer4 = ConvBlock(256, 512, stride=2, use_nonlocal=True)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(512, 256), nn.ELU(),
            nn.Dropout(0.3),
            nn.Linear(256, n_class),
        )

    def _log_mel(self, wav):
        mel = self.to_db(self.spec(wav))
        mean = mel.mean(dim=(1, 2), keepdim=True)
        std = mel.std(dim=(1, 2), keepdim=True)
        return (mel - mean) / (std + 1e-6)

    def embed(self, x):
        x = self._log_mel(x).unsqueeze(1)
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return self.pool(x).flatten(1)

    def forward(self, x):
        return self.classifier(self.embed(x))
