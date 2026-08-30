"""Faithful port of our own prior-year HW1 submission's SE-ResNet
(`SingerSENet`, task2_se_cnn.py), from
github.com/goog-msft-fb-nflx-nvda-aapl/NTU (CommE5070 hw1_submission).
A channel-ramped (32->64->128->256->512, 4 residual stages) ResNet with
Squeeze-and-Excitation gating in every block, contrasted in
deep_research/round5_prior_year_gap_and_latest_literature against this
project's own `se_resnet` (`src/models/se_resnet.py`, minzwon/sota-music-
tagging-models' 7-layer, fixed-max-256-channel SE-ResNet). Gemini's round-5
response argued the channel-ramp-to-512 pattern should generalize better on
~950 tracks than a fixed-max-width backbone; Qwen's argued the extra
capacity would just overfit. Neither response sourced this specific
disagreement to our data scale — ported faithfully here to let the training
run settle it directly, per project convention (test rather than trust or
dismiss either claim).

Same mel frontend and training-recipe settings as the original script:
n_mels=128, n_fft=2048, hop_length=256 (note: their CRNN script used
hop=512 — each of their three model scripts had its own preprocessing
constants; kept per-model here, not silently unified), f_min=20/f_max=8000,
top_db=80, per-sample mel normalization, 10s training crops.

Usage: python -m src.train --model singer_senet \
    --data_index_dir data/index --out_dir results/singer_senet \
    --optimizer adamw --label_smoothing 0.1 --lr 3e-4 --weight_decay 1e-4
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio


class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, channels // reduction),
            nn.ReLU(),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid(),
        )

    def forward(self, x):
        w = self.se(x).view(x.size(0), x.size(1), 1, 1)
        return x * w


class ConvSEBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ELU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
        )
        self.se = SEBlock(out_ch)
        self.shortcut = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
            nn.BatchNorm2d(out_ch),
        ) if in_ch != out_ch or stride != 1 else nn.Identity()

    def forward(self, x):
        out = self.conv(x)
        out = self.se(out)
        return F.elu(out + self.shortcut(x))


class SingerSENet(nn.Module):
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
        self.layer1 = nn.Sequential(ConvSEBlock(32, 64), ConvSEBlock(64, 64))
        self.layer2 = nn.Sequential(ConvSEBlock(64, 128, stride=2), ConvSEBlock(128, 128))
        self.layer3 = nn.Sequential(ConvSEBlock(128, 256, stride=2), ConvSEBlock(256, 256))
        self.layer4 = nn.Sequential(ConvSEBlock(256, 512, stride=2), ConvSEBlock(512, 512))
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
