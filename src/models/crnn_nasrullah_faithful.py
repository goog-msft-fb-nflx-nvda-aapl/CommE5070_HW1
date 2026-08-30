"""Faithful port of our own prior year's HW1 submission's CRNN (bidirectional
GRU + full-sequence attention pooling), from
github.com/goog-msft-fb-nflx-nvda-aapl/NTU (CommE5070 hw1_submission,
task2_dl_v2.py), itself loosely based on Nasrullah & Tan, "Musical artist
classification with convolutional recurrent neural networks," IJCNN 2019.

Motivation: our from-scratch ablations already tested attention pooling
(`CRNN_Attn`, bolted onto `sota_crnn`'s much smaller 32-dim GRU bottleneck —
see MATERIALS.md) and per-sample mel normalization (`sota_crnn_norm`)
independently, each on architectures that differ from the prior submission
in several other ways at once (mel resolution, GRU width/direction, conv
channel counts, segment length). Neither ablation faithfully isolates
"does this specific architecture, as actually run last year, reproduce that
year's numbers on our current (official, not self-split) val set" — this
model closes that gap by porting every architectural/training-recipe choice
from the original script as-is, changing only what's forced by unifying
into this repo's shared pipeline (this repo's official Artist20 train/val
split rather than a self-split "last album per artist"; song-level mean-
pooled softmax evaluation via src/evaluate.py rather than N=10 random-crop
TTA — kept separately testable, see src/analysis_tta_comparison.py).

Differences from `sota_crnn`/`CRNN_Attn`/`crnn_zain` this specifically
tests, all at once, matching the original submission exactly:
  - 128 mel bins, n_fft=2048, hop=512, f_min=20, f_max=8000, top_db=80
    (vs sota_crnn family's 96 mel / n_fft=512 / f_max=8000 / no top_db clip)
  - per-sample mel normalization: (mel - mel.mean()) / (mel.std() + 1e-6)
  - bidirectional GRU, hidden=256, 2 layers, internal dropout=0.3
    (vs sota_crnn's unidirectional GRU, hidden=32)
  - attention pooling over the *entire* GRU output sequence
  - 4 ConvBlocks (64/128/256/256 channels), each Conv-BN-ELU x2 + MaxPool +
    Dropout2d(0.1); last block pools (4,1) to compress frequency while
    keeping temporal resolution for the GRU/attention stage
  - classifier: Linear(512,256) -> ReLU -> Dropout(0.5) -> Linear(256, n_class)
  - trained on 10s chunks (`--model crnn_nasrullah_faithful` uses the
    "wave10s" dataset kind in src/train.py, chunk_samples=160000) instead of
    this repo's default 5s chunks

Usage: python -m src.train --model crnn_nasrullah_faithful \
    --data_index_dir data/index --out_dir results/crnn_nasrullah_faithful \
    --optimizer adamw --label_smoothing 0.1 --lr 3e-4 --weight_decay 1e-4
"""
import torch
import torch.nn as nn
import torchaudio


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, pool):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ELU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ELU(),
            nn.MaxPool2d(pool),
            nn.Dropout2d(0.1),
        )

    def forward(self, x):
        return self.block(x)


class CRNNNasrullahFaithful(nn.Module):
    def __init__(self, sample_rate=16000, n_fft=2048, hop_length=512, n_mels=128,
                 f_min=20.0, f_max=8000.0, gru_hidden=256, gru_layers=2, n_class=20):
        super().__init__()
        self.spec = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate, n_fft=n_fft, hop_length=hop_length,
            n_mels=n_mels, f_min=f_min, f_max=f_max,
        )
        self.to_db = torchaudio.transforms.AmplitudeToDB(top_db=80)

        self.cnn = nn.Sequential(
            ConvBlock(1, 64, pool=(2, 2)),
            ConvBlock(64, 128, pool=(2, 2)),
            ConvBlock(128, 256, pool=(2, 2)),
            ConvBlock(256, 256, pool=(4, 1)),
        )
        self.gru = nn.GRU(input_size=1024, hidden_size=gru_hidden, num_layers=gru_layers,
                           batch_first=True, dropout=0.3, bidirectional=True)
        self.attn = nn.Linear(gru_hidden * 2, 1)
        self.classifier = nn.Sequential(
            nn.Linear(gru_hidden * 2, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, n_class),
        )

    def _log_mel(self, wav):
        mel = self.to_db(self.spec(wav))  # (N, n_mels, T)
        mean = mel.mean(dim=(1, 2), keepdim=True)
        std = mel.std(dim=(1, 2), keepdim=True)
        return (mel - mean) / (std + 1e-6)

    def embed(self, x):
        x = self._log_mel(x).unsqueeze(1)  # (N, 1, n_mels, T)
        x = self.cnn(x)  # (N, 256, F', T')
        n, c, f, t = x.shape
        x = x.permute(0, 3, 1, 2).contiguous().view(n, t, c * f)
        x, _ = self.gru(x)  # (N, T', 512)
        attn_w = torch.softmax(self.attn(x), dim=1)
        return (x * attn_w).sum(dim=1)  # (N, 512)

    def forward(self, x):
        return self.classifier(self.embed(x))
