"""FCN and CRNN, ported from minzwon/sota-music-tagging-models
(training/model.py), MIT-licensed. Method 1 in the assignment doc.

Original citations:
- FCN: Choi et al. 2016, "Automatic tagging using deep convolutional neural
  networks."
- CRNN: Choi et al. 2017, "Convolution recurrent neural networks for music
  classification."

Adaptation from the original (multi-label tagging) to this assignment
(single-label 20-way artist classification): final `Sigmoid` replaced with
raw logits (paired with `nn.CrossEntropyLoss` instead of the original's
per-tag BCE), `n_class` defaults to 20. Everything else (spectrogram
front-end params, conv/pool/RNN layer shapes) is unchanged.

Scope note: `FCN`'s pooling schedule ((2,4),(2,4),(2,4),(3,5),(4,4) — a
total 1280x time downsample) needs at least ~20s of audio at this n_fft/hop
to avoid the time axis collapsing to 0; it was originally trained on long
MTAT clips. Our unified chunking (src/data/dataset.py) uses 5s chunks
across every method for a fair, consistent comparison, which `FCN`
structurally cannot consume — verified by a dry run (RuntimeError: output
size too small). `FCN` is kept here for completeness/citation but is not
used in the trained model comparison; `CRNN` (same repo, same Method 1
citation) is used instead and runs fine at 5s.
"""
import torch
import torch.nn as nn
import torchaudio

from .common import Conv_1d, Conv_2d, Res_2d


class FCN(nn.Module):
    def __init__(self, sample_rate=16000, n_fft=512, f_min=0.0, f_max=8000.0, n_mels=96, n_class=20):
        super().__init__()
        self.spec = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate, n_fft=n_fft, f_min=f_min, f_max=f_max, n_mels=n_mels
        )
        self.to_db = torchaudio.transforms.AmplitudeToDB()
        self.spec_bn = nn.BatchNorm2d(1)

        self.layer1 = Conv_2d(1, 64, pooling=(2, 4))
        self.layer2 = Conv_2d(64, 128, pooling=(2, 4))
        self.layer3 = Conv_2d(128, 128, pooling=(2, 4))
        self.layer4 = Conv_2d(128, 128, pooling=(3, 5))
        self.layer5 = Conv_2d(128, 64, pooling=(4, 4))

        self.dense = nn.Linear(64, n_class)
        self.dropout = nn.Dropout(0.5)

    def embed(self, x):
        x = self.spec(x)
        x = self.to_db(x)
        x = x.unsqueeze(1)
        x = self.spec_bn(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        return x.view(x.size(0), -1)

    def forward(self, x):
        emb = self.embed(x)
        return self.dense(self.dropout(emb))


class ShortChunkCNN_Res(nn.Module):
    """Won et al. 2020 SMC, "Evaluation of CNN-based automatic music tagging
    models" — a VGG-ish 7-block residual CNN, highlighted as a top exemplar
    in lecture02_classification.md slides 77-79. Deeper/narrower receptive
    field than FCN/CRNN above; residual connections via `Res_2d` (already
    ported in common.py for other models)."""

    def __init__(self, n_channels=128, sample_rate=16000, n_fft=512, f_min=0.0, f_max=8000.0,
                 n_mels=128, n_class=20):
        super().__init__()
        self.spec = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate, n_fft=n_fft, f_min=f_min, f_max=f_max, n_mels=n_mels
        )
        self.to_db = torchaudio.transforms.AmplitudeToDB()
        self.spec_bn = nn.BatchNorm2d(1)

        self.layer1 = Res_2d(1, n_channels, stride=2)
        self.layer2 = Res_2d(n_channels, n_channels, stride=2)
        self.layer3 = Res_2d(n_channels, n_channels * 2, stride=2)
        self.layer4 = Res_2d(n_channels * 2, n_channels * 2, stride=2)
        self.layer5 = Res_2d(n_channels * 2, n_channels * 2, stride=2)
        self.layer6 = Res_2d(n_channels * 2, n_channels * 2, stride=2)
        self.layer7 = Res_2d(n_channels * 2, n_channels * 4, stride=2)

        self.dense1 = nn.Linear(n_channels * 4, n_channels * 4)
        self.bn = nn.BatchNorm1d(n_channels * 4)
        self.dense2 = nn.Linear(n_channels * 4, n_class)
        self.dropout = nn.Dropout(0.5)
        self.relu = nn.ReLU()

    def embed(self, x):
        x = self.spec(x)
        x = self.to_db(x)
        x = x.unsqueeze(1)
        x = self.spec_bn(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = self.layer7(x)
        x = x.squeeze(2)

        if x.size(-1) != 1:
            x = nn.MaxPool1d(x.size(-1))(x)
        x = x.squeeze(2)

        x = self.dense1(x)
        x = self.bn(x)
        return self.relu(x)

    def forward(self, x):
        emb = self.embed(x)
        return self.dense2(self.dropout(emb))


class SampleCNN(nn.Module):
    """Lee et al. 2017 SMC, "Sample-level deep convolutional neural networks
    for music auto-tagging using raw waveforms" — end-to-end, operates
    directly on raw audio samples rather than a spectrogram front-end.
    Matches lecture02_classification.md's "sample-level CNN" / "end-to-end
    learning" framing (slides 47d, 71, 75) — architecturally the most
    distinct from-scratch option in this project (every other model starts
    from a mel-spectrogram)."""

    def __init__(self, n_class=20):
        super().__init__()
        self.layer1 = Conv_1d(1, 128, shape=3, stride=3, pooling=1)
        self.layer2 = Conv_1d(128, 128, shape=3, stride=1, pooling=3)
        self.layer3 = Conv_1d(128, 128, shape=3, stride=1, pooling=3)
        self.layer4 = Conv_1d(128, 256, shape=3, stride=1, pooling=3)
        self.layer5 = Conv_1d(256, 256, shape=3, stride=1, pooling=3)
        self.layer6 = Conv_1d(256, 256, shape=3, stride=1, pooling=3)
        self.layer7 = Conv_1d(256, 256, shape=3, stride=1, pooling=3)
        self.layer8 = Conv_1d(256, 256, shape=3, stride=1, pooling=3)
        self.layer9 = Conv_1d(256, 256, shape=3, stride=1, pooling=3)
        self.layer10 = Conv_1d(256, 512, shape=3, stride=1, pooling=3)
        self.layer11 = Conv_1d(512, 512, shape=1, stride=1, pooling=1)
        self.dropout = nn.Dropout(0.5)
        self.dense = nn.Linear(512, n_class)

    def embed(self, x):
        x = x.unsqueeze(1)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = self.layer7(x)
        x = self.layer8(x)
        x = self.layer9(x)
        x = self.layer10(x)
        x = self.layer11(x)
        return x.squeeze(-1)

    def forward(self, x):
        emb = self.embed(x)
        return self.dense(self.dropout(emb))


class CRNN(nn.Module):
    """`channel_mult` scales conv/GRU widths for a from-scratch capacity
    sweep (small-data overfitting-vs-undercapacity question raised, without
    a sourced citation, in deep_research/round3_from_scratch_improvement/response_qwen.md — tested
    directly here rather than accepted or dismissed on priors, per user
    instruction). Default 1.0 exactly reproduces the original (already
    trained, 0.762 val top1) architecture."""

    def __init__(self, sample_rate=16000, n_fft=512, f_min=0.0, f_max=8000.0, n_mels=96, n_class=20,
                 channel_mult=1.0, normalize_mel=False):
        super().__init__()
        c1, c2, c3, c4 = (max(8, round(c * channel_mult)) for c in (64, 128, 128, 128))
        gru_hidden = max(4, round(32 * channel_mult))
        self.normalize_mel = normalize_mel  # per-sample (mel-mean)/std, matching the
        # user's prior-run pipeline (dataset.py there normalizes each mel
        # chunk individually) — tested directly here rather than assumed
        # redundant with our models' existing BatchNorm2d(1) input norm.

        self.spec = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate, n_fft=n_fft, f_min=f_min, f_max=f_max, n_mels=n_mels
        )
        self.to_db = torchaudio.transforms.AmplitudeToDB()
        self.spec_bn = nn.BatchNorm2d(1)

        self.layer1 = Conv_2d(1, c1, pooling=(2, 2))
        self.layer2 = Conv_2d(c1, c2, pooling=(3, 3))
        self.layer3 = Conv_2d(c2, c3, pooling=(4, 4))
        self.layer4 = Conv_2d(c3, c4, pooling=(4, 4))

        self.layer5 = nn.GRU(c4, gru_hidden, 2, batch_first=True)

        self.dropout = nn.Dropout(0.5)
        self.dense = nn.Linear(gru_hidden, n_class)

    def embed(self, x):
        x = self.spec(x)
        x = self.to_db(x)
        if self.normalize_mel:
            n = x.size(0)
            flat = x.reshape(n, -1)
            mean = flat.mean(dim=1).view(n, 1, 1)
            std = flat.std(dim=1).view(n, 1, 1)
            x = (x - mean) / (std + 1e-6)
        x = x.unsqueeze(1)
        x = self.spec_bn(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = x.squeeze(2)
        x = x.permute(0, 2, 1)
        x, _ = self.layer5(x)
        return x[:, -1, :]

    def forward(self, x):
        emb = self.embed(x)
        return self.dense(self.dropout(emb))


class CRNN_DropBlock(nn.Module):
    """Same as `CRNN` but with `DropBlock2d` (src/models/dropblock.py) after
    each conv block instead of relying only on dropout+weight-decay for
    regularization — Qwen round-3 flagged DropBlock as "specifically
    validated for audio spectrograms" without a citation; tested directly
    rather than accepted or dismissed on that claim alone."""

    def __init__(self, sample_rate=16000, n_fft=512, f_min=0.0, f_max=8000.0, n_mels=96, n_class=20,
                 drop_prob=0.1, block_size=5):
        super().__init__()
        from .dropblock import DropBlock2d

        self.spec = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate, n_fft=n_fft, f_min=f_min, f_max=f_max, n_mels=n_mels
        )
        self.to_db = torchaudio.transforms.AmplitudeToDB()
        self.spec_bn = nn.BatchNorm2d(1)

        self.layer1 = Conv_2d(1, 64, pooling=(2, 2))
        self.layer2 = Conv_2d(64, 128, pooling=(3, 3))
        self.layer3 = Conv_2d(128, 128, pooling=(4, 4))
        self.layer4 = Conv_2d(128, 128, pooling=(4, 4))
        self.db1 = DropBlock2d(drop_prob, block_size)
        self.db2 = DropBlock2d(drop_prob, block_size)
        self.db3 = DropBlock2d(drop_prob, block_size)
        self.db4 = DropBlock2d(drop_prob, block_size)

        self.layer5 = nn.GRU(128, 32, 2, batch_first=True)

        self.dropout = nn.Dropout(0.5)
        self.dense = nn.Linear(32, n_class)

    def embed(self, x):
        x = self.spec(x)
        x = self.to_db(x)
        x = x.unsqueeze(1)
        x = self.spec_bn(x)

        x = self.db1(self.layer1(x))
        x = self.db2(self.layer2(x))
        x = self.db3(self.layer3(x))
        x = self.db4(self.layer4(x))

        x = x.squeeze(2)
        x = x.permute(0, 2, 1)
        x, _ = self.layer5(x)
        return x[:, -1, :]

    def forward(self, x):
        emb = self.embed(x)
        return self.dense(self.dropout(emb))


class CRNN_Attn(nn.Module):
    """Same conv/GRU stack as `CRNN` above, but with additive attention
    pooling over the GRU's full output sequence instead of taking only the
    last time step — matching the pooling strategy in the user's own prior
    submission for this assignment (their CRNN used attention pooling).
    Directly tests whether pooling strategy, not just architecture, explains
    part of the gap to their numbers."""

    def __init__(self, sample_rate=16000, n_fft=512, f_min=0.0, f_max=8000.0, n_mels=96, n_class=20):
        super().__init__()
        self.spec = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate, n_fft=n_fft, f_min=f_min, f_max=f_max, n_mels=n_mels
        )
        self.to_db = torchaudio.transforms.AmplitudeToDB()
        self.spec_bn = nn.BatchNorm2d(1)

        self.layer1 = Conv_2d(1, 64, pooling=(2, 2))
        self.layer2 = Conv_2d(64, 128, pooling=(3, 3))
        self.layer3 = Conv_2d(128, 128, pooling=(4, 4))
        self.layer4 = Conv_2d(128, 128, pooling=(4, 4))

        self.gru = nn.GRU(128, 32, 2, batch_first=True)
        self.attn = nn.Linear(32, 1)

        self.dropout = nn.Dropout(0.5)
        self.dense = nn.Linear(32, n_class)

    def embed(self, x):
        x = self.spec(x)
        x = self.to_db(x)
        x = x.unsqueeze(1)
        x = self.spec_bn(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = x.squeeze(2)
        x = x.permute(0, 2, 1)
        x, _ = self.gru(x)  # (N, T, 32)

        scores = self.attn(x).squeeze(-1)  # (N, T)
        weights = torch.softmax(scores, dim=1).unsqueeze(-1)  # (N, T, 1)
        return (x * weights).sum(dim=1)  # (N, 32)

    def forward(self, x):
        emb = self.embed(x)
        return self.dense(self.dropout(emb))
