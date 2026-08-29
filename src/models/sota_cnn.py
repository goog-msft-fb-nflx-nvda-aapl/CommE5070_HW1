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
"""
import torch.nn as nn
import torchaudio

from .common import Conv_2d


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


class CRNN(nn.Module):
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

        self.layer5 = nn.GRU(128, 32, 2, batch_first=True)

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
        x, _ = self.layer5(x)
        return x[:, -1, :]

    def forward(self, x):
        emb = self.embed(x)
        return self.dense(self.dropout(emb))
