"""SE-ResNet: ShortChunkCNN_Res (minzwon/sota-music-tagging-models)'s 7-block
residual CNN with Squeeze-and-Excitation gating added after each block's
residual add (Hu et al., "Squeeze-and-Excitation Networks," CVPR 2018).
Directly modeled on the user's own prior submission for this assignment
(task2_se_cnn.py), where an SE-ResNet was their second-best individual model
(0.675 val top1) and a nontrivial contributor to their ensemble. From
scratch — no pretrained weights.
"""
import torch.nn as nn
import torchaudio

from .common import Res_2d_SE


class SEResNet(nn.Module):
    def __init__(self, n_channels=128, sample_rate=16000, n_fft=512, f_min=0.0, f_max=8000.0,
                 n_mels=128, n_class=20, se_reduction=16):
        super().__init__()
        self.spec = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate, n_fft=n_fft, f_min=f_min, f_max=f_max, n_mels=n_mels
        )
        self.to_db = torchaudio.transforms.AmplitudeToDB()
        self.spec_bn = nn.BatchNorm2d(1)

        self.layer1 = Res_2d_SE(1, n_channels, stride=2, se_reduction=se_reduction)
        self.layer2 = Res_2d_SE(n_channels, n_channels, stride=2, se_reduction=se_reduction)
        self.layer3 = Res_2d_SE(n_channels, n_channels * 2, stride=2, se_reduction=se_reduction)
        self.layer4 = Res_2d_SE(n_channels * 2, n_channels * 2, stride=2, se_reduction=se_reduction)
        self.layer5 = Res_2d_SE(n_channels * 2, n_channels * 2, stride=2, se_reduction=se_reduction)
        self.layer6 = Res_2d_SE(n_channels * 2, n_channels * 2, stride=2, se_reduction=se_reduction)
        self.layer7 = Res_2d_SE(n_channels * 2, n_channels * 4, stride=2, se_reduction=se_reduction)

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
