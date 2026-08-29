"""CRNN2D, ported from Keras/TF to PyTorch from
ZainNasrullah/music-artist-classification-crnn (src/models.py). Method 2-1
in the assignment doc:
  Nasrullah & Zhao, "Musical Artist Classification with Convolutional
  Recurrent Neural Networks," IJCNN 2019.

Input: precomputed log-mel spectrogram (batch, n_mels=128, T), same
convention as src/models/confound_crnn.py (both repos converge on the same
4-block conv + pool schedule: (2,2),(4,2),(4,2),(4,2)).

Porting notes:
- Keras `BatchNormalization(axis=frequency_axis)` applied directly to the
  raw (freq, time) input is reproduced as `BatchNorm1d(n_mels)` on the
  (N, n_mels, T) tensor before the channel dim is added — same trick as
  bill317996's `Bn0`.
- Keras `Conv2D(..., padding='same')` with a 3x3 kernel == PyTorch
  `padding=1`.
- Crucially, `GRU(32, return_sequences=False)` in Keras returns only the
  **last** time step, unlike bill317996's PyTorch port (which reshapes the
  *whole* GRU2 output sequence into the embedding). We reproduce that
  distinction faithfully here: the classifier head operates on a 32-dim
  final hidden state, not a flattened sequence.
- `Softmax` output dropped in favor of raw logits + `nn.CrossEntropyLoss`.
"""
import torch.nn as nn


class CRNN2D(nn.Module):
    def __init__(self, n_class=20, n_mels=128):
        super().__init__()
        filters = [64, 128, 128, 128]
        pools = [(2, 2), (4, 2), (4, 2), (4, 2)]

        self.bn0 = nn.BatchNorm1d(n_mels)

        blocks = []
        in_ch = 1
        for out_ch, pool in zip(filters, pools):
            blocks.append(
                nn.Sequential(
                    nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
                    nn.ELU(),
                    nn.BatchNorm2d(out_ch),
                    nn.MaxPool2d(pool, stride=pool),
                    nn.Dropout2d(0.1),
                )
            )
            in_ch = out_ch
        self.conv_blocks = nn.ModuleList(blocks)

        self.gru1 = nn.GRU(filters[-1], 32, batch_first=True)
        self.gru2 = nn.GRU(32, 32, batch_first=True)
        self.dropout = nn.Dropout(0.3)
        self.dense = nn.Linear(32, n_class)

    def embed(self, x):
        x = self.bn0(x)
        x = x.unsqueeze(1)  # (N, 1, n_mels, T)

        for block in self.conv_blocks:
            x = block(x)
        # (N, C, freq, T') -> (N, T', freq, C) -> (N, T', freq*C)
        x = x.permute(0, 3, 2, 1)
        x = x.reshape(x.size(0), x.size(1), -1)

        x, _ = self.gru1(x)
        x, _ = self.gru2(x)
        return x[:, -1, :]

    def forward(self, x):
        emb = self.embed(x)
        return self.dense(self.dropout(emb))
