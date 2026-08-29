"""CRNN2D_elu2, ported from bill317996/Singer-identification-in-artist20
(model.py) — the official reference implementation for Hsieh et al.,
"Addressing the confounds of accompaniments in singer identification,"
ICASSP 2020 (the paper that defines the artist20 album-level split used by
this assignment, per readme.md footnote [1]). This is Method 2-2 in the
assignment doc, and also our Task-2 "from scratch" core model (Method 1's
family/slice_length convention is shared, see src/data/dataset.py).

Input: precomputed log-mel spectrogram (batch, n_mels=128, T=157), matching
this repo's `utility.py` (`sr=16000, n_mels=128, n_fft=2048, hop_length=512,
slice_length=157`).

Deviations from the original, both documented here rather than silently
applied:
- We port `CRNN2D_elu2` (not the plainer `CRNN2D_elu` also present in the
  repo) because it is the variant whose padding keeps the frequency axis
  divisible through all four (2,2)/(4,2)/(4,2)/(4,2) pooling stages for
  T=157 — `CRNN2D_elu` collapses the frequency axis to 0 at the last pool
  and cannot run at this chunk length. `CRNN2D_elu2`'s `emb_size=288` also
  falls out exactly from this shape (9 time steps x 32 GRU hidden units),
  confirming it's the intended, working configuration.
- The original's `forward(x, h)` re-randomizes the GRU's initial hidden
  state every batch (never carried across batches or seeded meaningfully);
  we use PyTorch's default zero-initialized hidden state, which is
  behaviorally equivalent for a non-persistent, freshly-random `h`.
- `Softmax` is dropped in favor of returning raw logits (paired with
  `nn.CrossEntropyLoss`), for numerical stability; `nn.CrossEntropyLoss`
  already applies log-softmax internally.
"""
import torch
import torch.nn as nn


class CRNN2D_elu2(nn.Module):
    def __init__(self, n_class=20, n_mels=128, chunk_frames=157):
        super().__init__()
        self.elu = nn.ELU()

        self.Bn0 = nn.BatchNorm1d(n_mels)

        self.Conv1 = nn.Conv2d(1, 64, (3, 3), padding=(1, 1))
        self.Conv1_1 = nn.Conv2d(64, 64, (3, 3), padding=(1, 1))
        self.Bn1 = nn.BatchNorm2d(64)
        self.mp1 = nn.MaxPool2d((2, 2), stride=(2, 2))
        self.drop1 = nn.Dropout2d(p=0.1)

        self.Conv2 = nn.Conv2d(64, 128, (3, 3), padding=(1, 1))
        self.Conv2_1 = nn.Conv2d(128, 128, (3, 3), padding=(1, 1))
        self.Bn2 = nn.BatchNorm2d(128)
        self.mp2 = nn.MaxPool2d((4, 2), stride=(4, 2))
        self.drop2 = nn.Dropout2d(p=0.1)

        self.Conv3 = nn.Conv2d(128, 128, (3, 3), padding=(1, 1))
        self.Conv3_1 = nn.Conv2d(128, 128, (3, 3), padding=(1, 1))
        self.Bn3 = nn.BatchNorm2d(128)
        self.mp3 = nn.MaxPool2d((4, 2), stride=(4, 2))
        self.drop3 = nn.Dropout2d(p=0.1)

        self.Conv4 = nn.Conv2d(128, 128, (3, 3), padding=(1, 1))
        self.Conv4_1 = nn.Conv2d(128, 128, (3, 3), padding=(1, 1))
        self.Bn4 = nn.BatchNorm2d(128)
        self.mp4 = nn.MaxPool2d((4, 2), stride=(4, 2))
        self.drop4 = nn.Dropout2d(p=0.1)

        self.gru1 = nn.GRU(128, 32, num_layers=1, batch_first=True)
        self.gru2 = nn.GRU(32, 32, num_layers=1, batch_first=True)
        self.drop5 = nn.Dropout(p=0.3)

        with torch.no_grad():
            dummy = torch.zeros(1, n_mels, chunk_frames)
            emb_size = self._features(dummy).shape[1]
        self.linear1 = nn.Linear(emb_size, n_class)

    def _features(self, x):
        x = self.Bn0(x)
        x = x[:, None, :, :]

        x = self.drop1(self.mp1(self.Bn1(self.elu(self.Conv1_1(self.Conv1(x))))))
        x = self.drop2(self.mp2(self.Bn2(self.elu(self.Conv2_1(self.Conv2(x))))))
        x = self.drop3(self.mp3(self.Bn3(self.elu(self.Conv3_1(self.Conv3(x))))))
        x = self.drop4(self.mp4(self.Bn4(self.elu(self.Conv4_1(self.Conv4(x))))))

        x = x.transpose(1, 3)
        x = torch.reshape(x, (x.size(0), x.size(1), -1))

        x, _ = self.gru1(x)
        x, _ = self.gru2(x)
        x = self.drop5(x)

        return torch.reshape(x, (x.size(0), -1))

    def embed(self, x):
        return self._features(x)

    def forward(self, x):
        emb = self._features(x)
        return self.linear1(emb)
