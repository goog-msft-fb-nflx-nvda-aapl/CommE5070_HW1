"""Same-track two-crop dataset for SimCLR-style contrastive pretraining, per
the round-4 Deep Research consensus recipe (Perplexity's response, most
concretely sourced): two independently-sampled, independently-augmented
crops from the same training track form a positive pair. Train-split only,
same as every other from-scratch component — self-supervised on our own
949 tracks, no external data or weights, so it stays from-scratch-eligible.

Yields raw waveform crops (not precomputed mel) so it plugs directly into
`sota_crnn` (our best from-scratch model, `CRNN` in sota_cnn.py, which
computes its own mel internally) — Perplexity's recommendation to "reuse
your strongest existing CRNN first" rather than confound the SSL-vs-no-SSL
comparison with a new architecture too.
"""
import random

import torch
from torch.utils.data import Dataset

from .dataset import CHUNK_SAMPLES, _augment_waveform, _load_waveform, load_index


class TwoCropSSLDataset(Dataset):
    def __init__(self, index_path, chunk_samples=CHUNK_SAMPLES):
        self.records = load_index(index_path)
        self.chunk_samples = chunk_samples

    def _one_crop(self, wav):
        if wav.numel() < self.chunk_samples:
            crop = torch.nn.functional.pad(wav, (0, self.chunk_samples - wav.numel()))
        else:
            start = random.randint(0, wav.numel() - self.chunk_samples)
            crop = wav[start : start + self.chunk_samples]
        return _augment_waveform(crop)  # independent stochastic augmentation per view

    def __len__(self):
        return len(self.records)

    def __getitem__(self, i):
        wav = _load_waveform(self.records[i]["path"])
        return self._one_crop(wav), self._one_crop(wav)
