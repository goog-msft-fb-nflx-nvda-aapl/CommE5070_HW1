"""Cross-song vocal/instrumental remix training, per Hsieh et al. ICASSP
2020 (bill317996/Singer-identification-in-artist20) — the technique behind
their CRNN's largest reported gain (+7-8pp song-level F1 going from
Origin-only to Origin+Vocal-only+Remix combined training). Confirmed as the
top from-scratch-compatible lever for this exact architecture/dataset by two
independent Deep Research follow-ups (deep_research_response_3_*.md).

Each `__getitem__` call stochastically picks one of three "views" for that
training index, approximating Hsieh et al.'s combined training pool without
materializing 3x the dataset:
  - the original mixture (label's own track)
  - the vocals-only separation of that same track
  - a remix: that track's separated vocals + a *different*, randomly-chosen
    training track's separated accompaniment (drums+bass+other), label stays
    with the vocal source. This is the piece that breaks the album/
    production confound — accompaniment can come from any other training
    track (deliberately not restricted to "different album of the same
    artist" as in the original paper, for simplicity; still strictly
    train-split-only, never touching val/test).

Requires the vocals (src/data/separate_vocals.py) and accompaniment
(src/data/separate_accompaniment.py) stems to already be extracted for the
train split, in the same record order as data/index/train.json.

Usage: see src/train.py's --remix flag.
"""
import json
import random

import torch

from src.data.dataset import MEL_CHUNK_FRAMES, _load_waveform, _log_mel_transform, _spec_augment

CHUNK_SAMPLES = 80000  # 5s @ 16kHz, matches src/data/dataset.py


def _random_chunk(wav, chunk_samples):
    if wav.numel() < chunk_samples:
        return torch.nn.functional.pad(wav, (0, chunk_samples - wav.numel()))
    start = random.randint(0, wav.numel() - chunk_samples)
    return wav[start : start + chunk_samples]


class RemixMelChunkTrainDataset(torch.utils.data.Dataset):
    def __init__(self, orig_index_path, vocals_index_path, accompaniment_index_path,
                 chunk_frames=MEL_CHUNK_FRAMES, p_origin=1 / 3, p_vocal_only=1 / 3, augment=True):
        with open(orig_index_path) as f:
            self.orig_records = json.load(f)
        with open(vocals_index_path) as f:
            self.vocal_records = json.load(f)
        with open(accompaniment_index_path) as f:
            self.accompaniment_records = json.load(f)
        assert len(self.orig_records) == len(self.vocal_records) == len(self.accompaniment_records)

        self.chunk_frames = chunk_frames
        self.p_origin = p_origin
        self.p_vocal_only = p_vocal_only  # remix probability = 1 - p_origin - p_vocal_only
        self.augment = augment
        self.mel, self.to_db = _log_mel_transform()

    def __len__(self):
        return len(self.orig_records)

    def _to_log_mel(self, wav):
        return self.to_db(self.mel(wav.unsqueeze(0))).squeeze(0)

    def __getitem__(self, i):
        label = self.orig_records[i]["label_idx"]
        r = random.random()

        if r < self.p_origin:
            wav = _load_waveform(self.orig_records[i]["path"])
            wav = _random_chunk(wav, CHUNK_SAMPLES)
        elif r < self.p_origin + self.p_vocal_only:
            wav = _load_waveform(self.vocal_records[i]["path"])
            wav = _random_chunk(wav, CHUNK_SAMPLES)
        else:
            j = random.randrange(len(self.orig_records))
            while j == i:
                j = random.randrange(len(self.orig_records))
            vocal = _random_chunk(_load_waveform(self.vocal_records[i]["path"]), CHUNK_SAMPLES)
            accompaniment = _random_chunk(_load_waveform(self.accompaniment_records[j]["path"]), CHUNK_SAMPLES)
            gain = random.uniform(0.7, 1.3)
            wav = (vocal + gain * accompaniment).clamp(-1.0, 1.0)

        log_mel = self._to_log_mel(wav)
        n_frames = log_mel.size(1)
        if n_frames < self.chunk_frames:
            log_mel = torch.nn.functional.pad(log_mel, (0, self.chunk_frames - n_frames))
        elif n_frames > self.chunk_frames:
            start = random.randint(0, n_frames - self.chunk_frames)
            log_mel = log_mel[:, start : start + self.chunk_frames]

        if self.augment:
            log_mel = _spec_augment(log_mel)

        return log_mel, label
