"""PyTorch datasets for artist20.

Two chunk representations are used across the ported reference methods:

- Mel-chunk (CRNN family: bill317996 / ZainNasrullah / FGNL): precomputed
  log-mel spectrogram, n_mels=128, n_fft=2048, hop_length=512, matching
  bill317996/Singer-identification-in-artist20's `utility.py` preprocessing
  (their `slice_length=157` frames ~= 5.02s chunks). We compute mel via
  torchaudio (GPU-friendly) with `norm="slaney", mel_scale="slaney"` to match
  librosa's defaults used in the original repo.
- Waveform-chunk (sota-music-tagging-models / SSL frontends / speaker
  frontend): raw 16kHz waveform, fixed-length chunks. The original
  sota-music-tagging-models repo computes its own mel internally
  (n_fft=512, n_mels=96) from a raw-waveform chunk, so we feed it raw audio
  and let the model's own frontend do the rest. We standardize all
  raw-waveform chunks to 5s (matching the mel-chunk duration) for a fair,
  consistent comparison across methods in this unified framework.
"""
import json
import random

import torch
import torchaudio
from torch.utils.data import Dataset

SR = 16000
CHUNK_SECONDS = 5.0
CHUNK_SAMPLES = int(SR * CHUNK_SECONDS)  # 80000

MEL_N_FFT = 2048
MEL_HOP = 512
MEL_N_MELS = 128
MEL_CHUNK_FRAMES = 157  # ~5.02s at hop=512, sr=16000, matches bill317996


def load_index(path):
    with open(path) as f:
        return json.load(f)


def _load_waveform(path):
    wav, sr = torchaudio.load(path)
    if wav.size(0) > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != SR:
        wav = torchaudio.functional.resample(wav, sr, SR)
    return wav.squeeze(0)  # (T,)


def _fixed_length_chunks(wav, chunk_samples):
    """Non-overlapping chunks, dropping a shorter tail chunk."""
    n_chunks = wav.numel() // chunk_samples
    if n_chunks == 0:
        # pad short tracks up to one chunk
        pad = chunk_samples - wav.numel()
        wav = torch.nn.functional.pad(wav, (0, pad))
        return wav.unsqueeze(0)
    wav = wav[: n_chunks * chunk_samples]
    return wav.view(n_chunks, chunk_samples)


def _augment_waveform(wav):
    """Cheap augmentation for from-scratch training, per the assignment doc's
    Data Augmentation section (torchaudio) — gain jitter + light additive
    noise. Kept simple/fast since it runs on every training sample."""
    gain_db = random.uniform(-6.0, 3.0)
    wav = wav * (10 ** (gain_db / 20))
    if random.random() < 0.5:
        noise = torch.randn_like(wav) * (wav.abs().mean() * random.uniform(0.01, 0.05))
        wav = wav + noise
    return wav.clamp(-1.0, 1.0)


class WaveformTrainDataset(Dataset):
    """One random chunk per track per __getitem__ call, for training."""

    def __init__(self, index_path, chunk_samples=CHUNK_SAMPLES, augment=True):
        self.records = load_index(index_path)
        self.chunk_samples = chunk_samples
        self.augment = augment

    def __len__(self):
        return len(self.records)

    def __getitem__(self, i):
        rec = self.records[i]
        wav = _load_waveform(rec["path"])
        if wav.numel() < self.chunk_samples:
            wav = torch.nn.functional.pad(wav, (0, self.chunk_samples - wav.numel()))
        else:
            start = random.randint(0, wav.numel() - self.chunk_samples)
            wav = wav[start : start + self.chunk_samples]
        if self.augment:
            wav = _augment_waveform(wav)
        return wav, rec["label_idx"]


class WaveformEvalDataset(Dataset):
    """All non-overlapping chunks of every track, for val/test inference.

    Each item is (chunks[n_chunks, chunk_samples], label_idx_or_None, track_key).
    Use a batch_size=1 DataLoader (chunk counts differ per track) and let the
    eval loop flatten/aggregate.
    """

    def __init__(self, index_path, chunk_samples=CHUNK_SAMPLES, has_labels=True):
        self.records = load_index(index_path)
        self.chunk_samples = chunk_samples
        self.has_labels = has_labels

    def __len__(self):
        return len(self.records)

    def __getitem__(self, i):
        rec = self.records[i]
        wav = _load_waveform(rec["path"])
        chunks = _fixed_length_chunks(wav, self.chunk_samples)
        label = rec["label_idx"] if self.has_labels else -1
        key = rec.get("id", rec.get("path"))
        return chunks, label, key


def _log_mel_transform():
    mel = torchaudio.transforms.MelSpectrogram(
        sample_rate=SR,
        n_fft=MEL_N_FFT,
        hop_length=MEL_HOP,
        n_mels=MEL_N_MELS,
        norm="slaney",
        mel_scale="slaney",
    )
    to_db = torchaudio.transforms.AmplitudeToDB(stype="power", top_db=None)
    return mel, to_db


def _spec_augment(log_mel, freq_mask_width=20, time_mask_width=25, n_freq_masks=2, n_time_masks=2):
    """SpecAugment (Park et al. 2019) on a (n_mels, T) log-mel chunk — per the
    assignment doc's Data Augmentation section. Masks are filled with the
    chunk's own mean (~silence in log-mel/dB space), not zero."""
    log_mel = log_mel.clone()
    fill = log_mel.mean()
    n_mels, n_frames = log_mel.shape
    for _ in range(n_freq_masks):
        w = random.randint(0, min(freq_mask_width, n_mels - 1))
        if w == 0:
            continue
        f0 = random.randint(0, n_mels - w)
        log_mel[f0 : f0 + w, :] = fill
    for _ in range(n_time_masks):
        w = random.randint(0, min(time_mask_width, n_frames - 1))
        if w == 0:
            continue
        t0 = random.randint(0, n_frames - w)
        log_mel[:, t0 : t0 + w] = fill
    return log_mel


class MelChunkTrainDataset(Dataset):
    """One random log-mel chunk (128, 157) per track per __getitem__ call."""

    def __init__(self, index_path, chunk_frames=MEL_CHUNK_FRAMES, augment=True):
        self.records = load_index(index_path)
        self.chunk_frames = chunk_frames
        self.mel, self.to_db = _log_mel_transform()
        self.augment = augment

    def __len__(self):
        return len(self.records)

    def __getitem__(self, i):
        rec = self.records[i]
        wav = _load_waveform(rec["path"])
        log_mel = self.to_db(self.mel(wav.unsqueeze(0))).squeeze(0)  # (n_mels, T)
        n_frames = log_mel.size(1)
        if n_frames < self.chunk_frames:
            pad = self.chunk_frames - n_frames
            log_mel = torch.nn.functional.pad(log_mel, (0, pad))
        else:
            start = random.randint(0, n_frames - self.chunk_frames)
            log_mel = log_mel[:, start : start + self.chunk_frames]
        if self.augment:
            log_mel = _spec_augment(log_mel)
        return log_mel, rec["label_idx"]


class MelChunkEvalDataset(Dataset):
    """All non-overlapping log-mel chunks of every track, for val/test inference."""

    def __init__(self, index_path, chunk_frames=MEL_CHUNK_FRAMES, has_labels=True):
        self.records = load_index(index_path)
        self.chunk_frames = chunk_frames
        self.has_labels = has_labels
        self.mel, self.to_db = _log_mel_transform()

    def __len__(self):
        return len(self.records)

    def __getitem__(self, i):
        rec = self.records[i]
        wav = _load_waveform(rec["path"])
        log_mel = self.to_db(self.mel(wav.unsqueeze(0))).squeeze(0)  # (n_mels, T)
        n_frames = log_mel.size(1)
        n_chunks = max(1, n_frames // self.chunk_frames)
        if n_frames < self.chunk_frames:
            log_mel = torch.nn.functional.pad(log_mel, (0, self.chunk_frames - n_frames))
            chunks = log_mel.unsqueeze(0)
        else:
            log_mel = log_mel[:, : n_chunks * self.chunk_frames]
            chunks = log_mel.unfold(1, self.chunk_frames, self.chunk_frames).permute(1, 0, 2)
        label = rec["label_idx"] if self.has_labels else -1
        key = rec.get("id", rec.get("path"))
        return chunks, label, key
