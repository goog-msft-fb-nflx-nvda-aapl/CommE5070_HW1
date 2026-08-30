"""Directly tests the user's prior-run TTA claim (single random crop ->
10 random crops averaged gave CRNN 0.65->0.77) against our own pipeline,
rather than assuming it doesn't apply here (per user instruction: don't
reject a path on assumption, ablate it). Compares three eval strategies on
the same trained checkpoint:
  A. single random crop (matches their pre-TTA baseline)
  B. N independent random crops averaged (matches their TTA exactly)
  C. our current full-track non-overlapping-tile averaging (existing default)

Usage:
    python -m src.analysis_tta_comparison --model sota_crnn --checkpoint results/sota_crnn/best.pt \
        --data_index_dir data/index --out_path results/analysis/tta_comparison.json
"""
import argparse
import json
import random

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from src.checkpoint_utils import load_checkpoint
from src.data.dataset import CHUNK_SAMPLES, MEL_CHUNK_FRAMES, _load_waveform, _log_mel_transform, load_index
from src.evaluate import aggregate_predict, compute_metrics
from src.train import MODEL_REGISTRY


class RandomCropEvalDataset(Dataset):
    """Yields (chunks[n_crops, ...], label, key) via n_crops INDEPENDENT
    random crops (with replacement) per track — matching the user's prior
    run's TTA method exactly, as opposed to our default's non-overlapping
    full-track tiling."""

    def __init__(self, index_path, kind, n_crops, seed=42):
        self.records = load_index(index_path)
        self.kind = kind
        self.n_crops = n_crops
        self.rng = random.Random(seed)
        if kind == "mel":
            self.mel, self.to_db = _log_mel_transform()

    def __len__(self):
        return len(self.records)

    def _one_crop(self, wav):
        if self.kind == "mel":
            log_mel = self.to_db(self.mel(wav.unsqueeze(0))).squeeze(0)
            n_frames = log_mel.size(1)
            if n_frames < MEL_CHUNK_FRAMES:
                return torch.nn.functional.pad(log_mel, (0, MEL_CHUNK_FRAMES - n_frames))
            start = self.rng.randint(0, n_frames - MEL_CHUNK_FRAMES)
            return log_mel[:, start : start + MEL_CHUNK_FRAMES]
        else:
            if wav.numel() < CHUNK_SAMPLES:
                return torch.nn.functional.pad(wav, (0, CHUNK_SAMPLES - wav.numel()))
            start = self.rng.randint(0, wav.numel() - CHUNK_SAMPLES)
            return wav[start : start + CHUNK_SAMPLES]

    def __getitem__(self, i):
        rec = self.records[i]
        wav = _load_waveform(rec["path"])
        crops = torch.stack([self._one_crop(wav) for _ in range(self.n_crops)])
        return crops, rec["label_idx"], rec.get("id", rec["path"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(MODEL_REGISTRY.keys()))
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--data_index_dir", required=True)
    ap.add_argument("--out_path", required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    with open(f"{args.data_index_dir}/labels.json") as f:
        labels = json.load(f)
    n_class = len(labels)

    model_fn, kind = MODEL_REGISTRY[args.model]
    model = model_fn(n_class).to(args.device)
    load_checkpoint(model, args.checkpoint, args.device)
    model.eval()

    val_path = f"{args.data_index_dir}/val.json"

    # A: single random crop
    ds_1 = RandomCropEvalDataset(val_path, kind, n_crops=1)
    _, trues_a, probs_a = aggregate_predict(model, ds_1, args.device, n_class)
    metrics_a, _ = compute_metrics(trues_a, probs_a, n_class)

    # B: 10 independent random crops averaged
    ds_10 = RandomCropEvalDataset(val_path, kind, n_crops=10)
    _, trues_b, probs_b = aggregate_predict(model, ds_10, args.device, n_class)
    metrics_b, _ = compute_metrics(trues_b, probs_b, n_class)

    # C: our default full-track non-overlapping tiling
    from src.data.dataset import MelChunkEvalDataset, WaveformEvalDataset

    ds_c = MelChunkEvalDataset(val_path) if kind == "mel" else WaveformEvalDataset(val_path)
    _, trues_c, probs_c = aggregate_predict(model, ds_c, args.device, n_class)
    metrics_c, _ = compute_metrics(trues_c, probs_c, n_class)

    result = {
        "A_single_random_crop": metrics_a,
        "B_10_random_crops_averaged": metrics_b,
        "C_our_full_tile_average": metrics_c,
    }
    with open(args.out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
