"""Shuffled-label sanity check for speaker_frontend, per the Deep Research
follow-up: with training labels randomly permuted (val labels left
untouched), a correctly-isolated train/val pipeline should collapse to
near-chance (1/20 = 5%) val accuracy. A result well above chance would
indicate some form of leakage between the "train" and "val" partitions (e.g.
overlapping audio, a broken split) rather than genuine signal in the ECAPA
embedding — this check doesn't overlap with the cosine-centroid/linear-probe
check (src/analysis_head_swap.py), which rules out MLP-specific overfitting
but not a split-leakage bug.

Usage (hw1_ssl_env, torch>=2.6):
    python -m src.analysis_shuffled_label_check --data_index_dir data/index --out_dir results/analysis
"""
import argparse
import json
import os
import random

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.data.dataset import WaveformTrainDataset, WaveformEvalDataset
from src.evaluate import evaluate_metrics_only
from src.models.speaker_frontend import SpeakerEmbeddingProbe


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_index_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    with open(f"{args.data_index_dir}/labels.json") as f:
        labels = json.load(f)
    n_class = len(labels)

    train_ds = WaveformTrainDataset(f"{args.data_index_dir}/train.json")
    random.seed(args.seed)
    original_labels = [r["label_idx"] for r in train_ds.records]
    shuffled = original_labels.copy()
    random.shuffle(shuffled)
    for rec, new_label in zip(train_ds.records, shuffled):
        rec["label_idx"] = new_label

    val_ds = WaveformEvalDataset(f"{args.data_index_dir}/val.json")

    model = SpeakerEmbeddingProbe(n_class=n_class).to(args.device)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=6, drop_last=True)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    history = []
    for epoch in range(args.epochs):
        model.train()
        total_loss, n_batches = 0.0, 0
        for x, y in train_loader:
            x, y = x.to(args.device), y.to(args.device)
            opt.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            opt.step()
            total_loss += loss.item()
            n_batches += 1
        metrics = evaluate_metrics_only(model, val_ds, args.device, n_class)
        print(f"epoch={epoch} loss={total_loss / max(1, n_batches):.4f} "
              f"val_top1={metrics['top1']:.4f} val_top3={metrics['top3']:.4f}")
        history.append({"epoch": epoch, "train_loss": total_loss / max(1, n_batches), **metrics})

    result = {
        "final_val_top1": history[-1]["top1"],
        "final_val_top3": history[-1]["top3"],
        "best_val_top1": max(h["top1"] for h in history),
        "chance_top1": 1.0 / n_class,
        "history": history,
    }
    with open(os.path.join(args.out_dir, "shuffled_label_check.json"), "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps({k: v for k, v in result.items() if k != "history"}, indent=2))


if __name__ == "__main__":
    main()
