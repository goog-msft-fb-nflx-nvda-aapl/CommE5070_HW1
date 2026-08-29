"""Unified, config-driven trainer for Task 2's deep-learning models.

Usage:
    python -m src.train --model confound_crnn --data_index_dir data/index \
        --out_dir results/confound_crnn --epochs 60 --batch_size 32
"""
import argparse
import json
import os
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.data.dataset import (
    MelChunkEvalDataset,
    MelChunkTrainDataset,
    WaveformEvalDataset,
    WaveformTrainDataset,
)
from src.evaluate import evaluate_and_save, evaluate_metrics_only
from src.models.confound_crnn import CRNN2D_elu2
from src.models.crnn_zain import CRNN2D
from src.models.nonlocal_fgnl import CRNN_FGNL
from src.models.sota_cnn import CRNN as SotaCRNN

def _build_ssl_frontend(n_class):
    from src.models.ssl_frontend import SSLLinearProbe

    return SSLLinearProbe(n_class=n_class)


def _build_speaker_frontend(n_class):
    from src.models.speaker_frontend import SpeakerEmbeddingProbe

    return SpeakerEmbeddingProbe(n_class=n_class)


MODEL_REGISTRY = {
    "confound_crnn": (lambda n_class: CRNN2D_elu2(n_class=n_class), "mel"),
    "crnn_zain": (lambda n_class: CRNN2D(n_class=n_class), "mel"),
    "fgnl": (lambda n_class: CRNN_FGNL(n_class=n_class), "mel"),
    "sota_crnn": (lambda n_class: SotaCRNN(n_class=n_class), "wave"),
    # ssl_frontend / speaker_frontend need torch>=2.6 (HF safetensors-only
    # torch.load policy) — run these with the separate `hw1_ssl_env` conda
    # env on gsm-gpu2, not the main `hw1_singer_env`. See
    # src/models/ssl_frontend.py's docstring / EXPERIMENT_LOG.md.
    "ssl_frontend": (_build_ssl_frontend, "wave"),
    "speaker_frontend": (_build_speaker_frontend, "wave"),
}


def build_datasets(model_kind, index_dir):
    train_path = os.path.join(index_dir, "train.json")
    val_path = os.path.join(index_dir, "val.json")
    if model_kind == "mel":
        return MelChunkTrainDataset(train_path), MelChunkEvalDataset(val_path)
    return WaveformTrainDataset(train_path), WaveformEvalDataset(val_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(MODEL_REGISTRY.keys()))
    ap.add_argument("--data_index_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    with open(os.path.join(args.data_index_dir, "labels.json")) as f:
        labels = json.load(f)

    model_fn, kind = MODEL_REGISTRY[args.model]
    model = model_fn(len(labels)).to(args.device)

    train_ds, val_ds = build_datasets(kind, args.data_index_dir)
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, drop_last=True, persistent_workers=args.num_workers > 0,
    )

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    best_top1 = -1.0
    best_epoch = -1
    history = []
    ckpt_path = os.path.join(args.out_dir, "best.pt")

    for epoch in range(args.epochs):
        model.train()
        t0 = time.time()
        total_loss = 0.0
        n_batches = 0
        for x, y in train_loader:
            x, y = x.to(args.device), y.to(args.device)
            opt.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            opt.step()
            total_loss += loss.item()
            n_batches += 1

        train_loss = total_loss / max(1, n_batches)
        metrics = evaluate_metrics_only(model, val_ds, args.device, len(labels))
        elapsed = time.time() - t0

        print(f"[{args.model}] epoch={epoch} loss={train_loss:.4f} "
              f"val_top1={metrics['top1']:.4f} val_top3={metrics['top3']:.4f} ({elapsed:.0f}s)")
        history.append({"epoch": epoch, "train_loss": train_loss, **metrics})

        if metrics["top1"] > best_top1:
            best_top1 = metrics["top1"]
            best_epoch = epoch
            torch.save({"model_state": model.state_dict(), "epoch": epoch, "metrics": metrics}, ckpt_path)

        if epoch - best_epoch >= args.patience:
            print(f"early stop at epoch {epoch} (best={best_epoch}, top1={best_top1:.4f})")
            break

    with open(os.path.join(args.out_dir, "history.json"), "w") as f:
        json.dump(history, f, indent=2)

    # reload best checkpoint for the final confusion-matrix / metrics artifact
    model.load_state_dict(torch.load(ckpt_path, map_location=args.device, weights_only=True)["model_state"])
    final_metrics = evaluate_and_save(model, val_ds, args.device, labels, args.out_dir, tag=args.model)

    with open(os.path.join(args.out_dir, "summary.json"), "w") as f:
        json.dump({"model": args.model, "best_epoch": best_epoch, **final_metrics}, f, indent=2)

    print(f"done. best_epoch={best_epoch} best_top1={final_metrics['top1']:.4f}. checkpoint={ckpt_path}")


if __name__ == "__main__":
    main()
