"""Weighted-average ensemble over trained from-scratch models. Directly
motivated by the user's prior run on this assignment: 3-model ensemble beat
the best individual model by 0.714->0.825 val top1. Grid-searches integer
weights on val, then applies to val (report) and optionally test (submission).

Usage:
    python -m src.ensemble --data_index_dir data/index \
        --models confound_crnn:results/confound_crnn/best.pt \
                 crnn_zain:results/crnn_zain/best.pt \
                 sota_crnn:results/sota_crnn/best.pt \
                 short_chunk_cnn:results/short_chunk_cnn/best.pt \
        --out_dir results/ensemble
"""
import argparse
import itertools
import json
import os

import numpy as np
import torch

from src.checkpoint_utils import load_checkpoint
from src.data.dataset import MelChunkEvalDataset, WaveformEvalDataset
from src.evaluate import aggregate_predict, compute_metrics, plot_confusion_matrix
from src.train import MODEL_REGISTRY


def get_probs(model_name, checkpoint, index_dir, split, device, n_class):
    model_fn, kind = MODEL_REGISTRY[model_name]
    model = model_fn(n_class).to(device)
    load_checkpoint(model, checkpoint, device)
    model.eval()

    path = f"{index_dir}/{split}.json"
    has_labels = split != "test"
    ds = (MelChunkEvalDataset(path, has_labels=has_labels) if kind == "mel"
          else WaveformEvalDataset(path, has_labels=has_labels))
    keys, trues, probs = aggregate_predict(model, ds, device, n_class)
    return keys, trues, probs


def grid_search_weights(all_probs, trues, n_class, max_weight=3):
    """All_probs: list of (N, n_class) arrays, one per model. Try every
    integer weight combination in [0, max_weight] (excluding all-zero),
    return the best by val top1."""
    n_models = len(all_probs)
    best_weights, best_top1 = None, -1.0
    for weights in itertools.product(range(max_weight + 1), repeat=n_models):
        if sum(weights) == 0:
            continue
        mix = sum(w * p for w, p in zip(weights, all_probs)) / sum(weights)
        preds = mix.argmax(axis=1)
        top1 = float((preds == trues).mean())
        if top1 > best_top1:
            best_top1 = top1
            best_weights = weights
    return best_weights, best_top1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_index_dir", required=True)
    ap.add_argument("--models", nargs="+", required=True, help="name:checkpoint_path pairs")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--max_weight", type=int, default=3)
    ap.add_argument("--test_out_path", default=None, help="if set, also write test-set predictions here")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    with open(f"{args.data_index_dir}/labels.json") as f:
        labels = json.load(f)
    n_class = len(labels)

    parsed = [tuple(m.split(":", 1)) for m in args.models]
    names = [n for n, _ in parsed]

    print(f"gathering val predictions for: {names}")
    val_probs, trues_ref, keys_ref = [], None, None
    for name, ckpt in parsed:
        keys, trues, probs = get_probs(name, ckpt, args.data_index_dir, "val", args.device, n_class)
        if trues_ref is None:
            trues_ref, keys_ref = trues, keys
        else:
            assert list(trues) == list(trues_ref), f"{name}'s val order doesn't match — can't ensemble"
        val_probs.append(probs)
        top1 = float((probs.argmax(axis=1) == trues).mean())
        print(f"  {name}: val_top1={top1:.4f}")

    weights, best_top1 = grid_search_weights(val_probs, trues_ref, n_class, args.max_weight)
    print(f"best weights: {dict(zip(names, weights))} -> val_top1={best_top1:.4f}")

    mix = sum(w * p for w, p in zip(weights, val_probs)) / sum(weights)
    metrics, cm = compute_metrics(trues_ref, mix, n_class)
    plot_confusion_matrix(cm, labels, os.path.join(args.out_dir, "confusion_matrix_ensemble.png"),
                           title=f"ensemble — val confusion matrix (top1={metrics['top1']:.3f})")

    result = {"models": names, "weights": list(weights), "val_metrics": metrics,
              "individual_val_top1": {n: float((p.argmax(axis=1) == trues_ref).mean())
                                       for n, p in zip(names, val_probs)}}
    with open(os.path.join(args.out_dir, "ensemble_result.json"), "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))

    if args.test_out_path:
        print("gathering test predictions...")
        test_probs = []
        test_keys_ref = None
        for name, ckpt in parsed:
            keys, _, probs = get_probs(name, ckpt, args.data_index_dir, "test", args.device, n_class)
            if test_keys_ref is None:
                test_keys_ref = keys
            test_probs.append(probs)
        mix_test = sum(w * p for w, p in zip(weights, test_probs)) / sum(weights)
        top3_idx = mix_test.argsort(axis=1)[:, ::-1][:, :3]
        out = {key: [labels[i] for i in idxs] for key, idxs in zip(test_keys_ref, top3_idx)}
        with open(args.test_out_path, "w") as f:
            json.dump(out, f, indent=2)
        print(f"wrote {len(out)} test predictions to {args.test_out_path}")


if __name__ == "__main__":
    main()
