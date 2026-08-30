"""Calibration analysis (ECE, multiclass Brier score, reliability diagrams)
over already-trained models' val predictions, per the round-6 Deep Research
follow-up (deep_research/round6_error_analysis_and_report_depth/
{prompt,response_*}.md) — all four engines flagged this as a natural Part-3
addition, directly motivated by the own-voice demo's qualitative contrast
(models with similar accuracy gave very differently *confident* wrong
answers on the out-of-distribution Sia clip; this quantifies that across
the whole val set rather than one anecdote).

For each named system (reuses `src.analysis_significance`'s SYSTEMS spec —
a single model, or an ensemble = model list + integer weights straight from
a results/ensemble*/ensemble_result.json), computes:
  - ECE (Expected Calibration Error): val split into 15 equal-width
    confidence bins on the top1 predicted probability; per-bin
    |accuracy - mean confidence|, weighted by bin size.
  - Multiclass Brier score: mean squared error between the full predicted
    probability vector and the one-hot true label, averaged over tracks
    (standard multiclass generalization of the binary Brier score).
  - Reliability diagram (accuracy vs. confidence per bin, bar chart against
    the y=x perfect-calibration line).

Usage:
    python -m src.analysis_calibration --data_index_dir data/index \
        --out_dir results/analysis
"""
import argparse
import json
import os

import numpy as np
import torch

from src.analysis_significance import SYSTEMS, system_probs_and_trues

N_BINS = 15


def expected_calibration_error(confidences, correct, n_bins=N_BINS):
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    bins = []
    n = len(confidences)
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        in_bin = (confidences > lo) & (confidences <= hi) if i > 0 else (confidences >= lo) & (confidences <= hi)
        count = int(in_bin.sum())
        if count == 0:
            bins.append({"range": [float(lo), float(hi)], "count": 0, "accuracy": None, "confidence": None})
            continue
        bin_acc = float(correct[in_bin].mean())
        bin_conf = float(confidences[in_bin].mean())
        ece += (count / n) * abs(bin_acc - bin_conf)
        bins.append({"range": [float(lo), float(hi)], "count": count, "accuracy": bin_acc, "confidence": bin_conf})
    return float(ece), bins


def multiclass_brier_score(probs, trues, n_class):
    one_hot = np.zeros_like(probs)
    one_hot[np.arange(len(trues)), trues] = 1.0
    return float(np.mean(np.sum((probs - one_hot) ** 2, axis=1)))


def plot_reliability_diagrams(all_bins, out_path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(all_bins)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4.5), squeeze=False)
    for ax, (name, bins) in zip(axes[0], all_bins.items()):
        centers = [(b["range"][0] + b["range"][1]) / 2 for b in bins if b["count"] > 0]
        accs = [b["accuracy"] for b in bins if b["count"] > 0]
        counts = [b["count"] for b in bins if b["count"] > 0]
        ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="perfect calibration")
        ax.bar(centers, accs, width=1.0 / N_BINS, alpha=0.7, edgecolor="black",
               label="observed accuracy")
        for c, a, n_pts in zip(centers, accs, counts):
            ax.annotate(str(n_pts), (c, a), textcoords="offset points", xytext=(0, 3), fontsize=7, ha="center")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel("confidence (top1 predicted probability)")
        ax.set_ylabel("accuracy")
        ax.set_title(name)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_index_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--systems", nargs="+",
                     default=["sota_crnn", "sota_crnn_wide", "ensemble4_14model", "speaker_frontend"],
                     help="names from src.analysis_significance.SYSTEMS")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    with open(f"{args.data_index_dir}/labels.json") as f:
        n_class = len(json.load(f))

    results = {"n_bins": N_BINS, "systems": {}}
    all_bins = {}

    for name in args.systems:
        trues, probs = system_probs_and_trues(SYSTEMS[name], args.data_index_dir, args.device, n_class)
        preds = probs.argmax(axis=1)
        correct = (preds == trues)
        confidences = probs.max(axis=1)

        ece, bins = expected_calibration_error(confidences, correct)
        brier = multiclass_brier_score(probs, trues, n_class)
        acc = float(correct.mean())
        mean_conf = float(confidences.mean())

        print(f"  {name}: acc={acc:.4f} mean_confidence={mean_conf:.4f} "
              f"ECE={ece:.4f} Brier={brier:.4f}")

        results["systems"][name] = {
            "accuracy": acc, "mean_confidence": mean_conf,
            "ece": ece, "brier_score": brier, "bins": bins,
        }
        all_bins[name] = bins

    with open(os.path.join(args.out_dir, "calibration.json"), "w") as f:
        json.dump(results, f, indent=2)

    plot_path = os.path.join(args.out_dir, "reliability_diagrams.png")
    plot_reliability_diagrams(all_bins, plot_path)

    print(f"wrote {os.path.join(args.out_dir, 'calibration.json')} and {plot_path}")


if __name__ == "__main__":
    main()
