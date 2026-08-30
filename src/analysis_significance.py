"""Paired-bootstrap CIs and McNemar's exact test over already-trained models'
val predictions, per the round-6 Deep Research follow-up
(deep_research/round6_error_analysis_and_report_depth/{prompt,response_*}.md)
— all four engines independently converged on this as the top-priority
addition, given how small this project's reported ensemble gains have been
(0.853 -> 0.857 -> 0.861 top1 across three ensemble versions) relative to
the ~0.43pp one val track is worth on our 231-track val set.

For each named "system" (a single model, or an ensemble = model list +
integer weights straight from a results/ensemble*/ensemble_result.json),
reconstructs per-track top1 correctness via src.ensemble.get_probs, then for
each requested pair reports: accuracy point estimate for both, the paired
bootstrap 95% CI on the accuracy difference (same resampled track indices
for both systems), and McNemar's exact (binomial) test p-value on the
discordant pairs.

Usage:
    python -m src.analysis_significance --data_index_dir data/index \
        --out_dir results/analysis
"""
import argparse
import itertools
import json
import os

import numpy as np
import torch
from scipy.stats import binomtest

from src.ensemble import get_probs

N_BOOTSTRAP = 5000
SEED = 42


def load_ensemble_spec(path):
    with open(path) as f:
        d = json.load(f)
    return list(zip(d["models"], d["weights"]))


# (name, spec) — spec is a list of (model_name, weight) pairs; weight=1 for
# a lone model. Comparisons below reference these by name.
SYSTEMS = {
    "sota_crnn": [("sota_crnn", 1)],
    "speaker_frontend": [("speaker_frontend", 1)],
    "sota_crnn_wide": [("sota_crnn_wide", 1)],
    "singer_senet": [("singer_senet", 1)],
    "ensemble2_9model": load_ensemble_spec("results/ensemble2/ensemble_result.json"),
    "ensemble3_12model": load_ensemble_spec("results/ensemble3/ensemble_result.json"),
    "ensemble4_14model": load_ensemble_spec("results/ensemble4/ensemble_result.json"),
}

COMPARISONS = [
    ("sota_crnn_wide", "singer_senet"),
    ("sota_crnn_wide", "ensemble4_14model"),
    ("ensemble2_9model", "ensemble3_12model"),
    ("ensemble3_12model", "ensemble4_14model"),
    ("ensemble2_9model", "ensemble4_14model"),
]


def system_probs_and_trues(spec, data_index_dir, device, n_class):
    """spec: list of (model_name, weight). Returns (trues, mixed_probs)."""
    trues_ref, keys_ref = None, None
    total = None
    weight_sum = 0
    for model_name, weight in spec:
        if weight == 0:
            continue
        keys, trues, probs = get_probs(model_name, f"results/{model_name}/best.pt",
                                        data_index_dir, "val", device, n_class)
        if trues_ref is None:
            trues_ref, keys_ref = trues, keys
        else:
            assert list(trues) == list(trues_ref), f"{model_name}'s val order doesn't match"
        total = probs * weight if total is None else total + probs * weight
        weight_sum += weight
    return trues_ref, total / weight_sum


def paired_bootstrap_ci(correct_a, correct_b, n_iter=N_BOOTSTRAP, seed=SEED):
    rng = np.random.default_rng(seed)
    n = len(correct_a)
    diffs = np.empty(n_iter)
    for i in range(n_iter):
        idx = rng.integers(0, n, size=n)
        diffs[i] = correct_a[idx].mean() - correct_b[idx].mean()
    return float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))


def mcnemar_exact(correct_a, correct_b):
    a_right_b_wrong = int(np.sum(correct_a & ~correct_b))
    a_wrong_b_right = int(np.sum(~correct_a & correct_b))
    n_discordant = a_right_b_wrong + a_wrong_b_right
    if n_discordant == 0:
        return a_right_b_wrong, a_wrong_b_right, 1.0
    p = binomtest(min(a_right_b_wrong, a_wrong_b_right), n_discordant, 0.5).pvalue
    return a_right_b_wrong, a_wrong_b_right, float(p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_index_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    with open(f"{args.data_index_dir}/labels.json") as f:
        n_class = len(json.load(f))

    print("computing val probs for every system in this run's comparison set...")
    needed = sorted({name for pair in COMPARISONS for name in pair})
    correctness = {}
    accuracy = {}
    for name in needed:
        trues, probs = system_probs_and_trues(SYSTEMS[name], args.data_index_dir, args.device, n_class)
        preds = probs.argmax(axis=1)
        correct = (preds == trues)
        correctness[name] = correct
        accuracy[name] = float(correct.mean())
        print(f"  {name}: top1={accuracy[name]:.4f} (n={len(correct)})")

    results = {"n_val_tracks": len(next(iter(correctness.values()))),
               "n_bootstrap": N_BOOTSTRAP, "accuracy": accuracy, "comparisons": []}

    for a, b in COMPARISONS:
        ca, cb = correctness[a], correctness[b]
        ci_lo, ci_hi = paired_bootstrap_ci(ca.astype(float), cb.astype(float))
        a_right_b_wrong, a_wrong_b_right, p = mcnemar_exact(ca, cb)
        entry = {
            "a": a, "b": b,
            "acc_a": accuracy[a], "acc_b": accuracy[b],
            "acc_diff": accuracy[a] - accuracy[b],
            "bootstrap_95ci_diff": [ci_lo, ci_hi],
            "mcnemar_a_right_b_wrong": a_right_b_wrong,
            "mcnemar_a_wrong_b_right": a_wrong_b_right,
            "mcnemar_exact_p": p,
            "significant_at_0.05": bool(p < 0.05),
        }
        results["comparisons"].append(entry)
        print(f"{a} vs {b}: diff={entry['acc_diff']:+.4f}, "
              f"bootstrap 95% CI=[{ci_lo:+.4f}, {ci_hi:+.4f}], "
              f"McNemar p={p:.4f} ({'significant' if p < 0.05 else 'NOT significant'})")

    with open(os.path.join(args.out_dir, "significance.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"wrote {os.path.join(args.out_dir, 'significance.json')}")


if __name__ == "__main__":
    main()
