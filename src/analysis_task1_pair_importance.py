"""Pair-conditional feature-group importance for Task 1, per the round-6
Deep Research follow-up (deep_research/round6_error_analysis_and_report_depth/
{prompt,response_*}.md) — all four engines gave the same recipe: global
permutation importance (already computed in `src/analysis_feature_ablation.py`)
answers "what matters overall," not "what actually separates artist A from
artist B specifically." This restricts the analysis to the SVM(RBF)
classifier's most-confused artist pairs and measures each feature group's
group-wise permutation importance (Breiman, "Random Forests," Machine
Learning 2001, sec. 10 — the permutation-importance procedure this
generalizes group-wise) within that binary subproblem specifically.

Procedure per confused pair (artist_i, artist_j):
  1. Restrict to that pair's TRAIN tracks only, fit a fresh binary SVM(RBF)
     (same hyperparameters as `src/classical_ml.py`) with its own
     StandardScaler fit on that binary subset (not the global one — this is
     a self-contained binary subproblem, not a slice of the global model).
  2. Evaluate baseline accuracy on that pair's VAL tracks.
  3. For each feature group (same GROUPS as `analysis_feature_ablation.py`):
     shuffle that group's columns across the val rows (breaking their
     row-track association while preserving every other column), re-predict,
     repeat `--n_repeats` times, and report the mean accuracy drop
     (baseline - permuted) as that group's importance for distinguishing
     this specific pair.

Confused pairs are chosen from the SVM(RBF)'s own val confusion matrix
(symmetric off-diagonal count (i,j)+(j,i)), refit here from the cached
feature matrices rather than reusing a saved confusion-matrix image.

Usage:
    python -m src.analysis_task1_pair_importance --cache_dir results/task1/feature_cache \
        --data_index_dir data/index --out_dir results/analysis --top_k_pairs 5
"""
import argparse
import json
import os

import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# same block layout as src/classical_ml.py::extract_features /
# src/analysis_feature_ablation.py — kept in sync deliberately, not imported,
# since classical_ml.py doesn't expose these as constants.
BLOCK_DIMS = [
    ("mfcc", 20), ("mfcc_delta", 20), ("mfcc_delta2", 20),
    ("chroma", 12), ("contrast", 7), ("centroid", 1),
    ("bandwidth", 1), ("rolloff", 1), ("zcr", 1), ("tonnetz", 6),
]
GROUPS = {
    "timbre_mfcc": ["mfcc", "mfcc_delta", "mfcc_delta2"],
    "harmonic_tonal": ["chroma", "tonnetz"],
    "spectral_shape": ["centroid", "bandwidth", "rolloff", "zcr"],
    "sub_band_energy": ["contrast"],
}
SEED = 42


def block_slices():
    slices, offset = {}, 0
    for name, dim in BLOCK_DIMS:
        width = 2 * dim
        slices[name] = slice(offset, offset + width)
        offset += width
    return slices, offset


def group_columns(group_blocks, slices):
    cols = []
    for b in group_blocks:
        s = slices[b]
        cols.extend(range(s.start, s.stop))
    return np.array(cols)


def top_confused_pairs(X_train, y_train, X_val, y_val, n_class, k):
    scaler = StandardScaler()
    Xtr, Xva = scaler.fit_transform(X_train), scaler.transform(X_val)
    clf = SVC(kernel="rbf", C=10.0, gamma="scale", probability=True, random_state=0)
    clf.fit(Xtr, y_train)
    preds = clf.predict(Xva)
    cm = confusion_matrix(y_val, preds, labels=range(n_class))
    pair_counts = {}
    for i in range(n_class):
        for j in range(n_class):
            if i == j:
                continue
            pair = tuple(sorted((i, j)))
            pair_counts[pair] = pair_counts.get(pair, 0) + cm[i, j]
    ranked = sorted(pair_counts.items(), key=lambda kv: kv[1], reverse=True)
    return [p for p, c in ranked if c > 0][:k], cm


def pair_group_importance(X_train, y_train, X_val, y_val, pair, slices, total_dim, n_repeats, rng):
    i, j = pair
    train_mask = np.isin(y_train, pair)
    val_mask = np.isin(y_val, pair)
    Xtr, ytr = X_train[train_mask], y_train[train_mask]
    Xva, yva = X_val[val_mask], y_val[val_mask]

    scaler = StandardScaler()
    Xtr_s = scaler.fit_transform(Xtr)
    Xva_s = scaler.transform(Xva)

    clf = SVC(kernel="rbf", C=10.0, gamma="scale", random_state=0)
    clf.fit(Xtr_s, ytr)
    baseline_acc = float((clf.predict(Xva_s) == yva).mean())

    group_drop = {}
    for group_name, blocks in GROUPS.items():
        cols = group_columns(blocks, slices)
        drops = []
        for _ in range(n_repeats):
            Xva_perm = Xva_s.copy()
            perm_idx = rng.permutation(len(Xva_perm))
            Xva_perm[:, cols] = Xva_perm[perm_idx][:, cols]
            acc = float((clf.predict(Xva_perm) == yva).mean())
            drops.append(baseline_acc - acc)
        group_drop[group_name] = {"mean_accuracy_drop": float(np.mean(drops)),
                                   "std_accuracy_drop": float(np.std(drops))}

    return baseline_acc, group_drop, int(len(ytr)), int(len(yva))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache_dir", required=True)
    ap.add_argument("--data_index_dir", default="data/index")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--top_k_pairs", type=int, default=5)
    ap.add_argument("--n_repeats", type=int, default=30)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    rng = np.random.default_rng(SEED)

    train_data = np.load(os.path.join(args.cache_dir, "train.npz"), allow_pickle=True)
    val_data = np.load(os.path.join(args.cache_dir, "val.npz"), allow_pickle=True)
    X_train, y_train = train_data["X"], train_data["y"]
    X_val, y_val = val_data["X"], val_data["y"]

    with open(os.path.join(args.data_index_dir, "labels.json")) as f:
        labels = json.load(f)
    n_class = len(labels)

    slices, total_dim = block_slices()
    assert total_dim == X_train.shape[1], f"block dims ({total_dim}) != cached feature width ({X_train.shape[1]})"

    pairs, full_cm = top_confused_pairs(X_train, y_train, X_val, y_val, n_class, args.top_k_pairs)
    print(f"top {len(pairs)} confused pairs (by symmetric val misclassification count):")

    results = {"full_confusion_matrix": full_cm.tolist(), "labels": labels, "pairs": []}
    for i, j in pairs:
        name_i, name_j = labels[i], labels[j]
        baseline_acc, group_drop, n_train, n_val = pair_group_importance(
            X_train, y_train, X_val, y_val, (i, j), slices, total_dim, args.n_repeats, rng
        )
        top_group = max(group_drop.items(), key=lambda kv: kv[1]["mean_accuracy_drop"])
        print(f"  {name_i} vs {name_j}: n_train={n_train} n_val={n_val} "
              f"binary_baseline_acc={baseline_acc:.4f} "
              f"most_important_group={top_group[0]} (drop={top_group[1]['mean_accuracy_drop']:.4f})")
        results["pairs"].append({
            "artist_i": name_i, "artist_j": name_j,
            "n_train": n_train, "n_val": n_val,
            "binary_baseline_accuracy": baseline_acc,
            "group_accuracy_drop": group_drop,
            "most_important_group": top_group[0],
        })

    with open(os.path.join(args.out_dir, "task1_pair_importance.json"), "w") as f:
        json.dump(results, f, indent=2)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    group_names = list(GROUPS.keys())
    pair_labels = [f"{p['artist_i']}\nvs\n{p['artist_j']}" for p in results["pairs"]]
    data = np.array([[p["group_accuracy_drop"][g]["mean_accuracy_drop"] for g in group_names]
                      for p in results["pairs"]])

    fig, ax = plt.subplots(figsize=(2.2 * len(pair_labels) + 2, 5))
    x = np.arange(len(pair_labels))
    width = 0.8 / len(group_names)
    for gi, g in enumerate(group_names):
        ax.bar(x + gi * width, data[:, gi], width, label=g)
    ax.set_xticks(x + width * (len(group_names) - 1) / 2)
    ax.set_xticklabels(pair_labels, fontsize=8)
    ax.set_ylabel("mean accuracy drop when group is permuted\n(binary val subset)")
    ax.set_title(f"pair-conditional feature-group importance (top {len(pairs)} confused pairs)")
    ax.legend(fontsize=8)
    ax.axhline(0, color="black", linewidth=0.8)
    fig.tight_layout()
    plot_path = os.path.join(args.out_dir, "task1_pair_importance.png")
    fig.savefig(plot_path, dpi=150)
    print(f"wrote {os.path.join(args.out_dir, 'task1_pair_importance.json')} and {plot_path}")


if __name__ == "__main__":
    main()
