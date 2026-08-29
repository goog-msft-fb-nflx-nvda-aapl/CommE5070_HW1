"""Feature-group ablation for Task 1, per the Deep Research follow-up: which
hand-crafted feature families actually drive the SVM-RBF's 59.3% top1?
Reuses the cached feature matrices from src/classical_ml.py (no
re-extraction — that took ~4.5 hours the first time) and slices out each
named block, matching classical_ml.extract_features's exact concatenation
order: [mfcc, mfcc_delta, mfcc_delta2, chroma, contrast, centroid, bandwidth,
rolloff, zcr, tonnetz], each contributing a (mean, std) pair.

Usage:
    python -m src.analysis_feature_ablation --cache_dir results/task1/feature_cache \
        --out_dir results/analysis
"""
import argparse
import json
import os

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import top_k_accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# (name, per-frame dim) in extract_features's concatenation order; each
# contributes 2*dim columns (mean then std) to the flat vector.
BLOCK_DIMS = [
    ("mfcc", 20),
    ("mfcc_delta", 20),
    ("mfcc_delta2", 20),
    ("chroma", 12),
    ("contrast", 7),
    ("centroid", 1),
    ("bandwidth", 1),
    ("rolloff", 1),
    ("zcr", 1),
    ("tonnetz", 6),
]

GROUPS = {
    "timbre_mfcc": ["mfcc", "mfcc_delta", "mfcc_delta2"],
    "harmonic_tonal": ["chroma", "tonnetz"],
    "spectral_shape": ["centroid", "bandwidth", "rolloff", "zcr"],
    "sub_band_energy": ["contrast"],
}


def block_slices():
    slices = {}
    offset = 0
    for name, dim in BLOCK_DIMS:
        width = 2 * dim  # mean + std
        slices[name] = slice(offset, offset + width)
        offset += width
    return slices, offset


def group_columns(group_blocks, slices):
    cols = []
    for b in group_blocks:
        s = slices[b]
        cols.extend(range(s.start, s.stop))
    return np.array(cols)


def fit_eval(X_train, y_train, X_val, y_val, n_class):
    scaler = StandardScaler()
    Xtr = scaler.fit_transform(X_train)
    Xva = scaler.transform(X_val)
    clf = SVC(kernel="rbf", C=10.0, gamma="scale", probability=True, random_state=0)
    clf.fit(Xtr, y_train)
    probs = clf.predict_proba(Xva)
    preds = probs.argmax(axis=1)
    top1 = float((preds == y_val).mean())
    top3 = float(top_k_accuracy_score(y_val, probs, k=3, labels=range(n_class)))
    return top1, top3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--data_index_dir", default="data/index")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    train_data = np.load(os.path.join(args.cache_dir, "train.npz"), allow_pickle=True)
    val_data = np.load(os.path.join(args.cache_dir, "val.npz"), allow_pickle=True)
    X_train, y_train = train_data["X"], train_data["y"]
    X_val, y_val = val_data["X"], val_data["y"]

    with open(os.path.join(args.data_index_dir, "labels.json")) as f:
        labels = json.load(f)
    n_class = len(labels)

    slices, total_dim = block_slices()
    assert total_dim == X_train.shape[1], f"block dims ({total_dim}) don't match cached feature width ({X_train.shape[1]})"

    results = {"all_features": None, "single_group_only": {}, "leave_one_group_out": {}}

    top1, top3 = fit_eval(X_train, y_train, X_val, y_val, n_class)
    results["all_features"] = {"top1": top1, "top3": top3, "n_dims": X_train.shape[1]}
    print(f"all_features: top1={top1:.4f} top3={top3:.4f} (n={X_train.shape[1]})")

    for group_name, blocks in GROUPS.items():
        cols = group_columns(blocks, slices)
        top1, top3 = fit_eval(X_train[:, cols], y_train, X_val[:, cols], y_val, n_class)
        results["single_group_only"][group_name] = {"top1": top1, "top3": top3, "n_dims": len(cols)}
        print(f"only {group_name}: top1={top1:.4f} top3={top3:.4f} (n={len(cols)})")

    for group_name, blocks in GROUPS.items():
        drop_cols = set(group_columns(blocks, slices).tolist())
        keep_cols = np.array([c for c in range(total_dim) if c not in drop_cols])
        top1, top3 = fit_eval(X_train[:, keep_cols], y_train, X_val[:, keep_cols], y_val, n_class)
        results["leave_one_group_out"][f"minus_{group_name}"] = {"top1": top1, "top3": top3, "n_dims": len(keep_cols)}
        print(f"minus {group_name}: top1={top1:.4f} top3={top3:.4f} (n={len(keep_cols)})")

    # RandomForest permutation importance, aggregated to the group level
    scaler = StandardScaler()
    Xtr = scaler.fit_transform(X_train)
    Xva = scaler.transform(X_val)
    rf = RandomForestClassifier(n_estimators=500, random_state=0, n_jobs=-1)
    rf.fit(Xtr, y_train)
    perm = permutation_importance(rf, Xva, y_val, n_repeats=10, random_state=0, n_jobs=-1)

    group_importance = {}
    for group_name, blocks in GROUPS.items():
        cols = group_columns(blocks, slices)
        group_importance[group_name] = float(perm.importances_mean[cols].sum())
    results["rf_permutation_importance_by_group"] = group_importance
    print("RF permutation importance by group:", json.dumps(group_importance, indent=2))

    with open(os.path.join(args.out_dir, "feature_group_ablation.json"), "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
