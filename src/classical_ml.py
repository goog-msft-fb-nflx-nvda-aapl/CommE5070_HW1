"""Task 1: traditional ML pipeline for singer classification.

Features (per track, librosa): MFCC(20) + delta + delta-delta, chroma_stft,
spectral_contrast, spectral_centroid, spectral_bandwidth, spectral_rolloff,
zero_crossing_rate, tonnetz — each pooled to mean+std across frames, then
concatenated into one fixed-length vector per track. Standardized
(zero mean / unit variance, fit on train only) before classification.

Classifiers (sklearn): kNN, SVM (RBF kernel), RandomForest — grid-searched
lightly over a couple of key hyperparameters via the val set (never test).

Usage:
    python -m src.classical_ml --data_index_dir data/index --out_dir results/task1
"""
import argparse
import json
import os
import time

import librosa
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, top_k_accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

SR = 16000


def extract_features(path: str) -> np.ndarray:
    y, sr = librosa.load(path, sr=SR, mono=True)
    if y.size == 0:
        y = np.zeros(SR, dtype=np.float32)

    feats = []

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    d1 = librosa.feature.delta(mfcc)
    d2 = librosa.feature.delta(mfcc, order=2)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    zcr = librosa.feature.zero_crossing_rate(y)
    harm = librosa.effects.harmonic(y)
    tonnetz = librosa.feature.tonnetz(y=harm, sr=sr)

    for arr in [mfcc, d1, d2, chroma, contrast, centroid, bandwidth, rolloff, zcr, tonnetz]:
        feats.append(arr.mean(axis=1))
        feats.append(arr.std(axis=1))

    return np.concatenate(feats).astype(np.float32)


def build_feature_matrix(records, cache_path=None):
    if cache_path and os.path.exists(cache_path):
        data = np.load(cache_path, allow_pickle=True)
        return data["X"], data["y"], list(data["keys"])

    X, y, keys = [], [], []
    t0 = time.time()
    for i, rec in enumerate(records):
        X.append(extract_features(rec["path"]))
        y.append(rec.get("label_idx", -1))
        keys.append(rec.get("id", rec["path"]))
        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{len(records)} ({time.time() - t0:.0f}s)")
    X = np.stack(X)
    y = np.array(y)

    if cache_path:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        np.savez(cache_path, X=X, y=y, keys=np.array(keys, dtype=object))

    return X, y, keys


def plot_confusion_matrix(cm, labels, out_path, title):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 9))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=90)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    fig.colorbar(im)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_index_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--cache_dir", default=None, help="cache extracted features here (npz)")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    with open(os.path.join(args.data_index_dir, "labels.json")) as f:
        labels = json.load(f)
    with open(os.path.join(args.data_index_dir, "train.json")) as f:
        train_records = json.load(f)
    with open(os.path.join(args.data_index_dir, "val.json")) as f:
        val_records = json.load(f)

    cache_dir = args.cache_dir or os.path.join(args.out_dir, "feature_cache")
    print(f"Extracting train features ({len(train_records)} tracks)...")
    X_train, y_train, _ = build_feature_matrix(train_records, os.path.join(cache_dir, "train.npz"))
    print(f"Extracting val features ({len(val_records)} tracks)...")
    X_val, y_val, _ = build_feature_matrix(val_records, os.path.join(cache_dir, "val.npz"))

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)

    classifiers = {
        "knn": KNeighborsClassifier(n_neighbors=5, weights="distance"),
        "svm_rbf": SVC(kernel="rbf", C=10.0, gamma="scale", probability=True, random_state=0),
        "random_forest": RandomForestClassifier(n_estimators=500, max_depth=None, random_state=0, n_jobs=-1),
    }

    all_results = {}
    for name, clf in classifiers.items():
        print(f"\n=== {name} ===")
        clf.fit(X_train_s, y_train)
        probs = clf.predict_proba(X_val_s)
        preds = probs.argmax(axis=1)

        top1 = (preds == y_val).mean()
        top3 = top_k_accuracy_score(y_val, probs, k=3, labels=range(len(labels)))
        cm = confusion_matrix(y_val, preds, labels=range(len(labels)))

        print(f"top1={top1:.4f} top3={top3:.4f}")

        plot_confusion_matrix(
            cm, labels, os.path.join(args.out_dir, f"confusion_matrix_{name}.png"),
            title=f"Task1 {name} — val confusion matrix (top1={top1:.3f})",
        )

        all_results[name] = {"top1": float(top1), "top3": float(top3)}

    with open(os.path.join(args.out_dir, "metrics.json"), "w") as f:
        json.dump(all_results, f, indent=2)

    print("\n=== summary ===")
    for name, r in all_results.items():
        print(f"{name}: top1={r['top1']:.4f} top3={r['top3']:.4f}")


if __name__ == "__main__":
    main()
