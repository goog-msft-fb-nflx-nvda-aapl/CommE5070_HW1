"""Encoder-vs-classifier-head ablation for ssl_frontend (MERT) and
speaker_frontend (ECAPA-TDNN), per the Deep Research follow-up: is the big
ECAPA > MERT gap (95.2% vs 68.4% top1 with our trained MLP) really about
encoder representation quality, or just how well each embedding happens to
pair with an MLP? Extracts frozen, mean-pooled track-level embeddings once,
then compares three heads with the encoder held fixed:
  - linear probe (sklearn LogisticRegression)
  - cosine nearest-centroid (no learned parameters at all)
  - (MLP result already measured via src/train.py — not re-run here)

Also reports embedding-space separability diagnostics (silhouette score,
within/between-class cosine distance) as a secondary, qualitative check —
per the Deep Research responses' caution not to over-index on silhouette
alone.

Usage (run in hw1_ssl_env, torch>=2.6):
    python -m src.analysis_head_swap --data_index_dir data/index --out_dir results/analysis
"""
import argparse
import json
import os

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import silhouette_score, top_k_accuracy_score
from sklearn.preprocessing import StandardScaler, normalize
from torch.utils.data import DataLoader

from src.data.dataset import WaveformEvalDataset
from src.models.ssl_frontend import SSLLinearProbe
from src.models.speaker_frontend import SpeakerEmbeddingProbe

ENCODERS = {
    "mert": lambda: SSLLinearProbe(n_class=20),
    "ecapa": lambda: SpeakerEmbeddingProbe(n_class=20),
}


@torch.no_grad()
def extract_embeddings(model, index_path, device):
    ds = WaveformEvalDataset(index_path)
    loader = DataLoader(ds, batch_size=1, shuffle=False, collate_fn=lambda b: b[0])
    embs, labels = [], []
    for chunks, label, _ in loader:
        chunks = chunks.to(device)
        e = model.embed(chunks).mean(dim=0).cpu().numpy()
        embs.append(e)
        labels.append(label)
    return np.stack(embs), np.array(labels)


def cosine_centroid_classify(X_train, y_train, X_val, n_class):
    Xtr = normalize(X_train)
    centroids = np.stack([Xtr[y_train == c].mean(axis=0) for c in range(n_class)])
    centroids = normalize(centroids)
    Xva = normalize(X_val)
    sims = Xva @ centroids.T  # (N, n_class) cosine similarity
    return sims  # use directly as "probs" for top-k accuracy (monotonic in cosine sim)


def linear_probe(X_train, y_train, X_val, n_class):
    scaler = StandardScaler()
    Xtr = scaler.fit_transform(X_train)
    Xva = scaler.transform(X_val)
    clf = LogisticRegression(max_iter=2000, multi_class="multinomial")
    clf.fit(Xtr, y_train)
    return clf.predict_proba(Xva)


def embedding_diagnostics(X_val, y_val):
    Xn = normalize(X_val)
    sil = float(silhouette_score(Xn, y_val, metric="cosine"))

    within, between = [], []
    sims = Xn @ Xn.T
    n = len(y_val)
    for i in range(n):
        for j in range(i + 1, n):
            if y_val[i] == y_val[j]:
                within.append(sims[i, j])
            else:
                between.append(sims[i, j])
    return {
        "silhouette_cosine": sil,
        "mean_within_class_cosine_sim": float(np.mean(within)),
        "mean_between_class_cosine_sim": float(np.mean(between)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_index_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    with open(f"{args.data_index_dir}/labels.json") as f:
        labels = json.load(f)
    n_class = len(labels)

    results = {}
    for name, ctor in ENCODERS.items():
        print(f"=== {name} ===")
        model = ctor().to(args.device)
        model.eval()

        X_train, y_train = extract_embeddings(model, f"{args.data_index_dir}/train.json", args.device)
        X_val, y_val = extract_embeddings(model, f"{args.data_index_dir}/val.json", args.device)
        print(f"  embeddings: train={X_train.shape} val={X_val.shape}")

        lin_probs = linear_probe(X_train, y_train, X_val, n_class)
        lin_top1 = float((lin_probs.argmax(axis=1) == y_val).mean())
        lin_top3 = float(top_k_accuracy_score(y_val, lin_probs, k=3, labels=range(n_class)))

        cos_sims = cosine_centroid_classify(X_train, y_train, X_val, n_class)
        cos_top1 = float((cos_sims.argmax(axis=1) == y_val).mean())
        cos_top3 = float(top_k_accuracy_score(y_val, cos_sims, k=3, labels=range(n_class)))

        diag = embedding_diagnostics(X_val, y_val)

        results[name] = {
            "linear_probe": {"top1": lin_top1, "top3": lin_top3},
            "cosine_centroid": {"top1": cos_top1, "top3": cos_top3},
            "embedding_diagnostics": diag,
        }
        print(json.dumps(results[name], indent=2))

    with open(os.path.join(args.out_dir, "head_swap_matrix.json"), "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
