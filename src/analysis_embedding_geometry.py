"""Embedding-geometry analysis beyond t-SNE, per the round-6 Deep Research
follow-up (deep_research/round6_error_analysis_and_report_depth/
{prompt,response_*}.md) — all four engines independently noted that a t-SNE
plot alone is a qualitative/illustrative tool (its axes and distances are
not meaningful in absolute terms — see van der Maaten & Hinton, "Visualizing
Data using t-SNE," JMLR 2008, sec. 4 on why perplexity/inter-cluster
distances shouldn't be over-interpreted), and is not on its own a rigorous
representation-quality metric.

Computes two standard, well-established diagnostics directly on each
model's track-level embedding space (mean-pooled across a track's chunks,
same convention as `src/tsne_viz.py`):
  - Silhouette score (Rousseeuw, "Silhouettes: a graphical aid to the
    interpretation and validation of cluster analysis," J. Comput. Appl.
    Math. 1987) over cosine distance, using the true artist label as the
    cluster assignment — standard scikit-learn `silhouette_score`, no
    modification.
  - Intra-class vs. inter-class mean pairwise cosine distance: for every
    pair of tracks, whether they share an artist label or not, averaged
    separately — a lower intra-class / higher inter-class distance
    indicates the embedding space separates artists better, independent of
    downstream classifier accuracy.

Usage:
    python -m src.analysis_embedding_geometry --data_index_dir data/index \
        --models sota_crnn_wide:results/sota_crnn_wide/best.pt \
                 singer_senet:results/singer_senet/best.pt \
                 speaker_frontend:results/speaker_frontend/best.pt \
                 confound_crnn:results/confound_crnn/best.pt \
        --out_dir results/analysis
"""
import argparse
import json
import os

import numpy as np
import torch
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_distances
from torch.utils.data import DataLoader

from src.checkpoint_utils import load_checkpoint
from src.data.dataset import CHUNK_SAMPLES_10S, MelChunkEvalDataset, WaveformEvalDataset
from src.train import MODEL_REGISTRY


@torch.no_grad()
def collect_embeddings(model, eval_dataset, device):
    model.eval()
    loader = DataLoader(eval_dataset, batch_size=1, shuffle=False, collate_fn=lambda b: b[0])
    embs, labels = [], []
    for chunks, label, _ in loader:
        chunks = chunks.to(device)
        e = model.embed(chunks).mean(dim=0).cpu().numpy()
        embs.append(e)
        labels.append(label)
    return np.stack(embs), np.array(labels)


def intra_inter_cosine_distance(embs, label_idx):
    dist = cosine_distances(embs)
    n = len(label_idx)
    same = label_idx[:, None] == label_idx[None, :]
    iu = np.triu_indices(n, k=1)  # upper triangle, exclude diagonal (self-distance=0)
    same_pairs = same[iu]
    dists = dist[iu]
    intra = dists[same_pairs]
    inter = dists[~same_pairs]
    return float(intra.mean()), float(inter.mean()), int(intra.size), int(inter.size)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_index_dir", required=True)
    ap.add_argument("--models", nargs="+", required=True, help="name:checkpoint_path pairs")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--split", default="val", choices=["train", "val"])
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    with open(f"{args.data_index_dir}/labels.json") as f:
        labels = json.load(f)

    split_path = f"{args.data_index_dir}/{args.split}.json"
    results = {}

    for spec in args.models:
        name, ckpt = spec.split(":", 1)
        model_fn, kind = MODEL_REGISTRY[name]
        model = model_fn(len(labels)).to(args.device)
        load_checkpoint(model, ckpt, args.device)

        if kind == "mel":
            ds = MelChunkEvalDataset(split_path)
        elif kind == "wave10s":
            ds = WaveformEvalDataset(split_path, chunk_samples=CHUNK_SAMPLES_10S)
        else:
            ds = WaveformEvalDataset(split_path)

        embs, label_idx = collect_embeddings(model, ds, args.device)
        sil = float(silhouette_score(embs, label_idx, metric="cosine"))
        intra, inter, n_intra, n_inter = intra_inter_cosine_distance(embs, label_idx)

        print(f"  {name}: silhouette={sil:.4f} intra_cos_dist={intra:.4f} "
              f"inter_cos_dist={inter:.4f} gap={inter - intra:.4f}")

        results[name] = {
            "silhouette_score_cosine": sil,
            "intra_class_mean_cosine_distance": intra,
            "inter_class_mean_cosine_distance": inter,
            "inter_minus_intra_gap": inter - intra,
            "n_intra_pairs": n_intra, "n_inter_pairs": n_inter,
            "n_tracks": len(label_idx), "embed_dim": int(embs.shape[1]),
        }

    with open(os.path.join(args.out_dir, "embedding_geometry.json"), "w") as f:
        json.dump(results, f, indent=2)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = list(results.keys())
    fig, axes = plt.subplots(1, 2, figsize=(5 * len(names) / 2 + 3, 4.5))

    sils = [results[n]["silhouette_score_cosine"] for n in names]
    axes[0].bar(names, sils)
    axes[0].set_ylabel("silhouette score (cosine)")
    axes[0].set_title("cluster separation (higher = better)")
    axes[0].tick_params(axis="x", rotation=30)

    x = np.arange(len(names))
    width = 0.35
    intras = [results[n]["intra_class_mean_cosine_distance"] for n in names]
    inters = [results[n]["inter_class_mean_cosine_distance"] for n in names]
    axes[1].bar(x - width / 2, intras, width, label="intra-class (same artist)")
    axes[1].bar(x + width / 2, inters, width, label="inter-class (different artist)")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(names, rotation=30)
    axes[1].set_ylabel("mean cosine distance")
    axes[1].set_title("intra- vs. inter-class distance")
    axes[1].legend(fontsize=8)

    fig.tight_layout()
    plot_path = os.path.join(args.out_dir, "embedding_geometry.png")
    fig.savefig(plot_path, dpi=150)
    print(f"wrote {os.path.join(args.out_dir, 'embedding_geometry.json')} and {plot_path}")


if __name__ == "__main__":
    main()
