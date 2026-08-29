"""t-SNE visualization of a trained model's track-level embeddings
(mean-pooled across a track's chunks via `model.embed(...)`), colored by
singer.

Usage:
    python -m src.tsne_viz --model confound_crnn --checkpoint results/confound_crnn/best.pt \
        --data_index_dir data/index --out_path results/confound_crnn/tsne.png
"""
import argparse
import json

import numpy as np
import torch
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader

from src.data.dataset import MelChunkEvalDataset, WaveformEvalDataset
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(MODEL_REGISTRY.keys()))
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--data_index_dir", required=True)
    ap.add_argument("--out_path", required=True)
    ap.add_argument("--split", default="val", choices=["train", "val"])
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    with open(f"{args.data_index_dir}/labels.json") as f:
        labels = json.load(f)

    model_fn, kind = MODEL_REGISTRY[args.model]
    model = model_fn(len(labels)).to(args.device)
    ckpt = torch.load(args.checkpoint, map_location=args.device, weights_only=True)
    model.load_state_dict(ckpt["model_state"])

    split_path = f"{args.data_index_dir}/{args.split}.json"
    ds = MelChunkEvalDataset(split_path) if kind == "mel" else WaveformEvalDataset(split_path)

    embs, label_idx = collect_embeddings(model, ds, args.device)
    coords = TSNE(n_components=2, perplexity=min(30, max(5, len(embs) // 4)), random_state=0, init="pca").fit_transform(embs)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 9))
    cmap = plt.get_cmap("tab20")
    for i, name in enumerate(labels):
        mask = label_idx == i
        ax.scatter(coords[mask, 0], coords[mask, 1], s=18, color=cmap(i / 20), label=name)
    ax.set_title(f"t-SNE of {args.model} track embeddings ({args.split} set)")
    ax.legend(fontsize=6, ncol=2, loc="center left", bbox_to_anchor=(1.0, 0.5))
    fig.tight_layout()
    fig.savefig(args.out_path, dpi=150, bbox_inches="tight")
    print(f"wrote {args.out_path}")


if __name__ == "__main__":
    main()
