"""Song-level evaluation: aggregates chunk-level model outputs per track
(mean-pooled softmax probability across all chunks of a track) into top-1 /
top-3 accuracy and a confusion matrix.
"""
import json
import os

import numpy as np
import torch
from sklearn.metrics import confusion_matrix, top_k_accuracy_score
from torch.utils.data import DataLoader


@torch.no_grad()
def aggregate_predict(model, eval_dataset, device, n_class):
    """eval_dataset yields (chunks[n_chunks, ...], label_idx_or_-1, key).
    Returns (keys, true_labels[N] or None, probs[N, n_class])."""
    model.eval()
    loader = DataLoader(eval_dataset, batch_size=1, shuffle=False, collate_fn=lambda b: b[0])

    keys, trues, probs = [], [], []
    for chunks, label, key in loader:
        chunks = chunks.to(device)
        logits = model(chunks)
        p = torch.softmax(logits, dim=1).mean(dim=0).cpu().numpy()
        keys.append(key)
        trues.append(label)
        probs.append(p)

    probs = np.stack(probs)
    trues = np.array(trues)
    has_labels = bool((trues >= 0).all())
    return keys, (trues if has_labels else None), probs


def compute_metrics(true_labels, probs, n_class):
    preds = probs.argmax(axis=1)
    top1 = float((preds == true_labels).mean())
    top3 = float(top_k_accuracy_score(true_labels, probs, k=3, labels=range(n_class)))
    cm = confusion_matrix(true_labels, preds, labels=range(n_class))
    return {"top1": top1, "top3": top3}, cm


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


def evaluate_metrics_only(model, eval_dataset, device, n_class):
    """Cheap per-epoch check: no plot/file I/O."""
    _, trues, probs = aggregate_predict(model, eval_dataset, device, n_class)
    metrics, _ = compute_metrics(trues, probs, n_class)
    return metrics


def evaluate_and_save(model, eval_dataset, device, labels, out_dir, tag):
    os.makedirs(out_dir, exist_ok=True)
    keys, trues, probs = aggregate_predict(model, eval_dataset, device, len(labels))
    metrics, cm = compute_metrics(trues, probs, len(labels))
    plot_confusion_matrix(cm, labels, os.path.join(out_dir, f"confusion_matrix_{tag}.png"),
                           title=f"{tag} — val confusion matrix (top1={metrics['top1']:.3f})")
    with open(os.path.join(out_dir, f"metrics_{tag}.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    return metrics
