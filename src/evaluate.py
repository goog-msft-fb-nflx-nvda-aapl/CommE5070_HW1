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
def aggregate_predict(model, eval_dataset, device, n_class, tta_shifts=None):
    """eval_dataset yields (chunks[n_chunks, ...], label_idx_or_-1, key).
    Returns (keys, true_labels[N] or None, probs[N, n_class]).

    `tta_shifts`: optional list of fractional time-shifts (e.g. [0.25, -0.25])
    for test-time augmentation. Each chunk is additionally evaluated after a
    circular roll along its last (time) axis by that fraction of its length,
    and softmax probabilities are averaged across shifts *and* chunks — a
    temporal-jitter multi-crop ensemble, modality-agnostic (works identically
    for precomputed log-mel chunks [n_mels, T] and raw waveform chunks
    [n_samples]). A roll preserves all information (unlike a mask/crop), so
    it's safe to use at test time even though the model never saw shifted
    inputs during training. Per lecture02_classification.md's data
    augmentation section — extending the idea to inference time.
    """
    model.eval()
    loader = DataLoader(eval_dataset, batch_size=1, shuffle=False, collate_fn=lambda b: b[0])
    shifts = [0.0] + list(tta_shifts) if tta_shifts else [0.0]

    keys, trues, probs = [], [], []
    for chunks, label, key in loader:
        chunks = chunks.to(device)
        view_probs = []
        for frac in shifts:
            shift = int(round(frac * chunks.shape[-1]))
            view = torch.roll(chunks, shifts=shift, dims=-1) if shift != 0 else chunks
            logits = model(view)
            view_probs.append(torch.softmax(logits, dim=1))
        p = torch.cat(view_probs, dim=0).mean(dim=0).cpu().numpy()
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


def evaluate_metrics_only(model, eval_dataset, device, n_class, tta_shifts=None):
    """Cheap per-epoch check: no plot/file I/O."""
    _, trues, probs = aggregate_predict(model, eval_dataset, device, n_class, tta_shifts=tta_shifts)
    metrics, _ = compute_metrics(trues, probs, n_class)
    return metrics


def evaluate_and_save(model, eval_dataset, device, labels, out_dir, tag, tta_shifts=None):
    os.makedirs(out_dir, exist_ok=True)
    keys, trues, probs = aggregate_predict(model, eval_dataset, device, len(labels), tta_shifts=tta_shifts)
    metrics, cm = compute_metrics(trues, probs, len(labels))
    plot_confusion_matrix(cm, labels, os.path.join(out_dir, f"confusion_matrix_{tag}.png"),
                           title=f"{tag} — val confusion matrix (top1={metrics['top1']:.3f})")
    with open(os.path.join(out_dir, f"metrics_{tag}.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    return metrics
