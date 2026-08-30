"""Diagnostic analyses over the from-scratch CRNN family, per the Deep
Research follow-up (deep_research/round2_sota_context_and_perartifact_ablations/{prompt,response_*}.md):

1. Ensemble diversity: pairwise Cohen's kappa, disagreement rate, and an
   oracle-ensemble upper bound across confound_crnn / crnn_zain / sota_crnn /
   fgnl — tests whether the 57-67% band reflects a shared data/input
   bottleneck (highly correlated errors) or genuine architectural diversity
   (complementary errors, ensemble headroom).
2. Vocal-separation error attribution: per-artist accuracy delta and a
   McNemar-style fix/damage/consensus transition matrix between
   confound_crnn (raw mixture) and confound_crnn_vocals (demucs vocals-only).

Usage:
    python -m src.analysis_ensemble --data_index_dir data/index --out_dir results/analysis
"""
import argparse
import json
import os

import numpy as np
import torch

from src.checkpoint_utils import load_checkpoint
from src.data.dataset import MelChunkEvalDataset, WaveformEvalDataset
from src.evaluate import aggregate_predict
from src.train import MODEL_REGISTRY

CRNN_FAMILY = ["confound_crnn", "crnn_zain", "sota_crnn", "fgnl"]


def get_predictions(model_name, checkpoint_path, data_index_dir, device, index_dir_override=None):
    with open(f"{data_index_dir}/labels.json") as f:
        labels = json.load(f)
    model_fn, kind = MODEL_REGISTRY[model_name]
    model = model_fn(len(labels)).to(device)
    load_checkpoint(model, checkpoint_path, device)
    model.eval()

    val_index_dir = index_dir_override or data_index_dir
    val_path = f"{val_index_dir}/val.json"
    ds = MelChunkEvalDataset(val_path) if kind == "mel" else WaveformEvalDataset(val_path)
    keys, trues, probs = aggregate_predict(model, ds, device, len(labels))
    preds = probs.argmax(axis=1)
    return keys, trues, preds, labels


def cohens_kappa(a, b, n_classes):
    n = len(a)
    po = np.mean(a == b)
    pe = 0.0
    for c in range(n_classes):
        pe += (np.mean(a == c)) * (np.mean(b == c))
    if pe == 1.0:
        return 1.0
    return (po - pe) / (1 - pe)


def ensemble_diversity(preds_by_model, trues, labels):
    names = list(preds_by_model.keys())
    n_classes = len(labels)
    correctness = {name: (preds_by_model[name] == trues) for name in names}

    pairwise = {}
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            kappa = cohens_kappa(preds_by_model[a], preds_by_model[b], n_classes)
            disagreement = float(np.mean(preds_by_model[a] != preds_by_model[b]))
            both_wrong = float(np.mean((~correctness[a]) & (~correctness[b])))
            pairwise[f"{a}_vs_{b}"] = {
                "cohens_kappa": kappa,
                "disagreement_rate": disagreement,
                "double_fault_rate": both_wrong,
            }

    any_correct = np.any(np.stack(list(correctness.values())), axis=0)
    oracle_acc = float(np.mean(any_correct))

    # simple probability-average ensemble is not available here (only argmax
    # preds saved) — majority vote instead
    stacked_preds = np.stack([preds_by_model[n] for n in names])  # (M, N)
    majority = np.array([np.bincount(stacked_preds[:, i], minlength=n_classes).argmax()
                          for i in range(stacked_preds.shape[1])])
    majority_acc = float(np.mean(majority == trues))

    individual_acc = {name: float(np.mean(correctness[name])) for name in names}

    return {
        "individual_top1": individual_acc,
        "pairwise": pairwise,
        "oracle_ensemble_top1": oracle_acc,
        "majority_vote_ensemble_top1": majority_acc,
    }


def vocal_separation_attribution(keys_raw, trues_raw, preds_raw, keys_voc, trues_voc, preds_voc, labels):
    # keys differ (raw vs vocals index point to different file paths) but are
    # the *same* val tracks in the *same* order (val.json / index_vocals/val.json
    # were built from the same source list) — assert same true-label sequence.
    assert list(trues_raw) == list(trues_voc), "raw/vocals val sets are out of sync"

    correct_raw = preds_raw == trues_raw
    correct_voc = preds_voc == trues_voc

    transition = {
        "consensus_correct": int(np.sum(correct_raw & correct_voc)),
        "separation_fix": int(np.sum(~correct_raw & correct_voc)),
        "separation_damage": int(np.sum(correct_raw & ~correct_voc)),
        "consensus_incorrect": int(np.sum(~correct_raw & ~correct_voc)),
    }

    per_artist = {}
    for i, name in enumerate(labels):
        mask = trues_raw == i
        if mask.sum() == 0:
            continue
        acc_raw = float(np.mean(correct_raw[mask]))
        acc_voc = float(np.mean(correct_voc[mask]))
        per_artist[name] = {"raw": acc_raw, "vocals": acc_voc, "delta": acc_voc - acc_raw, "n": int(mask.sum())}

    return {"transition_matrix": transition, "per_artist": per_artist}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_index_dir", required=True)
    ap.add_argument("--vocals_index_dir", default="data/index_vocals")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print("=== gathering CRNN-family predictions ===")
    preds_by_model, trues_ref, labels = {}, None, None
    for name in CRNN_FAMILY:
        ckpt = f"results/{name}/best.pt"
        keys, trues, preds, labels = get_predictions(name, ckpt, args.data_index_dir, args.device)
        preds_by_model[name] = preds
        if trues_ref is None:
            trues_ref = trues
        else:
            assert list(trues) == list(trues_ref), f"{name}'s val order doesn't match"
        print(f"{name}: top1={np.mean(preds == trues):.4f}")

    diversity = ensemble_diversity(preds_by_model, trues_ref, labels)
    with open(os.path.join(args.out_dir, "ensemble_diversity.json"), "w") as f:
        json.dump(diversity, f, indent=2)
    print(json.dumps(diversity, indent=2))

    print("\n=== vocal-separation attribution ===")
    keys_raw, trues_raw, preds_raw, _ = get_predictions(
        "confound_crnn", "results/confound_crnn/best.pt", args.data_index_dir, args.device
    )
    keys_voc, trues_voc, preds_voc, _ = get_predictions(
        "confound_crnn", "results/confound_crnn_vocals/best.pt", args.vocals_index_dir, args.device
    )
    attribution = vocal_separation_attribution(keys_raw, trues_raw, preds_raw, keys_voc, trues_voc, preds_voc, labels)
    with open(os.path.join(args.out_dir, "vocal_separation_attribution.json"), "w") as f:
        json.dump(attribution, f, indent=2)
    print(json.dumps(attribution, indent=2))


if __name__ == "__main__":
    main()
