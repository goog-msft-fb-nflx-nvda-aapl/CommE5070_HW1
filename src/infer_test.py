"""Produces the assignment's `R13921031.json` test-set submission:
    {"001": [top1, top2, top3], "002": [...], ...}
from a trained checkpoint, matching readme.md / test_pred.json's schema
exactly (string-keyed track ids, artist-name strings as predictions, ordered
by descending confidence).

Usage:
    python -m src.infer_test --model confound_crnn --checkpoint results/confound_crnn/best.pt \
        --data_index_dir data/index --out_path R13921031.json
"""
import argparse
import json

import torch

from src.checkpoint_utils import load_checkpoint
from src.data.dataset import MelChunkEvalDataset, WaveformEvalDataset
from src.evaluate import aggregate_predict
from src.train import MODEL_REGISTRY


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(MODEL_REGISTRY.keys()))
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--data_index_dir", required=True)
    ap.add_argument("--out_path", required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    with open(f"{args.data_index_dir}/labels.json") as f:
        labels = json.load(f)

    model_fn, kind = MODEL_REGISTRY[args.model]
    model = model_fn(len(labels)).to(args.device)
    load_checkpoint(model, args.checkpoint, args.device)
    model.eval()

    test_path = f"{args.data_index_dir}/test.json"
    if kind == "mel":
        test_ds = MelChunkEvalDataset(test_path, has_labels=False)
    else:
        test_ds = WaveformEvalDataset(test_path, has_labels=False)

    keys, _, probs = aggregate_predict(model, test_ds, args.device, len(labels))

    top3_idx = probs.argsort(axis=1)[:, ::-1][:, :3]
    result = {key: [labels[i] for i in idxs] for key, idxs in zip(keys, top3_idx)}

    # sanity: keys must be "001".."233" zero-padded ids, matching readme.md's convention
    with open(args.out_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"wrote {len(result)} predictions to {args.out_path}")
    print("sample:", dict(list(result.items())[:2]))


if __name__ == "__main__":
    main()
