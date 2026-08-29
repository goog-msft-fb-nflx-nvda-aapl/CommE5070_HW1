"""Re-saves a checkpoint containing only trainable parameters, for
frozen-backbone models (ssl_frontend, speaker_frontend). Their `best.pt` as
saved by `src/train.py` currently contains `state_dict()`, which includes
the *frozen* pretrained backbone (MERT / ECAPA-TDNN) — large, third-party,
and re-downloaded fresh at inference time via `from_pretrained`/
`from_hparams` anyway. The assignment forbids uploading other people's model
weights, and a multi-hundred-MB checkpoint doesn't fit the Google Drive
upload path available here (inline base64) — so strip it down to just the
head before distributing.

Usage:
    python -m src.strip_checkpoint --in_path results/speaker_frontend/best.pt \
        --out_path results/speaker_frontend/best_head_only.pt --model speaker_frontend
"""
import argparse

import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_path", required=True)
    ap.add_argument("--out_path", required=True)
    args = ap.parse_args()

    ckpt = torch.load(args.in_path, map_location="cpu", weights_only=True)
    full_state = ckpt["model_state"]

    # keep only the classifier head's own parameters (prefix "head.")
    head_state = {k: v for k, v in full_state.items() if k.startswith("head.")}
    assert head_state, "no 'head.*' keys found — check the checkpoint's key names"

    dropped = len(full_state) - len(head_state)
    print(f"kept {len(head_state)}/{len(full_state)} tensors ({dropped} frozen-backbone tensors dropped)")

    torch.save({"head_state": head_state, "epoch": ckpt.get("epoch"), "metrics": ckpt.get("metrics")}, args.out_path)

    import os

    print(f"wrote {args.out_path} ({os.path.getsize(args.out_path) / 1e6:.2f} MB, "
          f"was {os.path.getsize(args.in_path) / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
