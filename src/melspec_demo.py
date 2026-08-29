"""Mel-spectrogram visualization + top-3 inference for a single user-supplied
audio clip (own singing voice, or a recording of a singer you like).

Usage:
    python -m src.melspec_demo --audio_path /path/to/clip.mp3 \
        --model confound_crnn --checkpoint results/confound_crnn/best.pt \
        --data_index_dir data/index --out_dir results/demo
"""
import argparse
import json
import os

import torch

from src.checkpoint_utils import load_checkpoint
from src.data.dataset import SR, _load_waveform, _log_mel_transform, _fixed_length_chunks, CHUNK_SAMPLES
from src.train import MODEL_REGISTRY


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio_path", required=True)
    ap.add_argument("--model", required=True, choices=list(MODEL_REGISTRY.keys()))
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--data_index_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    with open(f"{args.data_index_dir}/labels.json") as f:
        labels = json.load(f)

    model_fn, kind = MODEL_REGISTRY[args.model]
    model = model_fn(len(labels)).to(args.device)
    load_checkpoint(model, args.checkpoint, args.device)
    model.eval()

    wav = _load_waveform(args.audio_path)

    # mel-spectrogram plot (always, regardless of model input type — this is
    # the assignment's visualization requirement, independent of the model)
    mel, to_db = _log_mel_transform()
    log_mel = to_db(mel(wav.unsqueeze(0))).squeeze(0).numpy()

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(12, 5))
    im = ax.imshow(log_mel, origin="lower", aspect="auto", cmap="magma")
    ax.set_xlabel("frame")
    ax.set_ylabel("mel bin")
    ax.set_title(f"log-mel spectrogram: {os.path.basename(args.audio_path)}")
    fig.colorbar(im, format="%+2.0f dB")
    fig.tight_layout()
    mel_path = os.path.join(args.out_dir, "melspectrogram.png")
    fig.savefig(mel_path, dpi=150)
    plt.close(fig)

    # inference, chunked the same way as val/test
    if kind == "mel":
        n_frames = log_mel.shape[1]
        chunk_frames = 157
        import numpy as np

        n_chunks = max(1, n_frames // chunk_frames)
        arr = log_mel[:, : n_chunks * chunk_frames] if n_frames >= chunk_frames else np.pad(
            log_mel, ((0, 0), (0, chunk_frames - n_frames))
        )
        chunks = torch.from_numpy(arr).float().unfold(1, chunk_frames, chunk_frames).permute(1, 0, 2) \
            if n_frames >= chunk_frames else torch.from_numpy(arr).float().unsqueeze(0)
    else:
        chunks = _fixed_length_chunks(wav, CHUNK_SAMPLES)

    with torch.no_grad():
        logits = model(chunks.to(args.device))
        probs = torch.softmax(logits, dim=1).mean(dim=0).cpu().numpy()

    top3_idx = probs.argsort()[::-1][:3]
    top3 = [(labels[i], float(probs[i])) for i in top3_idx]

    result = {"audio_path": args.audio_path, "model": args.model, "top3": top3}
    with open(os.path.join(args.out_dir, "prediction.json"), "w") as f:
        json.dump(result, f, indent=2)

    print(f"mel-spectrogram: {mel_path}")
    print("top-3 predictions:")
    for name, p in top3:
        print(f"  {name}: {p:.4f}")


if __name__ == "__main__":
    main()
