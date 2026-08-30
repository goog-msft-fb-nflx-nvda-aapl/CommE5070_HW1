"""Extracts the accompaniment (non-vocal: drums+bass+other) stem for the
training split, to pair with the vocals already extracted by
src/data/separate_vocals.py. Together these enable cross-song vocal/
instrumental remixing (src/data/remix_dataset.py) — per Hsieh et al.,
ICASSP 2020 (bill317996/Singer-identification-in-artist20), the technique
that gave their CRNN its largest reported gain (+7-8pp song-level F1),
confirmed as the top from-scratch-compatible lever by two independent Deep
Research follow-ups (deep_research/round3_from_scratch_improvement/response_*.md).

Train split only — remix pairs are constructed only from training data, per
"never use val/test for training".

Usage:
    python -m src.data.separate_accompaniment --data_index_dir data/index \
        --accompaniment_root data/accompaniment_train --out_index data/index_vocals/train_accompaniment.json
"""
import argparse
import json
import os

import torch
import torchaudio
from demucs.apply import apply_model
from demucs.pretrained import get_model

SR = 16000


def separate_accompaniment(model, demucs_sr, path, device):
    wav, sr = torchaudio.load(path)
    if wav.size(0) == 1:
        wav = wav.repeat(2, 1)
    if sr != demucs_sr:
        wav = torchaudio.functional.resample(wav, sr, demucs_sr)

    with torch.no_grad():
        sources = apply_model(model, wav.unsqueeze(0).to(device), device=device)[0]  # (stems, 2, T)
    vocal_idx = model.sources.index("vocals")
    accompaniment = torch.stack([s for i, s in enumerate(sources) if i != vocal_idx]).sum(dim=0)
    accompaniment = accompaniment.mean(dim=0, keepdim=True).cpu()  # mono

    if demucs_sr != SR:
        accompaniment = torchaudio.functional.resample(accompaniment, demucs_sr, SR)
    return accompaniment


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_index_dir", required=True)
    ap.add_argument("--accompaniment_root", required=True)
    ap.add_argument("--out_index", required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    os.makedirs(args.accompaniment_root, exist_ok=True)
    os.makedirs(os.path.dirname(args.out_index), exist_ok=True)

    with open(os.path.join(args.data_index_dir, "train.json")) as f:
        records = json.load(f)

    model = get_model("htdemucs").to(args.device)
    model.eval()
    demucs_sr = model.samplerate

    new_records = []
    for i, rec in enumerate(records):
        out_path = os.path.join(args.accompaniment_root, f"{i:05d}.wav")

        if not os.path.exists(out_path):
            accompaniment = separate_accompaniment(model, demucs_sr, rec["path"], args.device)
            torchaudio.save(out_path, accompaniment, SR)

        new_rec = dict(rec)
        new_rec["path"] = out_path
        new_records.append(new_rec)

        if (i + 1) % 25 == 0:
            print(f"  {i + 1}/{len(records)}")

    with open(args.out_index, "w") as f:
        json.dump(new_records, f, indent=2)

    print(f"wrote {len(new_records)} accompaniment-only records to {args.out_index}")


if __name__ == "__main__":
    main()
