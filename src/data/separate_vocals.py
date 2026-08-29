"""Vocal-separation preprocessing ablation.

Directly implements the assignment doc's "Source separation" augmentation
item, and mirrors bill317996/Singer-identification-in-artist20's (Method
2-2) approach of stripping backing instrumentation before classification —
their repo used `open-unmix`; we use `demucs` (actively maintained,
comparable quality) for the same purpose: isolate the vocal stem so the
classifier can't lean on album/production/instrumentation cues (the
"confound" the ICASSP 2020 paper targets), only the singer's voice.

Runs demucs (htdemucs, 4-stem: vocals/drums/bass/other) over every track in
an index file, caches the extracted vocals-only stem as a 16kHz mono wav
under `--vocals_root`, and writes a new index with `path` remapped to the
separated file (everything else — label, split — unchanged), so it's a
drop-in swap for src/train.py / src/classical_ml.py.

Usage:
    python -m src.data.separate_vocals --data_index_dir data/index --split train \
        --vocals_root data/vocals --out_index data/index_vocals/train.json
"""
import argparse
import json
import os

import torch
import torchaudio
from demucs.apply import apply_model
from demucs.pretrained import get_model

SR = 16000


def separate_track(model, demucs_sr, path, device):
    wav, sr = torchaudio.load(path)
    if wav.size(0) == 1:
        wav = wav.repeat(2, 1)  # demucs expects stereo
    if sr != demucs_sr:
        wav = torchaudio.functional.resample(wav, sr, demucs_sr)

    with torch.no_grad():
        sources = apply_model(model, wav.unsqueeze(0).to(device), device=device)[0]  # (stems, 2, T)
    vocals = sources[model.sources.index("vocals")].mean(dim=0, keepdim=True).cpu()  # mono

    if demucs_sr != SR:
        vocals = torchaudio.functional.resample(vocals, demucs_sr, SR)
    return vocals


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_index_dir", required=True)
    ap.add_argument("--split", required=True, choices=["train", "val", "test"])
    ap.add_argument("--vocals_root", required=True)
    ap.add_argument("--out_index", required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    os.makedirs(args.vocals_root, exist_ok=True)
    os.makedirs(os.path.dirname(args.out_index), exist_ok=True)

    with open(os.path.join(args.data_index_dir, f"{args.split}.json")) as f:
        records = json.load(f)

    model = get_model("htdemucs").to(args.device)
    model.eval()
    demucs_sr = model.samplerate

    new_records = []
    for i, rec in enumerate(records):
        src_path = rec["path"]
        out_path = os.path.join(args.vocals_root, f"{i:05d}.wav")

        if not os.path.exists(out_path):
            vocals = separate_track(model, demucs_sr, src_path, args.device)
            torchaudio.save(out_path, vocals, SR)

        new_rec = dict(rec)
        new_rec["path"] = out_path
        new_records.append(new_rec)

        if (i + 1) % 25 == 0:
            print(f"  {i + 1}/{len(records)}")

    with open(args.out_index, "w") as f:
        json.dump(new_records, f, indent=2)

    print(f"wrote {len(new_records)} vocals-only records to {args.out_index}")


if __name__ == "__main__":
    main()
