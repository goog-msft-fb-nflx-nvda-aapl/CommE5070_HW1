"""Bonus / exploratory Baseline 2: zero-shot singer classification with a
pretrained audio-language model (Qwen2-Audio-7B-Instruct), per the
assignment doc's "Audio Language Model" baseline
(Kong et al./Xu et al., Audio Flamingo 3 / Qwen-Audio). No training: the
model is prompted with the audio clip plus the closed list of 20 candidate
artist names and asked to rank its top 3 guesses.

This is a best-effort exploratory baseline, not a trained model — chat-model
outputs are free text, so ranking quality depends on the model reliably
naming exactly the given candidates in the requested format; a small amount
of output parsing/fallback is needed and is not expected to be as robust as
a trained classifier's calibrated probabilities.

Usage:
    python -m src.audio_llm_baseline --data_index_dir data/index --split val \
        --out_path results/audio_llm/val_predictions.json --limit 40
"""
import argparse
import json
import re

import torch
import torchaudio

MODEL_ID = "Qwen/Qwen2-Audio-7B-Instruct"
SR = 16000


def build_prompt(labels):
    options = ", ".join(labels)
    return (
        "You are a music expert. Listen to this short audio clip and identify which "
        f"artist/singer is performing, choosing ONLY from this exact list: {options}. "
        "Reply with your top 3 guesses as a comma-separated list, most likely first, "
        "using the exact spelling from the list above. Reply with nothing else."
    )


def parse_top3(text, labels):
    label_set = {l.lower(): l for l in labels}
    found = []
    # try comma/newline-separated parse first
    for tok in re.split(r"[,\n]", text):
        key = tok.strip().lower()
        if key in label_set and label_set[key] not in found:
            found.append(label_set[key])
    if len(found) < 3:
        # fallback: substring search, preserving list order of first mention
        for name in labels:
            if name.lower() in text.lower() and name not in found:
                found.append(name)
    return (found + [labels[0]] * 3)[:3]  # pad defensively, never crash on a bad response


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_index_dir", required=True)
    ap.add_argument("--split", default="val", choices=["val", "test"])
    ap.add_argument("--out_path", required=True)
    ap.add_argument("--limit", type=int, default=None, help="cap number of tracks (this is slow)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    import os

    os.makedirs(os.path.dirname(args.out_path) or ".", exist_ok=True)

    from transformers import Qwen2AudioForConditionalGeneration, AutoProcessor

    with open(f"{args.data_index_dir}/labels.json") as f:
        labels = json.load(f)
    with open(f"{args.data_index_dir}/{args.split}.json") as f:
        records = json.load(f)
    if args.limit:
        records = records[: args.limit]

    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = Qwen2AudioForConditionalGeneration.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16, device_map=args.device
    )
    model.eval()

    prompt_text = None  # built per-labels below, constant across tracks
    system_prompt = build_prompt(labels)

    results = {}
    for i, rec in enumerate(records):
        wav, sr = torchaudio.load(rec["path"])
        if wav.size(0) > 1:
            wav = wav.mean(dim=0, keepdim=True)
        if sr != SR:
            wav = torchaudio.functional.resample(wav, sr, SR)
        clip = wav.squeeze(0)[: SR * 15].numpy()  # first 15s, keeps inference tractable

        conversation = [
            {"role": "user", "content": [{"type": "audio", "audio_url": "clip"}, {"type": "text", "text": system_prompt}]}
        ]
        chat_prompt = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
        inputs = processor(text=chat_prompt, audio=[clip], sampling_rate=SR, return_tensors="pt").to(args.device)

        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=64)
        text = processor.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0]

        key = rec.get("id", rec.get("path"))
        results[key] = {"top3": parse_top3(text, labels), "raw": text, "true": rec.get("label")}

        if (i + 1) % 10 == 0:
            print(f"  {i + 1}/{len(records)}")

    with open(args.out_path, "w") as f:
        json.dump(results, f, indent=2)

    if all("true" in r and r["true"] for r in results.values()):
        top1 = sum(r["true"] == r["top3"][0] for r in results.values()) / len(results)
        top3 = sum(r["true"] in r["top3"] for r in results.values()) / len(results)
        print(f"zero-shot top1={top1:.4f} top3={top3:.4f} (n={len(results)})")

    print(f"wrote {args.out_path}")


if __name__ == "__main__":
    main()
