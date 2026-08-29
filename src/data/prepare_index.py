"""Builds a flat track-level manifest from the artist20 dataset layout.

Expected layout (see the assignment's readme.md):
    <data_root>/train.json           # list of relative paths, e.g. "./train_val/aerosmith/Draw_the_Line/01-....mp3"
    <data_root>/val.json             # same format
    <data_root>/train_val/<artist>/<album>/*.mp3
    <data_root>/test/<001-233>.mp3

Usage:
    python -m src.data.prepare_index --data_root /path/to/artist20 --out_dir data/index
"""
import argparse
import json
import os


def artist_from_path(rel_path: str) -> str:
    # "./train_val/aerosmith/Draw_the_Line/01-....mp3" -> "aerosmith"
    parts = [p for p in rel_path.split("/") if p not in (".", "")]
    idx = parts.index("train_val")
    return parts[idx + 1]


def build_split(data_root: str, json_name: str) -> list[dict]:
    with open(os.path.join(data_root, json_name)) as f:
        rel_paths = json.load(f)
    records = []
    for rel_path in rel_paths:
        abs_path = os.path.normpath(os.path.join(data_root, rel_path.lstrip("./")))
        records.append({"path": abs_path, "label": artist_from_path(rel_path)})
    return records


def build_test(data_root: str) -> list[dict]:
    test_dir = os.path.join(data_root, "test")
    records = []
    for fname in sorted(os.listdir(test_dir)):
        if not fname.endswith(".mp3"):
            continue
        track_id = os.path.splitext(fname)[0]
        records.append({"path": os.path.join(test_dir, fname), "id": track_id})
    return records


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    train = build_split(args.data_root, "train.json")
    val = build_split(args.data_root, "val.json")
    test = build_test(args.data_root)

    labels = sorted({r["label"] for r in train})
    assert len(labels) == 20, f"expected 20 artists, got {len(labels)}: {labels}"
    assert set(r["label"] for r in val) <= set(labels), "val has artists not seen in train"

    label2idx = {label: i for i, label in enumerate(labels)}

    for split_name, records in [("train", train), ("val", val)]:
        for r in records:
            r["label_idx"] = label2idx[r["label"]]

    with open(os.path.join(args.out_dir, "labels.json"), "w") as f:
        json.dump(labels, f, indent=2)
    with open(os.path.join(args.out_dir, "train.json"), "w") as f:
        json.dump(train, f, indent=2)
    with open(os.path.join(args.out_dir, "val.json"), "w") as f:
        json.dump(val, f, indent=2)
    with open(os.path.join(args.out_dir, "test.json"), "w") as f:
        json.dump(test, f, indent=2)

    print(f"train={len(train)} val={len(val)} test={len(test)} labels={len(labels)}")


if __name__ == "__main__":
    main()
