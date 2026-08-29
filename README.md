# CommE5070 HW1 — Singer Classification (Artist20)

Work in progress. Structure and instructions will be filled in as each part lands; see `MATERIALS.md` for a running index of results/plots/citations, and `configs/` for per-experiment configs.

## Dataset
[Artist20](http://labrosa.ee.columbia.edu/projects/artistid/): 20 artists, 949 train / 231 val / 233 test tracks, 16kHz mono mp3, full songs. Album-level train/val/test split, per Hsieh et al., "Addressing the confounds of accompaniments in singer identification," ICASSP 2020.

Not included in this repo (per assignment rules) — point `--data_root` at your local copy, with the same layout as provided (`train.json`, `val.json`, `train_val/<artist>/<album>/*.mp3`, `test/*.mp3`).

## Setup
```
pip install -r requirements.txt
```

## Tasks
- Task 1 (traditional ML): `src/classical_ml.py`
- Task 2 (deep learning): `src/train.py` (config-driven, see `configs/`)
- Test-set inference: `src/infer_test.py`
- t-SNE embedding viz: `src/tsne_viz.py`
- Mel-spectrogram + own-recording demo: `src/melspec_demo.py`

## Methods implemented
Ported faithfully from the original authors' released code where noted, inside one shared training/eval pipeline.

| Model | Basis |
|---|---|
| Traditional ML (kNN/SVM/RandomForest) | hand-crafted librosa/torchaudio features |
| CNN (scratch) | minzwon/sota-music-tagging-models |
| CRNN | ZainNasrullah/music-artist-classification-crnn |
| Non-local CNN | ian-k-1217/Fully-Generalized-Non-Local-Network |
| Vocal-separation ablation | bill317996/Singer-identification-in-artist20 |
| SSL-frontend linear probe | pretrained self-supervised audio model (MERT/HuBERT) |
| Speaker-embedding frontend | rssr25/voice-recognition-speak-sing |

(Details, citations, and results per model to follow in `MATERIALS.md` as each lands.)

## Inference / grading
See `readme` (TA-facing) for exact inference instructions once finalized.
