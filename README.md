# CommE5070 HW1 — Singer Classification (Artist20)

Work in progress. Structure and instructions will be filled in as each part lands; see `MATERIALS.md` for a running index of results/plots/citations, and `configs/` for per-experiment configs.

## Dataset
[Artist20](http://labrosa.ee.columbia.edu/projects/artistid/): 20 artists, 949 train / 231 val / 233 test tracks, 16kHz mono mp3, full songs. Album-level train/val/test split, per Hsieh et al., "Addressing the confounds of accompaniments in singer identification," ICASSP 2020.

Not included in this repo (per assignment rules) — point `--data_root` at your local copy, with the same layout as provided (`train.json`, `val.json`, `train_val/<artist>/<album>/*.mp3`, `test/*.mp3`).

## Setup
```
pip install -r requirements.txt
```

## Usage
```bash
# 1. build the track-level manifest from the raw dataset layout
python -m src.data.prepare_index --data_root /path/to/artist20 --out_dir data/index

# 2. Task 1: traditional ML (kNN / SVM / RandomForest)
python -m src.classical_ml --data_index_dir data/index --out_dir results/task1

# 3. Task 2: deep learning, pick a model from src/train.py's MODEL_REGISTRY
#    (confound_crnn, crnn_zain, sota_crnn, fgnl, ssl_frontend, speaker_frontend)
python -m src.train --model confound_crnn --data_index_dir data/index --out_dir results/confound_crnn

# 4. test-set predictions, in the assignment's {"001": [top1,top2,top3], ...} schema
python -m src.infer_test --model confound_crnn --checkpoint results/confound_crnn/best.pt \
    --data_index_dir data/index --out_path STUDENT_ID.json

# 5. t-SNE of a trained model's track embeddings
python -m src.tsne_viz --model confound_crnn --checkpoint results/confound_crnn/best.pt \
    --data_index_dir data/index --out_path results/confound_crnn/tsne.png

# 6. mel-spectrogram + inference demo on a single clip (own voice / a singer you like)
python -m src.melspec_demo --audio_path /path/to/clip.mp3 --model confound_crnn \
    --checkpoint results/confound_crnn/best.pt --data_index_dir data/index --out_dir results/demo
```

`ssl_frontend` and `speaker_frontend` need `torch>=2.6` (HF's `safetensors`-only
`torch.load` policy blocks older `transformers`/`torch` combos on some pretrained
checkpoints) — see `src/models/ssl_frontend.py`'s docstring if installing fresh.

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
