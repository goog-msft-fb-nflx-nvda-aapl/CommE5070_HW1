# CommE5070 HW1 — Singer Classification (Artist20)

## Problem statement

This is our submission for CommE5070 ("Deep Learning for Music Analysis and
Generation") HW1: singer classification on the [Artist20](http://labrosa.ee.columbia.edu/projects/artistid/)
dataset — 20 artists, 949 train / 231 val / 233 test tracks, 16kHz mono mp3,
full songs (not pre-chunked), split at the album level per Hsieh et al.,
"Addressing the confounds of accompaniments in singer identification,"
ICASSP 2020, specifically so a model can't cheat by learning
album/production characteristics instead of the singer's voice.

The assignment has two tasks, evaluated separately:

- **Task 1 — traditional ML.** Hand-crafted audio features (MFCC, chroma,
  spectral contrast/centroid/bandwidth/rolloff, ZCR, tonnetz, etc. — no
  learned/pretrained feature extractors) into a classical classifier
  (kNN/SVM/RandomForest). Per the TA's clarification (`TA_discussion.md`
  #3), this task's accuracy **does not count toward the grade** — it's
  graded on whether the feature/importance analysis is reasonable and
  thorough, not the raw number. We treat the ablation and permutation-
  importance writeup in `MATERIALS.md` as the actual deliverable here, not
  the classifier's val accuracy.
- **Task 2 — deep learning from scratch.** Both the encoder and the
  classifier head must be trained from scratch on Artist20's train split
  only — per the TA's clarification (`TA_discussion.md` #4), a pretrained
  encoder (frozen or fine-tuned) is explicitly **baseline-only, not eligible
  as the graded submission**, even if it scores higher. We implement several
  from-scratch architectures (faithfully ported from published reference
  code where noted), plus two pretrained-encoder baselines for comparison,
  and submit a weighted ensemble of from-scratch models as the graded
  result.

**Evaluation.** Every model predicts at the *song* level: a track is split
into non-overlapping 5s chunks, each chunk gets a softmax over the 20
artists, and the chunk-level probabilities are mean-pooled into one
track-level prediction (`src/evaluate.py::aggregate_predict`). We report
top1 and top3 accuracy on val throughout, and submit top3 predictions per
test track in the assignment's required JSON schema.

**Deliverables**: this repo (code + configs + full experiment log), test-set
predictions (`R13921031.json`, top-3 per track), and a report assembled from
`MATERIALS.md`. Test-set predictions are made only for Task 2, per the
assignment's Task 1 spec (val-only).

## Dataset

Not included in this repo (per assignment rules) — point `--data_root` at
your local copy, with the same layout as provided (`train.json`, `val.json`,
`train_val/<artist>/<album>/*.mp3`, `test/*.mp3`).

## Hardware used

We trained on a Linux server with 4x H200 NVL-class GPUs and 128 CPU cores,
running most jobs in parallel across GPUs via `tmux` + `conda`. Nothing in
the code assumes this specific setup — a single modern GPU with enough VRAM
for one model at a time will reproduce every result here, just serially
rather than in parallel; the largest single model comfortably fits on
consumer-GPU-class VRAM. See `requirements.txt` for the software
environment (all libraries are pinned to minimum versions, nothing bespoke
to our hardware).

## Setup

```
pip install -r requirements.txt
```

(`torch>=2.6` is required across the board — two of the models use
pretrained checkpoints that fail to load under HF's safetensors-only
`torch.load` policy on older torch. If disk/dependency constraints make a
single unified env awkward, those two models — `ssl_frontend` and
`speaker_frontend` — are also fine to install and run in a separate env
from the from-scratch models; see `src/models/ssl_frontend.py`'s docstring.)

## Usage

```bash
# 1. build the track-level manifest from the raw dataset layout
python -m src.data.prepare_index --data_root /path/to/artist20 --out_dir data/index

# 2. Task 1: traditional ML (kNN / SVM / RandomForest)
python -m src.classical_ml --data_index_dir data/index --out_dir results/task1

# 3. Task 2: deep learning, pick a model from src/train.py's MODEL_REGISTRY
python -m src.train --model sota_crnn_wide --data_index_dir data/index --out_dir results/sota_crnn_wide

# 4. test-set predictions, in the assignment's {"001": [top1,top2,top3], ...} schema
#    (see `readme` for the actual graded-submission command, which is an
#    ensemble of several trained models rather than a single --model/--checkpoint pair)
python -m src.infer_test --model sota_crnn_wide --checkpoint results/sota_crnn_wide/best.pt \
    --data_index_dir data/index --out_path R13921031.json

# 5. t-SNE of a trained model's track embeddings
python -m src.tsne_viz --model sota_crnn_wide --checkpoint results/sota_crnn_wide/best.pt \
    --data_index_dir data/index --out_path results/sota_crnn_wide/tsne.png

# 6. mel-spectrogram + inference demo on a single clip (own voice / a singer you like)
python -m src.melspec_demo --audio_path /path/to/clip.mp3 --model sota_crnn_wide \
    --checkpoint results/sota_crnn_wide/best.pt --data_index_dir data/index --out_dir results/demo
```

`ssl_frontend` and `speaker_frontend` need `torch>=2.6` specifically (HF's
`safetensors`-only `torch.load` policy blocks older `transformers`/`torch`
combos on some pretrained checkpoints) — see `src/models/ssl_frontend.py`'s
docstring if installing fresh.

## Methods implemented

Ported faithfully from the original authors' released code where noted,
inside one shared training/eval pipeline (`src/train.py`, `src/evaluate.py`).

| Model | Basis |
|---|---|
| Traditional ML (kNN/SVM/RandomForest) | hand-crafted librosa/torchaudio features |
| `confound_crnn` | bill317996/Singer-identification-in-artist20 (Hsieh et al., ICASSP 2020) |
| `crnn_zain` | ZainNasrullah/music-artist-classification-crnn |
| `fgnl` (non-local CNN) | ian-k-1217/Fully-Generalized-Non-Local-Network |
| `sota_crnn` and capacity/architecture variants (`_wide`/`_narrow`/`_norm`/`_attn`/`_dropblock`/`_supcon`/`_swa`) | minzwon/sota-music-tagging-models |
| `short_chunk_cnn` | minzwon/sota-music-tagging-models |
| `sample_cnn` | minzwon/sota-music-tagging-models |
| `se_resnet` | minzwon/sota-music-tagging-models (Squeeze-and-Excitation) |
| `confound_crnn_vocals` / `confound_crnn_remix` | vocal-separation and cross-song vocal/instrumental remix ablations, per Hsieh et al. |
| `ssl_pretrain` → `sota_crnn_ssl_finetune` | from-scratch SimCLR-style contrastive pretraining on our own train split (self-supervised, no external data/weights) |
| **Ensemble (graded submission)** | weighted average of the from-scratch models above, weights grid-searched on val |
| `ssl_frontend` (baseline only) | pretrained self-supervised audio model (MERT) |
| `speaker_frontend` (baseline only) | pretrained speaker-embedding model (ECAPA-TDNN) |

Full per-model results, ablations, and citations: `MATERIALS.md`. Full
chronological dev log (what we tried, what broke, what we fixed):
`EXPERIMENT_LOG.md`. Deep-research prompts and responses that informed
architecture/data/training-recipe decisions: `deep_research/`.

## Inference / grading

See `readme` (TA-facing) for the exact inference command for the graded
submission and its measured val accuracy.
