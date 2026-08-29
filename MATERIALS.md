# Materials index (for report handoff)

Running index of everything needed for the final `STUDENT_ID_report.html` (to be
assembled separately, on Claude web, from these materials). Update as each
experiment lands.

## Task 1 — traditional ML
- Features: MFCC(20)+delta+delta2, chroma, spectral contrast/centroid/bandwidth/
  rolloff, ZCR, tonnetz — mean+std pooled per track, StandardScaler, kNN/SVM(RBF)/
  RandomForest.
- Results: TBD — still extracting features on gsm-gpu2 (slow: full-song HPSS for
  tonnetz is expensive per track, no incremental caching so it's running to
  completion rather than being restarted).

## Task 2 — deep learning
All models: chunked 5s inputs, song-level prediction via mean-pooled softmax
across a track's non-overlapping chunks. Val-set top1/top3 as of last check
(gsm-gpu2, tmux `hw1_singer`; **not final** — some still training):

| model | method | val top1 | val top3 | status |
|---|---|---|---|---|
| speaker_frontend | Method 3 (ECAPA-TDNN + probe) | **0.896** | — | done (best_epoch=32) |
| ssl_frontend | Method 2-4 / Baseline 1 (MERT + probe) | 0.684 | — | done (best_epoch=36) |
| sota_crnn | Method 1 (CRNN, minzwon) | 0.584 | — | done (best_epoch=73) |
| confound_crnn | Task-2 core / Method 2-2 (CRNN2D_elu2) | ~0.60 | ~0.80 | training, epoch ~73/80 |
| crnn_zain | Method 2-1 (Zain CRNN2D) | ~0.55 | ~0.80 | training, epoch ~66/80 |
| fgnl | Method 2-3 (non-local net) | ~0.50 | ~0.81 | training, epoch ~68/80, slowest riser |

Best model so far: **speaker_frontend** (frozen VoxCeleb-pretrained ECAPA-TDNN +
MLP probe) — notably stronger than the music-domain SSL (MERT) and all
from-scratch CRNNs, on this ~950-track training set. Worth discussing in the
report: pretrained speaker-verification embeddings transferring from spoken to
sung voice (Method 3's premise) outperformed training from scratch here, echoing
what the deep-research synthesis below found in the literature.

## Deep Research synthesis (relayed by user, 2026-08-29, 3 engines: Gemini,
Perplexity, Qwen — full responses in `deep_research_response_*.md`)

**On vocal source separation (our ablation is running, `results/*_vocals`
TBD)**: the three engines *disagree* with each other, which is itself the
finding worth reporting — don't assume a direction, measure it:
- Gemini's sources report raw-mixture Artist20 baselines of only ~56-60%,
  jumping to 80-85%+ after vocal separation.
- Perplexity cites the original bill317996/Hsieh et al. paper directly: vocal-
  only *alone* actually **reduced** song-level F1 from 0.67 to 0.61 vs. the raw
  mixture; only the *combined* (original + vocal-only + cross-song remix) data
  condition improved to 0.74. Separation-alone is not reliably beneficial.
- Qwen expects separation to show "marginal or negative effect size" due to
  demucs artifacts (musical noise, phase smearing) destroying fine formant
  structure, and flags pitch-shifting in particular as *detrimental* to singer
  ID (a singer's F0 habits are an identity cue, not noise to augment away).
- **Takeaway for our own vocals-only ablation once it finishes**: report the
  actual delta vs. `confound_crnn` trained on raw mixtures, and interpret an
  improvement *or* a regression as informative either way — don't editorialize
  toward the "separation helps" prior most people walk in with.

**Best candidate not implemented this pass**: `SonyCSLParis/ssl-singer-identity`
(Torres, Lattner & Richard, ISMIR 2023 / arXiv:2401.05064) — an EfficientNet-B0
encoder trained *specifically* on isolated singing voice via BYOL/VICReg/
contrastive objectives at 44.1kHz, MIT-licensed, ungated, with a
`load_model(name, input_sr=16000)` API that auto-resamples. Recommended by 2 of
3 engines as the single strongest architecture addition, with reported
singer-ID accuracy up to 81% and 2.16% EER on singer-similarity in their own
evaluations (not on Artist20 specifically). Not integrated here because it
isn't pip-packaged (`from singer_identity import load_model` requires cloning
the repo onto `PYTHONPATH`, not just `pip install`) and the 6-model comparison
plus Task 1 already fully covers the assignment's required methods — flagged
here as the clearest next step if more time is available, rather than adding
integration risk this late in a already-parallel 8-job run.

**Other candidates surfaced** (see `deep_research_response_*.md` for full
detail — not pursued, listed for report completeness):
- Whisper encoder (pooled intermediate layers) as a third SSL paradigm distinct
  from MERT (music) and ECAPA (speech) — 2 of 3 engines flagged this as
  competitive for singing voice specifically, given Whisper's massive weakly-
  supervised training data includes singing.
- EfficientAT (`fschmid56/EfficientAT`, MobileNetV3/DyMN distilled from PaSST) —
  lightweight, good small-data inductive bias.
- Cross-song vocal/instrumental remixing (mix a singer's separated vocal with a
  *different* song's backing track, same label) — called out by all 3 engines
  as the highest-value augmentation for breaking the album/production confound,
  stronger than separation alone. Not implemented (would need a second demucs
  pass over accompaniment stems plus a remix-pairing scheme).
- Metric-learning heads (ArcFace / prototypical / supervised-contrastive) over
  a frozen encoder instead of plain softmax — cheap to try, flagged as a
  training-objective ablation rather than a new encoder.

## Visualizations
- t-SNE embedding plot: TBD (once a final best model is picked)
- Mel-spectrogram + own-recording inference: TBD (blocked on user-supplied clip,
  intentionally last per user instruction)

## Citations
(see per-file docstrings in `src/models/*.py` for full citations; summary:)
- Hsieh et al., "Addressing the confounds of accompaniments in singer
  identification," ICASSP 2020 — Artist20 dataset + album split + confound_crnn
- Choi et al. 2016/2017 (FCN/CRNN) via minzwon/sota-music-tagging-models
- Nasrullah & Zhao, IJCNN 2019 (CRNN2D) via ZainNasrullah/music-artist-
  classification-crnn
- "Fully Generalized Non-Local Network for Singer Identification," AAAI 2021,
  via ian-k-1217/Fully-Generalized-Non-Local-Network
- Li et al., MERT (m-a-p/MERT-v1-95M)
- Desplanques et al. 2020, ECAPA-TDNN, via speechbrain/spkrec-ecapa-voxceleb;
  premise from Rathnayake et al., ICASSP 2022 (speak/sing domain adaptation)
- Torres, Lattner & Richard, ISMIR 2023 (arXiv:2401.05064) — cited, not
  implemented (see Deep Research section above)

## Links
- GitHub: https://github.com/goog-msft-fb-nflx-nvda-aapl/CommE5070_HW1
- Checkpoint (Google Drive): TBD
