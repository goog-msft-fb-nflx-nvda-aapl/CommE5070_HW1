# Materials index (for report handoff)

Running index of everything needed for the final `R13921031_report.html` (to be
assembled separately, on Claude web, from these materials). Update as each
experiment lands.

## Task 1 — traditional ML
- Features: MFCC(20)+delta+delta2, chroma, spectral contrast/centroid/bandwidth/
  rolloff, ZCR, tonnetz — mean+std pooled per track, StandardScaler, kNN/SVM(RBF)/
  RandomForest.
- Results (val set, `results/task1/metrics.json` + confusion matrices):

  | classifier | top1 | top3 |
  |---|---|---|
  | kNN (k=5, distance-weighted) | 0.403 | 0.623 |
  | **SVM (RBF)** | **0.593** | **0.831** |
  | RandomForest (500 trees) | 0.550 | 0.766 |

  SVM(RBF) on hand-crafted features alone (59.3% top1) is competitive with —
  and beats several of — the from-scratch deep architectures below (sota_crnn
  58.4%, fgnl 57.1%), a genuinely interesting result worth calling out in the
  report: careful feature engineering + a classical classifier remains a strong
  baseline on a dataset this small.
- Took ~4.5 hours on gsm-gpu2 (single-threaded librosa HPSS/tonnetz on
  full-length songs is slow, and `SVC(probability=True)`'s internal 5-fold CV
  compounded it) — noted for next time: chunk/parallelize extraction and cache
  incrementally rather than only at the end.

## Task 2 — deep learning
All models: chunked 5s inputs, song-level prediction via mean-pooled softmax
across a track's non-overlapping chunks. Final val-set results:

| model | method | val top1 | val top3 |
|---|---|---|---|
| **speaker_frontend** | Method 3 (ECAPA-TDNN + probe) | **0.952** | **0.987** |
| ssl_frontend | Method 2-4 / Baseline 1 (MERT + probe) | 0.684 | — |
| confound_crnn | Task-2 core / Method 2-2 (CRNN2D_elu2, raw mixture) | 0.671 | 0.823 |
| confound_crnn_vocals | Method 2-2 ablation (CRNN2D_elu2, demucs vocals-only) | 0.671 | 0.857 |
| crnn_zain | Method 2-1 (Zain CRNN2D) | 0.619 | 0.835 |
| sota_crnn | Method 1 (CRNN, minzwon) | 0.584 | — |
| fgnl | Method 2-3 (non-local net) | 0.571 | 0.853 |

(`confound_crnn`'s number here was corrected after an unrelated bug — see
"Ghost retrain" note just below the vocal-separation writeup. The originally-
reported 0.649/0.853 no longer reflects the checkpoint actually on disk.)

Best model overall: **speaker_frontend** (frozen VoxCeleb-pretrained ECAPA-TDNN
+ MLP probe) — notably stronger than the music-domain SSL (MERT) and every
from-scratch CRNN, on this ~950-track training set. Confusion matrix
(`results/speaker_frontend/confusion_matrix_speaker_frontend.png`) is strongly
diagonal; t-SNE (`results/speaker_frontend/tsne.png`) shows clean, tight
per-artist clusters. Report angle: pretrained speaker-verification embeddings
transferring from spoken to sung voice (Method 3's premise) beat training from
scratch here by a wide margin, echoing the deep-research synthesis below.

**Bug caught and fixed mid-run**: the first `speaker_frontend` training
(89.6% top1) had a real correctness bug — `requires_grad=False` freezes a
backbone's *weights*, but not BatchNorm's `running_mean`/`running_var`
buffers, which still update on every forward pass in `.train()` mode. ECAPA-
TDNN uses BatchNorm extensively, so the "frozen" backbone was silently
drifting its normalization statistics onto our small training set every
epoch. Caught only because stripping the backbone out of the checkpoint for
a small redistributable file (see Links below) produced wildly different
predictions than the full checkpoint — a fresh pretrained download no longer
matched what was actually trained. Fixed by forcing `backbone.eval()` inside
a `train()` override (already present in `ssl_frontend.py` for a different
reason — MERT is transformer/LayerNorm-only so was incidentally unaffected);
retrained, and the *properly* frozen version scored **higher** (95.2% vs
89.6%) — the drifted stats were actively hurting, not helping. Old
(buggy-but-still-technically-valid) run kept at
`results/speaker_frontend_buggy_bn/` for the record. General lesson saved to
memory for future projects.

**Vocal-separation ablation result** (Method 2-2 / doc's Source Separation
section): demucs vocals-only training (`confound_crnn_vocals`) tied the same
architecture trained on raw mixtures on top1 (0.671 both) and beat it on top3
by **+3.5pp (0.857 vs 0.823)** — a real but narrow, metric-dependent effect,
nowhere near the Gemini source's claimed 25pp jump, and not a clean "separation
wins" story either (top1 literally identical). See `results/analysis/` for the
per-artist breakdown and fix/damage transition matrix that explains what that
tie is actually made of.

**"Ghost retrain" — a data-integrity incident, corrected**: `confound_crnn`'s
originally-reported 0.649/0.853 turned out to be stale. While Task 1's ~4.5-hour
feature extraction was still running in tmux window `:0`, a `src.train
--model confound_crnn` command got queued as pending input in that same busy
window (sent there before a dedicated window was created and used instead for
the real run). tmux doesn't discard queued input — it fires once the pane frees
up. When Task 1 finally finished hours later, that queued command silently
fired and **retrained `confound_crnn` a second time into the same `--out_dir`**,
overwriting the first run's checkpoint/log/summary with a different (different-
seed, otherwise identical-config) result. Not noticed until a later
cross-check ([[feedback_tmux_queued_input]], saved to memory) found the
checkpoint's `best_epoch` no longer matched what had been recorded right after
the original run. The checkpoint currently on disk (0.671/0.823) is verified
self-consistent across the training log, `summary.json`, and a fresh
independent re-evaluation — it's what's actually reproducible, so it's what's
reported here; the original 0.649/0.853 run no longer exists to recover. Not a
correctness bug in the model or pipeline, and doesn't affect the graded
submission (that's from `speaker_frontend`, trained in its own dedicated
window, unaffected) — but worth disclosing plainly rather than quietly editing
the number.

## Deep Research synthesis (relayed by user, 2026-08-29, 3 engines: Gemini,
Perplexity, Qwen — full responses in `deep_research_response_*.md`)

**On vocal source separation** — the three engines *disagreed* with each
other going in, which is itself the finding worth reporting, and our own
measured result (`confound_crnn_vocals`: top1 tied, +3.5pp top3, see Task 2
table above) landed closer to "mixed/metric-dependent" than any single prior:
- Gemini's sources report raw-mixture Artist20 baselines of only ~56-60%,
  jumping to 80-85%+ after vocal separation — we saw nothing like this
  magnitude.
- Perplexity cites the original bill317996/Hsieh et al. paper directly: vocal-
  only *alone* actually **reduced** song-level F1 from 0.67 to 0.61 vs. the raw
  mixture; only the *combined* (original + vocal-only + cross-song remix) data
  condition improved to 0.74 — our top1 was flat rather than reduced, so also
  doesn't match their alone-vs-raw direction.
- Qwen expects separation to show "marginal or negative effect size" — we saw
  marginal, and on top3 specifically positive, not negative.
- **Takeaway**: none of the three secondhand literature summaries predicted
  our actual result correctly in both direction and magnitude — a clean
  illustration of why the measured, architecture-specific result is what
  belongs in the report, not any single source's claim. (The follow-up Deep
  Research round, see below, converged on the same message from a different
  angle: published Artist20 separation results range from clearly-helps to
  clearly-hurts depending on protocol, so ours landing in between isn't an
  outlier.)

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

## Bonus — Baseline 2, zero-shot audio-LLM
Qwen2-Audio-7B-Instruct, prompted with a 15s clip + the closed list of 20
artist names, asked to rank its top-3 guesses (no training/fine-tuning).
On 40 val tracks: **top1=0.475, top3=0.700** — below the trained models but
surprisingly close to several from-scratch CRNNs (crnn_zain 0.619,
confound_crnn 0.671), and far above the 5%/15% random-chance floor for 20-way
top1/top3, for a model that saw zero gradient steps on this dataset.
(First pass under-scored this at 0.325/0.525 due to an answer-parsing bug —
the prompt asked the model to reply with underscored label names like
"led_zeppelin", which it naturally spelled as "Led Zeppelin"; the parser
didn't normalize spaces vs. underscores before matching. Fixed by presenting
space-separated names in the prompt and normalizing both sides before
matching — worth remembering generally: always sanity-check an LLM-as-judge
or LLM-output parser against a few raw examples before trusting the score.)
Results: `results/audio_llm/val_predictions.json` (includes raw model text
per track for inspection).

## Visualizations
- t-SNE embedding plot: `results/speaker_frontend/tsne.png` (best model, val
  set) — clean, tight per-artist clusters.
- Confusion matrices: `results/speaker_frontend/confusion_matrix_*.png` (best
  model, strongly diagonal), plus one per Task-1 classifier and per Task-2
  model in their respective `results/<name>/` dirs.
- Mel-spectrogram + own-recording inference: `results/demo/melspectrogram.png`
  + `results/demo/prediction.json`. Input: user-supplied clip of "Unstoppable"
  by Sia (`unstoppable_sia.m4a`, recorded via Voice Memos — not a
  train_val/test artist, so this is a genuine out-of-distribution probe, not
  a memorized track). Best model's (`speaker_frontend`) top-3: fleetwood_mac
  (0.469), roxette (0.232), tori_amos (0.159). Discussion angle: all three are
  female-fronted (Sia is female) or alto/mezzo-range acts within the 20-artist
  set — with the true singer entirely absent from the label space, the model
  falls back to nearest-neighbor vocal timbre among what it knows, which is
  exactly the behavior you'd hope for from a well-calibrated closed-set
  classifier facing an open-set input.

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
- Checkpoint (Google Drive): TBD — needs a manual upload, see EXPERIMENT_LOG.md
  ("Google Drive checkpoint upload — final status")
