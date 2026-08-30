# Materials index (for report handoff)

Running index of everything needed for the final `R13921031_report.html` (to be
assembled separately, on Claude web, from these materials). Update as each
experiment lands.

**TA clarifications (`TA_discussion.md`, read 2026-08-30) that shape this
document's framing:**
- Task 1's accuracy **doesn't count toward the grade at all** — TA grades
  only on whether the analysis process is reasonable and thorough. (Feature-
  group ablation / permutation importance below is exactly what's being
  graded, not the raw SVM number.)
- Task 2 requires training **both encoder and classifier from scratch**; a
  pretrained encoder (frozen or fine-tuned) is explicitly "baseline only, not
  the graded submission" (TA_discussion.md #4). This means `ssl_frontend`
  (MERT) and `speaker_frontend` (ECAPA-TDNN) — despite `speaker_frontend`
  scoring far higher than anything else in this project — are **baselines**,
  not eligible as the Task-2 submission. The graded/submitted result is a
  **weighted ensemble of from-scratch models** (85.3% val top1 — see
  Task 2 below), all individually eligible under this rule.
  See `readme`'s "Running inference" section.

**Retrain note (2026-08-30)**: the from-scratch scores below went through two
passes. Pass 1 (undertrained — all models hit an 80-epoch cap without ever
triggering early stopping, val accuracy still swinging ±8-10pp epoch to
epoch, and no data augmentation at all) gave the weak scores reported
earlier. After diagnosing this, added SpecAugment (mel models) / gain-jitter
+noise (waveform model), a cosine LR schedule, weight decay, and a much
higher epoch cap (300, patience 40) — pass 2's numbers below reflect that.
Also added a cross-song vocal/instrumental remix ablation
(`confound_crnn_remix`), the top lever identified by a further Deep Research
round specifically on improving these from-scratch numbers
(`deep_research/round3_from_scratch_improvement/response_*.md`).

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

  SVM(RBF) on hand-crafted features alone (59.3% top1) was initially
  competitive with several from-scratch deep architectures before their
  retrain (see Task 2 below) — a genuinely interesting result worth calling
  out in the report even though the from-scratch models have since pulled
  ahead: careful feature engineering + a classical classifier remains a
  strong baseline on a dataset this small, and a good sanity-check floor.
  (Per the TA clarification above, this accuracy number itself isn't what's
  graded — the feature-group ablation and permutation-importance analysis in
  "Deep-dive ablations" below, which explains *why* this number is what it
  is, is the actual graded content.)
- Took ~4.5 hours on our training server (single-threaded librosa HPSS/tonnetz on
  full-length songs is slow, and `SVC(probability=True)`'s internal 5-fold CV
  compounded it) — noted for next time: chunk/parallelize extraction and cache
  incrementally rather than only at the end.

## Task 2 — deep learning
All models: chunked 5s inputs, song-level prediction via mean-pooled softmax
across a track's non-overlapping chunks. Val-set results across every
architecture/ablation trained (pass-2 pipeline: SpecAugment/waveform
augmentation, cosine LR schedule, 300-epoch cap / 40-epoch patience unless
noted):

| model | eligibility | basis / ablation | val top1 | val top3 |
|---|---|---|---|---|
| **sota_crnn_wide** | **graded submission** | Method 1 CRNN, channel_mult=1.5 capacity sweep | **0.805** | **0.922** |
| sota_crnn_swa (best non-averaged ckpt) | from-scratch | Method 1 CRNN + AdamW + label-smoothing + SWA run (the SWA average itself was worse — see below) | 0.784 | 0.861 |
| short_chunk_cnn | from-scratch | ShortChunkCNN_Res (minzwon, lecture02-spotlighted) | 0.766 | 0.866 |
| sota_crnn_norm | from-scratch ablation | Method 1 CRNN + per-sample mel normalization | 0.766 | 0.879 |
| sota_crnn_ssl_finetune | from-scratch | Method 1 CRNN, SimCLR/NT-Xent-pretrained then fine-tuned | 0.766 | 0.853 |
| sota_crnn (original) | from-scratch | Method 1 CRNN, plain Adam | 0.762 | 0.900 |
| se_resnet | from-scratch | SEResNet (new arch, SE gating, per user's prior-run design) | 0.758 | 0.866 |
| sota_crnn_adamw_ls | from-scratch ablation | Method 1 CRNN + AdamW + label smoothing 0.1 | 0.753 | 0.892 |
| sota_crnn_supcon | from-scratch ablation | Method 1 CRNN + AdamW/LS + SupCon auxiliary loss | 0.736 | 0.853 |
| sota_crnn_dropblock | from-scratch ablation | Method 1 CRNN + AdamW/LS + DropBlock | 0.697 | 0.866 |
| crnn_zain | from-scratch | Method 2-1 (Zain CRNN2D) | 0.710 | 0.848 |
| fgnl (no-augment retry) | from-scratch | Method 2-3 non-local net, augmentation disabled | 0.710 | 0.870 |
| confound_crnn_remix | from-scratch ablation | Method 2-2 remix ablation (CRNN2D_elu2, cross-song remix) | 0.693 | 0.840 |
| confound_crnn | from-scratch | Task-2 core / Method 2-2 (CRNN2D_elu2, raw mixture) | 0.693 | 0.827 |
| sample_cnn | from-scratch | SampleCNN (minzwon, raw-waveform end-to-end) | 0.693 | 0.823 |
| sota_crnn_narrow | from-scratch ablation | Method 1 CRNN, channel_mult=0.5 capacity sweep | 0.654 | 0.835 |
| sota_crnn_attn | from-scratch ablation | Method 1 CRNN + attention pooling (vs. last-GRU-state) | 0.654 | 0.810 |
| sota_crnn_swa (SWA-averaged) | from-scratch ablation | same run as above, weights actually averaged | 0.749 | 0.866 |
| speaker_frontend | baseline (pretrained encoder) | Method 3 (ECAPA-TDNN + probe) | 0.952 | 0.987 |
| ssl_frontend | baseline (pretrained encoder) | Method 2-4 / Baseline 1 (MERT + probe) | 0.684 | — |

(`confound_crnn_vocals` and `fgnl`'s earlier lr=1e-4 retry are from the
undertrained pass-1 pipeline / a since-superseded hypothesis respectively —
kept in `results/*_v1_undertrained/`, `results/fgnl_v3_lr1e4_worse/` for the
record, not in the table above.)

**Ensemble (the actual graded submission): weighted average of 9 from-
scratch models** (`src/ensemble.py`, `results/ensemble2/ensemble_result.json`)
— grid-searched integer weights 0-2 over confound_crnn, crnn_zain, sota_crnn,
sota_crnn_wide, short_chunk_cnn, se_resnet, fgnl, sample_cnn,
sota_crnn_norm. Best: sota_crnn×2 + sota_crnn_wide×1 + short_chunk_cnn×1 +
sota_crnn_norm×1 → **val top1=0.853, top3=0.905** — +4.8pp over
`sota_crnn_wide` alone, the largest single lever in the whole from-scratch
comparison, and directly validated twice now (this project's first, smaller
ensemble at +3.0pp; the user's own prior run at +11pp). Directly regenerates
`results/R13921031.json`. Caveat stated plainly, not hidden: a 9-model,
0-2 integer-weight grid search (3^9 ≈ 19683 combinations) against only 231
val tracks carries real overfit-to-val risk in the *weight selection*
specifically — the individual models' own numbers are the more robust
per-architecture comparison; the ensemble weights should be read as "a
reasonable combination that measurably helps," not as a precisely-tuned
optimum.

**Best single from-scratch model: `sota_crnn_wide`**
— the winning architecture (Method 1 CRNN) scaled to 1.5x channel width via
the capacity sweep (`channel_mult` in `sota_cnn.py`'s `CRNN`), from scratch,
no pretrained weights. This is the one item on the ablation queue that
Qwen's round-3 response got backwards without a citation (it argued smaller
models would generalize better on ~950 tracks) — tested directly rather than
trusted or dismissed, and the opposite direction won: **wider, not narrower**
(narrow: 0.654, original 1x: 0.762, wide 1.5x: 0.805). Per TA_discussion.md
#4, this is the model that actually counts as the Task-2 deliverable —
*not* the Artist20 paper's own architecture (`confound_crnn`), and not
`sota_crnn` at its original width either.

**Ablation takeaways** (12 variants of/around the winning `sota_crnn`
architecture, all from the same starting point):
- **Helped**: more capacity (wide, +4.3pp), per-sample mel normalization
  (+0.4pp), plain AdamW+label-smoothing (mixed — see below), SSL pretraining
  (+0.4pp, within the "small positive" band Gemini's round-4 response
  predicted, nowhere near CLMR's 5.6pp reference point).
- **Hurt**: attention pooling (-10.8pp vs. last-GRU-state — the opposite of
  what our own prior-year run's design choice would predict, a genuine,
  measured negative result, not assumed **on the backbone actually tested**;
  see "Prior-year submission, revisited" below — this ablation bolted
  attention onto `sota_crnn`'s small 32-hidden-dim unidirectional GRU, not
  the larger bidirectional-256 GRU the prior run actually used, so it
  doesn't yet settle whether attention pooling itself is the culprit), less
  capacity (-10.8pp), SupCon
  auxiliary loss (-2.6pp vs. the AdamW/LS-only version), DropBlock (-5.6pp
  vs. AdamW/LS-only), SWA weight-averaging (-3.5pp vs. its own run's
  best non-averaged checkpoint — averaging late-training weights made this
  particular model *worse*, not more robust).
- **Mixed**: AdamW+label-smoothing alone (0.753) actually landed *below*
  plain Adam (0.762) on this specific architecture at this width — but
  every AdamW/LS-based ablation on top of it (norm, dropblock, supcon) used
  it as their shared base, so it's not directly comparable to isolate; the
  capacity-sweep variants (narrow/wide) used AdamW/LS too and still showed
  channel width as the dominant effect either way.

**Prior-year submission, revisited (2026-08-30).** We re-checked our own
prior-year submission for this same assignment
(`github.com/goog-msft-fb-nflx-nvda-aapl/NTU`) after it looked like it might
beat this project's ensemble. Two findings:
1. **Not directly comparable.** That submission self-splits `train_val` by
   taking each artist's alphabetically-last album as validation (946/234
   tracks), not this project's official assignment-provided split (949/231)
   — the two val accuracies are measured on different tracks. Under the
   report's own guessed scoring formula (top1 + 0.5×top3), this project's
   ensemble is *not* behind (0.853+0.5×0.905=**1.305** vs.
   0.825+0.5×0.949=**1.2995**), but that comparison is still cross-split and
   should be read as "not obviously worse," not "ahead."
2. **A real, previously-unclosed gap.** The prior submission's actual CRNN
   is architecturally larger and different from anything in this project's
   ablation table: bidirectional GRU (hidden=256, 2 layers) vs. `sota_crnn`'s
   unidirectional 32-hidden-dim GRU; attention pooling over the full
   sequence; 4 Conv-BN-ELU blocks up to 256 channels vs. `sota_crnn`'s
   96-mel/512-fft frontend; 10s training crops vs. our 5s; per-sample mel
   normalization; `f_min=20`/`f_max=8000`/`top_db=80`. Our existing
   `sota_crnn_attn` and `sota_crnn_norm` ablations tested those last two
   ingredients individually, but bolted onto `sota_crnn`'s much smaller
   backbone — so the "attention pooling hurt" result above is confounded by
   backbone capacity, not necessarily a clean refutation. Ported the prior
   submission's exact architecture as `crnn_nasrullah_faithful`
   (`src/models/crnn_nasrullah_faithful.py`) and launched training
   (`results/crnn_nasrullah_faithful/`) — update this section with the
   result once it lands, and fold into `src/ensemble.py`'s candidate pool if
   it's competitive. (Their 10-random-crop TTA method specifically is *not*
   being re-adopted — already directly tested and shown weaker than this
   project's default full-track averaging, see the ensemble/TTA section
   above.)

**`fgnl`**: two retries (lr=1e-4, then no-augment-at-all) both under-
performed the original pass-2 run's 52.4%... except the no-augment retry
actually recovered to 0.710 — *matching* crnn_zain and beating the
original augmented fgnl run. So augmentation specifically (not the LR) was
what hurt this architecture — confirms the hypothesis from the earlier
FGNL paper's actual published recipe as a constant LR of 1e-4, not the 1e-3
we used for every model uniformly — a 10x mismatch that plausibly explains
instability in a non-local-attention architecture more LR-sensitive than a
plain CRNN. Retry at lr=1e-4 launched, in progress; this table will be
updated once it finishes.

**Undertraining root-cause (why the numbers moved this much)**: prompted by
the user asking why Task 2's score looked "extremely low," checked the first
pass's training logs directly — every one of the 4 from-scratch models hit
its 80-epoch cap at `best_epoch` 73-77, **never once triggering early
stopping** (patience=15), with val accuracy still swinging ±8-10pp epoch to
epoch. Combined with zero data augmentation despite the assignment doc asking
for it, this was undertraining, not an architecture ceiling. Fixed
(`src/data/dataset.py`'s `_spec_augment`/`_augment_waveform`,
`src/train.py`'s cosine `LR` scheduler + weight decay + epochs=300/
patience=40) and retrained all 4 — 3 of 4 improved substantially
(sota_crnn +17.8pp, crnn_zain +9.1pp, confound_crnn +2.2pp), confirming the
diagnosis; fgnl regressed, addressed above. This also means the
"graded model" ranking flipped: `sota_crnn` (Method 1, not Artist20-specific)
now clearly beats `confound_crnn` (the Artist20 paper's own architecture) —
worth discussing in the report as a genuinely interesting result in its own
right, not just a training-fix footnote.

**Best model including baselines: `speaker_frontend`** (frozen VoxCeleb-
pretrained ECAPA-TDNN + MLP probe, 95.2%/98.7%) — dramatically stronger than
every from-scratch model and the music-domain SSL baseline, but per the TA's
clarification this is a **baseline only**, not eligible as the Task-2
submission (see readme). Still the most interesting single result in this
project and worth leading with in the report's discussion section — the
sanity-checking and encoder-vs-head analysis under "Deep-dive ablations"
below is what makes it a defensible, non-suspicious finding rather than just
a big number. Confusion matrix
(`results/speaker_frontend/confusion_matrix_speaker_frontend.png`) is
strongly diagonal; t-SNE (`results/speaker_frontend/tsne.png`) shows clean,
tight per-artist clusters.

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

**Vocal-separation ablation result (pass 1, superseded — see note)**: demucs
vocals-only training (`confound_crnn_vocals`) tied the pass-1 raw-mixture
`confound_crnn` on top1 (0.671 both) and beat it on top3 by +3.5pp
(0.857 vs 0.823). Not rerun after the pass-2 augmentation/scheduler fix
(`confound_crnn` moved to 0.693/0.827 — see "Undertraining root-cause" in the
Task 2 section), so this specific comparison is now stale; the per-artist
breakdown and fix/damage transition matrix in `results/analysis/` are still
qualitatively informative (real, large, roughly-canceling per-artist swings
from separation) even though the base numbers have moved. The separately-run
`confound_crnn_remix` ablation (cross-song remix, a different and more
targeted separation-based technique, see Task 2 above) *was* run on the
pass-2 pipeline: 0.693/0.840 vs pass-2 raw-mixture's 0.693/0.827 — same
top1, +1.3pp top3, a smaller effect than the ~7-8pp the source paper reports,
discussed further below.

**"Ghost retrain" — a data-integrity incident, corrected (historical, now
further superseded)**: `confound_crnn`'s originally-reported 0.649/0.853
turned out to be stale. While Task 1's ~4.5-hour feature extraction was still
running in tmux window `:0`, a `src.train --model confound_crnn` command got
queued as pending input in that same busy window (sent there before a
dedicated window was created and used instead for the real run). tmux doesn't
discard queued input — it fires once the pane frees up. When Task 1 finally
finished hours later, that queued command silently fired and **retrained
`confound_crnn` a second time into the same `--out_dir`**, overwriting the
first run's checkpoint/log/summary with a different (different-seed,
otherwise identical-config) result — landing at 0.671/0.823
([[feedback_tmux_queued_input]], saved to memory). That number has since been
superseded again, deliberately this time, by the pass-2 augmented retrain
(0.693/0.827) described in "Undertraining root-cause" above. Neither
incident affects the graded submission (`sota_crnn`, trained independently) —
kept here as a paper trail, not because the number still matters on its own.

## Deep Research synthesis (relayed by user, 2026-08-29, 3 engines: Gemini,
Perplexity, Qwen — full responses in `deep_research/round1_sota_and_architecture_survey/response_*.md`)

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

**Other candidates surfaced** (see `deep_research/round1_sota_and_architecture_survey/response_*.md` for full
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

## Deep-dive ablations (round 2 — Deep Research follow-up, 2026-08-30)

A second, follow-up Deep Research round (4 engines this time: Gemini,
Perplexity, ChatGPT, Qwen — `deep_research/round2_sota_context_and_perartifact_ablations/prompt.md` /
`deep_research/round2_sota_context_and_perartifact_ablations/response_*.md`) asked for (1) the actual Artist20 SOTA
context for our numbers and (2) specific ablations per method. Results below;
scripts are `src/analysis_*.py`, raw JSON in `results/analysis/`.

**Timing note**: the CRNN-family ablations here (ensemble diversity, feature-
group ablation, vocal-separation attribution) were computed against the
*pass-1* (undertrained) checkpoints, before the augmentation/LR-schedule fix
described in the Task 2 section above. The `speaker_frontend`/`ssl_frontend`
head-swap matrix and shuffled-label check are unaffected (those models
weren't touched by the from-scratch retrain). The qualitative findings below
(what drives the SVM, whether errors are correlated across CRNN
architectures, how separation redistributes errors) still hold — small-data
CRNNs don't change *character* between a 67% and 76% run — but the specific
numbers reflect the pre-retrain checkpoints; not rerun given time budget.

### Is 95.2% top1 plausible / SOTA?

All 4 engines converged on the same shape of answer, worth stating plainly:
**there is no single clean Artist20 leaderboard number to compare against.**
Published work reports song-level F1 (not top-1 accuracy) under varying
protocols, with directly-comparable album-split systems clustering around
**0.6-0.86 F1** (Nasrullah & Zhao CRNN ~0.60-0.67, Hsieh et al. CRNNM+aug
0.75, FGNL 0.73-0.83 depending on input, an X-vector+CRNN hybrid ~0.81-0.86 —
that last one is the closest precedent to our approach, since it's also a
speaker-recognition-derived timbral feature). A couple of much higher numbers
exist (KNN-Net 99%, a GMM-UBM system 97.5%) but every engine flagged those as
**not safely comparable** — ambiguous train/test protocol description,
internally inconsistent numbers within their own papers, or a split that
isn't clearly the standard album-level one. Given that, our own number is
best stated as: **95.2%/98.7% top1/top3, validation set, frozen ECAPA-TDNN +
trained head, on raw mixtures** — very likely at or above the best
directly-comparable published results, but not claimed as outright "SOTA"
without a common protocol to verify against.

**Is a speaker-verification embedding winning here a fluke?** No — the
literature has real precedent (the X-vector+CRNN hybrid at ~0.86 F1 above),
and ChatGPT's framing is worth keeping: the actual novel/interesting claim
isn't "speaker embeddings work for singers" (already shown elsewhere) but
that *a frozen, off-the-shelf, modern ECAPA checkpoint is already this
discriminative with nothing more than a small trained head* — no
task-specific fine-tuning, no separation preprocessing, no melody/X-vector
fusion engineering.

**Sanity checks run** (the single most-recommended follow-up across all 4
responses was: does the embedding alone — no MLP — already explain this?):

| encoder | head | top1 | top3 |
|---|---|---|---|
| ECAPA-TDNN | trained MLP (original result) | 0.952 | 0.987 |
| ECAPA-TDNN | linear probe (no hidden layer) | **0.952** | 0.978 |
| ECAPA-TDNN | cosine nearest-centroid (**zero trainable parameters**) | **0.848** | 0.922 |
| MERT | trained MLP (original result) | 0.684 | — |
| MERT | linear probe | 0.688 | 0.866 |
| MERT | cosine nearest-centroid (zero params) | 0.567 | 0.740 |

(`results/analysis/head_swap_matrix.json`.) This is decisive: a **linear**
probe matches the MLP exactly (95.2%), and even a **parameter-free**
nearest-centroid classifier gets 84.8% — there is no trained classifier
sophisticated enough to be "overfitting" or "memorizing" here, since one of
the three conditions has no learnable parameters at all. The ECAPA embedding
itself, not the head, is what's carrying nearly all the signal. Embedding
geometry confirms it directly: silhouette score (cosine) 0.181 for ECAPA vs.
0.001 for MERT; mean within-class vs. between-class cosine similarity gap of
0.158 for ECAPA vs. just 0.022 for MERT — MERT's embedding space barely
separates artists at all, consistent with it encoding general polyphonic/
genre structure rather than vocal identity, exactly as its training objective
(masked acoustic-token modeling over full mixtures) would predict.

Also ran the **shuffled-label sanity check** all 4 engines recommended for a
different failure mode than the head-swap (split/pipeline leakage, not
classifier overfitting): retrained `speaker_frontend` with training labels
randomly permuted, val labels untouched. Result
(`results/analysis/shuffled_label_check.json`): val top1 collapses to
**3.9%** (best across 40 epochs: 11.3%), right at the 5.0% chance floor for
20-way classification. The pipeline cannot "cheat" its way to a high score —
confirms the genuine 95.2% on correctly-labeled data isn't an artifact of a
broken train/val split or file-overlap bug.

Caveats we did *not* fully close out, flagged for the report rather than
silently assumed away (per Gemini's/ChatGPT's checklist): we don't have
access to the assignment's actual held-out **test-set labels** (only the TA
does), so val is the best unbiased estimate available to us, not a true
locked-test number; and we did not run an instrumental-only-audio control to
positively rule out the embedding partly keying on production/instrumentation
rather than pure vocal identity (would need a second demucs pass to isolate
the non-vocal stems — not done, time-boxed out).

### Classical ML: what drives the 59.3% SVM result

Feature-group ablation on the already-cached feature matrices (no
re-extraction needed — `results/analysis/feature_group_ablation.json`):

| condition | top1 | top3 |
|---|---|---|
| all features (178 dims) | 0.593 | 0.831 |
| **MFCC + Δ + ΔΔ only** (120 dims) | 0.550 | 0.758 |
| chroma + tonnetz only (36 dims) | 0.286 | 0.537 |
| spectral shape only — centroid/bandwidth/rolloff/ZCR (8 dims) | 0.277 | 0.481 |
| spectral contrast only (14 dims) | 0.320 | 0.580 |
| minus MFCC/Δ/ΔΔ | 0.468 | 0.684 |
| minus chroma/tonnetz | 0.550 | 0.788 |
| minus spectral shape | 0.597 | 0.831 |
| minus spectral contrast | 0.558 | 0.797 |

MFCC (+deltas) alone gets 55.0% — almost the whole story by itself — and
removing it costs the most (59.3%→46.8%, -12.5pp), confirming timbral/
vocal-tract features are the dominant driver, exactly the hypothesized
mechanism. Spectral-shape features (centroid/bandwidth/rolloff/ZCR) are
essentially dead weight for the SVM — removing them doesn't hurt at all
(59.3%→59.7%, a marginal *improvement*). RandomForest permutation importance
agrees: MFCC group 0.232, spectral contrast 0.106, spectral shape 0.047,
chroma/tonnetz 0.027 (harmonic/key information contributes least — sensible,
since backing-track harmony varies far more than a singer's timbre within
one artist's catalog).

### CRNN family: data-bottleneck vs. architecture, via ensemble diversity

Rather than the (expensive) full learning-curve ablation, ran the cheaper
diagnostic all 4 engines also recommended: pairwise error correlation and an
oracle-ensemble upper bound across the 4 from-scratch models
(`results/analysis/ensemble_diversity.json`):

| model | val top1 |
|---|---|
| confound_crnn | 0.671 |
| crnn_zain | 0.619 |
| sota_crnn | 0.584 |
| fgnl | 0.571 |

Pairwise Cohen's κ ranges 0.54-0.63 (moderate agreement, real but incomplete
overlap — 35-43% disagreement rate on any given pair). The striking number:
**oracle-ensemble accuracy (correct if *any* one of the 4 models gets it
right) is 83.1%**, a full **+16pp over the best individual model (67.1%)** —
but a naive majority vote only reaches 68.4%, barely above the single best
model. That gap between "oracle" and "majority vote" is the real finding:
there's substantial complementary signal spread across these 4 architectures
that a smarter combination (calibrated/weighted ensemble, or a stacked
meta-classifier) could likely capture, but simple voting can't — a concrete,
scoped follow-up if there's ever a next round on this project. The narrow
57-67% band these models all land in, combined with real (not illusory)
error diversity, is more consistent with each architecture hitting variations
on the same ~950-track data ceiling than with one architecture being clearly
better-suited to the task.

### Vocal separation: what a top1 tie is actually made of

Per-track transition matrix between `confound_crnn` (raw) and
`confound_crnn_vocals` (demucs vocals-only), both on the same 231 val tracks
(`results/analysis/vocal_separation_attribution.json`):

| | vocals-only correct | vocals-only wrong |
|---|---|---|
| **raw correct** | 118 (stable) | 37 (separation *damage*) |
| **raw wrong** | 37 (separation *fix*) | 39 (both wrong) |

37 fixed, 37 damaged — an almost exact wash, which is *why* top1 came out
numerically identical (0.671 both) despite each model getting different
songs right. This is a much more informative story than "no effect": vocal
separation is not neutral per-track, it's a large, roughly-canceling
redistribution of errors. Per-artist deltas make this vivid — swings from
**+81pp** (green_day: 0%→81% — presumably a dense, distorted-guitar mix where
separation genuinely removes a confound) to **-73pp** (u2: 73%→0%) and -44pp
(prince), -42pp (fleetwood_mac). A handful of artists (dave_matthews_band,
suzanne_vega, tori_amos) improve substantially and cleanly; a similarly-sized
handful (u2, prince, fleetwood_mac, garth_brooks, madonna, queen) get
noticeably worse. This is consistent with the Deep Research responses'
shared warning: separation artifacts (musical noise, phase smearing, lost
harmonic context) can hurt as much as removing accompaniment helps,
per-artist, even when the aggregate looks flat.

### Zero-shot Qwen2-Audio: not re-run

The suggested self-consistency / multi-crop / structured-prompt experiments
were not run this pass (time-boxed out in favor of the encoder/classical-ML
ablations above, which had clearer, cheaper payoff using already-computed
artifacts). Flagged as the most promising concrete next step for that
baseline specifically: majority-vote over 5 samples at temperature>0, per
ChatGPT/Perplexity's shared recommendation.

## Deep-dive round 3 — improving the from-scratch Task-2 score specifically

Triggered directly by the pass-1 (undertrained) numbers looking "extremely
low" — a third Deep Research round (2 engines: Perplexity, Qwen —
`deep_research/round3_from_scratch_improvement/prompt.md` / `deep_research/round3_from_scratch_improvement/response_*.md`) asked
specifically what's needed to close the gap to the original papers, scoped
to techniques that stay from-scratch-eligible per the TA's constraint.

**Reading the two responses critically**: Perplexity's numbers are directly
sourced (quotes an actual results table from Hsieh et al. with per-condition
F1 values, cites the FGNL paper's PDF for its LR=1e-4 claim) and internally
consistent. Qwen's response contains several precise-sounding but *uncited*
hyperparameter claims (e.g. "Hsieh et al. used 200 epochs, weight decay
1e-5, 5 augmented versions per track"; "FGNL used cosine decay + warmup, 150
epochs, batch 64") that don't match what was directly read from
bill317996's actual source repo earlier in this project, and that Perplexity
either contradicts or doesn't corroborate. Treated Qwen's specific numbers
as low-confidence/possibly fabricated and relied on Perplexity's more
carefully-sourced claims wherever they conflict — a useful general reminder
that "two sources agree" isn't validation when one of them can't show its
work.

**What both responses converge on regardless**: cross-song vocal/
instrumental remixing is *the* dataset-specific technique — it's literally
the mechanism in Hsieh et al.'s own paper (the architecture our
`confound_crnn` ports), reported worth ~7-8pp song-level F1 over origin-only
training, and both responses rank it above generic augmentation, mixup,
auxiliary losses, or architecture resizing as the highest-value next step
for this specific dataset/confound. Implemented as `confound_crnn_remix` (see
Task 2 table above) — the effect in our setup was much smaller than the
literature's (+1.3pp top3, flat top1) rather than the hoped-for 7-8pp.
Plausible reasons, not fully disentangled: (a) our remix pairs any two
training tracks at random rather than "different album, same or matched
artist" as Hsieh et al. specifically did, a real implementation
simplification; (b) Perplexity's own caution — quoting Hsieh et al.'s table
directly — that *remix-only* training underperforms *origin-only*, and the
literature's gain comes specifically from the combined origin+vocal+remix
pool, which is what we implemented (1/3 each), so this isn't obviously the
gap; (c) the augmentation/LR-schedule fix already captured much of the
achievable gain before remix was layered on, so there may simply be less
headroom left for remix to add on top of an already-fixed pipeline. Recorded
honestly rather than tuned until it produced the paper's number.

**FGNL's LR mismatch**: Perplexity's sourced claim that the original FGNL
paper used a constant LR of 1e-4 (not the 1e-3 applied uniformly to every
model in our first retrain pass) is a plausible, specific explanation for why
FGNL was the one architecture that got *worse* under the fix rather than
better — non-local/attention mechanisms are typically more LR-sensitive than
plain CNNs. Retry at lr=1e-4 launched; see Task 2 table for the outcome.

**Not pursued this pass** (noted for completeness, time-boxed out): a
capacity sweep (both responses suggested testing whether a smaller
CRNN2D_elu2/FGNL — Qwen's specific parameter-count suggestions weren't
corroborated by Perplexity and are treated with the same skepticism as its
other uncited numbers — reduces overfitting on ~950 tracks); supervised
contrastive/metric-learning auxiliary losses; confidence-filtered majority
voting for song-level aggregation (FGNL's own paper's approach, distinct
from our mean-pooled-softmax aggregation).

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
- t-SNE embedding plots: `results/sota_crnn/tsne.png` (the graded
  from-scratch model after the retrain, val set — clean, well-separated
  per-artist clusters, consistent with its 76.2% top1) and
  `results/speaker_frontend/tsne.png` (the pretrained-encoder baseline —
  still tighter/cleaner, consistent with 95.2% top1).
  `results/confound_crnn/tsne.png` (pass-2, 69.3% top1, some visible
  overlap e.g. madonna/tori_amos, prince/queen) is also available for a
  three-way comparison across the accuracy range. Cluster quality tracks
  accuracy exactly as it should across all three.
- Confusion matrices: one per Task-1 classifier and per Task-2 model in their
  respective `results/<name>/` dirs — `results/sota_crnn/` (graded model,
  strongly diagonal) and `results/speaker_frontend/` (baseline,
  near-perfectly diagonal) are the two to lead with.
- Mel-spectrogram + own-recording inference: input is a user-supplied clip of
  "Unstoppable" by Sia (`unstoppable_sia.m4a`, recorded via Voice Memos — not
  a train_val/test artist, so this is a genuine out-of-distribution probe,
  not a memorized track). Ran on the graded model and the baseline for
  contrast:
  - `results/demo_sota_crnn/` (graded `sota_crnn`): top-3 madonna (0.224),
    fleetwood_mac (0.124), roxette (0.108) — lower-confidence, flatter
    distribution than the baseline below, consistent with a smaller from-
    scratch model facing a genuinely out-of-distribution input.
  - `results/demo/` (baseline `speaker_frontend`): top-3 fleetwood_mac
    (0.469), roxette (0.232), tori_amos (0.159) — all female-fronted/alto-
    range acts, a sensible nearest-neighbor-in-timbre fallback for an
    unknown singer.
  - `results/demo_confound_crnn/` (pass-2 `confound_crnn`, kept for
    reference): top-3 madonna (0.458), radiohead (0.175), prince (0.114).
  - Discussion angle: none of the three models fully agree with each other
    on this out-of-distribution input (though madonna, fleetwood_mac, and
    roxette all show up across at least two of the three), which is itself
    informative — with the true singer entirely absent from the 20-artist
    label space, "top-3" here reflects each model's own notion of
    vocal-timbre similarity, not a right/wrong answer, and the partial
    disagreement is a visible symptom of each model's differing
    representation quality rather than a bug in any of them.

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
