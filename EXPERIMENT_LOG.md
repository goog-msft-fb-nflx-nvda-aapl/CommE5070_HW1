# Experiment log

## 2026-08-30 — 14-model ensemble is the new graded submission (0.861/0.913)

`results/ensemble4/`: grid search over all 14 trained models found
**sota_crnn×1 + sota_crnn_wide×1 + sample_cnn×1 + sota_crnn_norm×2 +
crnn_nasrullah_faithful×1 + crnn_nasrullah_asp×2 + singer_senet×2 → val
top1=0.861, top3=0.913** — up from the 12-model ensemble's 0.857/0.913
(top1 improved, top3 flat). `singer_senet` and both prior-year-CRNN ports
earned real weight this time; `nonlocal_singernet` still didn't (weight 0).

Ensemble trajectory this session, each step a new architecture not a
re-tuned weight search on the same pool: 9 models → 0.853/0.905, 12 models
(+ round-5's three ablations) → 0.857/0.913, 14 models (+ singer_senet/
nonlocal_singernet) → 0.861/0.913. Promoted to the graded submission —
`results/R13921031.json` regenerated (233 tracks, schema-checked),
`readme`/`MATERIALS.md` updated with the new command and numbers.

At this point the ensemble candidate pool covers every architecture either
this project or our own prior-year submission has produced, plus every
round-5 deep-research suggestion that was cheap enough to test directly
(ArcFace, ASP, both channel-ramped backbones). Remaining un-pursued items,
noted for completeness rather than as an active queue: DAMFF (checked,
task-mismatched, not implemented — see above), a self-consistency/majority-
vote improvement to the zero-shot audio-LLM baseline (already flagged
`MATERIALS.md`, time-boxed out earlier), and the t-SNE/own-voice demo
(blocked on the user supplying a clip).


## 2026-08-30 — singer_senet ties best single model; nonlocal_singernet finished; 14-model ensemble launched

`singer_senet` and `nonlocal_singernet` (faithful ports of the prior-year
submission's channel-ramped SE-ResNet / classic non-local ResNet — see the
entry below this one for what they're testing) finished:

| model | val top1 | val top3 |
|---|---|---|
| `singer_senet` | **0.805** (186/231, exactly tying `sota_crnn_wide`) | 0.883 |
| `nonlocal_singernet` | 0.775 | 0.874 |

`singer_senet` is the first model in this project to match `sota_crnn_wide`
on top1 while being architecturally unrelated (channel-ramped 32->512
SE-ResNet, no recurrence at all, vs. `sota_crnn_wide`'s CRNN) — this is
exactly the kind of result that should help ensemble diversity, per both
round-5 responses' (correctly, this time) shared claim that architectural
diversity matters more than raw individual accuracy. `nonlocal_singernet`
underperforms both, similar to `crnn_nasrullah_faithful`'s pattern (another
of the prior submission's individually-weaker-but-different architectures).

Net read on the Gemini/Qwen disagreement this was meant to settle: **mixed,
leaning Gemini** — porting the channel-ramped backbones was worth it for
`singer_senet` specifically (ties best single model, adds real diversity),
but not for `nonlocal_singernet` (classic non-local attention underperforms
both `fgnl`'s more parameter-efficient generalized non-local block and the
plain channel-ramped SE variant) — so Qwen's caution about non-local
specifically holds, while Gemini's optimism about channel-ramped backbones
generally holds for the SE variant. Neither engine's response distinguished
between the two prior-year architectures this cleanly; worth noting in the
report as a case where the literature-review layer couldn't have predicted
which specific variant would work without the training run.

Re-ran the ensemble grid search over all 14 trained models (`results/
ensemble4/`, 3^14 ≈ 4.78M weight combinations) — results pending.


## 2026-08-30 — DAMFF checked and not pursued (another mismatched round-5 citation)

While the two prior-year-architecture ports (`singer_senet`,
`nonlocal_singernet`) trained, checked whether Gemini's round-5-cited DAMFF
("Dual Attention-based Multi-scale Feature Fusion," works-cited #22,
archives.ismir.net/ismir2023/paper/000023.pdf) was worth porting as a fourth
new architecture. Fetched and read the actual paper rather than trusting
Gemini's paraphrase (same verify-before-building standard as the faithful
architecture ports) — it's real and correctly a 2023 ISMIR paper, but it's
**"Dual Attention-based Multi-scale Feature Fusion Approach for Dynamic
Music Emotion Recognition"** (Zhang, Yang, Zhang, Luo): a valence-arousal
*regression* method for continuous emotion prediction over time (BiLSTM +
FC regression head, evaluated on MER1101/DEAM2015), not a singer/artist
classification paper and not validated on anything like Artist20's task or
scale. Gemini's response presented it as directly relevant precedent for
"small-data audio classification" without noting the task/domain mismatch —
a second instance of the same citation-quality issue as footnote 12 (see the
round-5 entry below): a real, correctly-linked paper, but its applicability
here was overstated, not fabricated outright. The underlying idea (parallel
multi-scale conv branches + a channel+spatial dual-attention fusion module,
"SCAM") is architecturally adaptable to classification in principle, but
without a task-relevant precedent to anchor confidence in the port, and with
GPU capacity already committed to two better-grounded ablations, **not
implemented this pass** — noted here so it isn't silently dropped, and can
be revisited if the current queue clears with idle GPU time to spare.


## 2026-08-30 — resolving the Gemini/Qwen non-local/SE-ResNet disagreement directly

Round-5's one unresolved item: Gemini recommended porting the prior-year
submission's channel-ramped (32->512) SE-ResNet and classic Wang et
al.-style non-local ResNet as ensemble members; Qwen argued the opposite
(claimed our existing `se_resnet`/`fgnl` are already the more parameter-
efficient, better-suited-to-950-tracks choice, and that porting a naive
channel-ramp risks overfitting). Neither response sourced the disagreement
to a citation about our actual data scale. Per project convention, settling
disagreements like this by running the experiment rather than picking a
side on priors.

Re-cloned the prior-year submission (sparse checkout, same as before) and
read `task2_se_cnn.py`/`task2_nonlocal.py` in full this time (the earlier
investigation only had partial `grep` context for these two, not enough to
port faithfully — re-fetched to avoid guessing at architecture details).
Ported both exactly: `src/models/singer_senet.py` (`SingerSENet` —
Conv7x7 stem, 4 residual stages 32->64->128->256->512 with SE gating in
every `ConvSEBlock`) and `src/models/nonlocal_singernet.py`
(`NonLocalSingerNet` — same stem/stage structure, but with a classic
Wang et al. non-local block, *not* the FGNL block this project's `fgnl`
already uses, inserted into stages 2-4). Both keep their original scripts'
mel frontend (n_mels=128, n_fft=2048, **hop=256** — narrower hop than the
CRNN port's hop=512, kept faithfully per-model rather than silently
unified), per-sample mel normalization, and 10s crops; training recipe
standardized to this project's shared cosine+AdamW+300-epoch/40-patience
convention rather than each original script's own scheduler (OneCycleLR /
plain CosineAnnealingLR at 120 epochs), matching how `crnn_nasrullah_faithful`
was handled.

Smoke-tested both locally (forward+backward, correct output shapes:
SingerSENet 11.4M params, NonLocalSingerNet 5.9M params) before launching.
`--model singer_senet` on GPU 0 (batch_size=32), `--model
nonlocal_singernet` on GPU 1 (batch_size=16, smaller — the non-local block's
O(N^2) attention over a hop=256 feature map is more memory-hungry than the
other architectures trained so far). tmux windows `singer_senet` /
`nonlocal_singernet`. Results pending — will fold into MATERIALS.md and
re-run the ensemble grid search (now against 14 candidates) once both land.


## 2026-08-30 — round-5 ablations finished; new 12-model ensemble, new graded submission

All three round-5 ablations finished:

| model | val top1 | val top3 |
|---|---|---|
| `crnn_nasrullah_faithful` (faithful prior-year CRNN port, 10s + BiGRU-256 + attention) | 0.779 | 0.866 |
| `sota_crnn_wide_arcface` (ArcFace margin head on `sota_crnn_wide`) | 0.801 | 0.896 |
| `crnn_nasrullah_asp` (attentive statistics pooling, same backbone as the faithful port) | 0.762 | 0.861 |

None individually beats `sota_crnn_wide` (0.805/0.922). This is a stronger
negative result than the original `sota_crnn_attn` ablation: this time the
backbone has exactly the capacity (bidirectional GRU-256) both round-5
responses said was the missing ingredient for attention pooling to work, and
it still underperforms our simpler architecture's plain last-GRU-state
pooling. The "attention pooling only hurt because of a narrow backbone"
hypothesis (ours, corroborated by both engines) does not survive this direct
test — updated the ablation-takeaways bullet in MATERIALS.md to reflect
that this is now evidence *against* full-sequence attention pooling for
this task/data scale specifically, not an unresolved confound anymore. ASP
(mean+std) also underperformed plain attention on the identical backbone.

Re-ran the ensemble grid search (`src/ensemble.py`) over all 12 trained
models (the original 9 + these 3), `results/ensemble3/`. New best: **sota_crnn×2
+ short_chunk_cnn×1 + sota_crnn_wide_arcface×1 → val top1=0.857, top3=0.913**
— beats the previous 9-model ensemble (0.853/0.905) on both metrics, despite
none of the 3 new individual models beating `sota_crnn_wide` outright.
`sota_crnn_wide` and `sota_crnn_norm` both dropped to weight 0; only
`sota_crnn_wide_arcface` from the new batch earned nonzero weight (1) —
`crnn_nasrullah_faithful`/`crnn_nasrullah_asp` did not make the cut, despite
being the most architecturally distinct of the three. Grid search now covers
3^12 ≈ 531,441 weight combinations (up from 3^9 ≈ 19,683) — noted in
MATERIALS.md that the overfit-to-val-in-weight-selection risk is
correspondingly larger; the individual per-architecture numbers remain the
more robust comparison.

This is now the graded submission — regenerating `results/R13921031.json`
via `src/ensemble.py`'s `--test_out_path` (in progress as of this entry);
update `readme` and the checkpoint list once it lands.


## 2026-08-30 — round-5 responses read; ArcFace + ASP ablations launched

Gemini and Qwen's round-5 responses landed
(`deep_research/round5_prior_year_gap_and_latest_literature/response_{gemini,qwen}.md`).
Both agree the "attention pooling hurt" result is plausibly a capacity
artifact (narrow backbone, not attention pooling itself) and that 10s chunks
should outperform 5s for singer ID specifically — both already being tested
by the `crnn_nasrullah_faithful` run launched just before these came back
(83 epochs in, val 0.706/0.853 and climbing, ahead of `sota_crnn`'s
comparable-epoch trajectory).

Both also agree architectural ensemble diversity (CRNN + SE-ResNet +
Non-local, per our prior-year run) should beat hyperparameter-variant
ensembles of one backbone family — consistent with our own measured
+4.8pp (9 same-family variants) vs. the prior run's own +11pp (3 distinct
architectures). Where they **disagree**: Qwen says don't port the prior
submission's classic Wang-et-al. non-local ResNet / channel-ramped SE-ResNet
(claims our `fgnl`, a more parameter-efficient singer-ID-specific paper, is
already the "modern evolution" and a naive channel-ramp to 512 would overfit
949 tracks); Gemini's own comparison table rates a channel-ramped
non-local ResNet "Very High" generalization and recommends porting it as
ensemble Phase 3. Neither engine sources this specific disagreement to a
citation about *our* data scale — flagged as unresolved, not adopted either
way yet. Lower priority than the items below since our existing
`se_resnet`/`fgnl` already score competitively (0.758/0.710) with the prior
submission's individual numbers; revisit if GPU time allows once the
current queue clears.

**Citation-quality check** (per standing practice — verify hyperparameter/
literature claims rather than trust or dismiss on priors): Qwen's specific
numeric claim ("Kuo et al. AAAI 2021 segment-length ablation: 0.73 at 3s,
0.74 at 5s, 0.79 at 10s") uses bracket markers ([[176]] etc.) with no
resolvable bibliography — unverified, not adopted as a specific number
(though the *direction*, longer chunks helping, is exactly what
`crnn_nasrullah_faithful` is independently testing). Gemini's response does
include a real, checkable works-cited list (24 numbered links, several
legitimately relevant — e.g. arXiv:1901.04555 is actually the Nasrullah &
Zhao CRNN paper we already cite, and ISMIR 2023's "Dual Attention-based
Multi-scale Feature Fusion" paper is real and on-topic) — but footnote [12],
cited repeatedly for the *entire* ensemble-diversity/ambiguity-decomposition
argument (the quantitative claims backing "port more architectures"), traces
to arXiv:2507.03690, **"Graph Neural Networks for Electricity Load
Forecasting"** — completely unrelated to audio or ensembling theory. The
underlying ambiguity-decomposition math (bias-variance-covariance for
ensembles) is textbook-correct regardless, but the citation attached to it
is wrong/fabricated — treat the *qualitative* claim (architectural diversity
> hyperparameter diversity) as literature-plausible and already
independently corroborated by our own measurement, but don't cite footnote
12 itself as a source for anything audio-specific.

**Two new ablations launched**, techniques neither engine's core claims
needed us to trust blindly — both are cheap, directly testable, and target
this project's actual open questions:
1. **ArcFace / AAM-Softmax margin head** (`src/models/arcface_head.py`,
   `SotaCRNNArcFace` — `sota_crnn_wide`'s encoder + an angular-margin
   classifier instead of a plain linear layer, s=24/m=0.2, chosen as a
   from-scratch-eligible way to make the embedding production/album-
   invariant — directly on-topic for Hsieh et al.'s own confound framing,
   not previously tried in this project). `--model sota_crnn_wide_arcface`,
   new `requires_labels_in_forward` flag on the model class so
   `src/train.py`'s loop passes `y` into `forward()` only for models that
   need it (ArcFace needs labels to apply the margin during training; eval
   calls `model(x)` and gets plain scaled-cosine logits, unaffected).
   Smoke-tested (forward/backward, correct shapes) before launching. Launched
   on GPU 1, tmux `crnn_arcface`.
2. **Attentive Statistics Pooling (ASP)** (`CRNNNasrullahASP` in
   `src/models/crnn_nasrullah_faithful.py` — same conv/BiGRU trunk as
   `crnn_nasrullah_faithful`, weighted mean *and* weighted std concatenated
   instead of weighted-mean-only attention pooling). Deliberately built on
   the *same* already-capacity-matched backbone as the plain-attention run,
   not `sota_crnn`'s narrow one — a clean ASP-vs-plain-attention comparison
   that isn't confounded by backbone width the way this project's earlier
   `sota_crnn_attn` ablation was. Smoke-tested, launched on GPU 2, tmux
   `crnn_asp`.

Both wired into `src/ensemble.py`/`src/infer_test.py` automatically (no
model-specific changes needed there — they dispatch on `kind`, which was
already handled for `crnn_nasrullah_faithful`/`wave10s`, or falls through to
the generic `wave` branch for the ArcFace model). Check
`results/sota_crnn_wide_arcface/summary.json` and
`results/crnn_nasrullah_asp/summary.json` once done; fold into
`MATERIALS.md` and the ensemble candidate pool alongside
`crnn_nasrullah_faithful`.


## 2026-08-30 — prior-year submission re-investigated; faithful-CRNN gap found and launched

User asked us to re-investigate our own prior-year submission
(github.com/goog-msft-fb-nflx-nvda-aapl/NTU, `CommE5070.../hw1/hw1_submission`)
since its reported ensemble (0.825/0.949) looked like it might beat this
project's current ensemble (0.853/0.905). Cloned it (sparse checkout, partial
blobless clone — the `NTU` repo is ~800MB total across many unrelated
courses) and read `report.md` + all 4 model/inference scripts in full.

**Not a like-for-like comparison.** Their `dataset.py::get_split` self-splits
train_val by taking each artist's alphabetically-last album as "val"
(946 train / 234 val) — a different val set from this project's official,
assignment-provided `train.json`/`val.json` (949/231). Their val accuracy and
ours are measured on different tracks; a raw number comparison isn't valid
without re-running one pipeline on the other's split. Under the report's own
guessed scoring formula (top1 + 0.5×top3) our ensemble already edges theirs
slightly: ours 0.853+0.5×0.905=**1.305** vs theirs 0.825+0.5×0.949=**1.2995**
— but this is on different val sets, so read this as "not obviously behind,"
not "ahead."

**Real gap found, not previously closed.** Our existing ablations tested the
divergent *ingredients* of their CRNN (attention pooling via `sota_crnn_attn`,
per-sample mel norm via `sota_crnn_norm`) each bolted onto `sota_crnn`'s
architecture — a small, unidirectional, 32-hidden-dim GRU bottleneck (see
`src/models/sota_cnn.py`). Their actual CRNN is a *different, larger*
architecture entirely: bidirectional GRU (hidden=256, 2 layers), attention
pooling over the *whole* sequence, 4 Conv-BN-ELU blocks up to 256 channels,
10s training crops, f_min=20/f_max=8000, `top_db=80`. Nobody had ported this
whole thing faithfully as one model — so "attention pooling hurt (-10.8pp)"
in this project's ablation table tests attention pooling *on a low-capacity
backbone that isn't theirs*, not attention pooling itself. This confounds
the negative result, doesn't necessarily refute their approach.

Ported it faithfully as `src/models/crnn_nasrullah_faithful.py`
(`CRNNNasrullahFaithful`) — every architectural/training-recipe choice from
their `task2_dl_v2.py` copied as-is (see the model's docstring for the exact
line-by-line diff vs. `sota_crnn`/`CRNN_Attn`/`crnn_zain`), changing only
what's forced by using this repo's official split and song-level mean-pooled
eval instead of their self-split + 10-random-crop TTA (that TTA method is
separately already shown *not* to beat our default full-track averaging —
`src/analysis_tta_comparison.py`, see the 2026-08-30 "lecture02 scrutiny"
entry below — so intentionally not re-adopted here). Added `--model
crnn_nasrullah_faithful` to `src/train.py` (new `"wave10s"` dataset kind,
`CHUNK_SAMPLES_10S` in `src/data/dataset.py`; wired into `src/ensemble.py`
and `src/infer_test.py` too so it can join the ensemble once trained).
Smoke-tested locally (forward + embed pass, correct output shapes, 5.6M
params) before launching. Also queued but not yet ported: their SE-ResNet
(`SingerSENet`, 32→512 channels, 4-stage) and Non-local net
(`NonLocalSingerNet`, classic Wang et al. theta/phi/g block + SE, same
32→512 channel ramp) — both structurally different from our current
`se_resnet`/`fgnl` (different papers/depth-width tradeoffs), lower priority
than the CRNN gap since our `se_resnet`/`fgnl` already score competitively
with theirs on our own val set.

Launched: `python -m src.train --model crnn_nasrullah_faithful
--data_index_dir data/index --out_dir results/crnn_nasrullah_faithful
--optimizer adamw --label_smoothing 0.1 --lr 3e-4 --weight_decay 1e-4
--epochs 300 --patience 40`, tmux window `crnn_nasrullah` in the
`hw1_singer` session, GPU 0. Check `results/crnn_nasrullah_faithful/
summary.json` / `train_crnn_nasrullah.log` for the outcome; fold into
`MATERIALS.md`'s table and `src/ensemble.py`'s candidate pool once done.

Also drafted a 5th deep-research round
(`deep_research/round5_prior_year_gap_and_latest_literature/prompt.md`)
covering the problem statement, our current pipeline/architectures/
augmentation, this investigation's findings, and asking for a deep-dive
recommendation plus latest ICASSP/InterSpeech/AAAI/NeurIPS singer-ID papers.


## 2026-08-30 — all 14 ablation-queue jobs finished; new graded submission

Results (val, full table in MATERIALS.md): `sota_crnn_wide` (channel_mult=1.5
capacity sweep) is the new best single from-scratch model at **0.805/0.922**,
beating the original `sota_crnn` (0.762/0.900). A 9-model weighted ensemble
(`src/ensemble.py`, grid-searched weights) reaches **0.853/0.905** —
sota_crnn×2 + sota_crnn_wide×1 + short_chunk_cnn×1 + sota_crnn_norm×1 — now
the graded `R13921031.json` submission.

Per-ablation results (all variants of the `sota_crnn` architecture):
capacity up helped (+4.3pp), capacity down hurt (-10.8pp) — direct contradiction
of Qwen round-3's uncited "smaller is better" claim, resolved by testing
rather than trusting it; per-sample mel norm helped slightly (+0.4pp); SSL
pretraining (round-4 recipe) helped slightly (+0.4pp, in-line with the
"small positive, not a CLMR-sized jump" prior every round-4 response stated);
attention pooling hurt badly (-10.8pp); SupCon hurt (-2.6pp on top of
AdamW/LS); DropBlock hurt (-5.6pp on top of AdamW/LS); SWA's weight-averaged
checkpoint was worse than its own run's best non-averaged checkpoint
(-3.5pp). `fgnl` isolation confirmed SpecAugment specifically (not the LR)
was what hurt it after the pass-2 fix — no-augment retry recovered to 0.710,
matching crnn_zain.

Score under the assignment formula (top1 + 0.5×top3) on the graded ensemble:
0.853 + 0.5×0.905 = **1.305** (val), up from the original flagged-as-low
1.083.


## 2026-08-30 — full ablation queue launched (14 items, all from the batch-2 list)

Every item from the earlier master queue is now either done or running.
Added since batch 2: `src/supcon_loss.py` (SupCon auxiliary loss, smoke-
tested, launched as `sota_crnn_supcon`), `src/models/dropblock.py` +
`CRNN_DropBlock` (smoke-tested, launched as `sota_crnn_dropblock`),
`src/data/ssl_dataset.py` + `src/pretrain_ssl.py` (SimCLR/NT-Xent
pretraining reusing `sota_crnn`'s encoder, round-4 recipe — smoke-tested
both the pretrain loop and the `--init_encoder` fine-tune path; chained job
`ssl_pretrain` window runs 200-epoch pretrain then auto-launches fine-tune
into `sota_crnn_ssl_finetune`).

Running (13 windows + the ensemble/TTA-comparison already resolved):
short_chunk_cnn, sample_cnn, fgnl_noaug, sota_adamw_ls, se_resnet,
crnn_attn, crnn_narrow, crnn_wide, crnn_norm, sota_swa, sota_supcon,
crnn_dropblock, ssl_pretrain(→ssl_finetune). Check `results/<name>/
summary.json` (or `summary_swa.json` for the SWA run) for whichever have
landed — none had finished as of this entry. Training server load ~42/128 cores,
all 4 GPUs well under VRAM limits — no further capacity concerns.


## 2026-08-30 — batch 2: 11 concurrent jobs launched (user left GPU running unattended)

All on our training server, tmux session `hw1_singer`, epochs=300/patience=40 (patience
60 for SWA) unless noted, all with `--optimizer adamw --label_smoothing 0.1`
unless noted otherwise:

| window | model | notes |
|---|---|---|
| short_chunk_cnn | ShortChunkCNN_Res | plain (Adam, no LS) — new arch |
| sample_cnn | SampleCNN | plain — new arch |
| fgnl_lr4 (running noaug script) | fgnl | `--no_augment`, isolates whether SpecAugment specifically hurt fgnl |
| sota_adamw_ls | sota_crnn | AdamW + label_smoothing ablation vs. the plain-Adam 0.762 result |
| se_resnet | SEResNet | new arch (`src/models/se_resnet.py`), SE gating on Res_2d, per user's prior-run design |
| crnn_attn | sota_crnn_attn (`CRNN_Attn`) | attention pooling instead of last-GRU-state, per user's prior-run design |
| crnn_narrow | sota_crnn_narrow | channel_mult=0.5 capacity sweep |
| crnn_wide | sota_crnn_wide | channel_mult=1.5 capacity sweep |
| crnn_norm | sota_crnn_norm | per-sample mel normalization, per user's prior-run pipeline |
| sota_swa | sota_crnn + SWA | patience=60, swa_start_frac=0.7 |

Code added this batch: `src/ensemble.py` (weighted grid-search ensemble),
`src/analysis_tta_comparison.py` (single/random-10/full-tile TTA
comparison), `SqueezeExcite2d`/`Res_2d_SE` in `common.py`,
`src/models/se_resnet.py`, `CRNN_Attn`/capacity-mult/normalize_mel in
`sota_cnn.py`, `--optimizer`/`--label_smoothing`/`--swa*` flags in
`train.py`. One bug caught by smoke-testing before launch: `x.view()` on a
non-contiguous tensor in the normalize_mel path — fixed to `.reshape()`
before any real run used it.

**Not yet implemented** (queued, not skipped): DropBlock, supervised
contrastive (SupCon) auxiliary loss, SimCLR/BYOL self-supervised
pretraining (round-4 recipe). Picking up as GPU slots free / time allows.

Once each job's summary.json lands, results get folded into MATERIALS.md
with the full comparison table — check `results/<name>/summary.json` for
raw numbers before that happens.


## 2026-08-30 — ablation queue results (batch 1)

1. **Ensemble** (`src/ensemble.py`, `results/ensemble/ensemble_result.json`):
   grid-searched integer weights 0-3 over confound_crnn/crnn_zain/sota_crnn/
   fgnl(v1) on val. Best: crnn_zain×1 + sota_crnn×2 → **top1=0.7922,
   top3=0.8874**, +3.0pp over sota_crnn alone (0.7619). Smaller gain than the
   user's prior run's +11pp — plausible cause: our 4 models are more
   correlated (all CRNN-family, similar training recipe) vs their 3
   architecturally-distinct models; also 231 val tracks with a 4^4=256-point
   weight grid search carries real overfit-to-val risk, flagged not hidden.
6. **Random-crop TTA comparison** (`src/analysis_tta_comparison.py`,
   `results/analysis/tta_comparison_sota_crnn.json`), directly testing the
   user's prior-run claim rather than assuming it doesn't transfer: on
   sota_crnn val — single random crop **0.550**, 10 random crops averaged
   **0.723**, our default full-track non-overlapping-tile average **0.762**.
   Confirms their finding's *shape* (multi-crop >> single-crop, a real and
   large effect) but also confirms our existing default already captures
   more of that gain than their exact 10-crop method would, since it covers
   the whole track rather than a random subsample. Time-shift TTA (batch 0,
   see lecture02 section below) stays a negative result; this random-crop
   variant is a *positive but already-subsumed* result — no change needed
   to the submission pipeline.
2. **AdamW + label_smoothing=0.1** on sota_crnn: launched
   (`results/sota_crnn_adamw_ls/`, `train_sota_adamw_ls.log`), in progress.

Still queued (3-5, 7-13 from the list below): per-sample mel normalization,
10s segments, attention pooling, SE blocks, capacity sweep, DropBlock, SWA,
SupCon, SSL pretraining. Launching as GPU slots free.

## 2026-08-30 — lecture02 scrutiny + user's prior-run comparison + master ablation queue

**lecture02_classification.md scrutiny**: added `ShortChunkCNN_Res` and
`SampleCNN` (both minzwon/sota-music-tagging-models, spotlighted in the
lecture) — training. Added time-shift TTA to `src/evaluate.py` — measured on
`sota_crnn`: **hurt** slightly (76.2%→75.8% top1, flat top3). Not used for
submission; kept as a documented negative result, not deleted.

**User's prior run** (github.com/goog-msft-fb-nflx-nvda-aapl/NTU, same
assignment, downloaded to
`/private/tmp/.../scratchpad/prev_run/`): ensemble of 3 architectures
(CRNN/SE-ResNet/NonLocal, weighted 1/2/1) went 0.714 best-individual →
0.825 ensemble on their val set — the single largest lever in either
project. Their TTA (10 independent random crops averaged) reportedly gave
CRNN 0.65→0.77, but their *non-TTA baseline* was a single random 10s crop —
much weaker than our baseline (already full-song non-overlapping-chunk
averaging), so their result doesn't contradict our own TTA finding; it's
not evaluated against the same baseline. Still testing our own random-crop
variant directly rather than assuming per user instruction #2. Other
differences worth testing directly rather than assuming irrelevant:
per-sample mel normalization (mean/std), 10s segments (vs our 5s), label
smoothing, AdamW vs Adam, attention pooling vs last-GRU-state pooling, SE
blocks.

**Master ablation queue** (user: don't ask, list and run all, parallel where
GPU allows; don't reject anything on plausibility alone):
1. Weighted-average ensemble of all trained from-scratch models — highest
   validated lever, implementing first.
2. AdamW (decoupled weight decay) vs current Adam+L2 — cheap flag.
3. Label smoothing 0.1 — cheap flag.
4. Per-sample mel normalization (mean/std per chunk) — cheap flag.
5. 10s segments vs current 5s — needs dataset param change.
6. Random-crop TTA (N=10 independent crops, matching user's prior-run
   method exactly) vs single-crop baseline vs our full-tile averaging —
   direct 3-way comparison.
7. Attention pooling (vs last-GRU-timestep) for the CRNN family.
8. SE (squeeze-excitation) blocks — new architecture.
9. Capacity sweep (0.5x / 1x / 1.5x channels) — Qwen round-3, previously
   discounted for being uncited, now queued to actually test.
10. DropBlock — Qwen round-3.
11. SWA (stochastic weight averaging) — Qwen round-3.
12. Supervised contrastive (SupCon) auxiliary loss — Qwen/Perplexity round-3.
13. SimCLR/BYOL self-supervised pretraining (same-track crops, decoupled
    NT-Xent, reuse CRNN encoder) — round-4 consensus recipe (Perplexity's
    version most concretely sourced; Gemini's expectation table: +1-3pp
    realistic, +3-6pp good, >6pp unlikely — kept as a stated prior, not a
    reason to skip the experiment).
14. fgnl no-augment isolation run — already in progress, will resolve
    whether SpecAugment specifically is what hurt it.

## 2026-08-30 — Task 2 undertraining fix + remix ablation + graded model swap

User flagged the Task2 formula score (top1 + 0.5*top3, then 0.671+0.5*0.823=
1.083) as "extremely low" and asked to (1) review correctness, (2) draft a
survey prompt for further improvement, (3) confirm the allowed training data.

- (3) `train.json` only — `val.json` is validation-only, test folder
  off-limits, no outside data (both source docs agree).
- (1) All 4 from-scratch models' first training pass hit their 80-epoch cap
  without ever early-stopping (patience=15), val accuracy still swinging
  ±8-10pp epoch to epoch — undertraining, not an architecture ceiling. Also:
  zero data augmentation despite the assignment doc asking for it. Added
  SpecAugment (mel models) / gain-jitter+noise (waveform model) to
  `src/data/dataset.py`, a cosine LR schedule + weight decay to
  `src/train.py`, raised epochs to 300/patience 40, backed up the old
  checkpoints as `results/*_v1_undertrained/`, retrained all 4 in parallel.
  Results: sota_crnn 58.4%→76.2% (+17.8pp), crnn_zain 61.9%→71.0% (+9.1pp),
  confound_crnn 67.1%→69.3% (+2.2pp), fgnl 57.1%→52.4% (**regressed**).
- Also implemented cross-song vocal/instrumental remix augmentation
  (`src/data/remix_dataset.py`, `src/data/separate_accompaniment.py`) — the
  top from-scratch-compatible lever per a targeted third Deep Research round
  (`deep_research/round3_from_scratch_improvement/prompt.md`), since it's literally the technique from
  Hsieh et al.'s own paper (the architecture `confound_crnn` ports). Effect
  in our setup was much smaller than the literature's (+1.3pp top3, flat
  top1) — see MATERIALS.md for the honest writeup of why, not tuned to force
  a bigger number.
- fgnl's regression traced to a plausible cause: the original FGNL paper
  (per Perplexity's sourced claim, not corroborated but also not
  contradicted) used a constant LR of 1e-4; we'd applied 1e-3 uniformly to
  every model. Retry at lr=1e-4 launched (`train_fgnl_lr4.log`), in progress
  as of this entry.
- **Graded submission swapped**: `sota_crnn` (76.2%/90.0%) is now the best
  from-scratch model, clearly ahead of `confound_crnn` (69.3%/82.7%) — the
  Artist20 paper's own architecture didn't win. Regenerated
  `results/R13921031.json`, t-SNE, and confusion matrix from `sota_crnn`;
  `readme` and `MATERIALS.md` updated throughout.
- Minor false alarm during this work: `confound_crnn` and `confound_crnn_remix`
  landed on the *exact* same val top1 (0.6926, both at best_epoch=241) despite
  visibly different training logs throughout (different loss trajectories,
  different per-epoch top3, different per-epoch timing) — initially worried
  this was another queued-tmux-input-style bug, but confirmed both runs are
  genuinely independent (checked raw per-epoch log lines, confirmed differing
  top3 and loss at the "tied" epoch) — the tie is a real coincidence: cosine
  annealing naturally pushes both runs' best checkpoint toward the
  low-LR/late-epoch region, and with only 231 val tracks (232 possible top1
  values) two similarly-converged models landing on the same value isn't
  actually that improbable. No data-integrity issue this time.

## 2026-08-30 — TA clarifications (TA_discussion.md) changed the graded submission

User pointed to `TA_discussion.md` (Q&A thread from the course TA). Two
important clarifications that weren't obvious from the assignment doc alone:

1. **Task 1's accuracy doesn't count toward the grade at all** — TA grades
   only on whether the analysis process is reasonable/thorough. Retroactively
   validates prioritizing the feature-group ablation + permutation importance
   work over chasing a higher SVM number.
2. **Task 2 requires training both encoder and classifier from scratch** — a
   pretrained encoder (frozen or fine-tuned) is explicitly baseline-only, not
   eligible as the graded submission (TA_discussion.md #4: "task2 需要自己重新訓
   一個 encoder + classifier"). This directly contradicts what had been treated
   as the project's headline result: `speaker_frontend` (frozen ECAPA-TDNN +
   trained head, 95.2% top1) was the model used to generate the official test
   submission — not compliant per this clarification.

**Action taken**: swapped the graded/submitted model to `confound_crnn` (from
scratch, `CRNN2D_elu2`, 67.1%/82.3% top1/top3) — the best fully-from-scratch
model. Regenerated `results/R13921031.json` from it. Regenerated the t-SNE and
mel-spectrogram/inference demo using `confound_crnn` too (previously only had
these for `speaker_frontend`) so the "required" deliverables come from an
eligible model, while keeping `speaker_frontend`'s versions as baseline
comparisons — showing both side by side is more informative anyway (visual
cluster/accuracy gap tracks exactly as expected). Updated `readme` (TA-facing)
and `MATERIALS.md` throughout to distinguish "graded submission" (from-scratch
models) from "baseline" (`ssl_frontend`/`speaker_frontend`, pretrained
encoders) rather than silently relabeling — `speaker_frontend` is still the
most interesting single result in the project and is kept prominent in
`MATERIALS.md` as a baseline, just no longer framed as "the best model" for
grading purposes.

Also relevant, no action needed: dataset was already the corrected full-song
version (TA_discussion.md #1) and downloaded from the provided local copy, not
the Artist20 website — no issue there. Test set's 4 known-instrumental tracks
(074/119/169/206, TA_discussion.md #2) don't need special handling —
`infer_test.py` already produces a prediction for every track uniformly, and
the TA said those results won't affect scoring.

## 2026-08-29 — Student ID confirmed: R13921031

Renamed `results/STUDENT_ID.json` → `results/R13921031.json`, updated all
`STUDENT_ID` placeholder references in `README.md`/`readme`/`MATERIALS.md`/
`src/infer_test.py` to the real ID. `R13921031_report.html` still needs
building (deferred to the Claude-web pass over `MATERIALS.md`, per plan).

## 2026-08-29 — Google Drive checkpoint upload — final status: needs manual step

All 6 checkpoints stripped down to small files (largest 3.5MB, smallest 220KB —
see `results/*/best*.pt` locally) and confirmed correct. Tried uploading the
smallest one (220KB / 294KB base64) via `mcp__claude_ai_Google_Drive__create_file`
and hit a **hard wall**: the file has to be read into the assistant's context
first to pass as inline `base64Content`, and the `Read` tool caps at 256KB —
so even the smallest checkpoint (294KB base64-encoded) can't be relayed this
way. No chunked/append upload path exists on this tool either.

**Action needed from the user**: drag these files into a Google Drive folder and
set sharing to "anyone with the link" (a setting this MCP tool also can't set —
it only supports sharing to one specific email address):
- `results/speaker_frontend/best_head_only.pt` (220KB) — **the graded model**
- `results/ssl_frontend/best_head_only.pt` (810KB)
- `results/confound_crnn/best.pt`, `results/confound_crnn_vocals/best.pt`,
  `results/crnn_zain/best.pt`, `results/sota_crnn/best.pt`, `results/fgnl/best.pt`
  (1.6-3.5MB each, full checkpoints — these have no frozen backbone to strip)

Once uploaded, update `MATERIALS.md`'s Links section and the first page of the
report with the folder link. Not blocking anything else in the assignment —
everything else (code, results, test predictions) is already in the GitHub repo.

## 2026-08-29 — session 2 notes (mid-session, training in progress)

- Deep Research responses (3 engines, saved by user as `deep_research/round1_sota_and_architecture_survey/response_*.md`,
  now committed) synthesized into `MATERIALS.md`. Key finding to remember when writing
  up the vocal-separation ablation: the engines *disagree* on whether separation helps
  or hurts — report our own measured delta, don't assume a direction.
- **Google Drive checkpoint upload plan**: `mcp__claude_ai_Google_Drive__create_file`
  only accepts inline `base64Content`/`textContent` (no streaming upload of a large
  local file), and `share_file` only grants access to a specific `emailAddress`, not
  Drive's "anyone with the link" permission — so the upload path needs to be small
  files uploaded through this tool, not a multi-hundred-MB checkpoint. Fix: for
  `ssl_frontend`/`speaker_frontend` (frozen backbone + trainable head), the deliverable
  checkpoint should only contain the trainable head's `state_dict` (a few hundred KB),
  not the full model (which currently includes the frozen MERT/ECAPA backbone weights,
  ~100s of MB) — the inference script already re-downloads those backbones fresh from
  HuggingFace via `from_pretrained`, and the assignment explicitly forbids uploading
  third-party model weights anyway. `train.py`'s current checkpoint save does *not* do
  this yet (saves full `state_dict()` for every model) — write a small
  `strip_checkpoint.py` at wrap-up time to re-save only `requires_grad=True` params for
  the two frozen-backbone models before uploading. CRNN-family checkpoints (fully
  trained from scratch, 1.5-3.5MB each) don't need this — small enough to upload as-is.
  If Drive's "anyone with link" truly isn't reachable through this MCP tool, ask the
  user to flip that one setting manually after upload (should be a single click in the
  Drive UI), since the tool can't set it.

## 2026-08-29 — session 1: scaffold + reference ports (paused, offline)

**Status when paused: no GPU job running.** Our training server's tmux session `hw1_singer` finished
its one-off conda env setup (`pip install ... > install_status.txt` shows `INSTALL_DONE`)
and is sitting idle at a bash prompt — nothing to kill. No training has started yet.

### Done
- Access checks: our training server (4x H200 NVL GPUs, 128 CPU cores, all idle, 1.1TB free), GitHub
  (`goog-msft-fb-nflx-nvda-aapl`, switched active), HuggingFace (reachable, no token —
  fine for public models, will flag if something needed turns out gated).
- Repo scaffolded at `/Users/chun-feitan/Desktop/CommE5070/HW1/CommE5070_HW1/`, own git
  repo (root `~/Desktop` is an unrelated BlueWX repo — never touch that level), pushed to
  `https://github.com/goog-msft-fb-nflx-nvda-aapl/CommE5070_HW1` (public).
- Training server: new conda env `hw1_singer_env` (python 3.10), packages installed — torch
  2.5.1+cu121, torchaudio 2.5.1+cu121, librosa, scikit-learn, transformers,
  huggingface_hub, matplotlib, seaborn, pandas, tqdm, soundfile, demucs, speechbrain.
  New tmux session `hw1_singer` on our training server (separate from the pre-existing, unrelated
  `oita_exp` session — don't touch that one).
- Cloned the 5 reference repos read-only into the local scratchpad
  (`/private/tmp/claude-501/.../scratchpad/refs/`, not part of this git repo) to port
  architectures faithfully:
  - `minzwon/sota-music-tagging-models` (Method 1)
  - `ZainNasrullah/music-artist-classification-crnn` (Method 2-1)
  - `bill317996/Singer-identification-in-artist20` (Method 2-2 / dataset paper)
  - `ian-k-1217/Fully-Generalized-Non-Local-Network` (Method 2-3)
  - `rssr25/voice-recognition-speak-sing` (Method 3) — cloned, not yet read/ported
- Code written and committed to this repo so far:
  - `src/data/prepare_index.py` — builds `labels.json`/`train.json`/`val.json`/`test.json`
    manifests from the raw `train.json`/`val.json`/`test/` dataset layout. **Not yet run**
    (needs `--data_root` pointed at the local `Hw1/hw1/artist20/` dir, or a copy of it on
    our training server).
  - `src/data/dataset.py` — two chunk representations: `MelChunk{Train,Eval}Dataset`
    (log-mel, n_mels=128, n_fft=2048, hop=512, chunk_frames=157 ≈5.02s, matching
    bill317996's `utility.py` slicing) for the CRNN family, and
    `Waveform{Train,Eval}Dataset` (raw 16kHz, 5s chunks) for sota-music-tagging-models /
    SSL / speaker-embedding frontends. Eval datasets return *all* chunks of a track for
    song-level aggregation at inference — not yet wired into an `evaluate.py`.
  - `src/models/common.py` — `Conv_1d`/`Conv_2d`/`Res_2d`, ported verbatim from
    sota-music-tagging-models.
  - `src/models/sota_cnn.py` — `FCN` + `CRNN`, ported from sota-music-tagging-models
    (Method 1), sigmoid→logits adaptation documented in the file's docstring.
  - `src/models/confound_crnn.py` — `CRNN2D_elu2`, ported from
    bill317996/Singer-identification-in-artist20 (Method 2-2 **and** our Task-2 "from
    scratch" core model, since this is the artist20 paper's own architecture). Chose the
    `_elu2` variant over the plainer `_elu` because only `_elu2`'s padding keeps the
    frequency axis alive through all 4 pooling stages at `T=157` — documented in-file.
  - `src/models/crnn_zain.py` — `CRNN2D`, ported from ZainNasrullah's Keras model
    (Method 2-1). Faithfully reproduces `GRU(..., return_sequences=False)` returning only
    the last time step (a real difference from bill317996's flattened-sequence approach).
  - `src/models/nonlocal_fgnl.py` — `CRNN_FGNL`, ported from ian-k-1217's Keras/TF
    Fully-Generalized-Non-Local block (Method 2-3): multi-scale theta/phi/g branches,
    Gaussian pre-smoothing, `torch.roll`-diversified affinity maps, MoSE gating, residual
    projection. Most involved port so far — worth a fresh read-through before trusting it
    at training time.

### Not yet done (next steps, in order)
1. **Sanity-check the model files actually run** — no forward pass has been tested yet
   (no torch environment available locally; needs our training server's `hw1_singer_env` or a local
   dry run with dummy tensors). Do this *before* trusting any of the architecture ports.
2. `src/data/prepare_index.py` — run it, verify counts (train=949, val=231, test=233,
   labels=20).
3. Sync code + dataset to our training server (`~/hw1_singer/`), keeping to our own home directory only.
4. `src/classical_ml.py` (Task 1) — not started.
5. `src/train.py` (unified trainer) + `src/evaluate.py` (confusion matrix, top1/top3,
   song-level chunk aggregation) — not started. Needed before any model above can
   actually be trained/scored.
6. `src/infer_test.py` — produces the `STUDENT_ID.json` submission — not started. Get
   this working off the `confound_crnn` model first as an early safety-net submission.
7. Still to port: `src/models/ssl_frontend.py` (Method 2-4 / Baseline 1, e.g. MERT/HuBERT
   + linear probe) and `src/models/speaker_frontend.py` (Method 3, e.g. speechbrain
   ECAPA-TDNN + linear probe) — repos cloned but not yet read.
8. Vocal-separation ablation (demucs/open-unmix) — not started.
9. Baseline 2 (audio-LLM zero-shot bonus) — not started, feasibility unchecked.
10. t-SNE viz, mel-spectrogram + own-recording demo (**blocked on user supplying a voice
    clip**, deferred to last on purpose per user instruction), README/requirements
    finalization, Google Drive checkpoint upload, `MATERIALS.md` fill-in — all not started.

### Notes / gotchas for future me
- `.gitignore`'s `data/` pattern was originally unanchored and silently ate
  `src/data/.gitkeep` too — fixed to `/data/` (root-only). If a new top-level dir
  called `data` shows up unexpectedly untracked, check this first.
- Local git identity for this repo is set via `git config --local` to the personal
  account (`goog-msft-fb-nflx-nvda-aapl` / `r05921008@gmail.com`), not the BlueWX
  company one — don't let a global config override it.
- Student ID in filenames is still the placeholder `STUDENT_ID`.
