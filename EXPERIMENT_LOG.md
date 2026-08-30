# Experiment log

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
  (`deep_research_prompt_3.md`), since it's literally the technique from
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
and hit a **hard wall**: the file has to be read into my own context first to pass
as inline `base64Content`, and the `Read` tool caps at 256KB — so even the
smallest checkpoint (294KB base64-encoded) can't be relayed through me. No
chunked/append upload path exists on this tool either.

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

- Deep Research responses (3 engines, saved by user as `deep_research_response_*.md`,
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

**Status when paused: no GPU job running.** `gsm-gpu2` tmux session `hw1_singer` finished
its one-off conda env setup (`pip install ... > install_status.txt` shows `INSTALL_DONE`)
and is sitting idle at a bash prompt — nothing to kill. No training has started yet.

### Done
- Access checks: gsm-gpu2 (4x H200 NVL, all idle, 1.1TB free on `/home/jtan`), GitHub
  (`goog-msft-fb-nflx-nvda-aapl`, switched active), HuggingFace (reachable, no token —
  fine for public models, will flag if something needed turns out gated).
- Repo scaffolded at `/Users/chun-feitan/Desktop/CommE5070/HW1/CommE5070_HW1/`, own git
  repo (root `~/Desktop` is an unrelated BlueWX repo — never touch that level), pushed to
  `https://github.com/goog-msft-fb-nflx-nvda-aapl/CommE5070_HW1` (public).
- gsm-gpu2: new conda env `hw1_singer_env` (python 3.10), packages installed — torch
  2.5.1+cu121, torchaudio 2.5.1+cu121, librosa, scikit-learn, transformers,
  huggingface_hub, matplotlib, seaborn, pandas, tqdm, soundfile, demucs, speechbrain.
  New tmux session `hw1_singer` on gsm-gpu2 (separate from the pre-existing, unrelated
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
    gsm-gpu2).
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
   (no torch environment available locally; needs gsm-gpu2's `hw1_singer_env` or a local
   dry run with dummy tensors). Do this *before* trusting any of the architecture ports.
2. `src/data/prepare_index.py` — run it, verify counts (train=949, val=231, test=233,
   labels=20).
3. Sync code + dataset to gsm-gpu2 (`~/hw1_singer/`), keeping to `/home/jtan/` only.
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
