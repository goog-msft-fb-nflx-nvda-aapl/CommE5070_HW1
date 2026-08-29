# Deep Research prompt — improving Task 2's from-scratch score specifically

For relaying to Deep Research. Context for whoever answers this:

We're building a singer/artist identification pipeline on Artist20 (20
artists, 949 training tracks, album-level split, 16kHz mono full songs) for a
graduate course assignment. **Critical constraint, confirmed by the course
TA**: Task 2 requires training the model **fully from scratch** — both
encoder and classifier. A pretrained encoder (frozen or fine-tuned) is only
allowed as an optional baseline comparison, not the graded submission. So
this prompt is specifically about improving from-scratch performance — do
not recommend pretrained-encoder / transfer-learning approaches as the
answer, even though we know from separate experiments that they score far
higher (a frozen VoxCeleb ECAPA-TDNN speaker embedding + trained head gets
95.2% val top-1 here, kept only as a baseline, not eligible).

Also confirmed by the TA: only `train.json`'s 949 tracks may be used for
training (no pooling in the validation split, no outside data, no touching
the test set).

Current from-scratch results, all trained with Adam lr=1e-3, 80 epochs (no
LR schedule, no data augmentation — a mistake we've since identified: all 4
models were still improving noisily when training was cut off, never
triggering early stopping):

| model | basis | val top1 | val top3 |
|---|---|---|---|
| CRNN2D_elu2 | bill317996/Singer-identification-in-artist20 (Hsieh et al., ICASSP 2020 — the Artist20 dataset's own paper) | 0.671 | 0.823 |
| CRNN2D | ZainNasrullah/music-artist-classification-crnn (Nasrullah & Zhao, IJCNN 2019) | 0.619 | 0.835 |
| CRNN (sota-music-tagging-models) | minzwon/sota-music-tagging-models (Choi et al. 2017-style, originally for MTAT tagging, not Artist20-specific) | 0.584 | — |
| CRNN + Fully Generalized Non-Local block | ian-k-1217/Fully-Generalized-Non-Local-Network (AAAI 2021) | 0.571 | 0.853 |

We've already identified and are re-running with two fixes: (1) a cosine
learning-rate schedule + higher epoch cap (300, patience 40, up from 80/15)
so training actually converges instead of getting cut off mid-improvement,
and (2) basic augmentation — SpecAugment (time/frequency masking) on the
log-mel input for the 3 mel-chunk models, gain jitter + light additive noise
on the raw-waveform input for the sota-music-tagging-models CRNN.

## What we need

1. **Sanity-check our numbers against the literature for these specific
   architectures.** For CRNN2D_elu2 and the FGNL non-local network
   specifically (both published *on Artist20 itself*, under the album
   split): what training regime (epochs, optimizer, LR schedule, batch
   size, augmentation) did the original papers actually use to reach their
   reported numbers (Hsieh et al.'s ~0.75 F1 with shuffle-and-remix
   augmentation; FGNL's ~0.73-0.83 F1 depending on input)? Is there a
   specific, known training detail we're likely missing beyond LR schedule
   and basic augmentation — e.g. a particular data sampling strategy,
   warmup, specific augmentation recipe, or number of training epochs that
   would explain a meaningful part of the remaining gap?

2. **What's the strongest *from-scratch-eligible* technique published
   2022-2026 for small-scale (~1000 track), album-split-constrained singer/
   artist identification** that we haven't tried? Specifically excluding
   anything requiring a pretrained encoder or fine-tuning one — we need
   architectures/training tricks that stand on their own with only Artist20's
   949 training tracks. Candidates we're aware might be relevant but haven't
   verified are from-scratch-compatible: mixup, SpecAugment variants beyond
   basic time/freq masking, curriculum learning by chunk length, multi-task
   auxiliary losses (e.g. predicting album/song as an auxiliary head to
   explicitly encourage the network away from album shortcuts), stochastic
   weight averaging, self-distillation, or contrastive/metric-learning
   objectives trained from scratch (not requiring pretrained weights).

3. **Given the album-level split's small size, what's a defensible from-
   scratch architecture change (not just more training) most likely to move
   the needle** — e.g. is there a case for using less parameter-hungry
   architectures (fewer conv channels, smaller GRU hidden size) to reduce
   overfitting risk on ~950 tracks, versus the reverse (more capacity +
   heavier regularization)? Cite any Artist20-specific or comparably-sized-
   dataset evidence if available, not just general small-data ML advice.

4. **Cross-song vocal/instrumental remixing** was flagged as a
   high-value from-scratch-compatible augmentation in an earlier research
   round (mix a singer's separated vocal — we have demucs infrastructure
   already — with a *different* song's backing track, same label, to break
   the album/production confound). Is this worth prioritizing over the
   generic augmentations above given our specific numbers, or is basic
   SpecAugment/gain-jitter likely to capture most of the achievable gain for
   less implementation effort? A rough cost/benefit judgment is enough, not
   a full literature survey repeat.

Please prioritize concrete, implementable answers over a broad survey — we
have working training code and can add one or two specific things quickly,
not redesign the whole pipeline.
