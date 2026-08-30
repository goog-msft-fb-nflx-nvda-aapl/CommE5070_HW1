# Deep Research prompt — closing the gap to our own prior-year submission + latest literature

For relaying to Deep Research. Context for whoever answers this:

## Problem statement

We're building a singer/artist identification system on the **Artist20**
dataset (Kim & Whitman / Ellis lab; official split as re-released for our
graduate course) for a graduate-course assignment (CommE5070, "Deep Learning
for Music Analysis and Generation"): 20 artists, 949 training tracks / 231
validation tracks / 233 held-out test tracks, 16kHz mono, full songs (not
pre-chunked), split at the **album level** so a model can't cheat by
learning album/production characteristics instead of the singer's voice
(Hsieh et al., "Addressing the confounds of accompaniments in singer
identification," ICASSP 2020 — the paper this dataset split methodology
comes from). Evaluation is at the **song level**: predict per-5-second-chunk,
mean-pool softmax probabilities across all non-overlapping chunks of a
track, report top1/top3 accuracy.

Two graded parts:
- **Task 1** — traditional ML: hand-crafted features (no learned/pretrained
  feature extractors) into a classical classifier. Per our TA's
  clarification, Task 1's accuracy doesn't count toward the grade at all —
  only the feature/ablation analysis quality does. Not the focus of this
  prompt.
- **Task 2** — deep learning **from scratch only**. Per our TA's explicit
  clarification: both the encoder and the classifier head must be trained
  from scratch on the 949 training tracks; a pretrained encoder (frozen or
  fine-tuned) is baseline-only, not eligible as the graded submission, even
  if it scores far higher (a frozen ECAPA-TDNN speaker-embedding baseline we
  built scores 95.2% val top1 vs. our best from-scratch model's 85.3% — not
  usable as the actual submission). **This prompt is about improving the
  from-scratch result specifically.**

## Our current pipeline

- Input: 16kHz mono, 5s non-overlapping chunks at train and eval time
  (song-level prediction = mean-pooled softmax across all of a track's
  chunks). Log-mel spectrogram, typically 96-128 mel bins depending on
  architecture, computed via torchaudio.
- Augmentation: SpecAugment (time/freq masking) for mel-input models,
  gain-jitter + additive noise for raw-waveform-input models.
- Training: AdamW or Adam, cosine LR annealing, label smoothing (mixed
  results — see below), up to 300 epochs / 40-epoch early-stop patience.
- Architectures trained so far (all from scratch, val top1/top3 after fixing
  an earlier undertraining bug):

  | model | basis | val top1 | val top3 |
  |---|---|---|---|
  | sota_crnn_wide (channel_mult=1.5) | minzwon/sota-music-tagging-models CRNN, capacity-scaled | **0.805** | **0.922** |
  | short_chunk_cnn | minzwon/sota-music-tagging-models | 0.766 | 0.866 |
  | sota_crnn_norm | above + per-sample mel normalization | 0.766 | 0.879 |
  | sota_crnn_ssl_finetune | above, SimCLR/NT-Xent self-supervised pretrained (own data only) then fine-tuned | 0.766 | 0.853 |
  | sota_crnn (original) | minzwon CRNN, plain Adam | 0.762 | 0.900 |
  | se_resnet | minzwon SE-ResNet (7-layer, SE-gated) | 0.758 | 0.866 |
  | sota_crnn_adamw_ls | above + AdamW + label smoothing 0.1 | 0.753 | 0.892 |
  | sota_crnn_supcon | above + supervised contrastive auxiliary loss | 0.736 | 0.853 |
  | sota_crnn_dropblock | above + DropBlock | 0.697 | 0.866 |
  | crnn_zain | ZainNasrullah/music-artist-classification-crnn (Nasrullah & Zhao, IJCNN 2019) | 0.710 | 0.848 |
  | fgnl | "Fully Generalized Non-Local Network for singer identification," AAAI 2021 | 0.710 | 0.870 |
  | confound_crnn | Hsieh et al. ICASSP 2020's own CRNN (Artist20's source paper) | 0.693 | 0.827 |
  | confound_crnn_remix | above + cross-song vocal/instrumental remix augmentation (their own paper's biggest reported lever) | 0.693 | 0.840 |
  | sample_cnn | minzwon SampleCNN, raw-waveform end-to-end | 0.693 | 0.823 |
  | sota_crnn_narrow (channel_mult=0.5) | capacity-scaled down | 0.654 | 0.835 |
  | sota_crnn_attn | sota_crnn + attention pooling instead of last-GRU-state | 0.654 | 0.810 |

- **Best result: a 9-model weighted-average ensemble** (integer weights,
  grid-searched on val) of confound_crnn, crnn_zain, sota_crnn,
  sota_crnn_wide, short_chunk_cnn, se_resnet, fgnl, sample_cnn, sota_crnn_norm
  → **val top1=0.853, top3=0.905**.
- Ablations already tried and their measured direction (not assumed from
  priors — we test every claim, including uncited ones from earlier Deep
  Research rounds, rather than reject on plausibility):
  - Capacity up: **helped** (+4.3pp). Capacity down: **hurt** (-10.8pp) —
    the opposite of an uncited claim from an earlier round that smaller
    models generalize better on ~950 tracks.
  - Per-sample mel normalization: helped slightly (+0.4pp).
  - Self-supervised (SimCLR-style) pretraining on our own 949 tracks: helped
    slightly (+0.4pp), well under the ~5.6pp a literature example (CLMR)
    suggested as an optimistic ceiling.
  - Attention pooling (last-GRU-state → full-sequence attention pooling):
    **hurt badly** (-10.8pp) — but see the caveat below, this was tested on
    a low-capacity backbone, not a clean test of attention pooling per se.
  - SupCon auxiliary loss: hurt (-2.6pp). DropBlock: hurt (-5.6pp). SWA
    weight-averaging: hurt vs. the same run's best non-averaged checkpoint
    (-3.5pp).
  - Cross-song vocal/instrumental remixing (the Artist20 paper's own biggest
    reported lever, +7-8pp in their paper): only +1.3pp top3 in our setup,
    flat top1.
  - Test-time augmentation: our default (deterministic full-track
    non-overlapping-chunk averaging) already beats random-crop TTA methods
    we tested against it directly — no further TTA gain available via that
    route as far as we've found.

## Why this prompt: a specific gap just found

We re-investigated our own prior-year submission for this same assignment
(same student, different year's cohort/checkpoint of the course) after
noticing its reported ensemble looked competitive. Its best individual model
was a CRNN with: **bidirectional GRU (hidden=256, 2 layers)**, **attention
pooling over the full GRU output sequence**, 4 Conv-BN-ELU blocks up to
**256 channels**, **10-second** training crops (vs. our 5s), per-sample mel
normalization, `f_min=20/f_max=8000/top_db=80` mel-spectrogram settings,
AdamW + cosine LR + label smoothing 0.1. Ensembled 1:2:1 with an SE-ResNet
and a classic non-local-block ResNet (Wang et al.-style, channel-ramped
32→512), it reached what its own report described as 0.825 top1/0.949 top3
on a *self-split* validation set (not directly comparable to our official
split's numbers, but the top3 gap in particular is large enough to take
seriously).

The key finding: our existing "attention pooling hurt" and "per-sample
normalization helped only slightly" ablations tested those ingredients
bolted onto `sota_crnn`'s much smaller backbone (**unidirectional, 32-hidden-
dim GRU**) — not the larger bidirectional-256 backbone the prior submission
actually used. We've now ported that architecture faithfully as
`crnn_nasrullah_faithful` and launched training on our official split (in
progress as of this prompt), but we'd like your help going further:

## What we need

1. **Is there recent (2023-2026) published evidence, specifically for
   singer/artist identification or closely related small-N music/speaker
   classification (not generic music tagging with 10k+ tracks), on whether
   full-sequence attention pooling over a bidirectional RNN outperforms
   simpler pooling (mean/max/last-state) *as a function of backbone
   capacity/width*? We want to know if our earlier negative result is
   plausibly a capacity artifact (attention pooling needs more hidden
   capacity to earn its keep, and hurts on a narrow bottleneck by adding
   optimization difficulty without enough parameters to exploit it) or
   whether there's a different reason (e.g. attention pooling overfitting on
   small-N data by learning track-position-specific weights) that would
   still apply to our new wider backbone.

2. **10-second vs. 5-second (or other) chunk length for singer ID
   specifically**: is there a known relationship between chunk length and
   accuracy for this task (as opposed to genre/tagging tasks, where shorter
   chunks are often fine)? Longer chunks give the GRU more temporal context
   per training example but also mean fewer independent training examples
   per epoch from a small (949-track) pool — is there a citable
   sweet-spot finding, or a principled way to reason about this trade-off
   for datasets this size?

3. **Latest architectures and techniques (2023-2026) from ICASSP,
   Interspeech, AAAI, and NeurIPS** — as well as ISMIR/WASPAA/TASLP if
   relevant — specifically for singer/artist identification, singing-voice
   representation learning, or from-scratch small-data audio classification
   that we haven't already covered above (please skip re-covering: SpecAugment,
   generic capacity scaling, SWA, DropBlock, SupCon, self-supervised
   contrastive pretraining basics, cross-song remixing — we already have
   measured results for all of those). We're specifically interested in
   anything published in the last 2-3 years that's from-scratch-eligible
   (no pretrained weights, no external training data beyond our own 949
   tracks) and has a track record on small (order-of-1000-track) singer/
   speaker/artist identification benchmarks, not just large-scale ones.

4. **Our own non-local and SE-ResNet architectures are also structurally
   different from the prior submission's** (ours: `fgnl`, "Fully Generalized
   Non-Local Network," AAAI 2021, a singer-ID-specific paper; prior
   submission: a classic Wang et al.-style non-local block bolted onto a
   ResNet, channel-ramped 32→512, plus a separate SE-ResNet variant, also
   channel-ramped 32→512 in 4 stages). Given our current `fgnl`/`se_resnet`
   already score competitively with the prior submission's individual
   numbers on our own val set, is porting the prior submission's simpler
   non-local/SE architectures likely worth the effort, or is there a good
   reason to expect a channel-ramping ResNet-style backbone (deep+narrow →
   wide, few stages) to systematically outperform (or underperform) our
   current deeper, narrower-max-width architectures for this specific
   task/data scale? A grounded expectation matters more than an optimistic
   one.

5. **Ensemble diversity**: the prior submission's 3-way ensemble (three
   *architecturally distinct* models: CRNN, SE-ResNet, classic non-local
   ResNet) gained +11pp top1 over its best individual model; our current
   9-model ensemble (many of which are variants of the same `sota_crnn`
   backbone) gains +4.8pp over our best individual model. Is there
   literature on how much ensemble-diversity gain specifically comes from
   *architectural* diversity (different inductive biases) vs. *training-
   recipe* diversity (same architecture, different hyperparameters/seeds),
   for small ensembles (3-10 members) on small-N classification tasks? This
   would help us decide whether porting 1-2 more architecturally-distinct
   models is worth prioritizing over further hyperparameter ablations on
   architectures we already have.

Please prioritize concrete, sourced findings (specific paper/arXiv/GitHub
links) over plausible-sounding general claims — earlier rounds of this
research surfaced some uncited, likely-fabricated hyperparameter claims from
one engine, and we test every claim rather than trust or dismiss it on
priors alone, so an unsourced claim just costs us a training cycle to
verify rather than saving us one.
