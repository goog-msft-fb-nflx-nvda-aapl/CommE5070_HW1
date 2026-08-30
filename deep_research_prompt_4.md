# Deep Research prompt — from-scratch self-supervised pretraining recipe for Artist20

For relaying to Deep Research. Context for whoever answers this:

Same project as before: Artist20 singer/artist identification, 20 artists,
949 training tracks, album-level split, 16kHz mono full songs, graduate
course assignment. **Confirmed by the course TA**: Task 2 requires training
the model fully from scratch — no pretrained encoder or fine-tuning of one
is eligible as the graded submission (a frozen pretrained encoder is only
allowed as an optional baseline comparison, kept separate).

## Why this prompt

The course's own lecture notes on music classification (a slide deck
covering hand-crafted features, CNN/CRNN/sample-level-CNN architectures, and
a section on self-supervised learning) present a specific, on-topic
worked example: a sample-level CNN pretrained with SimCLR-style contrastive
learning (the CLMR method, Spijkervet & Burgoyne, ISMIR 2021), then
frozen-backbone linear-probed, **beats the same architecture trained purely
supervised from scratch** on their benchmark (49.6% supervised vs. 55.2%
SSL+linear-probe). Critically, **this stays from-scratch-eligible under our
constraint** as long as the contrastive pretraining uses only our own 949
training tracks and no external data or pretrained weights — it's
self-supervised pretraining on our own data, not transfer learning from an
outside source.

Current from-scratch results (val top1/top3, after fixing an earlier
undertraining bug — augmentation + cosine LR schedule + up to 300 epochs):

| model | basis | val top1 | val top3 |
|---|---|---|---|
| CRNN (minzwon/sota-music-tagging-models) | Choi et al. 2017-style | 0.762 | 0.900 |
| CRNN2D (ZainNasrullah) | Nasrullah & Zhao, IJCNN 2019 | 0.710 | 0.848 |
| CRNN2D_elu2 | Hsieh et al., ICASSP 2020 (Artist20's own paper) | 0.693 | 0.827 |
| Fully Generalized Non-Local Net | AAAI 2021 | 0.42-0.52 depending on config, unresolved regression under the augmentation fix |

We also have `ShortChunkCNN_Res` and `SampleCNN` (both from
minzwon/sota-music-tagging-models, the same repo as our best model above)
freshly added and training now — no results yet.

## What we need

1. **A concrete, implementable SimCLR/CLMR-style contrastive pretraining
   recipe for our exact setting**: which augmentation pairs are known to
   work well for *singer/artist identity* specifically (not generic music
   tagging) as the "positive pair" views — e.g. is pitch-shift a good view
   for identity-invariant contrastive learning, or does it risk removing
   identity-relevant signal (a singer's habitual pitch range) the same way
   it's debated as a supervised augmentation? What about different temporal
   crops of the *same track* as positives (standard CLMR approach) vs. crops
   from *different tracks of the same artist* (identity-specific, would need
   album-level care to avoid encoding production/album cues instead of
   voice) — is there published evidence either way for singer ID
   specifically, not just general music tagging?

2. **Concrete hyperparameters for a ~949-track pretraining set**: projection
   head size, temperature, batch size (SimCLR is known to want large
   batches — is there a known adaptation for small-data regimes?), number of
   pretraining epochs vs. fine-tuning epochs, and whether a shorter
   pretrain+fine-tune split outperforms allocating all compute to longer
   supervised training on a dataset this small (that's an open question we
   can't answer from priors alone).

3. **Backbone choice**: should the contrastive pretraining wrap one of our
   existing from-scratch encoders (e.g. reuse the `CRNN` above, add a
   projection head, pretrain, then fine-tune the same architecture
   end-to-end or with a frozen backbone + linear head), or is there a
   specific architecture published 2022-2026 for small-scale from-scratch
   audio SSL that's known to outperform reusing a supervised-style backbone
   for this purpose?

4. **Cost/benefit vs. our other options**: given we've already tried data
   augmentation + LR scheduling (large gains, +10-18pp on 2 of 4
   architectures) and cross-song vocal/instrumental remixing (small gain,
   +1.3pp top3 only), is SSL pretraining likely to be a similarly large
   lever, or is the CLMR example's ~5.6pp gain (49.6%→55.2%) a more
   realistic expectation for what a contrastive-pretraining pass could add
   here? A grounded expectation matters more than an optimistic one — we'd
   rather not spend a training cycle on this if the realistic ceiling is
   small.

5. Separately: any other 2023-2026 from-scratch-eligible technique
   specifically for small-N (~1000 track) audio classification you're aware
   of that wasn't covered in an earlier round (we already have answers on:
   architecture capacity sizing, cross-song remixing, generic augmentation,
   metric-learning auxiliary losses, SWA, mixup/label smoothing — please
   don't re-cover those unless directly relevant to the SSL recipe above).

Please prioritize a concrete, implementable recipe over a broad literature
survey — if you cite a specific paper's hyperparameters, make sure it's
actually sourced (a specific arXiv/GitHub link), not just plausible-sounding;
a previous round of this research surfaced some uncited, likely-fabricated
hyperparameter claims from one engine and we had to discount them.
