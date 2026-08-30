# Deep Research prompts — CommE5070 HW1 singer classification

Three prompts, each self-contained, for relaying to Deep Research (Gemini/OpenAI).
Findings feed into the ablation section of the final report — not blocking the
current training runs.

---

## Prompt 1: architectures beyond CRNN/non-local for small-scale singer/artist ID

We are training singer/artist identification models on the Artist20 dataset (20
artists, ~950 training tracks, album-level train/val/test split, 16kHz mono full
songs). Our current model comparison covers: hand-crafted-feature classical ML
(kNN/SVM/RandomForest), a from-scratch CNN and CRNN (Choi et al. 2016/2017 style,
via minzwon/sota-music-tagging-models), a CRNN with GRU-based temporal pooling
(Nasrullah & Zhao, IJCNN 2019), a CRNN with a non-local self-attention block
(the "Fully Generalized Non-Local Network," AAAI 2021), a frozen pretrained SSL
audio-encoder (MERT) + linear probe, and a frozen pretrained speaker-verification
embedding (ECAPA-TDNN, VoxCeleb) + linear probe.

Question: what architectures published 2023-2025 for music/artist/singer
identification, or closely related MIR classification tasks (music tagging,
speaker/artist retrieval, singing voice understanding) would be worth adding to
this comparison, specifically ones that:
1. Are known to perform well on small training sets (~1000 tracks, 20 classes) —
   not just large-scale (Million Song Dataset scale) benchmarks.
2. Have released code or weights we could realistically adapt in a few days.
3. Represent a genuinely different modeling idea from what we already have (not
   another CNN+GRU variant) — e.g. newer SSL backbones specifically pretrained on
   singing voice, contrastive/metric-learning approaches for speaker-like
   identity tasks, or lightweight transformer audio encoders.

For each candidate: name, paper/repo link, why it's a good fit for our setting,
and any known caveat (data/compute requirements, licensing, gating).

---

## Prompt 2: augmentation recipes for small, album-correlated singer ID datasets

Artist20's official split (Hsieh et al., ICASSP 2020) is album-level — one full
album held out for val, one for test, per artist — specifically to prevent
models from learning album/production "confound" shortcuts instead of vocal
identity. Even so, with only ~950 training tracks across 20 classes, overfitting
to instrumentation/production style rather than the singer's voice is a live
risk.

Question: what data augmentation and regularization strategies are most
effective, according to recent (2022-2025) singer identification / speaker
identification / small-scale audio classification literature, at forcing a
model to key on vocal identity rather than production/instrumentation cues?
Consider and evaluate (with citations if possible):
1. Vocal source separation as a preprocessing step (we already have a demucs-
   based vocals-only ablation planned/running) — what do recent papers report
   as the actual effect size of this on artist/singer ID accuracy, and are there
   known failure modes (separation artifacts hurting more than they help)?
2. Standard audio augmentations (pitch shift, time stretch, SpecAugment, mixup,
   background noise injection, EQ/production-style perturbation) — which of
   these specifically target the "instrumentation confound" versus which are
   generic regularizers?
3. Any singer-ID-specific augmentation tricks from recent papers (e.g.
   re-mixing vocals over a different backing track, synthetic re-production).

---

## Prompt 3: best available SSL checkpoints specifically for singing voice

We're using `m-a-p/MERT-v1-95M` (a general music-audio SSL model) as a frozen
frontend + linear probe for singer identification, and separately a VoxCeleb-
trained speaker-verification embedding (ECAPA-TDNN) under the premise from
Rathnayake et al. (ICASSP 2022, "Domain Adaptation for Speaker Recognition in
Singing and Spoken Voice") that spoken-voice speaker embeddings partially
transfer to singing voice despite the domain gap.

Question: as of 2025-2026, what are the strongest publicly available
self-supervised or supervised pretrained audio encoders specifically for
*singing voice* (not just general music, and not just spoken-voice speaker
verification) — e.g. anything pretrained on singing-voice corpora, or
fine-tuned/distilled from a music-SSL or speech-SSL base specifically for
singing voice understanding, singer identification, or singing voice
conversion (whose encoders sometimes transfer well to ID tasks)? For each:
name/checkpoint location (HuggingFace or otherwise), pretraining data/objective,
any reported singer-ID or singing-voice-classification numbers, and whether
weights are openly downloadable without gating.
