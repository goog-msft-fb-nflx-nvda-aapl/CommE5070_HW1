# Deep Research prompt — Artist20 SOTA + per-method deep-dive ablations

Single prompt covering two related questions, for relaying to Deep Research
(Gemini/OpenAI/etc.). Context for whoever answers this:

We've built a singer/artist identification pipeline on the Artist20 dataset
(20 artists, ~950 training tracks, album-level train/val/test split per
Hsieh et al. ICASSP 2020, 16kHz mono full songs) with the following methods
already implemented and measured (val-set top1/top3 accuracy in parens):

- Task 1, traditional ML: hand-crafted librosa features (MFCC+delta+delta2,
  chroma, spectral contrast/centroid/bandwidth/rolloff, ZCR, tonnetz),
  mean+std pooled per track → StandardScaler → kNN (0.403/0.623), SVM-RBF
  (0.593/0.831), RandomForest (0.550/0.766).
- CNN/CRNN from scratch (minzwon/sota-music-tagging-models CRNN) (0.584/—).
- CRNN, Nasrullah & Zhao IJCNN 2019 port (0.619/0.835).
- CRNN2D_elu2, Hsieh et al. ICASSP 2020 port — our Task-2 core model, on raw
  mixtures (0.649/0.853) and on demucs vocals-only audio (0.671/0.857).
- Fully Generalized Non-Local Network, AAAI 2021 port (0.571/0.853).
- Frozen MERT-v1-95M (music SSL) + MLP probe (0.684/—).
- Frozen ECAPA-TDNN (speechbrain/spkrec-ecapa-voxceleb, VoxCeleb speaker
  embeddings) + MLP probe — our best model by a wide margin (0.952/0.987).
- Zero-shot Qwen2-Audio-7B-Instruct (no training, prompted with a closed
  20-way artist list) — bonus baseline (0.475/0.700 on a 40-track subset).

We also have three earlier Deep Research responses on hand (architectures
beyond CRNN/non-local, augmentation/regularization strategies for the
album-confound problem, and singing-specific SSL checkpoints — including a
recommendation for `SonyCSLParis/ssl-singer-identity`, not yet implemented).
This prompt is a *follow-up*, not a repeat — please don't re-cover ground
already answered there (chromagram remixing, EfficientAT, Whisper-as-SSL,
etc.) unless directly relevant to a specific numeric SOTA claim.

## Part 1 — best known results specifically on Artist20

We want the actual SOTA leaderboard for *this specific dataset* (Artist20,
album-level split), not singer-ID in general:

1. What is the best published top-1 (and top-3, if reported) accuracy on
   Artist20 under the album-level split, and which paper/system achieves it?
   Please distinguish between:
   - Results using raw polyphonic mixtures as input.
   - Results using vocal-separated / vocals-only input.
   - Results using pretrained embeddings (speaker verification, SSL, or
     otherwise) versus models trained fully from scratch on Artist20 alone.
2. For each SOTA system found, report: architecture/encoder, classifier
   head, input representation (raw waveform / mel-spectrogram / separated
   vocals / embeddings), any pretraining data used, and the exact reported
   top-1/top-3 (or F1, if that's what's reported — note the metric).
3. Is our best result (95.2% top1 / 98.7% top3, frozen VoxCeleb speaker
   embeddings + a small trained MLP head, on raw — not separated — mixtures)
   in the range of, above, or below published SOTA? If it's surprisingly
   high, help us sanity-check *why* — e.g., is a linear/MLP probe over
   VoxCeleb speaker embeddings a known strong baseline for Artist20
   specifically that's under-cited relative to fancier architectures, or is
   there a known evaluation pitfall (e.g., embedding leakage, chunk-level
   vs. song-level scoring differences, val-set-only vs. true held-out test
   set) that could make a number this high suspicious and worth
   double-checking on our end?

## Part 2 — per-method ablation/deep-dive recommendations

For each of the following methods we've implemented, tell us the 2-3 most
informative ablations or analyses (specific encoder/classifier
combinations, probing experiments, or diagnostic checks — not generic
"try more data augmentation" advice) that would be most likely to (a)
explain *why* it performs the way it does relative to the others, and (b)
be genuinely interesting for a course report rather than incremental
hyperparameter tuning:

1. **Classical ML (SVM-RBF 59.3% top1)** — beats several from-scratch deep
   models. What ablation would best show *why* hand-crafted features do this
   well here (e.g., per-feature-group ablation — which of
   MFCC/chroma/contrast/tonnetz actually drives the accuracy; a
   feature-importance analysis for the RandomForest variant)?
2. **From-scratch CRNN family (CRNN2D_elu2, Zain's CRNN2D, FGNL,
   sota-music-tagging-models CRNN)** — all land in a fairly narrow
   57-67% top1 band despite architectural differences (non-local attention,
   different pooling/RNN configs). What's the most informative way to test
   whether they're bottlenecked by *training data volume* (~950 tracks)
   rather than architecture — e.g., a learning-curve ablation (subsample
   the training set), or a check of whether their errors are
   correlated/uncorrelated across models (ensemble-diversity analysis)?
3. **MERT (frozen SSL, 68.4% top1) vs. ECAPA-TDNN (frozen speaker embedding,
   95.2% top1)** — the speaker-verification embedding beats the music-SSL
   embedding by a wide margin. What encoder-vs-classifier-head ablation
   would best isolate *why* — e.g., swapping classifier heads (linear probe
   vs. our 2-layer MLP vs. cosine/prototype classifier) while holding the
   encoder fixed, to check whether the gap is really about the encoder's
   representation quality or just how well each embedding pairs with a
   simple head; or an embedding-space analysis (e.g., silhouette score /
   class separability of MERT vs. ECAPA embeddings on the val set, which we
   could compute ourselves from saved embeddings — is that a good
   diagnostic here, or would you recommend something else?).
4. **Vocal separation ablation (CRNN2D_elu2: 64.9% raw mixture → 67.1%
   vocals-only, a modest +2.2pp)** — smaller than several literature claims.
   What follow-up analysis would best characterize *what kind* of
   improvement this is — e.g., per-artist breakdown (does separation help
   uniformly, or mostly for specific artists with denser
   instrumentation/loud mixes?), or an error-overlap analysis (does the
   vocals-only model fix genuinely different tracks than it breaks, or is
   it a broad small shift)?
5. **Zero-shot Qwen2-Audio (47.5% top1, no training at all)** — surprisingly
   competitive with several trained from-scratch models. Is there a known
   way to meaningfully improve a zero-shot audio-LLM baseline like this
   without any gradient-based training (e.g., better prompting, few-shot
   in-context examples with reference clips if the model supports
   multi-turn audio context, self-consistency/majority-vote over multiple
   samples) that would be worth trying as a quick follow-up?

For anything in Part 2 that requires new code, a rough sketch of the
approach (not full implementation) is enough — we'll implement it
ourselves if it looks worthwhile.
