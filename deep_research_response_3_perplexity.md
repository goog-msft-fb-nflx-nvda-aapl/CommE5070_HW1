## Bottom line

Your 0.671 validation top-1 for `CRNN2D_elu2` is broadly consistent with the Artist20 literature once the metrics and evaluation units are aligned. The largest missing ingredient is probably not merely a learning-rate schedule: it is the **task-specific “Origin + Vocal-only + Shuffle-and-remix” training distribution**, together with the original paper’s **5-second segment training and song-level aggregation**. Hsieh et al. report 0.75 song-level F1 with that combination, while FGNL reports approximately 0.72–0.73 average song-level F1 on original audio and up to 0.83 best-run F1 depending on configuration and clip length. [cdn.aaai](https://cdn.aaai.org/ojs/17000/17000-13-20494-1-2-20210518.pdf)

For your next two experiments, I would prioritize:

1. **Demucs vocal/instrumental shuffle-and-remix, retaining original and vocal-only examples.**
2. **Ablation-controlled training recipe with balanced track/artist sampling, longer training, moderate regularization, and song-level aggregation.**

I would not prioritize contrastive learning, auxiliary album prediction, or a major architecture redesign before establishing that baseline.

## 1. Literature sanity check

### Hsieh et al. CRNNM

The important methodological point is that the published 0.75 result is not simply the same mel-CRNN trained longer. Hsieh et al. use:

- A four-convolution-layer CRNN with two GRU layers and a dense classifier.
- A second melody branch based on CREPE features.
- 5-second segments for training.
- Open-Unmix vocal separation.
- Three training datasets: original mixtures, vocal-only audio, and shuffled vocal/instrumental remixes.
- A combined “Data aug.” set containing all three datasets.
- Majority voting across segments for song-level prediction.
- Three independent runs averaged for the reported result.

Their key table is:

| Model | Training data | 5-sec F1 | Song F1 |
|---|---:|---:|---:|
| CRNN | Origin | 0.50 | 0.67 |
| CRNN | Vocal-only | 0.39 | 0.61 |
| CRNN | Remix | 0.39 | 0.65 |
| CRNN | Origin + Vocal-only + Remix | 0.47 | 0.74 |
| CRNNM | Origin | 0.53 | 0.69 |
| CRNNM | Origin + Vocal-only + Remix | 0.45 | 0.75 |

Thus, the published gain is primarily associated with the **combined data distribution**, not remixing alone. Remix-only training actually performs worse than Origin-only training at the segment level and does not improve song-level F1. [cdn.aaai](https://cdn.aaai.org/ojs/17000/17000-13-20494-1-2-20210518.pdf)

This is directly relevant to your experiment: if you train only on shuffled mixtures, you may reproduce the weaker “Remix” condition rather than the strong “Data aug.” condition.

### FGNL

The FGNL paper explicitly states that its weights are randomly initialized, so it is eligible under your TA’s rule. Its training recipe is comparatively conservative:

- Random initialization.
- Adam.
- Constant learning rate \(10^{-4}\).
- Batch normalization and dropout.
- Softmax cross-entropy.
- 3-, 5-, and 10-second clips.
- Three independent runs.
- Validation used for hyperparameter selection.
- Album-level splitting.
- Song-level prediction by majority voting, with low-confidence frames removed when confidence is below 0.5.

The paper does **not** describe cosine decay, warmup, mixup, SpecAugment, or a sophisticated curriculum. Therefore, your constant \(10^{-3}\) learning rate is not a faithful reproduction of the published FGNL setup: it is ten times larger than the reported \(10^{-4}\). [qmro.qmul.ac](https://qmro.qmul.ac.uk/xmlui/handle/123456789/105029)

Reported original-audio song-level F1 values were:

| Model | 3 sec | 5 sec | 10 sec |
|---|---:|---:|---:|
| CRNN | 0.57 | 0.55 | 0.58 |
| CRNN FGNL | 0.72 | 0.73 | 0.73 |
| CRNNM | 0.62 | 0.61 | 0.65 |
| CRNNM FGNL | 0.74 | 0.74 | 0.73 |

The best individual FGNL runs reached 0.82–0.83 song-level F1 for some settings. These are **song-level F1 values**, not directly comparable to your frame/chunk validation top-1. [qmro.qmul.ac](https://qmro.qmul.ac.uk/xmlui/handle/123456789/105029)

### What is probably missing?

The most consequential mismatches to check are:

1. **Metric mismatch.** Compare song-level majority-vote accuracy/F1, not only randomly sampled chunk top-1.
2. **Clip aggregation.** The papers use many 3-, 5-, or 10-second clips per song and aggregate them.
3. **Sampling policy.** Avoid allowing tracks with more usable chunks or longer duration to dominate training.
4. **Learning rate.** FGNL used \(10^{-4}\), not \(10^{-3}\).
5. **Data composition.** Hsieh’s strong result uses original, vocal-only, and remix data together.
6. **Confidence-aware voting.** FGNL discards predictions with confidence below 0.5 before voting.
7. **Melody information.** Hsieh’s strongest model includes a separate melody branch; this is an architectural/input feature absent from a mel-only reproduction.
8. **Run variance.** FGNL reports three runs and both average and best values, so a single validation run can differ materially.

I would verify the exact repository preprocessing before changing the network. In particular, confirm mel-bin count, hop length, normalization, chunk placement, padding, pooling order, and whether the validation set is sampled by track or by chunk. These details can easily produce larger differences than a modest architecture change.

## 2. Best next technique

### First priority: task-specific remixing

For this dataset, shuffle-and-remix is the strongest evidence-backed, from-scratch-compatible intervention. It directly targets the album/production confound: the vocal identity remains associated with the singer label while the backing track changes. Hsieh et al. explicitly motivate the method as breaking the bond between singer and accompaniment, and their combined training set improves song-level F1 from 0.67 to 0.74 for the CRNN and from 0.69 to 0.75 for CRNNM. [cdn.aaai](https://cdn.aaai.org/ojs/17000/17000-13-20494-1-2-20210518.pdf)

Use it as:

\[
\text{training pool}
=
\text{Origin}
+
\text{Vocal-only}
+
\text{Cross-song remix}.
\]

Do not replace the original data entirely.

A practical first implementation would be:

- For each training chunk, retain the original mixture.
- Add the separated vocal-only version.
- Add one remix using that vocal with an instrumental from another training song.
- Prefer a different artist for the instrumental, while preserving the vocal label.
- Randomize relative gain or target SNR mildly.
- Reject obviously invalid pairings with silence or severe separation artifacts.
- Never create remixes using validation or test tracks.

The Hsieh paper’s remix set has the same size as the original set, and the combined data-augmentation set is therefore approximately three times the Origin training set. [cdn.aaai](https://cdn.aaai.org/ojs/17000/17000-13-20494-1-2-20210518.pdf)

### Second priority: better temporal aggregation

Before adding a new objective, implement robust song-level inference:

- Generate overlapping 5–10 second windows.
- Average logits or log-probabilities across windows rather than majority-voting hard labels.
- Weight windows by vocal activity or prediction confidence.
- Optionally discard very low-vocalness windows.
- Report both chunk-level and song-level results.

This is especially important because the literature shows a large gap between segment-level and song-level scores. Hsieh et al. report CRNNM Data Augmentation at only 0.45 5-second F1 but 0.75 song-level F1.  The assignment’s validation metric should determine what you optimize, but song-level aggregation is still essential for a meaningful comparison. [cdn.aaai](https://cdn.aaai.org/ojs/17000/17000-13-20494-1-2-20210518.pdf)

### Lower priority: mixup and self-distillation

Mixup is reasonable as a cheap secondary experiment, but generic waveform or spectrogram mixup is less directly matched to the failure mode than source-aware remixing. A label-preserving mixture of two different artists is not naturally valid for a single-label singer classifier unless one label is treated as dominant or the loss is changed to a soft target.

Self-distillation, SWA, and contrastive learning are also plausible, but the evidence is weaker for this exact regime:

- Self-distillation results found in recent audio work often concern pretrained audio networks or much larger datasets.
- Contrastive learning usually benefits from many more tracks, augmentations, or external/unlabeled data.
- Album prediction as an auxiliary task could encourage album information rather than remove it, because album and artist are correlated.

If you try one objective-level change, use **supervised metric learning alongside cross-entropy**, not album prediction:

\[
\mathcal{L}
=
\mathcal{L}_{\mathrm{CE}}
+
\lambda \mathcal{L}_{\mathrm{SupCon}},
\]

with positives drawn from different chunks of the same artist and, ideally, different songs or albums. However, ensure that positive pairs are not just neighboring chunks from the same song; otherwise the model may learn song-production identity rather than singer identity.

## 3. Architecture recommendation

I would not immediately make FGNL larger. Your current results suggest that optimization, sampling, augmentation, and evaluation alignment are unresolved. Increasing capacity before fixing those factors risks improving training accuracy while worsening album-held-out generalization.

A defensible architecture experiment is a small capacity sweep:

| Variant | Purpose |
|---|---|
| 0.5× channels and smaller GRU | Test overfitting hypothesis. |
| Current model | Reference. |
| 1.5× channels or GRU size | Test under-capacity hypothesis. |

Keep the training recipe identical and use at least three seeds if feasible. Select based on song-level validation performance, not training loss.

For FGNL specifically, the paper’s ablations suggest that the gain is not merely from parameter count. CRNN FGNL has fewer parameters than CRNNM yet performs better in several settings, and the paper attributes the improvement to cross-position, cross-channel, and cross-layer interactions. The Gaussian smoothing and modified squeeze-and-excitation components also each contribute to generalization. [qmro.qmul.ac](https://qmro.qmul.ac.uk/xmlui/handle/123456789/105029)

Therefore, if you retain FGNL, the more defensible change is:

- Keep the FGNL module.
- Reduce its input representation or bottleneck width if memory/overfitting is a concern.
- Preserve Gaussian smoothing and MoSE.
- Add stronger data diversity rather than simply adding layers.
- Compare against a parameter-matched CRNN baseline.

A compact model may win if the current network memorizes production signatures, but the literature does not establish that smaller is consistently better on Artist20. The strongest Artist20-specific evidence instead favors better input factorization and temporal aggregation.

## 4. Remix versus generic augmentation

### Expected value

| Intervention | Expected benefit | Implementation cost | Recommendation |
|---|---:|---:|---|
| Cosine schedule / longer training | Medium | Low | Keep running. |
| SpecAugment | Low–medium | Low | Keep as baseline regularization. |
| Gain jitter + light noise | Low–medium | Low | Keep for waveform model. |
| Cross-song vocal/instrumental remix | High | Medium–high | Highest priority. |
| Melody branch | Medium–high | Medium | Strong second experiment if available. |
| Generic mixup | Uncertain | Low | Try only after remix. |
| Album auxiliary head | Potentially harmful | Medium | Do not prioritize. |
| From-scratch contrastive loss | Uncertain | Medium–high | Later experiment. |
| Larger architecture | Uncertain | Medium | Avoid until ablations are clean. |

The reason remixing deserves priority is not that generic augmentation is useless. SpecAugment can improve robustness to missing time-frequency regions, while gain/noise augmentation can reduce sensitivity to recording conditions. But neither directly breaks the strongest dataset-specific shortcut: the association between a singer and the instrumentation, production style, or album.

The prior Artist20 result is unusually specific: Origin + Vocal-only + Remix improves song-level F1 by roughly 7–8 percentage points over the corresponding Origin-only baselines in the cited experiments.  That makes remixing more than a generic augmentation guess. [cdn.aaai](https://cdn.aaai.org/ojs/17000/17000-13-20494-1-2-20210518.pdf)

## Recommended two-run plan

### Run A: faithful Hsieh-style baseline

Use the current mel CRNN or CRNN2D_elu2, but:

- Train on Origin + Vocal-only + Remix.
- Keep all three pools balanced.
- Use only training tracks to construct remix pairs.
- Train with 5-second chunks.
- Use a lower initial learning rate, such as \(10^{-4}\) for a faithful FGNL-style comparison.
- Use cosine decay only as a separate modern optimization variant.
- Apply moderate SpecAugment only to the training input.
- Evaluate with overlapping windows and logit averaging.
- Report chunk top-1, song top-1, macro-F1, and confidence-filtered voting.

This is the highest-value experiment because it simultaneously tests whether your gap is caused by data confounding and whether your metric is misaligned with the papers.

### Run B: controlled FGNL optimization

Keep the FGNL architecture unchanged and compare:

- Adam, constant \(10^{-4}\), 300 epochs.
- Adam, cosine decay from \(10^{-4}\).
- Same augmentation and same sampling in both conditions.
- Three seeds if computationally practical.

Do not compare your current \(10^{-3}\) run directly with the paper and conclude that FGNL underperforms. The published recipe used \(10^{-4}\), dropout, batch normalization, and song-level aggregation. [qmro.qmul.ac](https://qmro.qmul.ac.uk/xmlui/handle/123456789/105029)

The main conclusion I would expect is that **source-aware remixing plus correct temporal aggregation will matter more than adding a new loss or enlarging the encoder**. Contrastive learning and self-distillation are worth mentioning in a research discussion, but they should be fallback experiments after the Hsieh-style from-scratch pipeline has been reproduced.