# Recommended decision

**Run one carefully controlled SSL experiment, but treat it as a targeted ablation—not as the most likely major breakthrough.** For Artist20, I would use same-track temporal crops as the primary positive construction, weak-to-moderate vocal-preserving augmentations, a small-batch-friendly loss such as decoupled contrastive loss or VICReg/BYOL, and compare frozen linear probing against end-to-end fine-tuning.

The strongest singer-specific evidence supports SSL for learning singer representations, but it does **not** establish that CLMR-style pretraining will improve Artist20’s album-split top-1 accuracy by anything close to the large gains you obtained from fixing augmentation and optimization.

## 1. What the evidence supports

There are three particularly relevant results:

- **CLMR** applies SimCLR to raw music using two independently augmented views of the same audio fragment. Its reference implementation used SampleCNN, a 128-dimensional two-layer projection head, batch size 96, temperature \( \tau=0.5 \), Adam with learning rate \(3\times10^{-4}\), and 1,000 pretraining epochs. The original experiments were primarily music tagging, not singer identification. [archives.ismir](https://archives.ismir.net/ismir2021/paper/000084.pdf)
- **Yakura et al.** specifically studied self-supervised contrastive learning for singing voices. Their method uses pitch shifting and time stretching as transformations and deliberately learns representations sensitive to vocal timbre and singing expression. The authors released code with explicit pitch and stretch options. [github](https://github.com/hiromu/contrastive-singing-voices)
- **Torres, Lattner, and Richard** evaluated SimCLR-like contrastive learning, BYOL, VICReg, and related objectives for singer representations. Their best in-domain similarity result came from contrastive learning, while BYOL generalized best across several out-of-domain singing datasets. Their implementation and paper are unusually useful because they provide a concrete singing-specific recipe. [github](https://github.com/SonyCSLParis/ssl-singer-identity)

However, the Torres et al. data regime was much larger than Artist20: approximately 940 hours and 25,000 vocal tracks, compared with your 949 full songs. Their results therefore support the *direction* of the approach, not the likely magnitude of improvement on Artist20.

## 2. Positive-pair construction

### Primary recommendation: same-track, separated temporal crops

Use two independently sampled crops from the same training track:

1. Select one training song.
2. Sample crop \(a\) and crop \(b\) independently from that song.
3. Apply separate stochastic augmentations to the two crops.
4. Treat them as a positive pair.

This is close to the singing-voice method of Torres et al., which samples two random segments from the same source recording and then augments them independently. [staff.aist.go](https://staff.aist.go.jp/m.goto/PAPER/SIGMUS202209yakura.pdf)

For your setting, use:

- Crop duration: **4 seconds** initially.
- Minimum crop separation: preferably **1–2 seconds**, when the song is long enough.
- No pair may cross the album split.
- Do not use a crop from a different track as a positive in the first experiment.

Same-track positives have an important advantage: they do not inject artist labels into pretraining. Thus, the representation remains genuinely self-supervised, and the protocol cannot accidentally exploit artist labels in a way that complicates the course-assignment interpretation.

### Do not make different songs by the same artist the default positive

Artist-level positives are attractive because they directly encode the downstream invariant, but they introduce two risks:

- The model may learn album, mixing, mastering, or recording-session cues.
- The representation can collapse useful distinctions between songs while learning “artist identity” through production style.

There is also a more fundamental issue: using artist identity to select positives is no longer purely self-supervised in the usual sense. It is closer to supervised contrastive learning or label-informed metric learning. Unless your TA explicitly approves that formulation, keep it separate from the main SSL submission.

A useful optional ablation would be:

- **Track-positive SSL:** two crops from the same track.
- **Artist-positive contrastive learning:** crops from different tracks by the same artist, with all album constraints respected.

But report the second as a label-informed auxiliary experiment, not as the primary self-supervised result.

### Recommended augmentation policy

For singer identification, invariance should target recording conditions and modest musical variation without destroying vocal identity.

| Transformation | Recommendation | Rationale |
|---|---:|---|
| Random temporal crop | Always | Core positive construction; forces local-to-global identity learning. |
| Gain | Probability 0.5 | Removes loudness/mastering dependence. |
| Additive noise | Probability 0.3–0.5, mild | Robustness to recording noise; avoid severe corruption. |
| Time masking | Probability 0.3–0.5, up to 10–12.5% | Encourages use of distributed vocal cues. |
| Mild low/high-pass filtering | Probability 0.2–0.4 | Reduces dependence on exact production spectrum. |
| Reverb | Probability 0.2–0.4, mild | Helps separate identity from room/acoustic cues. |
| Polarity inversion | Optional, probability 0.5 | Safe for most identity tasks. |
| Delay/echo | Omit initially | Less clearly relevant to singer identity and can create unrealistic positives. |
| Pitch shift | Use only in a dedicated ablation | Potentially removes identity-relevant habitual pitch information. |
| Time stretch | Use mildly, dedicated ablation | Can remove expression/rhythm cues if too strong. |

### Pitch shifting: use carefully

Pitch shifting is **not automatically a good augmentation** for Artist20.

Yakura et al. explicitly use pitch shifting and time stretching for singing-voice representation learning.  Torres et al. also use formant-preserving pitch shifting, specifically avoiding naïve transposition because it changes singer timbre. Their pitch transformation samples pitch and range ratios and applies it with probability 0.5. [github](https://github.com/hiromu/contrastive-singing-voices)

That evidence supports the following conclusion:

- Pitch invariance is useful if the target is **vocal timbre or singer identity independent of song key**.
- Excessive or naïve pitch shifting can remove genuine singer cues, especially habitual register, vocal range, vibrato behavior, and register transitions.
- Formant-preserving pitch shifting is much safer than ordinary resampling-based transposition.

For the first Artist20 run, I would use **no pitch shift in the main recipe**, then run a small ablation with:

- Formant-preserving pitch shift only.
- Probability: 0.3–0.5.
- Maximum shift: approximately \(\pm 2\) semitones initially.
- Do not begin with CLMR’s \(\pm5\)-semitone range; that was validated for general music tagging, not singer identification. CLMR reports pitch shifts sampled from \([-5,+5]\) semitones, but its downstream task was tagging rather than singer identity. [archives.ismir](https://archives.ismir.net/ismir2021/paper/000084.pdf)

The key diagnostic is whether pitch-shift SSL improves **album-held-out** performance. If it improves random-song validation but hurts album-split validation, it is likely removing identity information or encouraging shortcut learning.

## 3. Concrete first recipe

### Data

Use only the 949 training tracks for SSL pretraining.

- Keep the album-level split exactly unchanged.
- Do not pretrain on validation or test tracks.
- Sample crops on the fly rather than materializing a fixed crop dataset.
- Use 16 kHz mono input, matching your supervised experiments.
- Start with 4-second crops.
- Ensure the two views of one pair are generated from the same original crop or from two nearby/random crops, but keep this choice fixed for the ablation.

I recommend two stages:

- **Stage A:** same source crop, two independent augmentations.
- **Stage B:** two independently sampled crops from the same track.

Stage B is the more useful test of track-level singer invariance, but Stage A is easier to optimize and provides a clean CLMR-style baseline.

### Architecture

Reuse your strongest existing CRNN first rather than introducing a new architecture.

Recommended structure:

```text
existing CRNN encoder
→ global temporal pooling
→ 512-dimensional representation h
→ 512 → 256 → 128 projection head
→ L2 normalization
→ contrastive loss
```

Discard the projection head after pretraining. Use \(h\), not \(z\), for downstream classification. CLMR also reports better downstream performance from the encoder representation rather than the projection output. [archives.ismir](https://archives.ismir.net/ismir2021/paper/000084.pdf)

If your CRNN does not naturally produce a fixed-dimensional vector, add adaptive average pooling over time before the projection head. Do not attach the contrastive loss to frame-level outputs unless you intentionally want a local representation objective.

### Objective

Use the following priority order:

1. **Decoupled contrastive loss**, if straightforward to implement.
2. Standard NT-Xent/InfoNCE as the reproducible baseline.
3. BYOL or VICReg as a second objective only if the contrastive run is unstable or highly batch-size-sensitive.

The singer-specific Torres et al. paper used a decoupled contrastive loss because it is easier to optimize with smaller batches and is less sensitive to temperature. [staff.aist.go](https://staff.aist.go.jp/m.goto/PAPER/SIGMUS202209yakura.pdf)

For your first run:

- Loss: decoupled NT-Xent.
- Temperature: \( \tau=0.2 \).
- Projection dimension: 128.
- Projection hidden dimension: 256 or 512.
- Batch size: **32 or 64 pairs**, whichever fits.
- Gradient accumulation: accumulate to an effective batch of 128 if possible.
- Optimizer: AdamW.
- Learning rate: \(1\times10^{-4}\) initially.
- Weight decay: \(1\times10^{-5}\).
- Gradient clipping: global norm 1.0.
- Mixed precision: yes, if already reliable in your code.
- Normalize projections before computing cosine similarity.

The published CLMR recipe used batch size 96, \( \tau=0.5 \), and Adam at \(3\times10^{-4}\), but it was trained on much larger music corpora and a different architecture.  The Torres singer-identity study used batch size 120, \( \tau=0.2 \), Adam at \(10^{-4}\), and weight decay \(10^{-5}\), although its EfficientNet-B0 backbone was initialized from ImageNet, which you must not do for the graded submission. [archives.ismir](https://archives.ismir.net/ismir2021/paper/000084.pdf)

### Small-batch handling

Do not assume that a physical batch of 32 makes the experiment invalid. The practical alternatives are:

- Gradient accumulation for the optimizer does **not** automatically increase the number of negatives unless embeddings are cached across accumulation steps.
- Use a memory queue or MoCo-style queue if you implement it carefully.
- Prefer decoupled contrastive loss, which was specifically motivated partly by smaller-batch behavior.
- Compare against BYOL or VICReg, which do not require large numbers of explicit negatives.

For a course assignment, I would avoid adding a queue in the first pass. It adds another moving part and makes the comparison harder to interpret. Use an actual batch of 32–64, report it clearly, and test objective stability.

## 4. Epoch allocation

Because your data consists of only 949 tracks but can generate many random crops, define an epoch by **number of sampled pairs**, not by one pass through full songs.

A practical schedule is:

### Pretraining

- 200–400 epochs.
- 100–200 sampled pairs per epoch, or approximately 100k–200k pair updates total.
- Early stopping based on a downstream validation probe, not contrastive loss alone.

Do not copy CLMR’s 1,000 epochs literally. CLMR trained on datasets containing vastly more independent fragments and songs; its large epoch count does not transfer directly to 949 tracks. [archives.ismir](https://archives.ismir.net/ismir2021/paper/000084.pdf)

### Linear probe

- Freeze the encoder.
- Use the 512-dimensional \(h\).
- Train a linear classifier for 50–150 epochs.
- Use the same album-level validation protocol.
- Report top-1 and top-3.

### Fine-tuning

Then initialize the same encoder from SSL and fine-tune end-to-end:

- Classifier learning rate: \(1\times10^{-3}\) or your best supervised value.
- Encoder learning rate: \(1\times10^{-4}\) to \(3\times10^{-4}\).
- Differential learning rates are preferable.
- Fine-tune for 100–200 epochs with your established cosine schedule.
- Retain the supervised augmentations that already improved results, but do not automatically retain the strongest SSL augmentations.

Run three downstream conditions:

| Condition | Purpose |
|---|---|
| Supervised scratch | Existing baseline. |
| Frozen SSL encoder + linear head | Measures representation quality directly. |
| SSL initialization + end-to-end fine-tuning | Measures practical classification benefit. |

The third condition is the most relevant comparison for your assignment. A frozen linear probe can understate the usefulness of a representation when the downstream task differs from the SSL objective.

## 5. Backbone choice

### Reuse CRNN first

Your current CRNN is already at 0.762 validation top-1, substantially ahead of the other tested architectures. That makes it the correct first backbone for an SSL experiment.

Using the same encoder gives you a clean comparison:

\[
\text{CRNN supervised from scratch}
\quad\text{vs.}\quad
\text{CRNN SSL-pretrained then fine-tuned}.
\]

A new architecture would confound the question of whether SSL helped.

### SampleCNN as a secondary experiment

SampleCNN is the most directly supported CLMR backbone. CLMR used a nine-block SampleCNN-style encoder, producing a 512-dimensional representation and a 128-dimensional projection. [archives.ismir](https://archives.ismir.net/ismir2021/paper/000084.pdf)

Your newly added SampleCNN is therefore worth testing, but I would not replace the CRNN experiment with it. SampleCNN may be a better fit for short raw-waveform crops, whereas your CRNN may better aggregate singer cues over time.

### EfficientNet-B0 is evidence, not a direct recipe

The Torres implementation uses EfficientNet-B0 over mel features and reports strong singer-representation results, but the published setup initialized the backbone with ImageNet weights.  For your assignment, that exact result is not eligible as a graded from-scratch result. [github](https://github.com/SonyCSLParis/ssl-singer-identity)

You could reproduce the architecture with random initialization, but it would be a new uncontrolled architecture comparison. Do this only after the CRNN/SampleCNN experiment.

### 2023–2026 alternatives

The recent literature contains larger masked-prediction and music-representation models, but these generally benefit from much larger corpora and are not obviously advantageous at 949 tracks. For this dataset size, a compact encoder plus a carefully designed objective is more defensible than reproducing a large modern music foundation model.

A particularly relevant non-contrastive alternative is **BYOL**, because the singer-specific study found it generalized better out of domain than contrastive learning, although contrastive learning was stronger for in-domain similarity.  It is worth adding if your initial contrastive experiment produces unstable or overly production-sensitive embeddings. [staff.aist.go](https://staff.aist.go.jp/m.goto/PAPER/SIGMUS202209yakura.pdf)

## 6. Expected benefit

Your current evidence changes the prior substantially:

- You already gained roughly 10–18 percentage points from fixing undertraining, augmentation, and scheduling.
- Remixing produced only a small gain.
- Your CRNN is already at 76.2% top-1.

That means the remaining error is likely harder than the earlier optimization error. I would set expectations as follows:

| Outcome | Plausibility |
|---|---|
| No improvement or a small decrease | High enough that it must be expected. |
| +1–3 percentage points top-1 | Realistic default expectation. |
| +3–6 percentage points | Plausible if SSL reduces album/production shortcuts. |
| More than +6 points | Possible, but should be treated as an unexpectedly strong result. |
| +10 points or more | Unlikely given the current 76.2% baseline. |

The CLMR gain of 49.6% to 55.2% is informative but not a reliable Artist20 forecast. It was a different dataset, task, split, architecture, and baseline regime. CLMR’s strongest evidence is that SSL can produce useful music representations and improve data efficiency; it does not demonstrate a singer-ID gain on a small album-level benchmark. [archives.ismir](https://archives.ismir.net/ismir2021/paper/000084.pdf)

The singer-specific evidence is more encouraging, but Torres et al. trained on approximately 940 hours of vocal data and 25,000 tracks.  Artist20 has far fewer independent recordings, so contrastive learning may repeatedly see different crops of the same production environment and learn album-specific structure unless the augmentations and evaluation protocol actively discourage it. [staff.aist.go](https://staff.aist.go.jp/m.goto/PAPER/SIGMUS202209yakura.pdf)

## 7. Most informative experiment matrix

If compute is limited, run this six-condition matrix:

| Run | Pretraining views | Objective | Downstream |
|---|---|---|---|
| A | None | — | CRNN supervised scratch |
| B | Same-track crops | NT-Xent | Frozen linear probe |
| C | Same-track crops | NT-Xent | End-to-end fine-tuning |
| D | Same-track crops + mild formant-preserving pitch shift | NT-Xent | End-to-end fine-tuning |
| E | Same-track crops | BYOL | End-to-end fine-tuning |
| F | Different tracks, same artist | Label-informed contrastive | End-to-end fine-tuning |

Use the same three random seeds for A–E. Run F only if permitted and label it separately.

For each run, also record:

- Album-level validation top-1/top-3.
- Random-song validation only as a diagnostic, not the primary result.
- Linear-probe performance.
- Performance by album.
- Confusion matrix.
- Performance on vocal-heavy versus instrumental-heavy segments, if available.
- Sensitivity to crop location.

A large random-split gain with little album-split gain would strongly suggest that SSL learned recording or album cues rather than transferable singer identity.

## 8. Additional eligible technique

The most promising additional technique not already in your list is **multi-view consistency without explicit artist positives**, using either BYOL or VICReg-style variance/covariance regularization.

Why it is relevant:

- It avoids dependence on a large negative batch.
- It can reduce the tendency of contrastive learning to push acoustically similar singers apart.
- The singer-specific comparison found BYOL particularly strong for out-of-domain generalization, while contrastive learning was strongest for in-domain similarity. [staff.aist.go](https://staff.aist.go.jp/m.goto/PAPER/SIGMUS202209yakura.pdf)

A practical variant is:

```text
same-track crops
→ shared CRNN encoder
→ 512-dimensional pooled representation
→ projector/predictor
→ BYOL loss
→ supervised fine-tuning
```

Use this only as a second objective after establishing the simpler contrastive baseline. Otherwise, the experiment becomes too broad for a graduate assignment.

## Bottom line

Implement **same-track two-crop SSL on the existing CRNN**, with a 128-dimensional projection head, batch size 32–64, decoupled NT-Xent at \( \tau=0.2 \), 200–400 pretraining epochs, and end-to-end fine-tuning afterward. Exclude pitch shifting from the main run; test mild formant-preserving pitch shifting as a separate ablation.

The scientifically defensible expectation is **approximately 1–3 percentage points**, with 3–6 points representing a strong success. If you can afford only one SSL experiment, use the CRNN with same-track crops and compare it directly against your current 0.762 supervised baseline.