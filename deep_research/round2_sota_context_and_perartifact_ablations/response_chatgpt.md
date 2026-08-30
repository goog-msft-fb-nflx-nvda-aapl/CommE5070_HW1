I did a literature audit focused specifically on **Artist20 + album-level evaluation**, and the main conclusion is that the apparent SOTA picture is much messier than a single leaderboard suggests.

## Executive conclusion

Your **95.2% top-1 / 98.7% top-3 from a frozen VoxCeleb ECAPA-TDNN embedding + MLP** is very strong, but I would **not yet claim it is Artist20 SOTA**.

The reason is not that the result looks implausibly good. It is that the published Artist20 literature mixes:

* album split vs. song/random split,
* song-level F1 vs. accuracy,
* frame/chunk-level vs. song-level aggregation,
* vocal-only vs. mixture input,
* and, importantly, validation/test protocols.

Under the **strict album-split, song-level comparison**, the strongest clearly documented systems I found are around **0.81–0.86 F1**, while some papers report **~0.99 accuracy** under protocols that are not sufficiently specified/comparable to safely put on the same leaderboard as your result. ([ResearchGate][1])

There is also a particularly important paper-level sanity check: **KNN-Net reports 99% Artist20 accuracy**, but its own paper says its Artist20 frame-block confusion matrix has only **85% average accuracy**, and it does not clearly document which of its validation/test subsets the Table-1 99% number corresponds to. I would therefore treat that 99% as a **reported result, not a clean apples-to-apples SOTA reference**. 

---

# Part 1 — Artist20 SOTA

### 1. The most relevant published results

| System                                          | Input                                                    | Pretraining                                         | Evaluation                                                 | Reported result                                                            | Comparability to yours                               |
| ----------------------------------------------- | -------------------------------------------------------- | --------------------------------------------------- | ---------------------------------------------------------- | -------------------------------------------------------------------------- | ---------------------------------------------------- |
| **Zhang et al., KNN-Net, 2021**                 | mixture spectrogram                                      | Artist20 training                                   | album split stated; 1-s blocks + song voting               | **99% accuracy**, 99% F1 in Table 1                                        | **Very important, but protocol/reporting ambiguity** |
| **Hsieh et al., 2020**                          | mixture / vocal-only / remix                             | none                                                | album split, song-level F1                                 | up to roughly the high-0.7 range depending on augmentation/model           | Comparable split, but different metric               |
| **Kuo et al., FGNL, AAAI 2021**                 | mixture or separated vocals, mel-spectrogram             | none                                                | album split, frame + song F1                               | up to **~0.74 average song F1**, with vocal-only best runs around **0.81** | Same core protocol, different metric                 |
| **MetaSID, 2022**                               | mel spectrogram + CRNN/domain adaptation                 | Artist20 source-domain training                     | album split                                                | **0.83 F1** for CRNN-CAN                                                   | Same split, different metric                         |
| **Zhang et al., 2022, Singer ID for Metaverse** | mixture mel + X-vector + Inception middle-level features | pretrained X-vector / ImageNet Inception components | album split                                                | **0.81 average F1; 0.86 best F1**                                          | Very relevant pretrained-feature baseline            |
| **Yamamoto, APSIPA 2023**                       | **Demucs vocals**, SSL embeddings                        | Wav2Vec2/WavLM/MERT/MapMusic2Vec                    | album split, 4 train albums / 1 val / 1 test               | **WavLM F1 61.9; MapMusic2Vec Top-2 71.5 / Top-3 77.0**                    | Very relevant SSL comparison                         |
| **GMM-UBM, Zhang et al.**                       | separated vocals + MFCC                                  | statistical speaker modeling                        | split not aligned clearly with modern album-split protocol | **97.5% accuracy**, **99.25% vocals-only**                                 | **Do not treat as strict album-SOTA**                |

Sources: KNN-Net ; Hsieh et al. ([arXiv][2]); FGNL ([Liner][3]); MetaSID ([ResearchGate][1]); Metaverse/X-vector model ([Emergent Mind][4]); Yamamoto ; GMM-UBM ([Springer Nature Link][5]).

### The KNN-Net result deserves special attention

The KNN-Net paper reports:

* CRNN: **94% accuracy**
* Attention-CRNN: **95%**
* Attention-CRNN-KNN: **99%**

for Artist20. 

It says each singer has six albums, with four used for training and the remaining two for validation/test. Predictions are made on **1-second blocks**, with an entire-song result obtained by voting. 

But the same paper then says its Artist20 **frame-block** confusion matrix has an average accuracy of **85%**. 

That creates a serious comparability question around the 99% headline number. I would cite it in a report as:

> “Zhang et al. report 99% Artist20 accuracy with KNN-Net, although the paper's evaluation description does not make the exact relationship between this value, the validation/test partitions, and the separately reported 85% frame-block accuracy sufficiently clear for a strict leaderboard comparison.”

That is much safer than saying “Artist20 SOTA = 99%.”

---

## The older 97.5% / 99.25% GMM-UBM result

The GMM-UBM paper reports:

* Artist20: **97.5% accuracy**
* Artist20NoBG: **99.25% accuracy**

with vocal separation and MFCC-based GMM-UBM modeling. ([Springer Nature Link][5])

However, its methodology says:

> “Then, the dataset is split into a training set and testing set.”

rather than adopting the now-standard four-albums-train / one-album-validation / one-album-test configuration. ([Scribd][6])

So I would **not** put 97.5/99.25 beside your 95.2 as though they were equivalent benchmark numbers.

---

# 2. What is actually closest to your experiment?

There are three especially important precedents.

### A. Singer-ID models using pretrained speaker representations

The 2022 “Singer Identification for Metaverse” paper is particularly relevant because it explicitly uses an **X-vector timbre representation from speaker-recognition technology**, together with a CRNN. The authors argue that the X-vector is useful because it can still distinguish singers under accompaniment. ([ResearchGate][7])

Its best Artist20 result is approximately:

**CRNN + X-vector + middle-level feature → 0.86 best F1 / 0.81 average F1.** ([Emergent Mind][4])

That means your observation that a speaker-oriented encoder is unusually good is **not without precedent**. It is actually consistent with a line of Artist20 work going back to speaker-derived representations.

The important novelty in your result is therefore less:

> “Speaker embeddings work for singers.”

and more:

> **“A frozen modern ECAPA speaker embedding is already so discriminative that a tiny closed-set classifier reaches 95.2% top-1 on mixture audio.”**

That is an interesting result.

---

### B. SSL comparison: speech SSL vs music SSL

Yamamoto's 2023 Artist20 experiment is almost tailor-made for your discussion. They compare:

* Wav2Vec 2.0
* WavLM
* MERT
* MapMusic2Vec

using an album split with **4 albums train / 1 validation / 1 test**, and use Demucs vocal separation. 

Their results:

| Model        |       F1 |    Top-2 |    Top-3 |
| ------------ | -------: | -------: | -------: |
| Wav2Vec2     |     60.0 |     70.7 |     76.3 |
| WavLM        | **61.9** |     70.2 |     76.4 |
| MERT         |     56.8 |     68.4 |     75.6 |
| MapMusic2Vec |     59.6 | **71.5** | **77.0** |



This is useful because it supports a very specific interpretation:

**speech-oriented pretrained representations can be better suited to singer identity than music-oriented SSL representations.**

Your ECAPA-vs-MERT gap is therefore scientifically interesting rather than merely “ECAPA happened to win.”

One warning: your **MERT 68.4 top-1 is not directly comparable with the Yamamoto 68.4 number**, because their 68.4 is **Top-2**, not Top-1. 

---

# 3. Is 95.2% suspicious?

### My assessment: **highly impressive, but plausible**

I would not conclude “leakage” merely from the magnitude.

There is a structural reason ECAPA can perform extraordinarily well here:

**Artist20 is a tiny 20-class closed-set problem, and the classes are famous recording artists with highly distinctive voices.**

ECAPA was explicitly trained to learn speaker-discriminative information using VoxCeleb 1+2, attentive statistical pooling, and an additive-margin classification objective. ([Hugging Face][8])

So you are effectively taking a representation specifically optimized for:

> “Who is speaking?”

and transferring it to:

> “Which singer is performing?”

That is almost exactly the right inductive bias for this benchmark.

And there is prior Artist20 evidence that X-vector/speaker representations are useful. ([ResearchGate][7])

### But I would perform these sanity checks before putting 95.2% in a headline table.

#### Check 1 — strict test-set evaluation

The biggest issue is that you currently describe the result as **validation-set** performance.

Do not compare a tuned 95.2% validation result to published **test-set** numbers.

The clean experiment should be:

**freeze the ECAPA encoder → make every hyperparameter decision on validation → lock everything → evaluate once on the held-out album test set.**

This is probably the single most important thing to do.

#### Check 2 — speaker/artist identity leakage

VoxCeleb is speaker-identification data, and the ECAPA checkpoint was trained on VoxCeleb 1+2. ([Hugging Face][8])

You should check whether any Artist20 artists themselves appear as speakers in the pretraining corpus or whether clips from the same commercial recordings somehow occur upstream.

I would explicitly document:

> “No Artist20 audio was used in ECAPA pretraining.”

and, ideally, investigate **identity overlap**, not merely file overlap.

#### Check 3 — song-level versus chunk-level aggregation

Make sure 95.2% means:

> one prediction for one complete held-out song

rather than:

> many 5-s/10-s chunks evaluated independently.

The latter can inflate the apparent sample count and produce a misleadingly precise number.

#### Check 4 — album isolation at every preprocessing stage

The split should happen **before**:

* embedding extraction,
* normalization,
* prototype construction,
* PCA,
* feature selection,
* hyperparameter selection.

For example, fitting StandardScaler/PCA on train+val+test would be leakage.

#### Check 5 — simple nearest-prototype baseline

This is particularly important.

Before accepting “ECAPA + MLP” as the reason for the 95.2%, evaluate:

**ECAPA → L2 normalize → class centroid → cosine nearest centroid**

with no learned classifier.

If that also gets ~90–95%, then the key result is plainly:

> **the ECAPA embedding itself already separates Artist20 artists extremely well.**

That would be more interesting than the MLP result.

---

# Part 2 — the ablations I would actually do

## 1. Classical ML — SVM-RBF 59.3%

Your instinct about feature-group ablation is exactly right, but I would make it slightly more rigorous.

### Ablation A — leave-one-feature-family-out

Run:

* all features
* −MFCC/deltas
* −chroma
* −spectral contrast
* −spectral shape: centroid/bandwidth/rolloff/ZCR
* −tonnetz

Keep the classifier and scaling identical.

Then also run **each feature family alone**.

This gives you two complementary questions:

> “What is indispensable?”

and

> “What information is sufficient by itself?”

The particularly interesting comparison is likely:

**MFCC-only vs chroma/tonnetz-only vs spectral-only vs everything.**

### Ablation B — RandomForest importance, but use permutation importance

I would prefer **permutation importance** over raw RF Gini importance because your feature families contain highly correlated variables such as MFCC coefficients and their deltas.

Aggregate importance at the feature-family level:

$$
I_g = \sum_{j\in g} I_j
$$

and plot the contribution of each group.

That produces a good report figure.

### Ablation C — SVM vs RF vs linear probe under identical feature subsets

For example:

| Features  | Linear SVM | RBF SVM | RF |
| --------- | ---------: | ------: | -: |
| MFCC      |            |         |    |
| MFCC+Δ+Δ² |            |         |    |
| +chroma   |            |         |    |
| all       |            |         |    |

This tells you whether the SVM's advantage is mostly due to the **feature representation** or to **nonlinear decision boundaries**.

That is much more informative than trying C/gamma values.

---

# 2. From-scratch CRNN family

Here I strongly recommend **learning curves over albums**, not random track subsampling.

### Ablation A — album-count learning curve

This is the cleanest experiment for your stated hypothesis.

For each artist, train using:

* 1 album
* 2 albums
* 3 albums
* 4 albums

while keeping the validation and test albums fixed.

Critically, subsample **whole albums**, not individual tracks.

Why?

Because the entire point of Artist20 is that production characteristics are correlated within albums. Randomly removing 25% of tracks can accidentally leave nearly identical production contexts in train and validation.

The question becomes:

$$
\text{Accuracy} = f(\text{number of independent albums})
$$

If all models improve substantially as the number of albums increases while remaining relatively parallel, that is strong evidence that **data volume / diversity**, rather than architecture, is the bottleneck.

### Ablation B — model error correlation

Your ensemble-diversity idea is also excellent.

For every test song, record:

$$
e_m =
\begin{cases}
0 & \text{correct}\\
1 & \text{incorrect}
\end{cases}
$$

Then calculate pairwise:

* disagreement rate
* Pearson correlation of error indicators
* Jaccard overlap of error sets
* double-fault measure

The important result is whether:

> CRNN2D, Nasrullah-CRNN, and FGNL make mostly the **same errors**.

If yes, that is powerful evidence that they share a **dataset/input limitation** rather than a model-architecture limitation.

If errors are highly complementary, then an ensemble experiment becomes interesting.

### Ablation C — oracle/soft ensemble

Do one simple experiment:

$$
p_{\text{ensemble}} = \frac{1}{M}\sum_m p_m
$$

Then compare:

* best individual
* probability-average ensemble
* oracle ensemble

The oracle accuracy answers:

> “How much headroom is hidden in complementary errors?”

That is a very nice course-report diagnostic.

---

# 3. MERT vs ECAPA — the most interesting section of your report

Your proposed classifier-head swap is **exactly the experiment I would prioritize**.

### Ablation A — same encoder, multiple heads

For each embedding:

**ECAPA**
→ linear softmax
→ 2-layer MLP
→ cosine nearest centroid
→ linear discriminant analysis / nearest-centroid baseline

and identically:

**MERT**
→ linear softmax
→ 2-layer MLP
→ cosine nearest centroid.

Then you obtain something like:

| Encoder | Linear | MLP | Cosine prototype |
| ------- | -----: | --: | ---------------: |
| MERT    |        |     |                  |
| ECAPA   |        |     |                  |

This cleanly separates:

$$
\text{performance}
\approx
\text{representation quality}
+
\text{head compatibility}.
$$

### Ablation B — frozen embeddings + classifier-capacity curve

Use:

* linear
* 1 hidden layer
* 2 hidden layers
* nearest-centroid

while keeping the embedding absolutely fixed.

I expect that if ECAPA is genuinely carrying identity information, its performance should be high even with the **linear classifier**.

That would be a much stronger scientific result than:

> “our MLP happened to get 95.2%.”

### Analysis C — embedding separability

Yes, **silhouette score is useful**, but I would not make it the primary diagnostic.

I'd use:

1. **intra-class / inter-class cosine distance distributions**
2. nearest-centroid accuracy
3. silhouette score
4. optionally a 2-D UMAP/t-SNE visualization

The most informative plot would be:

> ECAPA same-artist cosine similarity vs different-artist cosine similarity

compared with the same plot for MERT.

If ECAPA shows strongly separated distributions and MERT shows substantial overlap, you have a very intuitive explanation for the accuracy gap.

I would avoid using t-SNE as quantitative evidence; use it only as visualization.

---

# 4. Vocal separation: 64.9 → 67.1%

I would **not** do only an aggregate error-overlap analysis.

Do this three-part experiment.

### Analysis A — per-artist Δ accuracy

For each artist:

$$
\Delta_a =
Acc_{vocals,a} - Acc_{mix,a}
$$

Plot all 20 artists.

You may find:

* some artists improve +8–15 pp,
* some are unchanged,
* some actually get worse.

That is substantially more interesting than saying “+2.2 pp overall.”

### Analysis B — per-song transition matrix

For every test song classify it into:

| Mixture | Vocals  | Meaning               |
| ------- | ------- | --------------------- |
| correct | correct | stable                |
| wrong   | correct | **separation fix**    |
| correct | wrong   | **separation damage** |
| wrong   | wrong   | unresolved            |

Then report:

$$
\Delta =
N_{\text{fix}} - N_{\text{damage}}.
$$

This directly tells you whether +2.2 pp comes from a small set of genuinely difficult songs being rescued or from a broad tiny improvement.

### Analysis C — correlate improvement with vocal dominance

You already have the separated signal.

Compute something like:

$$
VDR =
\frac{E_{\text{vocals}}}
{E_{\text{mixture}}}
$$

or another simple vocal-to-mixture energy measure.

Then ask:

> Do songs with lower vocal dominance benefit more from separation?

That produces a plausible mechanistic explanation for the result.

This is better than merely saying “dense instrumentation hurts.”

Hsieh et al. explicitly motivate source separation around the accompaniment-confound hypothesis, and their experiments also analyze vocalness-related behavior. ([arXiv][2])

---

# 5. Qwen2-Audio zero-shot — 47.5%

I would **not spend much time on generic prompt engineering**.

The most interesting low-cost experiments are:

### Experiment A — self-consistency over random song excerpts

Sample, say, 5–10 different excerpts from the same song.

Run the same closed-set prompt independently.

Then aggregate:

$$
\hat y =
\operatorname{mode}
(\hat y_1,\ldots,\hat y_K).
$$

This tests whether the current 47.5% is limited by **unstable local evidence**.

### Experiment B — provide multiple excerpts in one query

Give the model several non-overlapping clips from the same song and ask for one artist prediction.

Conceptually:

> “These are multiple excerpts from the same song. Choose exactly one artist from the list.”

This tests whether temporal aggregation helps an audio-language model without training.

### Experiment C — prompt sensitivity

Keep the candidate list fixed and compare:

> “Who is the singer?”

vs.

> “Choose exactly one of these 20 artists based only on vocal identity.”

vs.

> “Ignore genre/instrumentation and identify the vocalist.”

I would only do this as a **small diagnostic**, not a large prompt sweep.

The scientific question is:

> Is 47.5% evidence of usable zero-shot singer representation, or mostly evidence that the model can exploit superficial musical/artist context?

---

# What I would prioritize for your course report

You have enough experiments already that I would **avoid adding five more architectures**.

The strongest report storyline would be:

### 1. Classical features unexpectedly strong

**Feature-group ablation → explain the 59.3%.**

### 2. From-scratch deep models hit a ceiling

**Album-count learning curve → test the small-data hypothesis.**

### 3. Pretraining changes the game

**MERT vs ECAPA with identical classifier heads → isolate representation quality.**

### 4. Separation has limited value

**Fix/damage transition matrix + per-artist Δ → explain the +2.2 pp.**

### 5. ECAPA result

**Linear/cosine/MLP probe + strict held-out test → establish whether 95.2% comes from the embedding itself.**

That last experiment is the most important one.

---

# One especially strong experiment I would add

There is a very clean experiment hiding in your current setup:

## ECAPA zero-shot nearest-centroid vs trained MLP

Do this:

```text
train albums
    ↓
ECAPA frozen
    ↓
one embedding/song
    ↓
class centroid for each of 20 artists
    ↓
cosine similarity
    ↓
nearest artist
```

No learned classifier at all.

Then compare:

```text
Random               5.0%
SVM handcrafted      59.3%
MERT + linear/MLP      ?
ECAPA + centroid       ?
ECAPA + linear         ?
ECAPA + MLP            95.2%
```

If ECAPA-centroid is already, say, **92–95%**, that is an excellent scientific story:

> **A speaker-verification representation learned outside the target task linearly separates the Artist20 singers with remarkable effectiveness, whereas substantially more specialized music models trained on Artist20 struggle under the album-confound setting.**

That is much more interesting than “we tried another neural network.”

---

## Bottom line for your report

I would currently phrase the SOTA conclusion approximately as:

> **There is no clean, universally comparable single SOTA number for Artist20 because published work reports different metrics and evaluation protocols. Under album-split evaluation, recent directly comparable systems generally report song-level F1 around 0.8–0.86, while some papers report substantially higher accuracy numbers under less clearly aligned protocols. Our 95.2% top-1 validation accuracy from a frozen VoxCeleb ECAPA-TDNN embedding is therefore exceptionally strong, but should not be called SOTA until confirmed on the locked held-out album-level test set and checked for upstream identity overlap.**

That conclusion is supported by the published Artist20 protocols/results I found. 

And one very useful framing point: **your 95.2% result is not obviously “too good to be true.”** There is already published Artist20 work using speaker-derived X-vectors, and the current ECAPA checkpoint is explicitly trained for speaker discrimination on VoxCeleb 1+2. ([ResearchGate][7])

The thing I would be most suspicious of is therefore **not the ECAPA representation itself, but the evaluation protocol**—especially validation vs. held-out test, chunk vs. song aggregation, and any upstream identity/data overlap.

[1]: https://www.researchgate.net/publication/360834074_MetaSID_Singer_Identification_with_Domain_Adaptation_for_Metaverse?utm_source=chatgpt.com "(PDF) MetaSID: Singer Identification with Domain Adaptation for Metaverse"
[2]: https://arxiv.org/abs/2002.06817 "Addressing the confounds of accompaniments in singer identification"
[3]: https://liner.com/review/positions-channels-and-layers-fully-generalized-nonlocal-network-for-singer?utm_source=chatgpt.com "Positions, Channels, and Layers: Fully Generalized Non-Local Network for Singer Identification [Quick Review]"
[4]: https://www.emergentmind.com/topics/performsinger?utm_source=chatgpt.com "PerformSinger: Multimodal Singing Synthesis"
[5]: https://link.springer.com/book/10.1007/978-981-13-8707-4?utm_source=chatgpt.com "Proceedings of the 6th Conference on Sound and Music Technology (CSMT): Revised Selected Papers | Springer Nature Link"
[6]: https://www.scribd.com/document/966400509/Proceedings-of-the-6th-Conference-on-Sound-and-Music-Technology-CSMT-Revised-Selected-Papers-Wei-Li-eBook-available-instantly?utm_source=chatgpt.com "Proceedings of The 6th Conference On Sound and Music Technology CSMT Revised Selected Papers Wei Li Ebook Available Instantly | PDF"
[7]: https://www.researchgate.net/publication/360834322_Singer_Identification_for_Metaverse_with_Timbral_and_Middle-Level_Perceptual_Features?utm_source=chatgpt.com "(PDF) Singer Identification for Metaverse with Timbral and Middle-Level Perceptual Features"
[8]: https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb?utm_source=chatgpt.com "speechbrain/spkrec-ecapa-voxceleb · Hugging Face"
