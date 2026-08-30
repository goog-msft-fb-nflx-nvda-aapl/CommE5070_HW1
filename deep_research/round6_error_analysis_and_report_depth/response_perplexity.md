## Part 1: Error analysis methodology

### 1. What the Artist20 literature actually establishes

The central methodological issue in Artist20 is not merely class imbalance or generic multiclass confusion. It is **accompaniment and production confounding**: because an artist’s songs often share genre, instrumentation, producers, and recording practices, a classifier can recognize the musical context rather than the singer. Hsieh et al. explicitly frame the problem this way and test vocal separation, vocal-only training, vocal–instrumental remixing, and vocal-melody features—not a formal musicological taxonomy of individual artist confusions. [arxiv](https://arxiv.org/abs/2002.06817)

Earlier Artist20 work makes a related point: MFCC-based systems can perform well partly because spectral features reflect instrumentation, which correlates with artist identity; chroma features add partially independent melodic/harmonic information.  Thus, Hsieh et al. is highly relevant for your **confound analysis**, but it does not appear to provide a reusable, pair-by-pair “why X is confused with Y” protocol. [ee.columbia](https://www.ee.columbia.edu/~dpwe/pubs/Ellis07-timbrechroma.pdf)

A strong report should therefore distinguish three hypotheses:

1. **Vocal-identity similarity:** timbre, register, formant structure, vibrato, phonation, rasp, breathiness, pitch dynamics.
2. **Musical-style similarity:** genre, harmonic language, rhythm, arrangement, instrumentation.
3. **Recording/production similarity:** microphone, compression, reverb, distortion, mastering, backing vocals, producer or album signature.

Do not describe a confusion pair as vocal similarity unless you have performed a vocal-isolated or controlled analysis supporting that interpretation.

### 2. Practical confusion-pair checklist

For each important directed pair \(A \rightarrow B\), create a short case sheet containing:

| Analysis layer | Concrete procedure | Interpretation |
|---|---|---|
| Confusion magnitude | Report \(n_{A\rightarrow B}\), \(P(\hat y=B\mid y=A)\), and the reverse \(B\rightarrow A\) rate | Separates a genuine symmetric similarity from a one-sided bias |
| Album concentration | Break the errors down by held-out album and track | A cluster in one album suggests production/song-content dependence |
| Segment localization | Run inference on multiple vocal-active windows and aggregate votes | Reveals whether errors occur in verses, choruses, instrumental gaps, or specific vocal techniques |
| Vocal presence | Compare original mix, vocal-separated audio, and optionally instrumental-only audio | Vocal-only improvement supports singer identity; instrumental-only performance exposes accompaniment leakage |
| Acoustic profile | Compare pitch range, median pitch, voiced/unvoiced ratio, spectral centroid, MFCC statistics, spectral tilt, ZCR, vibrato rate, and harmonicity | Provides measurable evidence for timbre/register hypotheses |
| Musical context | Compare tempo, key/chroma, chord profile, instrumentation, energy envelope, and genre/era annotations | Tests style and arrangement explanations |
| Recording context | Inspect reverb, compression, distortion, stereo width, backing vocals, and mastering characteristics | Identifies production artifacts |
| Human listening | Blindly present vocal-only and mixture excerpts to several listeners or a knowledgeable annotator | Provides qualitative support, but should not be treated as definitive evidence |
| Model agreement | Compare whether the same pair appears across architectures, features, and seeds | Stable pairs are stronger evidence than one-model accidents |

Use normalized confusion matrices, not only raw counts, because artists may contribute different numbers of validation tracks. Also report confidence intervals for per-artist recalls where feasible; a difference based on only a few tracks can otherwise look substantial.

A useful report structure is:

> “The dominant \(A\rightarrow B\) error occurred primarily on album C and disappeared/reduced after vocal separation. This suggests accompaniment or production leakage rather than stable vocal similarity. Conversely, if it persists in vocal-only audio and is concentrated in vocal-active chorus segments, a timbre/register explanation is more plausible.”

### 3. Feature importance tied to artist pairs

Global permutation importance is insufficient for artist-specific claims. Use **pair-conditional importance**:

1. Select a pair \(A,B\).
2. Restrict the evaluation set to examples whose true label is \(A\) or \(B\).
3. Compute baseline pairwise accuracy or balanced accuracy.
4. Permute one feature group—MFCC, chroma, spectral contrast, ZCR, tonnetz, etc.—within that subset.
5. Measure the drop:
   \[
   \Delta_{A,B,g}
   =
   \text{score}_{A,B}
   -
   \text{score}_{A,B\text{ with group }g\text{ permuted}}.
   \]
6. Repeat across bootstrap resamples and report a confidence interval.
7. Repeat in both directions, \(A\rightarrow B\) and \(B\rightarrow A\), because the discriminative cue may be asymmetric.

For multiclass models, use one-vs-one margins or probabilities rather than merely asking whether the prediction is correct. For example, calculate how permuting MFCCs changes:

\[
p(B\mid x)-p(A\mid x)
\]

for true-\(A\) examples. This tells you whether the feature group specifically changes the model’s \(A\)-versus-\(B\) decision.

Recommended feature groups:

- **MFCC and spectral contrast:** spectral envelope, instrumentation, recording coloration.
- **Chroma and tonnetz:** harmony, tonal profile, recurring chordal preferences.
- **ZCR and spectral flux:** noisiness, articulation, percussiveness, distortion, high-frequency activity.
- **Pitch and vocal features**, if available: register, melodic contour, vibrato, voiced fraction.

This interpretation should remain cautious. Ellis’s Artist20 analysis found MFCCs stronger than chroma overall and explicitly noted that MFCC performance can reflect instrumentation, while chroma contributes complementary harmonic information.  Therefore, “MFCCs distinguish Artist A from Artist B” does not automatically mean “the vocal timbres differ”; it may mean their arrangements or production differ. [ee.columbia](https://www.ee.columbia.edu/~dpwe/pubs/Ellis07-timbrechroma.pdf)

### 4. Significance testing on 231 tracks

Use **paired** methods because every model is evaluated on the same tracks.

#### Primary recommendation: paired bootstrap

For each model pair:

1. Store per-track top-1 correctness and top-3 correctness.
2. Resample the 231 tracks with replacement, preserving the same sampled indices for both systems.
3. Recompute the metric difference for each resample.
4. Report the 2.5th and 97.5th percentiles of the difference distribution.
5. Repeat with a fixed seed and at least several thousand resamples.

For your graded prediction metric,

\[
M = \text{top1} + 0.5\text{top3},
\]

bootstrap \(M_B-M_A\) directly. Also report the raw difference in correct top-1 and top-3 decisions. A difference of 0.4–0.5 percentage points corresponds to roughly one track at this sample size, so it should normally be described as a small observed gain unless the paired interval is clearly away from zero.

#### McNemar’s test

McNemar’s exact test is appropriate for comparing **top-1 correctness** of two classifiers on the same examples. It uses only the discordant cases:

- A correct, B incorrect.
- A incorrect, B correct.

It does not directly test top-3 accuracy or your weighted score. It is therefore a useful secondary test, not the sole analysis. McNemar is specifically designed for paired nominal outcomes on one test set. [aclanthology](https://aclanthology.org/P18-1128.pdf)

For top-3, use a paired bootstrap or a paired permutation/randomization test. For multiple model comparisons, adjust for multiplicity—Holm correction is a reasonable, interpretable choice. The multiple-McNemar literature specifically discusses exact McNemar testing with Bonferroni–Holm adjustment. [pmc.ncbi.nlm.nih](https://pmc.ncbi.nlm.nih.gov/articles/PMC2902578/)

Report:

- Absolute metric difference.
- 95% paired-bootstrap CI.
- McNemar exact \(p\)-value for top-1.
- Number of pairwise comparisons and multiplicity correction.
- Whether the test set was used for model selection.

Do not use independent-proportion tests, because they ignore the pairing and are less informative here.

***

## Part 2: Interpreting the Sia result

### 1. What can be said about “Unstoppable”

“Unstoppable” is generally categorized as **electropop**, not primarily pop-soul.  That matters because the song’s audible identity includes programmed production, compressed/driven drums, synth layers, vocal processing, and an anthemic dynamic build. [en.wikipedia](https://en.wikipedia.org/wiki/Unstoppable_(Sia_song))

The strongest credible vocal description in the sources is qualitative rather than a definitive song-specific range measurement. NPR describes Sia’s characteristic approach as moving from a low-register, processed sound to a large chorus delivered with exertion, rasp, strain, and a pushed belt.  A vocal-analysis source estimates “Unstoppable” around F♯3–B4, while another estimates a wider D3–A5 range but explicitly labels that estimate as pending verification.  These should be presented as estimates, not authoritative physiological classifications. [npr](https://www.npr.org/2018/11/06/664395501/sia-is-the-21st-centurys-most-resilient-songwriter)

A defensible description is:

> On “Unstoppable,” Sia uses a relatively low, speech-like and processed verse/pre-chorus delivery that expands into a forceful, high-energy chorus. Her recognizable sound combines a dark or throaty lower register, breathy/processed onset, audible rasp or grain, strong chest-dominant belting, rapid expressive vibrato, and a large pop-anthem production frame.

The phrase “breathy, powerful, raspy, belting” is reasonable as a descriptive summary, but avoid claiming a precise voice type such as mezzo-soprano or light-lyric soprano unless you label it as an informal vocal-coach classification. The available range sources disagree substantially.

### 2. What the model’s selected artists plausibly represent

There is not strong, source-quality evidence establishing a documented direct vocal comparison between Sia and each of Tori Amos, Freddie Mercury, Stevie Nicks, Madonna, and Marie Fredriksson. Therefore, your report should not claim that these artists are established vocal equivalents of Sia.

A more defensible interpretation is:

| Prediction | Plausible shared cue | Strength of interpretation |
|---|---|---|
| Tori Amos | Female singer-songwriter identity, expressive intensity, dark/bright register contrast, emotionally distinctive phrasing | Possible vocal-expression analogue, but not a documented one-to-one match |
| Freddie Mercury / Queen | Large dynamic range, forceful projection, dramatic anthemic phrasing, high-energy chorus | More likely delivery/anthemic-performance similarity than literal timbre; gender is not a sufficient explanation |
| Stevie Nicks / Fleetwood Mac | Grainy or breathy female timbre, dark lower register, emotive phrasing | Plausible timbral analogue, but arrangement and band-production cues may dominate |
| Madonna | Processed electropop production, rhythmic vocal delivery, pop-era similarity | Particularly plausible as production/genre similarity rather than vocal identity |
| Marie Fredriksson / Roxette | Powerful melodic pop-rock chorus, female vocal intensity, polished 1980s–1990s pop-rock production | Mixed explanation: chorus delivery plus production similarity |

The cross-model pattern is more informative than any individual label:

- `tori_amos`, `queen`, and `fleetwood_mac` appearing in the strongest model’s top three suggests a broad cluster involving expressive intensity, vocal grain, and anthem-like dynamics.
- `madonna` and `roxette` appearing in other models is consistent with electropop/pop-rock production, compressed vocal texture, and melodic chorus structure.
- The frozen ECAPA system’s strong `fleetwood_mac` probability may indicate speaker/timbre embedding similarity, but ECAPA is not a musicological voice-comparison oracle; mixture acoustics, singing style, and recording conditions can all affect it.
- The relatively flat top three from `sota_crnn_wide`—13.7%, 12.9%, 12.4%—is evidence of uncertainty, not evidence that Sia is genuinely equally close to those three artists.

NPR’s description of Sia’s pushed, raspy, effortful belt supports a broad comparison to artists characterized by expressive, forceful delivery, but it does not specifically validate Tori Amos, Mercury, Nicks, Madonna, or Fredriksson as documented matches. [npr](https://www.npr.org/2018/11/06/664395501/sia-is-the-21st-centurys-most-resilient-songwriter)

### 3. How to frame the OOD prediction

The correct scientific interpretation is not “the model identified Sia as Tori Amos.” It is:

> The closed-set classifier projected an unknown singer into the nearest available regions of its 20-class decision space.

The Hsieh et al. paper gives you a strong reason to suspect production and accompaniment effects: models can exploit non-vocal features when artists are associated with recurring musical contexts, and performance can deteriorate when a singer is placed in an unseen context. [arxiv](https://arxiv.org/abs/2002.06817)

The MIR literature also supports separating vocal and accompaniment information. Vocal-timbre modeling work explicitly treats accompaniment as a source of contamination and reports improved singer identification after reducing accompaniment influence.  Ellis’s Artist20 results similarly caution that MFCC-based artist recognition can exploit instrumentation correlated with artist labels. [ee.columbia](https://www.ee.columbia.edu/~dpwe/pubs/Ellis07-timbrechroma.pdf)

So the current evidence does **not** justify choosing between “timbre” and “production” as a general rule. The likely answer is model- and input-dependent:

- Mel-spectrogram CNN/CRNN models can use both voice and accompaniment.
- Speaker-embedding models may emphasize vocal identity, but singing and accompaniment can still distort embeddings.
- A phone recording introduces microphone response, room acoustics, background noise, accompaniment leakage, and possibly the singing voice of the person recording rather than Sia’s studio voice.
- Since the clip is a person singing along, the model may be identifying the amateur singer’s voice, the backing track, or the combined mixture—not Sia.

Run these controls before making a stronger claim:

1. Vocal-separate the clip and classify the separated vocal.
2. Classify the instrumental/backing component alone.
3. Mute or attenuate the backing track and compare predictions.
4. Compare the same singer on spoken voice, dry singing, and processed singing.
5. Pitch-shift the vocal while preserving timbre as much as possible.
6. Add Artist20-style songs from known artists with matched production and inspect whether predictions move toward the producer/genre cluster.
7. Compare logits or embeddings, not only softmax probabilities.

A concise report paragraph could be:

> The OOD Sia experiment should be interpreted as closed-set nearest-class behavior rather than recognition of Sia. “Unstoppable” is an electropop anthem whose vocal combines a darker, processed lower delivery with an exerted, grainy belt and large dynamic escalation. These properties may plausibly place the clip near artists such as Tori Amos, Stevie Nicks, or Freddie Mercury in a broad expressive-vocal space, while Madonna and Roxette may reflect electropop/pop-rock production and chorus structure. However, Artist20 research shows that accompaniment and production can become confounds, and the phone recording further mixes singer, backing track, and recording conditions. The model’s varied and relatively flat top-three distributions therefore demonstrate an interpretable but non-diagnostic OOD projection, not evidence of a verified vocal similarity.

***

## Part 3: Highest-value additions to the report

### Priority order

| Priority | Analysis | Recommendation |
|---|---|---|
| 1 | Confound and album diagnostics | Highest value |
| 2 | Paired uncertainty and significance | Highest value |
| 3 | Calibration | High value, especially for Sia |
| 4 | Pairwise feature attribution | High value for Task 1 |
| 5 | Embedding geometry | Useful if quantitatively tied to performance |
| 6 | Ablation presentation | Necessary for readability, but not itself a new experiment |

### 1. Statistical rigor

Definitely include paired bootstrap intervals for the weighted metric and exact McNemar tests for top-1. Do not frame every tiny improvement as meaningful. For each major comparison, show:

- Top-1 and top-3 point estimates.
- \(M=\text{top1}+0.5\text{top3}\).
- 95% CI for the paired difference.
- McNemar result for top-1.
- Number of seeds, if multiple runs exist.
- Whether the model was selected using that validation set.

Since you have many architectures and ensemble variants, distinguish **exploratory** comparisons from the final confirmatory comparison. Otherwise, the best observed variant among many trials is subject to selection bias.

### 2. Calibration

This is especially worthwhile because the Sia example uses softmax probabilities.

For each important model, report:

- Reliability diagram.
- Expected Calibration Error, with the number of bins stated.
- Brier score.
- Negative log-likelihood.
- Maximum probability on correct versus incorrect predictions.
- Accuracy as a function of confidence.
- Risk–coverage curve: accuracy when retaining only predictions above a confidence threshold.

Use validation data for temperature scaling and evaluate calibration on untouched test data. Do not calibrate on the same set used to report final calibration.

Most importantly, explain that softmax probabilities are **closed-set confidence scores**. A 46.9% probability for `fleetwood_mac` on Sia does not mean “46.9% probability that Sia is Fleetwood Mac”; it means the model assigns that fraction of its normalized closed-set score to that class. Calibration analysis will make this distinction concrete.

### 3. Album and production diagnostics

This should be one of the centerpiece analyses because album splitting is explicitly intended to reduce confounding. Artist20’s canonical setup uses albums as the split unit, and later work repeatedly describes album splitting as a way to prevent the model from exploiting album clues. [cdn.aaai](https://cdn.aaai.org/ojs/17000/17000-13-20494-1-2-20210518.pdf)

For each artist:

- Report accuracy by held-out album.
- Report the gap between the easiest and hardest album.
- Compare album-level accuracy with track-level accuracy.
- Calculate whether errors concentrate in particular albums.
- Compare predictions from original mix, vocal-only, and instrumental-only audio.
- Train or evaluate a simple album/production classifier as a diagnostic upper bound.
- Test whether embeddings cluster by artist or album more strongly.

A useful quantitative measure is the variance of album-level recalls within each artist. A strong overall result with extreme album variation suggests residual domain dependence.

Also perform a **leave-one-album-style stress test** if your split permits it: train on some albums and evaluate separately on each unseen album, without retuning. This is closer to the actual question “does the system generalize to a new recording context?”

### 4. Embedding geometry

Your proposed analysis is worthwhile, but avoid relying on t-SNE. Compute on the original embeddings:

- Mean within-artist cosine similarity.
- Mean between-artist cosine similarity.
- Within-artist dispersion.
- Between-centroid distance.
- Silhouette score.
- Davies–Bouldin or related cluster indices.
- Pairwise artist-centroid similarity matrix.
- Retrieval metrics such as Recall@1 and Recall@3 using nearest-neighbor embeddings.

Compute these separately:

- Overall.
- By album.
- On vocal-only versus mixture audio.
- For correct versus incorrect tracks.

The most persuasive result would be something like:

> Model B improves top-1 accuracy because its within-artist compactness improves on unseen albums while its between-artist separation remains stable.

Do not interpret a pretty t-SNE plot as evidence of representation quality; t-SNE can create apparent clusters through its projection procedure.

### 5. Ablation summary convention

Do not place 19 rows of loosely comparable experiments in one undifferentiated table. Use a hierarchical presentation:

1. **Task definition and common protocol**
   - Split, input duration, sampling rate, augmentation, optimizer, number of seeds.
2. **Task 1 feature-family table**
   - Feature group, dimensionality, classifier, top-1/top-3, macro-F1, key observation.
3. **Task 2 development table**
   - Architecture family, major change, parameter count, training protocol, validation results.
4. **Focused ablation table**
   - One factor at a time: width, recurrence, pooling, augmentation, loss, segment strategy.
5. **Final comparison**
   - Best single from-scratch model, pretrained baselines, zero-shot model, ensemble.
6. **Error and robustness table**
   - Album breakdown, per-artist recall, confidence/calibration, OOD behavior.

For each ablation, state the intended question in plain language:

> “Does temporal recurrence help after controlling for input representation?”

Then show the matched comparison, not merely the best result from different settings.

### Recommended final report additions

If time is limited, prioritize these five deliverables:

1. A paired-bootstrap/McNemar comparison of the final ensemble against its strongest single-model baseline.
2. Per-artist and per-album recall with confidence intervals.
3. Original, vocal-only, and instrumental-only confound tests on a representative subset.
4. Calibration plots and Brier/ECE results, including interpretation of the Sia probabilities.
5. Pair-conditional permutation importance for the top three Task-1 confusion pairs.

These additions would turn the report from a catalog of models into a defensible scientific analysis of **what the system learned, when it fails, and whether its performance reflects singer identity or recording context**.