The strongest defensible conclusion is that your **95.2% validation top-1 / 98.7% top-3 result is not directly comparable to most published Artist20 numbers**. Published work usually reports **song-level F1**, often under different clip lengths, voting rules, and train/test protocols; the best clearly documented album-split result I found is approximately **0.86 song-level F1 from CRNN + X-vector**, while Hsieh et al. report **0.75 song-level F1** for their CRNN-with-melody plus shuffle-and-remix system. Your result could be genuinely strong, but it requires a strict held-out test evaluation and leakage/protocol audit before calling it Artist20 SOTA.

## Part 1: Artist20 results

### What is actually published?

| System | Input / representation | Pretraining | Album-split result | Comparability |
|---|---|---|---|---|
| Ellis baseline | MFCC/chroma-style timbral features | None | About 54.5% accuracy in the original six-fold experiment | Different early protocol; useful historical baseline |
| Nasrullah & Zhao CRNN | Mel-spectrogram → 4 CNN layers → 2 GRUs → FC | Artist20 only | Average song-level F1 ≈ 0.674 for the 30-second frame setting; 0.603 for the 30-second frame-feature setting | Closest conventional baseline |
| Hsieh et al. CRNNM + data augmentation | Mel-spectrogram plus CREPE melody branch; CNN/GRU; open-unmix vocals and shuffle-and-remix augmentation | Open-Unmix and CREPE pretrained models; SID trained on Artist20 | **0.75 song-level F1** | Album split; songs evaluated by majority voting |
| FGNL CRNN | CRNN plus fully generalized non-local attention | Artist20 only | **0.73 average song-level F1** on original audio at 5 seconds; best run 0.82 | F1, not accuracy; three-run averages |
| FGNL CRNNM | CRNNM plus FGNL | CREPE plus Artist20 training | **0.74 average song-level F1** at 5 seconds on original audio; best run 0.81 | F1, not accuracy |
| FGNL with vocals-only input | CRNN/CRNNM plus FGNL | Open-Unmix plus Artist20 training | Up to **0.83 average song-level F1** for CRNNM-FGNL at 5 seconds in the reported table | Separation applied to train and test; F1 |
| CRNN-CAN / MetaSID | CRNN plus contrastive adaptation/domain adaptation | Artist20 training; domain-adaptation setup | Reported **0.83 song-level F1** on album split | Different task framing and protocol details |
| WaveNet classifier | Raw waveform → WaveNet-style network | Artist20 only | Reported average F1 ≈ **0.854** | Requires checking exact split and aggregation details |
| CRNN + X-vector | CRNN combined with speaker-style x-vector features | Speaker/audio pretraining, depending on implementation | Reported average F1 ≈ **0.86** | Potentially the strongest directly relevant embedding-based result, but source/protocol should be verified |
| Timbral i-vector system | MFCC statistics → i-vector modeling | Generic speaker/i-vector modeling | **84.31% accuracy**, 83.68% F1 | Six-fold album-based evaluation, but not necessarily the same fixed train/val/test split |
| Your ECAPA probe | Frozen ECAPA-TDNN embedding → MLP | VoxCeleb speaker verification | **95.2% top-1 / 98.7% top-3 on validation** | Not yet comparable unless evaluated on the fixed held-out test set with matching song-level aggregation |

The Hsieh paper explicitly states that the album split places songs from the same album exclusively in train, validation, or test, and reports results using 5-second segments with song-level majority voting. Its best reported system, CRNNM with data augmentation, reaches **0.75 song-level F1**. cite [arxiv](https://arxiv.org/abs/2002.06817)

The FGNL paper reports, for original mixtures, average song-level F1 values of 0.72–0.73 for CRNN-FGNL and 0.73–0.74 for CRNNM-FGNL depending on clip length. Its vocals-only setting reaches as high as 0.83 average F1 for CRNNM-FGNL at 5 seconds. cite [cdn.aaai](https://cdn.aaai.org/ojs/17000/17000-13-20494-1-2-20210518.pdf)

Other published systems report approximately 0.83 F1 for CRNN-CAN, 0.854 for a WaveNet classifier, and approximately 0.86 for a CRNN plus x-vector system. These numbers should be treated cautiously until their exact evaluation scripts, split files, confidence filtering, and song-level aggregation are aligned. cite [arxiv](http://arxiv.org/abs/2205.11821)

### Raw, vocals-only, and embeddings

The literature does not provide a single clean leaderboard because three dimensions are mixed together:

1. **Input condition**
   - Raw/polyphonic mixture.
   - Open-Unmix or another separator’s vocals-only output.
   - Mixture plus auxiliary melody features.
   - Learned embeddings such as x-vectors or i-vectors.

2. **Evaluation metric**
   - Top-1 accuracy.
   - Macro or weighted F1.
   - Frame-level F1.
   - Song-level F1 after majority voting or confidence filtering.

3. **Split and aggregation**
   - Fixed album-level train/validation/test split.
   - Six-fold album cross-validation.
   - Frame-level classification followed by song-level voting.
   - Occasionally unclear or inconsistently described split procedures.

Consequently, there is no reliable published “top-1 accuracy leaderboard” that can be compared numerically with your 95.2% unless the papers’ predictions are available and recomputed under your exact protocol.

### Is 95.2% suspicious?

It is **above the strongest clearly reported conventional album-split results**, but the comparison is provisional. ECAPA-TDNN is trained for speaker discrimination on VoxCeleb, so its embedding can encode highly useful vocal identity cues even though it was not trained for Artist20. Speaker-style embeddings are therefore a plausible explanation, not automatically evidence of leakage.

The main checks I would perform are:

- **Evaluate once on the untouched test partition.** Do not select the classifier, MLP depth, normalization, or threshold using test results.
- **Verify album disjointness by metadata and audio hashes.** Confirm that no duplicated master, remaster, alternate encoding, or near-duplicate track crosses partitions.
- **Check segmentation leakage.** All chunks from one song must remain in one partition. More subtly, do not fit scaling, PCA, normalization statistics, or calibration parameters using validation-plus-test data.
- **Aggregate at song level.** A track-level result should be obtained by aggregating chunk embeddings or chunk predictions, not by counting chunks as independent songs.
- **Compare embedding extraction modes.** Run ECAPA on:
  1. full mixture;
  2. vocals-only audio;
  3. isolated short vocal regions;
  4. instrumental-only audio.
- **Run an instrumental control.** If instrumental-only ECAPA embeddings remain highly predictive, the model is likely exploiting album/production/channel cues rather than primarily singer identity.
- **Run a shuffled-label control.** A high result after label shuffling indicates a pipeline or split bug.
- **Use artist-balanced confidence intervals.** With only 20 artists and a relatively small held-out set, a few songs can move the percentage substantially.
- **Repeat across seeds and fixed splits.** Report mean, standard deviation, and preferably bootstrap confidence intervals.
- **Compare a linear probe.** If a linear classifier already reaches roughly the same performance, the representation is strongly linearly separable. If the MLP adds a large jump, inspect whether it is overfitting the validation set.
- **Test a truly external set if possible.** A small set of live performances, alternate albums, or tracks absent from Artist20 is more informative than another in-dataset improvement.

A particularly important diagnostic is the **vocal-only versus instrumental-only ECAPA experiment**. Your raw-mixture result may be high because ECAPA is responding to vocal timbre, but it could also exploit recording conditions, mastering signatures, accompaniment, or artist-specific production regularities. Hsieh et al. specifically identify accompaniment as a confound and show that album splitting is intended to reduce, not necessarily eliminate, this issue. cite [arxiv](https://arxiv.org/pdf/2002.06817.pdf)

## Part 2: Highest-value ablations

### 1. Classical ML

Your most informative experiment is a **nested feature-group ablation**, not generic feature importance.

Use the same train/validation/test split and compare:

| Feature set | Purpose |
|---|---|
| MFCC only | Vocal spectral envelope and timbre |
| MFCC + deltas | Local temporal dynamics |
| Chroma only | Harmonic/key-related information |
| Spectral contrast + centroid/bandwidth/rolloff | Brightness and spectral shape |
| ZCR only | Noisiness and high-frequency activity |
| Tonnetz only | Tonal/harmonic relations |
| All groups | Full baseline |
| All groups without one group | Marginal contribution |

Run this with the same SVM-RBF hyperparameter-selection procedure. Report the mean and standard deviation across several seeds or folds if possible.

Then add two analyses:

- **Permutation importance and SHAP-style group importance for Random Forest**, aggregated by feature family rather than individual coefficient.
- **SVM versus simpler classifiers on identical features**: linear SVM, logistic regression, RBF SVM, and perhaps nearest-centroid. This separates “the features are good” from “the nonlinear decision boundary is good.”

The most report-worthy result would be something like: “MFCC plus spectral contrast explains most of the performance, while chroma and tonnetz contribute little,” or the opposite. Be careful not to interpret individual feature importance causally because correlated audio features can distribute importance arbitrarily.

### 2. From-scratch CRNN family

The strongest test of a data-volume bottleneck is a **learning curve with architecture held fixed**.

Use stratified song-level subsamples such as 10%, 25%, 50%, 75%, and 100% of the training songs, preserving album constraints. For every subset:

- keep the validation and test sets fixed;
- use the same number of training steps or report both fixed epochs and fixed optimizer updates;
- repeat each point with at least two or three seeds;
- plot top-1/F1 against the number of unique songs, not the number of randomly generated chunks.

Interpretation:

- If all architectures improve similarly as data increases, the bottleneck is probably data volume or label variability.
- If the curves saturate early and the models remain close, representation/input limitations may dominate.
- If FGNL or CRNNM gains only appear at larger data sizes, the extra capacity may need more supervision.
- If the largest model degrades at small sample sizes, this is evidence of variance/overfitting rather than architectural inferiority.

Your second experiment should be **cross-model error complementarity**. Generate one song-level prediction per model and compute:

- pairwise disagreement rate;
- Jaccard overlap of incorrect songs;
- double-fault measure: proportion of songs both models get wrong;
- oracle accuracy: whether at least one model is correct;
- majority-vote ensemble accuracy;
- pairwise correlation of confidence margins.

If the models all make the same errors, architecture is not addressing the dominant difficulty. If errors differ substantially but individual accuracy is similar, a simple calibrated ensemble may be valuable and the models are learning complementary cues.

A useful third analysis is **within-song temporal stability**: calculate prediction entropy over chunks, vocalness, and confidence. Published work shows that song-level voting can substantially exceed 5-second performance and that non-vocal segments are especially problematic. cite [arxiv](https://arxiv.org/pdf/2002.06817.pdf)

### 3. MERT versus ECAPA

Your proposed head-swap experiment is exactly the right first ablation. Use saved, frozen embeddings and compare:

- multinomial logistic regression or linear softmax;
- linear SVM;
- two-layer MLP;
- cosine prototype classifier;
- nearest-centroid classifier;
- optionally a supervised metric-learning head trained only on the Artist20 training set.

Keep preprocessing, chunking, pooling, early stopping, and evaluation identical. The key matrix is:

| Encoder | Linear | MLP | Cosine prototype |
|---|---:|---:|---:|
| MERT |  |  |  |
| ECAPA |  |  |  |

If ECAPA wins under every head, the gap is primarily representational. If the gap shrinks substantially with a better MERT head, the original comparison was partly head-dependent.

Silhouette score is a reasonable visualization statistic, but it should not be your principal diagnostic. It is sensitive to scaling, dimensionality, and the geometry induced by the metric. Prefer:

- **linear-probe accuracy**;
- class-centroid cosine separation;
- within-class versus between-class cosine distributions;
- Fisher discriminant ratio;
- \(k\)-NN accuracy in embedding space;
- linear CKA or RSA between embeddings;
- leave-one-album-out retrieval;
- artist-balanced confusion matrices.

Compute all metrics at both chunk and song level. For a song-level embedding, compare mean pooling, median pooling, attention-like confidence-weighted pooling, and majority vote over chunk predictions.

The most convincing analysis would be a **vocal/instrumental decomposition of embedding separability**:

1. compute ECAPA and MERT embeddings for the mixture;
2. compute them for vocals-only;
3. compute them for instrumental-only;
4. measure linear-probe accuracy and between/within-class distances.

If ECAPA’s advantage persists on vocals-only but collapses on instruments-only, that strongly supports a singer-identity explanation.

### 4. Vocal separation

The best analysis is a **paired per-song transition matrix**, not only a per-artist accuracy table.

For every test song, compare raw and vocals-only predictions:

| Raw | Vocals-only | Interpretation |
|---|---|---|
| Correct | Correct | Stable success |
| Wrong | Correct | Separation fixes the song |
| Correct | Wrong | Separation damages the song |
| Wrong | Wrong | Persistent error |

Report these counts overall and per artist. Also report the confidence change for each transition.

Then stratify songs by measurable acoustic properties:

- vocal-to-mixture energy ratio;
- estimated vocal activity fraction;
- instrumental density;
- spectral centroid or spectral flatness;
- separator artifact score;
- song duration and number of usable vocal chunks.

This can reveal whether the +2.2 percentage points is concentrated in dense mixes, low-vocal songs, or artists with particular production styles.

A third useful test is a **separation-quality dose-response analysis**. Rather than using only mixture and fully separated vocals, construct mixtures with different vocal gains or masks:

\[
x_\alpha = \alpha v + (1-\alpha)i
\]

for several values of \(\alpha\), or use soft separator masks. Evaluate the same trained model across the continuum. A smooth improvement suggests the model benefits from progressively increased vocal dominance; a non-monotonic curve suggests separator artifacts or loss of useful accompaniment cues.

The published results make a modest gain quite plausible: Hsieh et al. found that vocals-only training could be worse than original mixtures, while shuffle-and-remix augmentation produced the larger improvement. FGNL likewise reported that vocals-only could lower frame-level scores while improving song-level scores through confidence filtering and voting. cite [arxiv](https://arxiv.org/pdf/2002.06817.pdf)

### 5. Zero-shot Qwen2-Audio

The most worthwhile no-gradient improvement is to replace one closed-list prompt with a **structured audio evidence protocol**.

Try these conditions:

1. **One-vs-one or one-vs-few classification.** Ask the model whether the clip is artist A, B, or neither, then aggregate pairwise votes. This may reduce the difficulty of selecting among 20 names.
2. **Explicit output format.** Require exactly one artist name and forbid explanations, reducing parsing errors.
3. **Multiple temporal crops.** Query intro, verse, chorus, bridge, and outro separately, then majority-vote or confidence-weight the answers.
4. **Mixture versus vocals-only prompting.** Ask the model to identify the singer while explicitly prioritizing vocal timbre and ignoring instrumentation.
5. **Self-consistency.** Use several semantically equivalent prompts and aggregate normalized artist predictions.
6. **Candidate descriptions.** If the model has useful world knowledge, provide short neutral descriptions of each candidate; however, test this against a names-only prompt because descriptions can introduce textual priors.
7. **Reference-clip prompting.** Only attempt this if the exact Qwen2-Audio interface supports multiple audio turns or multimodal conversation context. Give one or more labeled reference clips and ask for nearest vocal identity, then test whether the model actually uses the audio rather than the labels.

The key control is to compare the model against:

- shuffled artist names;
- artist names replaced by arbitrary labels;
- audio-only prompts with no candidate list;
- multiple crops from the same song.

If performance changes dramatically when artist names are shuffled, the model may be relying on prior knowledge or textual associations rather than acoustic matching. Also report **coverage and abstention**, not only forced-choice accuracy: allow “uncertain/none of the above” on low-confidence clips if your interface supports it.

## Recommended report structure

For a strong course report, I would prioritize these experiments:

1. **ECAPA sanity audit:** held-out test, full-mixture/vocal-only/instrumental-only, linear versus MLP probe.
2. **CRNN learning curve:** fixed architecture, increasing numbers of training songs.
3. **Cross-model error complementarity:** determine whether the 57–67% models fail for the same reasons.
4. **Feature-group ablation:** establish why classical SVM remains competitive.
5. **Paired raw/vocal transition analysis:** explain the modest separation gain.
6. **Qwen crop-and-prompt ensemble:** a compact zero-shot follow-up.

The most important wording change for your results table is to label your current number as **“validation top-1/top-3 accuracy”**, not “Artist20 SOTA,” until you have run the fixed held-out test partition and converted the comparison to a common song-level protocol.