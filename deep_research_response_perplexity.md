## Prompt 1 — architectures worth adding

The strongest near-term addition is the **Torres–Lattner–Richard singer-identity encoder**, rather than another CRNN. It is explicitly trained for singer identity, has released checkpoints and code, and is designed to make embeddings invariant to pitch and lyrical content. The project provides BYOL, contrastive, VICReg, uniformity, and related models through Hugging Face; its default architecture is an EfficientNet-B0 with temporal average pooling, making it substantially lighter than large music transformers. [github](https://github.com/SonyCSLParis/ssl-singer-identity)

| Candidate | Modeling idea | Fit for Artist20 | Caveat |
|---|---|---|---|
| **Singer Identity Representation Learning / CVSM** | Self-supervised vocal-identity learning using BYOL, contrastive learning, VICReg, and uniformity-alignment objectives. | Best direct architectural addition. Frozen embedding plus linear probe is easy to compare against MERT and ECAPA; alternatively fine-tune the projection/classifier. Published results report singer-identification accuracy up to 81.01% on VocalSet/M4Singer-style evaluations, and contrastive models achieved 2.16% EER on singer similarity.  [arxiv](https://arxiv.org/html/2401.05064v1) | Trained primarily on isolated vocals at 44.1 kHz. Artist20 is mixed music at 16 kHz, so separation quality and resampling may materially affect results. It is not specifically benchmarked on album-split Artist20. |
| **Yakura et al. singing-voice contrastive learner** | Contrastive learning treats pitch-shifted/time-stretched versions as positive or negative examples to emphasize singer timbre and expression. | A genuinely different identity-learning objective. The reported top-1 accuracy was 63.08% across 500 singers, with the proposed transformations improving accuracy by 9.12%.  [dl.acm](https://dl.acm.org/doi/pdf/10.1109/TASLP.2022.3169627) | Older than your 2023–2025 target window and likely requires reproduction or locating the authors’ implementation. Dataset/task conditions may not match Artist20. |
| **MAEST** | Music Audio Efficient Spectrogram Transformer, pretrained for large-scale music-style tagging and usable as an embedding extractor. | Good lightweight-transformer control against MERT: it represents music-level semantics rather than recurrent temporal pooling. Public code and models are available, and the authors report strong open-model performance on music tagging.  [arxiv](https://arxiv.org/pdf/2309.16418.pdf) | Trained on approximately 3.3 million tracks for style tagging, not identity. It may encode genre, production, and album cues—the exact confounds of concern—so interpret it as a strong music-SSL baseline, not a singer-specific model. |
| **HTS-AT** | Hierarchical Swin-style spectrogram transformer with token-semantic attention. | A practical transformer architecture with roughly 30M parameters and public checkpoints. It is small enough to fine-tune or use as a frozen frontend in a few days.  [github](https://github.com/RetroCirce/HTS-Audio-Transformer) | General sound classification pretraining rather than music or singer identity. Its AudioSet bias may make it sensitive to instrumentation. |
| **Modern general audio SSL encoders** such as BEATs, AudioMAE, or AST | Masked acoustic modeling or spectrogram-transformer pretraining. | Useful as controls if you want to test whether gains come from transformer pretraining rather than singing-specific supervision. | They are not necessarily better than MERT for music, and their reported large-scale results do not establish small-sample Artist20 performance. Include only one to avoid an unfocused ablation. |
| **Metric-learning head over MERT/MAEST** | Prototypical, supervised-contrastive, ArcFace, or triplet objective instead of ordinary cross-entropy. | Requires little engineering and tests whether identity-aware supervision helps a frozen music encoder. Use song/segment positives from the same artist and album-disjoint validation. | This is a training-objective ablation rather than a new encoder. Avoid positives from the same album if the objective could simply memorize production style. |

### Recommended architecture subset

For a compact but defensible ablation, add:

1. **Singer Identity Representation Learning–BYOL or contrastive checkpoint**.
2. **MAEST frozen embeddings** as a music-transformer comparison.
3. **MERT plus supervised-contrastive or ArcFace head**.
4. Optionally **HTS-AT** if compute and implementation time permit.

The 2023 singing-voice SSL comparison is useful for selecting the frontend: it evaluates SSL models on singer identification, singing transcription, and singing-technique classification, and reports that SSL frontends can match or outperform conventional methods under limited labeled data. [arxiv](https://arxiv.org/abs/2306.12714)

## Prompt 2 — augmentation and regularization

The most directly relevant evidence is the Artist20 source-separation and remixing study. On the album-level split, vocal-only training actually reduced song-level F1 for the baseline CRNN from 0.67 to 0.61, while combining original, vocal-only, and cross-song remix data increased it to 0.74. The melody-enhanced model reached 0.75 with the combined augmentation. [arxiv](https://arxiv.org/pdf/2002.06817.pdf)

This is important: **vocal separation alone is not reliably beneficial**. Separation artifacts, missing vocal frames, backing vocals, and loss of useful natural vocal–instrument interaction can outweigh the benefit of removing accompaniment.

| Method | What it attacks | Recommendation |
|---|---|---|
| Vocal-only input | Directly suppresses instrumentation and production cues. | Keep as an ablation, but do not assume improvement. Report separation model, vocal activity, and artifact handling. |
| Cross-song vocal/instrumental remix | Breaks the statistical association between singer and backing track while preserving the singer label. | Highest-priority augmentation. Mix a singer’s separated vocal with another song’s accompaniment, retaining the vocal label. Include original and vocal-only examples rather than replacing the original data. |
| Pitch shift | Encourages invariance to song key and pitch range. | Use conservatively, for example ±1–2 semitones. Large shifts can alter timbre and create artifacts, especially for singing. |
| Time stretch | Reduces dependence on tempo and phrasing. | Use small factors, such as 0. nueve–1.1, and avoid aggressive stretching that changes vocal quality. |
| EQ / production perturbation | Directly attacks microphone, mastering, instrumentation balance, and production signatures. | Particularly appropriate here. Random low/high-shelf EQ, gain changes, mild compression, reverb, stereo narrowing, and simulated band-limiting are more targeted than generic noise. |
| Instrumental replacement | Removes singer–genre and singer–album correlations. | Strongest conceptual intervention when separation is usable. Match rough loudness and optionally key/tempo to avoid obviously unnatural mixtures. |
| Mixup | Generic interpolation regularizer. | Useful as a control, but ordinary waveform mixup can make labels ambiguous. Prefer feature-space mixup or mixup between examples of the same singer and carefully chosen different singers. |
| SpecAugment | Masks time/frequency regions in the spectrogram. | Good generic regularization, but it does not specifically remove production confounds. Frequency masking may remove vocal harmonics, so use moderate masks. |
| Noise injection | Improves robustness to noise and compression. | Generic regularization; less relevant to clean album confounds unless the deployment condition includes noisy audio. |
| Vocal-activity weighting | Prevents non-vocal sections from dominating the prediction. | Highly relevant. Use a vocal detector or separation-energy threshold, then pool only vocal-active frames or weight them more heavily. The Artist20 study observed that segment-level predictions were substantially weaker on non-vocal portions.  [arxiv](https://arxiv.org/pdf/2002.06817.pdf) |

### Practical recipe

For each training song:

- Generate the original mixture.
- Generate the separated-vocal version.
- With moderate probability, remix that vocal with an accompaniment from another training song, preferably from a different artist and album.
- Apply small pitch and time perturbations to the vocal branch.
- Apply independent EQ, loudness, compression, and reverb perturbations to the accompaniment.
- Train with vocal-aware frame pooling and song-level aggregation.
- Keep all transformations strictly within the training split.

The original Artist20 study used open-unmix and tested three data conditions—original, vocal-only, and shuffled remix. Its key result was that the **combined** data condition worked better than vocal-only or remix-only, suggesting that separation should be treated as an additional view rather than a universal replacement for the original mixture. [arxiv](https://arxiv.org/pdf/2002.06817.pdf)

### Failure modes to measure

Your report should separately quantify:

- Separation artifacts, especially musical noise and vocal formant distortion.
- Incorrect removal of breath sounds, harmonies, or vocal doubles.
- Non-vocal sections being classified from accompaniment.
- Distribution mismatch between separated training audio and unseparated test audio.
- Leakage caused by remixing tracks across album boundaries.

A useful diagnostic is to evaluate four matched conditions: original test audio, separated test audio, original-trained model, and remix-trained model. This distinguishes “the model learned identity better” from “the test preprocessing happened to be easier.”

## Prompt 3 — singing-specific checkpoints

### Strongest direct candidate

**Singer Identity Representation Learning / CVSM**

- **Repository:** `SonyCSLParis/ssl-singer-identity`.
- **Checkpoint access:** Hugging Face models loaded through the repository’s `load_model()` interface.
- **Objectives:** BYOL, decoupled contrastive learning, VICReg, uniformity-alignment, and related self-supervised objectives.
- **Architecture:** EfficientNet-B0 frontend with temporal average pooling; embeddings are 1,000-dimensional.
- **Training domain:** Large collections of isolated singing/vocal tracks, using augmentations intended to remove pitch and content dependence.
- **Reported results:** Contrastive models achieved 2.16% EER for singer similarity; BYOL and contrastive models reached 81.01% and 77.42% singer-identification accuracy in the reported experiments, while XLSR-53 remained the strongest overall baseline in those evaluations. [arxiv](https://arxiv.org/html/2401.05064v1)
- **Availability:** The repository states that pretrained models can be downloaded through Hugging Face and is MIT licensed. [github](https://github.com/SonyCSLParis/ssl-singer-identity)
- **Best Artist20 use:** Run the checkpoints on Demucs vocals at their native 44.1 kHz, then compare frozen embeddings, linear probe, and light fine-tuning. Also test the model on original mixtures to measure sensitivity to accompaniment.

### Other relevant models

| Model | Relevance to singing | Artist20 value | Availability/caveat |
|---|---|---|---|
| **Yakura et al. self-supervised singing representation** | Explicitly learns singer representations using pitch-shift and time-stretch transformations. | Strong methodological precedent for identity-invariant augmentation; reported 63.08% top-1 over 500 singers.  [dl.acm](https://dl.acm.org/doi/pdf/10.1109/TASLP.2022.3169627) | Older and less turnkey than the Torres et al. release; verify current checkpoint availability before promising it in the report. |
| **XLSR-53 / wav2vec 2.0 family** | Speech SSL, not singing-specific, but evaluated in singing-voice representation work. | Important control because it can outperform singing-specific models on some singer-identification evaluations.  [arxiv](https://arxiv.org/html/2401.05064v1) | Speech-domain bias and usually 16 kHz operation. It may encode phonetic or recording cues, so interpret carefully. |
| **MERT** | General music SSL; your current baseline. | Appropriate music-domain control, especially for mixed tracks. | Not singer-specific; performance may reflect genre and production. |
| **MAEST** | Large-scale music representation model for tagging and embedding extraction. | Stronger modern music-transformer control against MERT. | Trained for music style tagging on about 3.3M tracks, not singer identity. Public models and code are available.  [arxiv](https://arxiv.org/pdf/2309.16418.pdf) |
| **HTS-AT** | Efficient pretrained audio transformer. | Lightweight transformer architecture that can be frozen or fine-tuned. | General AudioSet/sound-classification pretraining, not singing-specific; public checkpoints are reported by the project.  [github](https://github.com/RetroCirce/HTS-Audio-Transformer) |
| **SVC semantic encoders** such as ContentVec, HuBERT, and related systems | Often used to represent singing content, lyrics, pitch, or speaker-independent information for singing voice conversion. | Potentially useful diagnostic embeddings, especially for separating content from identity. | Their intended representation is frequently speaker/singer-independent, so they should not automatically be expected to perform well for singer ID. The SVC literature emphasizes that these encoders are evaluated for content, melody, and speaker-independence rather than identity classification.  [arxiv](https://arxiv.org/abs/2310.11160) |

### Recommended experiment matrix

A clean, feasible matrix is:

| Frontend | Input | Head |
|---|---|---|
| MERT | Original mixture | Linear probe |
| MERT | Demucs vocals | Linear probe |
| Singer-ID BYOL | Demucs vocals | Linear probe |
| Singer-ID contrastive | Demucs vocals | Linear probe |
| MAEST | Original mixture | Linear probe |
| MERT | Original + vocal/remix views | Supervised-contrastive or ArcFace head |

Use identical song-level pooling, training splits, and classifier capacity. This avoids attributing gains to unequal temporal aggregation or parameter counts.

## Bottom line

For the report, the most defensible claims are:

- **Best new architecture:** the openly released singer-identity SSL encoder from Torres, Lattner, and Richard.
- **Best anti-confound augmentation:** separation followed by cross-song vocal–accompaniment remixing, retaining original and vocal-only views.
- **Most important negative result:** vocal-only preprocessing can hurt; separation is not automatically a singer-identity improvement.
- **Best transformer control:** MAEST, with HTS-AT as a smaller general-audio alternative.
- **Strongest checkpoint caveat:** singing-specific encoders are usually trained on isolated vocals at 44.1 kHz, whereas Artist20 contains mixed 16-kHz music under an album-level split. Their transfer should therefore be presented as an empirical test, not assumed superiority.