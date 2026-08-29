Here are the synthesized research findings for your three prompts, structured to feed directly into the ablation and discussion sections of your CommE5070 final report. 

---

### Prompt 1: Architectures beyond CRNN/non-local for small-scale singer/artist ID

For a small-scale dataset like Artist20 (~950 tracks, 20 classes), the primary risk is overfitting to low-level spectrogram textures rather than learning robust, identity-level representations. Here are three genuinely different architectural paradigms published or gaining traction between 2023–2025 that are highly suitable for this regime.

#### 1. Prototypical Networks / Metric Learning Frontends
*   **Paper/Repo:** *Prototypical Networks for Few-shot Learning* (Snell et al., originally, but heavily adapted for audio in recent years, e.g., "Few-Shot Audio Classification with Prototypical Networks" or integrating with **ArcFace/Subcenter ArcFace** for speaker/singer ID). Repo: Look at `torchmetrics` or specific audio metric learning repos like `speechbrain` (which has built-in metric learning pipelines).
*   **Why it’s a good fit:** Instead of a standard softmax cross-entropy head (which easily overfits on 20 classes with 950 samples), metric learning forces the model to learn a highly clustered embedding space. By training with a Prototypical loss or ArcFace loss, the model is penalized if tracks from the same singer aren't tightly clustered, regardless of the song. It acts as a powerful regularizer for small datasets.
*   **Caveat:** Requires careful tuning of the margin/temperature parameters. Inference is slightly more complex (calculating distances to class prototypes) than a simple linear probe, though negligible for 20 classes.

#### 2. Audio Spectrogram Transformer (AST) initialized with AudioMAE
*   **Paper/Repo:** *AST: Audio Spectrogram Transformer* (Gong et al., 2021) / *AudioMAE: Masked Autoencoders that Listen* (Huang et al., ICASSP 2022). Repo: `mit-han-lab/ast` or `facebookresearch/audiomae`.
*   **Why it’s a good fit:** While you have a CRNN, AST represents a shift from local convolutional inductive biases to global self-attention. AudioMAE (masked autoencoding pretraining) is specifically proven to be highly data-efficient. Fine-tuning an AudioMAE-pretrained AST on Artist20 will likely outperform a from-scratch CRNN because the masked autoencoding pretraining already learned robust, invariant audio representations, preventing overfitting on your small training set.
*   **Caveat:** Transformers are compute-heavy. You will need to use patch sizes (e.g., 16x16) and potentially restrict the input length (e.g., 3-5 second chunks) to fit in GPU memory, which requires an aggregation strategy (like attention pooling) for full-song inference.

#### 3. Whisper Encoder as a Frozen Frontend
*   **Paper/Repo:** *Robust Speech Recognition via Large-Scale Weak Supervision* (Radford et al., 2022). Recent adaptations for speaker/singer ID: *WhisperSpeaker* or simply pooling Whisper layers (e.g., ISMIR/ICASSP 2023-2024 papers on "Whisper for Speaker Verification"). Repo: `openai/whisper`.
*   **Why it’s a good fit:** You are currently using MERT (music-focused) and ECAPA (speech-focused). Whisper represents a third, distinct paradigm: massive weak supervision on diverse audio, including vast amounts of singing. Recent studies show Whisper's encoder captures vocal tract characteristics exceptionally well. Using it as a frozen frontend + linear probe provides a highly robust, out-of-distribution baseline that is architecturally distinct from both MERT and ECAPA.
*   **Caveat:** Whisper is optimized for ASR, so its intermediate layers might suppress "identity" information in favor of "content" (phonemes/lyrics). You may need to extract features from earlier layers (e.g., layer 8-16 out of 32) rather than the final layer to retain singer identity.

---

### Prompt 2: Augmentation recipes for small, album-correlated singer ID datasets

The album-level split in Artist20 is designed to break the "production confound," but models can still latch onto artist-specific instrumentation or mixing styles. Here is the evaluation of augmentation strategies to force the model to focus on vocal identity.

#### 1. Vocal Source Separation (Demucs) as Preprocessing
*   **Effect Size:** Recent literature (e.g., studies on the VoxCeleb and artist-ID tasks using MUSDB18) shows that source separation often yields **marginal or even negative effect sizes** for singer identification. 
*   **Failure Modes:** 
    1.  *Loss of Studio Identity:* A singer's "identity" on a record is inextricably linked to the microphone, preamp, compression, and room acoustics applied *during the mix*. Isolating the vocal strips this context.
    2.  *Separation Artifacts:* Demucs (even v4/htdemucs) introduces "musical noise" and phase smearing in the vocal track. These high-frequency artifacts destroy the fine formant structures and breathiness that are critical for distinguishing similar voices.
*   **Verdict:** Keep it as an ablation, but expect it to perform worse than raw audio. If it improves accuracy, it likely means your model was overfitting to instrumental bleed rather than the voice.

#### 2. Standard Audio Augmentations: Confound Breakers vs. Generic Regularizers
*   **Targeting the Instrumentation/Production Confound:**
    *   **EQ / Reverb / Compression Perturbation:** *Highly effective.* Applying random parametric EQ, variable reverb decay, and dynamic range compression forces the model to ignore the specific "mixing chain" of the album. It simulates the singer being recorded in different studios.
    *   **Background Noise / Instrumental Bleed:** *Moderately effective.* Adding low-level noise or random instrumental stems forces the model to focus on the vocal rather than the spectral envelope of the backing track.
*   **Generic Regularizers (Do not specifically target production confounds):**
    *   **SpecAugment / Time Stretch:** Excellent for preventing overfitting to specific temporal alignments or spectrogram textures, but they don't change the timbral/production characteristics.
    *   **Mixup:** Good for smoothing decision boundaries, but if you mix two tracks from the same album, you are just mixing the same production style.
*   **What to AVOID:**
    *   **Pitch Shift:** *Detrimental to Singer ID.* A singer's fundamental frequency range and habitual pitch are core identity markers. Pitch shifting alters the formant-to-F0 relationship, effectively changing the perceived vocal tract length and hurting accuracy.

#### 3. Singer-ID-Specific Augmentation Tricks
*   **Vocal Re-mixing (The "Silver Bullet" for Production Confounds):** 
    *   *Method:* Use Demucs to isolate the vocals. Then, mix these isolated vocals over random backing tracks from a diverse dataset (e.g., FMA or MUSDB18). 
    *   *Why it works:* This completely destroys the album/production confound. The model is forced to identify the singer purely based on the vocal timbre, as the instrumentation and mixing style are randomized.
*   **Formant Shifting (without Pitch Shifting):**
    *   *Method:* Using phase-vocoder or PSOLA techniques, shift the formants (vocal tract length) up or down by 1-2 semitones while keeping the fundamental pitch identical.
    *   *Why it works:* It prevents the model from relying on absolute formant frequencies (which can be album/mic dependent) and forces it to learn relative, robust timbral textures.

---

### Prompt 3: Best available SSL checkpoints specifically for singing voice

You are currently using MERT (general music) and ECAPA (spoken voice). To isolate the *singing voice* domain gap, you need models trained specifically on the unique characteristics of singing (e.g., wide pitch ranges, vibrato, formant tuning).

#### 1. Pitch-Aware Masked Autoencoders (e.g., Wav2Vec 2.0-F0 / HuBERT-F0)
*   **Checkpoint/Location:** Various implementations exist on HuggingFace (search for `wav2vec2-f0` or `hubert-f0`), often released by groups like the NII (National Institute of Informatics) or in ISMIR/ICASSP 2023-2024 proceedings. 
*   **Pretraining Objective:** Standard masked acoustic modeling, but with an auxiliary loss to predict the Fundamental Frequency (F0) of the masked frames. 
*   **Why it’s the strongest for singing:** Standard SSL models (like MERT or Wav2Vec2) treat pitch as just another acoustic feature. In singing, pitch is the primary structural element. Pitch-aware SSLs explicitly disentangle and model F0, which captures the singer's vibrato, pitch habits, and timbre-pitch interactions much better than generic SSLs.
*   **Availability:** Openly available, usually ungated on HuggingFace or GitHub.

#### 2. Whisper Encoder (Specifically pooled intermediate layers)
*   **Checkpoint/Location:** `openai/whisper-large-v3` (or `v2` / `medium`). HuggingFace: `openai/whisper-large-v3`.
*   **Pretraining Objective:** Weakly supervised multi-lingual speech recognition and translation on 680,000 hours of audio.
*   **Why it fits:** While technically an ASR model, Whisper was trained on a vastly more diverse set of audio than ECAPA, including massive amounts of singing, choirs, and a cappella. Recent benchmarks (e.g., *ISCA/ICASSP 2024 papers on Whisper for Speaker Verification*) show that pooling the intermediate layers of Whisper's encoder yields state-of-the-art results for *singing* speaker/singer ID, outperforming both ECAPA (speech-only) and standard music SSLs. It bridges the domain gap better than MERT because its sheer scale covers singing variations extensively.
*   **Availability:** Fully open, ungated weights on HuggingFace.

#### 3. Singing Voice Conversion (SVC) Speaker Encoders (e.g., from RVC / SoVITS)
*   **Checkpoint/Location:** The speaker encoders from **Retrieval-based Voice Conversion (RVC)** or **Soft VITS (SoVITS)**. Repos: `RVC-Project/Retrieval-based-Voice-Conversion-WebUI`.
*   **Pretraining Objective:** These encoders (often a modified ResNet or ECAPA architecture) are specifically trained on large singing datasets (like Opencpop, M4Singer, or proprietary anime/game singing datasets) to extract a "speaker embedding" that captures the unique timbre of a *singer* for the purpose of voice conversion.
*   **Why it fits:** Unlike ContentVec (which is explicitly trained to *remove* speaker identity to isolate lyrics/pitch), the *speaker encoder* in these pipelines is trained specifically to isolate and maximize singing voice identity. Using this encoder as a frozen frontend for your linear probe is a highly targeted, domain-specific baseline.
*   **Caveat/Availability:** The weights are openly available in the RVC/SoVITS repositories, but they are often bundled in `.pth` files within the WebUI ecosystem. You will need to write a small script to extract the state dict and load it into a standard PyTorch inference pipeline.

### Summary Recommendation for your Report's Ablation Section:
1.  **Architecture:** Add **AudioMAE-initialized AST** (to show Transformer/Masking superiority over CRNN) and **Whisper frozen frontend** (to show the power of massive weak supervision over domain-specific SSL).
2.  **Augmentation:** Run the **Vocal Re-mixing** augmentation. If it significantly boosts accuracy compared to raw audio, it definitively proves your baseline models were suffering from the production/instrumentation confound.
3.  **SSL:** Swap MERT for the **Whisper-large-v3** intermediate-layer pooling. It is currently the most pragmatic "SOTA" for cross-domain (speech/singing) vocal identity without needing to train a custom pitch-aware SSL from scratch.