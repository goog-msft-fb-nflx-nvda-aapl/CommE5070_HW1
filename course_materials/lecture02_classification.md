# Deep Learning for Music Analysis and Generation - Music Classification

## Table of Contents
* [Introduction and Reference Materials](#introduction-and-reference-materials)
* [Outline](#outline)
* [Music Classification: Basics and Tasks](#music-classification-basics-and-tasks)
* [ML-Based Music Classification and Hand-Crafted Audio Features](#ml-based-music-classification-and-hand-crafted-audio-features)
* [Audio Libraries and Demonstrations](#audio-libraries-and-demonstrations)
* [Advanced Feature Extraction and Analysis](#advanced-feature-extraction-and-analysis)
* [DL-Based Music Classification](#dl-based-music-classification)
* [Audio Data Augmentation](#audio-data-augmentation)
* [Exemplar DL Models](#exemplar-dl-models)
* [Evaluation Metrics and Training Tricks](#evaluation-metrics-and-training-tricks)
* [Music Foundation Models and Datasets](#music-foundation-models-and-datasets)

---

## Introduction and Reference Materials

### Slide 1
Music Classification (audio → labels)
Deep Learning for Music Analysis and Generation

### Slide 2
Music AI; or Music Information Research (MIR)
1. Music analysis
   * audio (exisiting song) -> Music semantic lableing -> labels
     * audio -> genre (classical)
     * audio -> emtion (yearning)
     * audio -> other atributes (slow/fast)
   * audio (exisiting song) -> Music transcription (audio2score) -> score
     * audio -> note (pitch, onset, offset)
     * audio -> instrument (flute, cello)
     * audio -> meter (4/4)
     * audio -> key (E-flat major)
   * music understanding
   * music search
   * music recommendation

2. Music generation
   * ( random seed + labels ) -> AI composer -> score
   * ( score + labels ) -> AI performer (score2audio) -> audio
   * MIDI generation
   * audio generation
   * MIDI-to-audio generation

### Slide 3
Reference 1: KAIST Course & ISMIR 2021 Tutorial
For fundamentals of deep learning and music classification
* https://mac.kaist.ac.kr/~juhan/gct634/Slides/05.%20music%20classification%20-%20deep%20learning.pdf
* https://music-classification.github.io/tutorial/landing-page.html

### Slide 4
Reference 2
* Deep Learning An MIT Press book (2016)
  by Ian Goodfellow and Yoshua Bengio and Aaron Courville
  https://www.deeplearningbook.org/
* Deep Learning - Foundations and Concepts(2024)
  by Christopher M. Bishop & Hugh Bishop
  https://link.springer.com/book/10.1007/978-3-031-45468-4
* Deep Learning 101 for Audio-based MIR (2024)
  by Geoffroy Peeters, Gabriel Meseguer-Brocal, Alain Riou & Stefan Lattner
  https://geoffroypeeters.github.io/deeplearning-101-audiomir_book/task_musicprocessing.html

---

## Outline

### Slide 5
Outline
* Music classification: Basics
* ML-based music classification (and hand-crafted audio features)
* DL-based music classification
* Music foundation models

---

## Music Classification: Basics and Tasks

### Slide 6
Different Classification Tasks
* Single-label vs multi-label
* Song-level vs instance-level
  * instance/chunk/clip/segment (various names)

https://music-classification.github.io/tutorial/part1_intro/what-is-music-classification.html

### Slide 7
Single-label vs Multi-label Classification
* Single-label classification
  * One-hot; one out of many (mutually exclusive classes)
  * Can be binary or multi-class classification
  * Activation function (in DL): softmax (sum-to-one)
  * Output interpretation: argmax
  * Loss function (in DL): categorical cross entropy (CE)

* Multi-label classification 
  * Multi-hot; some out of many (assigning an input to multiple classes simultaneously)
  * Activation function (in DL): sigmoid (each in [0,1], not sum-to-one)
  * Output interpretation: $\ge 0.5$ (or other thresholds)
  * Loss function in DL: binary cross entropy (BCE)

https://www.singlestore.com/blog/a-guide-to-softmax-activation-function/

### Slide 8
Song-label vs Instance-label Classification
* Song-level: Make a prediction for the entire song (or a long audio clips), without specifying the temporal location/activation of each class
* Instance-level: Need to mark the temporal location/activation of each class
* It’s related to the length of the model input
  * Make a prediction per STFT frame
  * Make a prediction every second
  * Make a prediction for a 30-second spectrogram
  * Make a prediction per musical note (variable-length)
  * Make a prediction per musical section (verse, chorus, etc) (variable-length)

### Slide 9
Different Classification Tasks
* The most explored music classification tasks in MIR
  * Genre classification [TC02]
  * Mood classification [KSM+10]
  * Instrument identificatoin [HBPD03]
  * Music tagging [Lam08]
* Many others
  * singer/composer classification
  * technique classification
  * audio event detection

https://music-classification.github.io/tutorial/part1_intro/what-is-music-classification.html

### Slide 10
Genre/Style Classification
* A conventional category that identifies some pieces of music as belonging to a shared tradition or set of conventions
* Evolve over time

### Slide 11
Emotion/Mood Classification/Regression
* Perceived vs. felt emotion
* Song-level or instance-level 
  * music emotion variation detection
* Classification vs. regression
  * arousal: energy or neuro-physiological stimulation level
  * valence: pleasantness or positive/negative affective states
  * popular taxonomy: 4Qs of the valence/arousal plane
* Inherently subjective 

Geneva Emotional Music Scale (GEMS) https://musemap.org/resources/gems
https://github.com/juansgomez87/datasets_emotion

### Slide 12
Instrument Classification/Detection
* Song-level or instance-level
  * singing activity detection
  * instrument activity detection
* Hierarchical taxonomy

> Ref: Krause et al., “Hierarchical classification of singing activity, gender, and type in complex music recordings,” ICASSP 2022
> Ref: Krause et al., “Hierarchical classification for instrument activity detection in orchestral music recordings,” TASLP 2023

### Slide 13
Music Tagging

MagnaTagATune (https://mirg.city.ac.uk/codeapps/the-magnatagatune-dataset)
> Ref: Law et al., “Evaluation of algorithms using games: the case of music annotation,” ISMIR 2009

* Top 50 by categories (source)
  * genre: classical, techno, electronic, rock, indian, opera, pop, classic, new age, dance, country, metal
  * instrument: guitar, strings, drums, piano, violin, vocal, synth, female, male, singing, vocals, no vocals, harpsichord, flute, no vocal, sitar, man, choir, voice, male voice, female vocal, harp, cello, female voice, choral
  * mood: slow, fast, ambient, loud, quiet, soft, weird
  * etc: beat, solo, beats

### Slide 14
Technique Classification
* Electric guitar
  * bend, vibrato, hammer-on, pull-off, slide
  * https://zenodo.org/record/1414806
* Singing voice
  * breathy, vibrato, vocal fry, etc
  * https://zenodo.org/record/1193957

### Slide 15
Technique Classification: Vibrato and Tremolo
* Tremolo: periodic variations in amplitude (amplitude modulations)
* Vibrato: periodic variations in frequency (frequency modulations)
  * Wind and bowed instruments generally use vibratos with an extent of less than half a semitone either side
  * Tremolo and vibrato do not necessarily evoke a perceived change in loudness or pitch of the tone

https://en.wikipedia.org/wiki/Vibrato
> Ref: Sundberg, “Acoustic and psychoacoustic aspects of vocal vibrato,” 1994

### Slide 16
Audio Event Detection

Audio Set (http://research.google.com/audioset/)
> Ref: Gemmeke et al., “Audio Set: An ontology and human-labeled dataset for audio events,” ICASSP 2017

* 527 audio classes
  * Over 2M audio clips from YouTube
  * Each 10 second
* Widely used benchmark for audio classification and audio captioning
* Can be useful for sound design 

### Slide 17
Audio Event Detection

https://github.com/qiuqiangkong/audioset_tagging_cnn
* Useful for 
  * Music/singing/speech detection; instrument activity detection

### Slide 18
Different Classification Tasks

| Task | Single-label? | Multi-label? | Song-level? | Instance-level? |
| --- | --- | --- | --- | --- |
| Genre analysis | | | | |
| Emotion analysis | | | | |
| Instrument analysis | | | | |
| Technical analysis | | | | |
| Audio event detection | | | | |

* It depends on your dataset and your problem formulation
* Single-label, song-level classification is the simplest setting (good for beginners)
* We can do instance level first, then aggregate the result into the song level 

---

## ML-Based Music Classification and Hand-Crafted Audio Features

### Slide 19
Outline
* Music classification: Basics
* ML-based music classification (and hand-crafted audio features)
* DL-based music classification
* Music foundation models

### Slide 20
ISMIR 2024 Test-of-Time (ToT) Awardee

### Slide 21
Musical Genre Classification of Audio Signals 
* “A musical genre is characterized by the common characteristics shared by its members. These characteristics typically are related to the instrumentation, rhythmic structure, and harmonic content of the music.”
* “In this paper, […] three feature sets for representing timbral texture, rhythmic content and pitch content are proposed.”
* “The performance and relative importance of the proposed features is investigated by training statistical pattern recognition classifiers using real-world audio collections. […] Using the proposed feature sets, classification of 61% for ten musical genres is achieved.”

https://www.cs.cmu.edu/~gtzan/work/pubs/tsap02gtzan.pdf

### Slide 22
The GTZAN Dataset
* “The dataset consists of 1000 audio tracks each 30 seconds long. It contains 10 genres, each represented by 100 tracks. The tracks are all 22,050Hz Mono 16-bit audio files in WAV format”

https://www.tensorflow.org/datasets/catalog/gtzan
* The genres are:
  * blues
  * classical
  * country
  * disco
  * hiphop
  * jazz
  * metal
  * pop
  * reggae
  * rock

### Slide 23
Representing the Audio as Features is Needed in ML/DL
* Given: $\{ (x_1,y_1) ,\dots, (x_N,y_N) \}$
  * $x_i \in \mathbb{R}^M$: feature representation; a “vector”
  * $y_i \in \{-1,+1\}$: class label

* training data ---> Feature extraction ---> feature
* training data ---> Manual annotation ---> ground truth
* feature, ground truth ---> Model training ---> model
* test data ---> Feature extraction ---> feature_test
* feature_test, model ---> Automatic Prediction ---> estimate

### Slide 24
The Three Feature Sets Used By GTZAN
* They are basically statistics
* Timbral texture features: Statistics from the waveform and magnitude spectrogram
  * Spectral centroid
  * Spectral rolloff
  * Spectral flux
  * Time domain zero crossings
  * MFCCs
  * Low-energy feature
* Rhythmic content features: Statistics from the result of a “beat estimator”
* Pitch content features: Statistics from the result of a “multi-pitch estimator”

### Slide 25
The Rhythmic Content Features Used By GTZAN
Fig. 2 shows a beat histogram for a 30-s excerpt of the song “Come Together” by the Beatles. The two main peaks of the BH correspond to the main beat at approximately 80 bpm and its first harmonic (twice the speed) at 160 bpm. Fig. 3 shows four beat histograms of pieces from different musical genres. The upper left corner, labeled classical, is the BH of an excerpt from “La Mer” by Claude Debussy. Because of the complexity of the multiple instruments of the orchestra there is no strong self-similarity and there is no clear dominant peak in the histogram. More strong peaks can be seen at the lower left corner, labeled jazz, which is an excerpt from a live performance by Dee Dee Bridgewater. The two peaks correspond to the beat of the song (70 and 140 bpm). The BH of Fig. 2 is shown on the upper right corner where the peaks are more pronounced because of the stronger beat of rock music

### Slide 26
The Rhythmic Content Features Used By GTZAN
* Rhythmic content features: Statistics from a “beat histogram”
  * A0, A1: relative amplitude (divided by the sum of amplitudes) of the first, and second histogram peak
  * RA: ratio of the amplitude of the second peak divided by the amplitude of the first peak
  * P1, P2: period of the first, second peak in bpm
  * SUM: overall sum of the histogram (indication of beat strength)

### Slide 27
The Timbral Texture Features Used By GTZAN
* Question: how to compute features/statistics from the spectrogram (or waveform)?
  * The number of features cannot be too large 
  * The features have to be somehow “meaningful”

### Slide 28
Spectral Centroid
* Each frame of a magnitude spectrogram is normalized and treated as a distribution over frequency bins, from which the mean (centroid) is extracted per frame
* A measure of spectral shape 
* Higher centroid values imply “brighter” textures with more high frequencies
* Other statistics can also be used – bandwidth, skewness, kurtosis

> Ref: Tzanetakis and Cook, “Musical genre classification of audio signals,” TASLP 2002

### Slide 29
Spectral Rolloff
* The frequency for a spectrogram bin such that at least roll_percent (0.85 by default) of the energy of the spectrum in a frame is contained in this bin and the bins below
* Can be used to approximate the maximum (or minimum) frequency by setting roll_percent to a value close to 1 (or 0)
* Another measure of spectral shape

> Ref: McFee et al., “librosa: Audio and music signal analysis in python,” 2015

### Slide 30
Spectral Contrast
* Each frame of a spectrogram is divided into multiple sub-bands
* Compute the mean energy for each sub-band
* Compare the mean energy in the top quantile (peak energy) to that of the bottom quantile (valley energy)
  * High contrast values generally correspond to clear, narrow-band signals, while low contrast values correspond to broad-band noise
* Alternatively: entropy of the sub-band mean energy

> Ref: McFee et al., “librosa: Audio and music signal analysis in python,” 2015

### Slide 31
Spectral Flux
* How quickly the power spectrum of a signal is changing over time
* Usually calculated as the L2-difference between two adjacent normalized spectra
* BTW, this feature is also often used for musical onset detection (more related to rhythm)

> Ref 1: https://en.wikipedia.org/wiki/Spectral_flux
> Ref 2: https://librosa.org/librosa_gallery/auto_examples/plot_superflux.html

### Slide 32
Mel-Spectrogram
* The Mel scale is a perceptual scale of pitches judged by listeners to be equal in distance from one another
* Finer resolution in the low-frequency range (NOT exactly logarithmic scale)
* Dimension reduction
  * linear scale: hundreds of frequency bins
  * mel scale: tens of frequency bands

### Slide 33
Mel-Spectrogram
https://music-classification.github.io/tutorial/part2_basics/input-representations.html

### Slide 34
Mel-frequency cepstral coefficients (MFCC)
* Procedure
  1. Compute the spectrogram
  2. Grouping the FFT bins according to the perceptually motivated Mel-filter bank 
     * Linear till 1,000 Hz and logarithmic above it
     * Usually with triangular overlapping windows
  3. Taking logs and DCT for uncorrelating the resulting features
* Compact representation of the spectrum (1,024-dim → 128 → 13 coefficients)
  * Somehow capture the energy distribution in the spectrum
  * Less interpretable

> Ref: McFee et al., “librosa: Audio and music signal analysis in python,” 2015

### Slide 35
The GTZAN Classifier
* Each 30-sec audio is represented as a single vector (that is composed of the timbre, rhythmic and pitch features)
* Then train a classifier using Gaussian mixture model (GMM) or K-nearest neighbor (K-NN) 
* Confusion matrix
  * Columns: GT  /  rows: predictions
  * “Classical music is misclassified as jazz music for pieces with strong rhythm from composers like Leonard Bernstein and George Gershwin.” 
  * “Rock music has the worst classification accuracy and is easily confused with other genres which is expected because of its broad nature.”

### Slide 36
BTW, the Mel-Spectrogram is Actually Quite Important…
* Usually treated as the “default” feature representation for musical audio, especially in the DL era
  * Easy to compute and understand
  * Reasonably rich information
  * Reasonable size
  * Can be used as input to computer vision (CV) models
  * Possible to go back from mel-spectrograms to waveforms via a “vocoder”
* Used in all types of tasks, for both music analysis and generation

---

## Audio Libraries and Demonstrations

### Slide 37
Library: Torchaudio
https://pytorch.org/audio/0.11.0/tutorials/audio_feature_extractions_tutorial.html

### Slide 38
Library: LibROSA
https://librosa.org/doc/latest/index.html
https://colab.research.google.com/github/stevetjoa/musicinformationretrieval.com/blob/gh-pages/ipython_audio.ipynb

### Slide 39
Library 1: LibROSA
Spectral features

| Function | Description |
| --- | --- |
| `chroma_stft(*[, y, sr, S, norm, n_fft, ...])` | Compute a chromagram from a waveform or power spectrogram. |
| `chroma_cqt(*[, y, sr, C, hop_length, fmin, ...])` | Constant-Q chromagram |
| `chroma_cens(*[, y, sr, C, hop_length, fmin, ...])` | Compute the chroma variant "Chroma Energy Normalized" (CENS) |
| `chroma_vqt(*[, y, sr, V, hop_length, fmin, ...])` | Variable-Q chromagram |
| `melspectrogram(*[, y, sr, S, n_fft, ...])` | Compute a mel-scaled spectrogram. |
| `mfcc(*[, y, sr, S, n_mfcc, dct_type, norm, ...])` | Mel-frequency cepstral coefficients (MFCCs) |
| `rms(*[, y, S, frame_length, hop_length, ...])` | Compute root-mean-square (RMS) value for each frame, either from the audio samples y or from a spectrogram S. |
| `spectral_centroid(*[, y, sr, S, n_fft, ...])` | Compute the spectral centroid. |
| `spectral_bandwidth(*[, y, sr, S, n_fft, ...])` | Compute p'th-order spectral bandwidth. |
| `spectral_contrast(*[, y, sr, S, n_fft, ...])` | Compute spectral contrast |
| `spectral_flatness(*[, y, S, n_fft, ...])` | Compute spectral flatness |
| `spectral_rolloff(*[, y, sr, S, n_fft, ...])` | Compute roll-off frequency. |
| `poly_features(*[, y, sr, S, n_fft, ...])` | Get coefficients of fitting an nth-order polynomial to the columns of a spectrogram. |
| `tonnetz(*[, y, sr, chroma])` | Compute the tonal centroid features (tonnetz) |
| `zero_crossing_rate(y, *[, frame_length, ...])` | Compute the zero-crossing rate of an audio time series. |
| ... | |

> Ref: McFee et al., “librosa: Audio and music signal analysis in python,” 2015

### Slide 40
Library 2: Audio Commons Audio Extractor
https://github.com/AudioCommons/ac-audio-extractor

### Slide 41
More on LibROSA
* Audio loading
* Time-domain processing
* Signal generation
* Magnitude scaling

https://librosa.org/doc/latest/index.html

### Slide 42
More on LibROSA
* Time unit conversion
* Frequency unit conversion
* Music notation

https://librosa.org/doc/latest/index.html

### Slide 43
More on LibROSA
* Spectral representations
* Harmoincs
* Spectral features
* Pitch and tuning
* Phythm features

https://librosa.org/doc/latest/index.html

### Slide 44
Demonstrations 1
https://colab.research.google.com/github/stevetjoa/musicinformationretrieval.com/blob/gh-pages/spectral_features.ipynb

### Slide 45
Demonstrations 2
https://www.ifs.tuwien.ac.at/~schindler/lectures/MIR_Feature_Extraction.html

---

## Advanced Feature Extraction and Analysis

### Slide 46
Spectral Features Can be Used to Build a Classifier in ML
* MFCC : Spectrogram ---> Mel Filter Bank ---> Log Compression ---> Discrete Cosine Transform
* MFCC ---> "Mean and Variance" ---> Classifier
* Procedure
  1. Compute the features per STFT frame (e.g., 13-D frame-level features) 
  2. Temporal pooling over time for each audio clip (e.g., by taking the mean and variance; leading to 26-D clip-level features)
  3. Use that as input to a classifier (e.g., random forest, or support vector machine)
* Limits
  * Clear physical meaning but not sophisticated enough and limited semantic meaning
  * Only the classifier is trainable; the features are hand-crafted

### Slide 47
“Feature Learning” in DL (the blocks inside the black lines are learned)

* (a) Feature engineering
  * Spectrogram → Mel Filter Bank → Log Compression → Discrete Cosine Transform → Mean and Variance → Classifier
* (b) Low-level feature learning
  * Spectrogram → Mel Filter Bank → Log Compression → [Affine Transform → Nonlinearity → ...] → Pooling → Classifier
* (c) Convolution neural networks
  * Spectrogram → Mel Filter Bank → Log Compression → [Affine Transform → Nonlinearity → Pooling → ...] → Classifier
* (d) End-to-end learning
  * Raw Waveform → [Affine Transform → Nonlinearity → Pooling → Affine Transform → Nonlinearity → Pooling → ...] → Classifier

> Ref: Nam et al., “Deep learning for audio-based music classification and tagging,” IEEE Signal Processing Magazine, 2019

### Slide 48
The Use of Hand-crafted Features in DL
* Hand-crafted features
  * Clear physical meaning
  * Interpretable
* Used as input to music classifiers in the early days
* Can be used alongside learned features (though not popular)
  * The “deep & wide” architecture
* Can be used as objective metrics or loss functions for DL models

> Ref: Schaffer et al., “Music separation enhancement with generative modeling,” ISMIR 2022
> Ref: Cheng et al., “Wide & deep learning for recommender systems,” DLRS 2016

### Slide 49
Moving Beyond STFT: Constant-Q Transform (CQT)
https://music-classification.github.io/tutorial/part2_basics/input-representations.html
* STFT (linearly-spaced frequencies)
* CQT (logarithmically-spaced, closer to human auditory perception)
  * Logarithmic frequency, logarithmic magnitude

### Slide 50
Constant-Q Transform (CQT)
* Good for pitch-related tasks (e.g., multi-pitch estimation or transcription)
  * Less timbre-sensitive

https://github.com/archinetai/cqt-pytorch
https://github.com/eloimoliner/CQT_pytorch
> Ref: Hung et al, “Multitask learning for frame-level instrument recognition,” ICASSP 2019

### Slide 51
Pitch Class Profile / Chromagram
https://musicinformationretrieval.com/chroma.html
* “A chroma vector(Wikipedia) is a typically a 12-element feature vector indicating how much energy of each pitch class, {C, C#, D, D#, E, ..., B}, is present in the signal”  
  * i.e., ignore octaves

### Slide 52
Pitch Class Profile / Chromagram
* Good for tasks such as cover song identification or chord recognition 

Pipeline Steps:
1. HPCP Extraction (Extract Harmonic Pitch Class Profiles from each song)
2. Global HPCP (Compute global HPCP representation)
3. Optimal Transposition Index (Determine the best transposition between songs)
4. Song Transposition (Apply transposition to align songs)
5. Binary Similarity Matrix (Build a binary similarity matrix from both songs)
6. Dynamic Programming Local Alignment (Align the songs using DP-based local alignment)
7. Post-processing (Refine the alignment result)
8. Output: Final distance metric between the two songs

> Ref 1: Muller et al, “Audio matching via chroma-based statistical features,” ISMIR 2005
> Ref 2: Serra et al, “Chroma binary similarity and local alignment applied to cover song identification,” TASLP 2008

https://essentia.upf.edu/tutorial_similarity_cover.html

### Slide 53
Pitch Class Profile / Chromagram
* Good for tasks such as cover song identification or chord recognition 

https://www.audiolabs-erlangen.de/resources/MIR/FMP/C5/C5S2_ChordRec_Templates.html
> Ref: Cho et al, “On the relative importance of individual components of chord recognition systems,” TASLP 2014

### Slide 54
ISMIR 2025 Test-of-Time (ToT) Awardee

### Slide 55
Different Features are Needed for Different Tasks
* Timbre representation: Spectrogram → mel-spectrogram → MFCC
* Harmonic representation: Spectrogram → CQT → chroma feature

| Step | Operation | Top Parameter / Variant | Bottom Parameter / Variant |
| --- | --- | --- | --- |
| 1 | Short-time Windowing | $\approx 50$ms | $\approx 800$ms |
| 2 | Affine Transformation | Mel-scale Filterbank | Constant-Q Filterbank |
| 3 | Non-linearity | Modulus / Log-Scaling | Modulus / Log-Scaling |
| 4 | Pooling | (None) | Octave Equivalence |
| 5 | Affine Transformation | Discrete Cosine Projection | (None) |
| 6 | Features | MFCC | Chroma |

> Ref: Humphrey et al, “Feature learning and deep architectures: new directions for music informatics,” JIIS 2013

### Slide 56
Different Features are Needed for Different Tasks
* We can combine different features to train a classifier for tasks such as music genre classification or music emotion classification

https://maelfabien.github.io/machinelearning/Speech10/#spectrogram

### Slide 57
Different Features are Needed for Different Tasks
* Timbre representation: Spectrogram → mel-spectrogram → MFCC
* Harmonic representation: Spectrogram → CQT → chroma feature
* Wait… why the window size is different?

> Ref: Humphrey et al, “Feature learning and deep architectures: new directions for music informatics,” JIIS 2013

### Slide 58
Recap: Math in STFT
* Frequency spacing: $F_s / N$
  * Longer window size ($N$) → finer frequency resolution (but larger resulting STFT) → can localize events along the frequency axis
* Temporal spacing: $H / F_s$
  * Smaller hop size ($H$) → finer temporal resolution (but larger resulting STFT) → can localize events along the time axis 
* If $H = N / R$ (e.g., $R=4$)
  * Temporal resolution: $N / (R F_s)$ (no good freq/time resolution at the same time)

### Slide 59
Real Example 1: Piano Transcription
* Curtis Hawthorne et al., “Onsets and Frames: Dual-objective piano transcription,” ISMIR 2018
  https://archives.ismir.net/ismir2018/paper/000019.pdf

Q: What is the frequency resolution? And the temporal resolution?
"Our onset and frame detectors are built upon the convolution layer acoustic model architecture presented in [13], with some modifications. We use librosa [15] to compute the same input data representation of mel-scaled spectrograms with log amplitude of the input raw audio with 229 logarithmically-spaced frequency bins, a hop length of 512, an FFT window of 2048, and a sample rate of 16kHz. We present the network with the entire input sequence, which allows us to feed the output of the convolutional frontend into a recurrent neural network (described below)."

### Slide 60
Real Example 2: Beat Tracking
* Sebastian Böck, Florian Krebs, and Gerhard Widmer, “Joint beat and downbeat tracking with recurrent neural networks,” ISMIR 2016
  http://www.cp.jku.at/research/papers/Boeck_etal_ISMIR_2016.pdf

120 BPM = 120 beats per minute = 2 beats per second (16th note in 4/4 meter: 125 ms)

### Slide 61
Different STFT Parameters are Needed for Different Tasks
* Timbre/rhythm related tasks tend to use smaller windows
* Pitch/harmony related tasks tend to use longer windows
* Make sure you use the right $F_s$, window size and hop size!
  * Especially when using models from open source projects
  * STFTs of the same matrix size may have different physical meanings

### Slide 62
Other Time-Frequency Representations
* STFT of multiple window sizes (often seen in DL papers)
* CQT
* Wavelets
* Scattering transform
* In DL, STFT is a common choice

---

## DL-Based Music Classification

### Slide 63
Outline
* Music classification: Basics
* ML-based music classification (and hand-crafted audio features)
* DL-based music classification
* Music foundation models

### Slide 64
Reference: KAIST Course & ISMIR 2021 Tutorial
For fundamentals of deep learning and music classification
* https://mac.kaist.ac.kr/~juhan/gct634/Slides/05.%20music%20classification%20-%20deep%20learning.pdf
* https://music-classification.github.io/tutorial/landing-page.html

### Slide 65
Feature Learning
(the blocks inside the black lines are learned)
* (a) Feature engineering
* (b) Low-level feature learning
* (c) Convolution neural networks
* (d) End-to-end learning

> Ref: Nam et al., “Deep learning for audio-based music classification and tagging,” IEEE Signal Processing Magazine, 2019

### Slide 66
Feature Learning by Convolutional Layers
> Ref: Kereliuk et al, “Deep learning and music adversaries,” IEEE Trans. Multimedia, 2015

### Slide 67
Downsampling Layers: Convolution
https://github.com/vdumoulin/conv_arithmetic

### Slide 68
Padding & Stride
From: https://d2l.ai/chapter_convolutional-neural-networks/padding-and-strides.html

### Slide 69
Pooling
From: https://d2l.ai/chapter_convolutional-neural-networks/pooling.html#pooling

### Slide 70
Convolution: Locality and Translation Invariance
* “Convolution pooling” may lead to translation invariance
* See Dr. Juhan Nam’s slides (GCT634)

### Slide 71
Different Convolution Approaches
* 1D CNNs
* 2D CNNs
* Sample-level CNNs

### Slide 72
1D CNNs vs. 2D CNNs
* 1D CNNs
  * The filter size of the first conv layer covers the entire frequency range (can cover multiple frames)
  * Fast to train
  * Time-invariant but not pitch-invariant
* 2D CNNs
  * Significantly increases the number of parameters and thus need more computational resources
  * More flexible and powerful
  * Might be pitch-invariant

### Slide 73
1D CNN
Front-end: 1D CNN
* Mel-spectrogram input filter with 128 (mel bin) × 4 (frames)
* Feature maps (256 → 256 → 256 → 512), max-pooling in time (4 → 2 → 2)
Back-end: global pooling (temporal summary) with mean, max, and ...

https://sander.ai/2014/08/05/spotify-cnns.html

### Slide 74
Input Audio Representation
https://music-classification.github.io/tutorial/part2_basics/input-representations.html
* Log-magnitude spectrogram
  * Be viewed as a raw audio representation 
  * Discard phase: the human auditory system is insensitive to phase information
  * Log compresssion: human perception of loudness is closer to a logarithmic scale
* Melspectrograms
  * Based on a Mel-scale, which is nonlinear and approximates human perception
  * Reduces the number of frequency band greatly (1,024 → 128)

### Slide 75
Sample-level CNN
* Work on audio samples (e.g., two or three samples) rather than a typical window size (e.g., 512 samples)
* Longer training time; need larger compute

> Ref: Lee et al., “Sample-level deep convolutional neural networks for music auto-tagging using raw waveforms,” SMC 2017

### Slide 76
Convolutional Recurrent Neural Networks
* See Dr. Juhan Nam’s slides (GCT634)
* Use RNN for temporal summary
* The CRNN model slightly outperforms the CNN models but it is slower

> Ref: Choi et al., “Convolutional recurrent neural networks for music classification,” ICASSP 2017
https://music-classification.github.io/tutorial/part3_supervised/architectures.html

### Slide 77
Short-Chunk CNN
https://github.com/minzwon/sota-music-tagging-models/blob/master/training/model.py
> Ref: Won et al., “Evaluation of CNN-based automatic music tagging models,” SMC 2020

### Slide 78
Short-Chunk CNN
> Ref: Won et al., “Evaluation of CNN-based automatic music tagging models,” SMC 2020
https://github.com/minzwon/sota-music-tagging-models/blob/master/training/modules.py

### Slide 79
Short-Chunk CNN and Others
https://github.com/minzwon/sota-music-tagging-models

### Slide 80
Exemplar Models
https://music-classification.github.io/tutorial/part3_supervised/architectures.html

### Slide 81
Exemplar Models
https://music-classification.github.io/tutorial/part3_supervised/architectures.html

---

## Audio Data Augmentation

### Slide 82
Audio Data Augmentations
https://music-classification.github.io/tutorial/part3_supervised/architectures.html

### Slide 83
Audio Degradation Toolbox
* Exemplar use case: simulate the case of smartphone recording

https://github.com/sevagh/audio-degradation-toolbox

### Slide 84
torchaudio_augmentations
https://pytorch.org/audio/stable/tutorials/audio_data_augmentation_tutorial.html
https://music-classification.github.io/tutorial/part3_supervised/tutorial.html

### Slide 85
pyrubberband
* Time stretch: make it faster/slower without changing the pitch
* Pitch shift
* (ps. There is a cool function called “timemap_stretch”; check it out yourself)

https://github.com/bmcfee/pyrubberband

### Slide 86
Sample Code
https://music-classification.github.io/tutorial/part3_supervised/tutorial.html

### Slide 87
Sample Code
https://music-classification.github.io/tutorial/part3_supervised/tutorial.html

---

## Exemplar DL Models

### Slide 88
Exemplar Model: PANNs
https://github.com/qiuqiangkong/audioset_tagging_cnn
> Ref: Kong et al., “PANNs: Large-scale pretrained audio neural networks for audio pattern recognition,” arXiv 2020

* From sound event detection
* Purely convolutional
* Can be used as a pre-trained model
  * Produce audio embeddings that have been used by CLAP (https://github.com/LAION-AI/CLAP) in learning audio-text joint embedding space for text-to-audio generation

### Slide 89
Exemplar Model: Audio Spectrogram Transformer 
https://github.com/YuanGongND/ast
* From sound event detection
* Use Vison Transformer (ViT) based architecture
  * The first convolution-free, purely attention-based model for audio classification
* May need larger amount of training data and compute

> Ref: Gong et al., “AST: Audio Spectrogram Transformer,” INTERSPEECH 2021

### Slide 90
Exemplar Model: HTS-AT
> Ref: Chen et al., “HTS-AT: A hierarchical token-semantic audio transformer for sound classification and detection,” ICASSP 2022
> Ref: Wu et al., “Large-scale contrastive language-audio pretraining with feature fusion and keyword-to-caption augmentation,” arXiv 2022

---

## Evaluation Metrics and Training Tricks

### Slide 91
Evaluation Metrics for Music Classification
* The output of DL-based classifiers are usually probabilities
  * multi-class classification: softmax
  * multi-label classification: sigmoid
* Probability vs decision
  * outputting probabilities is fine at training time
  * but, at inference time, need to “make decisions”
  * usually by thresholding (e.g., at 0.5)

### Slide 92
Evaluation Metrics for Music Classification
* Classification accuracy (top1, top3)
* Precision, recall, Fscore
  * obtained by varying the threshold
* ROC-AUC
  * “micro” vs “macro” average 

https://music-classification.github.io/tutorial/part2_basics/evaluation.html

### Slide 93
Tricks when Training DL Models
* Avoid overfitting
  * Dropout
  * Weight decay (L1, L2 norm of weights)
  * Early stopping
  * Reduce model size
  * Data augmentation
* Overfitting is better than underfitting
  * Scale up the model till it overfits
  * Then try to mitigate overfitting
* Try different learning rates and optimizers

---

## Music Foundation Models and Datasets

### Slide 94
Outline
* Music classification: Basics
* ML-based music classification (and hand-crafted audio features)
* DL-based music classification
* Music foundation models

### Slide 95
Music Foundation Models
https://github.com/nicolaus625/FM4Music
> Ref: Ma et al., “Foundation Models for Music: A Survey,” arXiv 2024

* “The term foundation model was coined to describe a multipurpose machine learning model that, rather than being trained for a single specific task, serves as the basis of multiple derived models that are able to perform a wide range of tasks”
* “Following this pre-training phase, foundation models can be adapted for various downstream tasks via a relatively lightweight finetuning or in-context learning stage, for example, using a labelled dataset that is orders of magnitude smaller than the pre-training data.”
* “FMs for music not only address data scarcity and reduce annotation costs, but also enhance generalisation in music information retrieval and creation.”

### Slide 96
Pre-Training Strategies
* “Foundation models are pre-trained in a self-supervised fashion on large-scale datasets, avoiding or minimising the need for labelled data”
* Mainstream strategies
  * Contrastive learning and clustering (e.g., SimCLR)
  * Generative pre-training (e.g., AE or VQVAE)
  * Masked modelling (e.g., BERT)

### Slide 97
Self-Supervised Learning (SSL)
* Do not need human labels (unlabeled)
* Can learn from large amount of data
* Can do “transfer learning”
  * Pre-train the model with SSL
  * Fine-tune with a few more layers on downstream tasks using supervised learning

https://music-classification.github.io/tutorial/part5_beyond/introduction.html
> Ref: Gui et al., “A survey on self-supervised learning: Algorithms, applications, and future trends,” TPAMI 2024

### Slide 98
Contrastive Predictive Coding
https://music-classification.github.io/tutorial/part5_beyond/methods.html
> Ref: van den Oord et al., “Representation learning with contrastive predictive coding,” arXiv 2018

* Learn by predicting the future in a learned latent space

### Slide 99
SimCLR
https://music-classification.github.io/tutorial/part5_beyond/methods.html
> Ref: Chen et al., “A simple framework for contrastive learning of visual representations,” ICML 2020

* “Draw closer” positive data pairs while “push away” negative data pairs
  * Positive pairs: different “augmented views” of the same instance (done by data augmentation)
  * Negative pairs: different instances 
* Metric learning
  * shared encoder (CNN+MLP), output an embedding vector
  * learn a similarity metric discriminatively

### Slide 100
SimCLR
* Training details
  * given a mini-batch of N samples, we will have 2N samples after data augmentation
  * given a positive pair, view the other 2(N-1) pairs as negative
  * infoNCE loss

### Slide 101
Exemplar Model: CLMR
https://music-classification.github.io/tutorial/part5_beyond/self-supervised-learning.html
> Ref: Spijkervet & Burgoyne, “Contrastive Learning of Musical Representations,” ISMIR 2021

### Slide 102
Sample Code
* Use sample-level CNN as the feature extractor 
  * 59,049 ($3^{10}$) samples at 22kHz (~2.68 seconds)
* Use SimCLR for learning
* Freeze the pre-trained feature extractor, fine-tune linear classifier
  * supervised SampleCNN: 49.6%
  * SSL CLMR + SampleCNN: 55.2%

https://music-classification.github.io/tutorial/part5_beyond/self-supervised-learning.html

### Slide 103
Sample Code
https://music-classification.github.io/tutorial/part5_beyond/self-supervised-learning.html

### Slide 104
Exemplar Model: MERT
https://github.com/yizhilll/MERT
* Use sample-level CNNs
  * 5-second clips as input
  * 75 embeddings per second
* Masked language modeling
  * acoustic teachers ($\mathcal{L}_A$)
    * predict HuBERT or Encodec
  * musical teachers ($\mathcal{L}_{CQT}$)
    * predict CQT
  * do both

> Ref: Li et al., “MERT: Acoustic music understanding model with large-scale self-supervised training,” ICLR 2024

### Slide 105
MARBLE Benchmark
Music Audio Representation Benchmark for universaL Evaluation
https://marble-bm.shef.ac.uk/

### Slide 106
MERT Outperforms CLMR
https://marble-bm.shef.ac.uk/

### Slide 107
BEATS
> Ref: Chen et al., “BEATs: Audio Pre-Training with Acoustic Tokenizers,” ICML 2023

### Slide 108
Copyright-free Music Audio Datasets
* Free Music Archive (FMA)
  * https://freemusicarchive.org/
  * https://github.com/mdeff/fma
* Jamendo dataset
  * https://www.jamendo.com/
  * https://github.com/MTG/mtg-jamendo-dataset
* FreeSound
  * https://freesound.org/
  * https://labs.freesound.org/datasets/

https://music-classification.github.io/tutorial/part2_basics/dataset.html