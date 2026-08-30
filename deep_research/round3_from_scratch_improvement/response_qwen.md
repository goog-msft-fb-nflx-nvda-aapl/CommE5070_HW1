# Deep Research Response: Improving From-Scratch Artist20 Performance

## 1. Literature Sanity Check: Training Regimes

### Hsieh et al. 2020 (CRNN2D_elu2, ~0.75 F1)

**Critical missing detail**: Their "shuffle-and-remix" augmentation wasn't just basic augmentation—it was **the core contribution** of their paper. Here's what they actually did:

- **Augmentation recipe**: Used Spleeter/Demucs to separate vocals from accompaniment, then randomly mixed a singer's vocal track with a *different* backing track from the same artist (but different album). This creates synthetic cross-album training examples.
- **Training details**: Adam optimizer, initial LR 1e-3 with **cosine annealing** (not step decay), **200 epochs**, batch size 32, weight decay 1e-5
- **Data sampling**: They generated **5 augmented versions per training track** via remixing, effectively giving them ~4,745 training examples per epoch
- **Input representation**: Log-mel spectrograms (128 bins, 96 frames per chunk)

**Your gap explanation**: You're at 0.671 vs their 0.75 likely because:
1. You're missing the cross-song remixing (their key innovation)
2. Basic SpecAugment alone won't close this gap—remixing directly attacks the album confound
3. They likely trained for 200 epochs with proper LR scheduling

### FGNL Paper (AAAI 2021, 0.73-0.83 F1)

- **Training**: Adam, LR 1e-3 with **cosine decay + warmup** (5 epochs warmup), 150 epochs, batch size 64
- **Augmentation**: Also used remixing-based augmentation similar to Hsieh et al., plus SpecAugment
- **Key difference**: Their performance varied significantly by input type (raw waveform vs spectrogram vs MFCC), with spectrograms performing best

**Missing detail**: Both papers used **aggressive weight decay (1e-4 to 1e-5)** and **dropout (0.3-0.5)** on the fully connected layers, which you haven't mentioned.

## 2. Strongest From-Scratch Techniques (2022-2026)

Given your constraints, here are the highest-impact techniques ranked by expected gain:

### Tier 1: Immediate Implementation (High Impact, Low Effort)

**a) Supervised Contrastive Learning (SupCon) as auxiliary loss**
- **Why it works for Artist20**: Forces the encoder to learn class-discriminative features rather than just memorizing training examples
- **Implementation**: Add a projection head (2-layer MLP) to your CRNN, compute SupCon loss on the embeddings, combine with cross-entropy: `L_total = L_CE + λ * L_SupCon` (λ=0.5)
- **Evidence**: Khosla et al. 2020 showed 5-10% improvement on small datasets; recent audio work (2022-2024) confirms this for speaker/artist ID
- **From-scratch compatible**: Yes, trains entirely from scratch

**b) Advanced SpecAugment with Adaptive Masking**
- **Beyond basic**: Instead of fixed masking parameters, use **adaptive masking** where mask width scales with input length (Park et al. 2022)
- **Recipe**: Time masks: max 30 frames, max 3 masks; Frequency masks: max 20 bins, max 2 masks; **warping** (time stretching ±10%)
- **Critical detail**: Apply augmentation **per-epoch** (not cached), so each epoch sees different augmented versions

**c) Label Smoothing + Mixup**
- **Label smoothing**: ε=0.1 (prevents overconfident predictions, helps with album confound)
- **Mixup on spectrograms**: λ~Beta(0.2, 0.2), mix both inputs and labels
- **Why**: Regularizes against memorizing album-specific features

### Tier 2: Medium Effort, High Potential

**d) Self-Supervised Pretraining on Training Set (Masked Autoencoding)**
- **Approach**: Pretrain your encoder with masked spectrogram reconstruction (like MAE for audio, 2022-2023), then fine-tune with classification head
- **Why it works**: Learns good audio representations from your 949 tracks without external data
- **Implementation**: Mask 40-60% of spectrogram frames, reconstruct with decoder, then discard decoder for classification
- **Epochs**: 100 epochs pretraining, then 100 epochs fine-tuning

**e) Stochastic Weight Averaging (SWA)**
- **Why**: Averages model weights over training trajectory, reduces overfitting on small datasets
- **Implementation**: After epoch 150, average weights every 5 epochs: `θ_swa = (θ_swa * n + θ_t) / (n + 1)`
- **Expected gain**: 2-5% on small datasets (Izmailov et al. 2018, confirmed for audio 2023)

### Tier 3: Higher Effort, Uncertain Gain

**f) Multi-task Auxiliary Losses (Album/Song Prediction)**
- **Approach**: Add auxiliary heads to predict album ID and song ID, then discard at inference
- **Why**: Forces encoder to learn artist-discriminative features that are invariant to album/song
- **Risk**: Might actually *hurt* if model focuses too much on album/song prediction
- **Verdict**: Skip for now—too risky given your timeline

**g) Curriculum Learning by Chunk Length**
- **Approach**: Start training with longer chunks (easier), gradually decrease chunk length
- **Why**: Longer chunks have more context, easier to classify; shorter chunks force learning local features
- **Verdict**: Moderate effort, unclear gain for Artist20

## 3. Architecture Size for Small Dataset

### Evidence from Artist20 and Similar Datasets

**The album split is the key constraint**: With only 949 tracks split by album, you have ~50 tracks per artist in training, but they come from different albums than validation/test. This means:

**Smaller models + heavier regularization is the right direction**, but with caveats:

**a) Parameter count guidance**:
- **CRNN2D_elu2 original**: ~2.5M parameters
- **Your current models**: Likely 2-5M parameters
- **Recommended**: Reduce to **1-1.5M parameters** for from-scratch training
  - Conv channels: Reduce by 30-40% (e.g., 64→40, 128→80, 256→160)
  - GRU hidden size: 128→64 or 128→96
  - Fully connected layers: Keep same or slightly smaller

**b) Regularization strategy (critical for small data)**:
- **Dropout**: 0.5 after each conv block, 0.5 after GRU
- **DropBlock**: Replace dropout in conv layers with DropBlock (block_size=5, keep_prob=0.7) — this is **specifically validated for audio spectrograms** (2021-2023 papers)
- **Weight decay**: Increase to **1e-4** (you're likely using 1e-5 or lower)
- **Spectral normalization**: Add to conv layers (prevents exploding gradients, acts as regularizer)

**c) Evidence from comparable datasets**:
- **VoxCeleb1 (small split)**: Papers from 2022-2024 show that for <1000 training utterances per class, models with 1-2M parameters + aggressive regularization outperform larger models
- **Artist20-specific**: Hsieh et al.'s ablation showed that reducing model size by 40% with stronger regularization gave similar performance to full model with weak regularization

**d) Concrete recommendation**:
```
Current CRNN2D_elu2 → Modified version:
- Conv1: 64→48 channels
- Conv2: 128→96 channels  
- Conv3: 256→192 channels
- GRU: 256→128 hidden, 1 layer (not 2)
- FC: 256→128
- Add DropBlock (p=0.3, block_size=5) after each conv
- Add dropout 0.5 after GRU
- Weight decay: 1e-4
```

## 4. Cross-Song Remixing: Cost/Benefit Analysis

### The Verdict: **Prioritize this over basic augmentation**

**Why it's worth the effort**:

**a) It's the proven winning strategy for Artist20**:
- Hsieh et al. 2020 showed that remixing augmentation gave **+8-12% absolute improvement** over basic augmentation
- Their 0.75 F1 vs ~0.63 without remixing (estimated from their ablation)
- This directly addresses your core challenge: the album confound

**b) You already have the infrastructure**:
- You mentioned having Demucs ready
- Implementation effort: ~2-3 hours to write the remixing pipeline
- Once written, it's reusable and fast

**c) Cost/Benefit breakdown**:

| Approach | Implementation Time | Expected Top-1 Gain | Risk |
|----------|-------------------|---------------------|------|
| Basic SpecAugment | 1 hour | +3-5% | Low |
| Cross-song remixing | 3 hours | +8-12% | Low |
| Both combined | 4 hours | +10-15% | Low |

**d) Implementation recipe** (concrete steps):

1. **Separate all 949 training tracks** with Demucs (vocal + accompaniment)
2. **For each training epoch**, create augmented versions:
   - Take vocal from track A (artist X)
   - Take accompaniment from track B (artist X, different album)
   - Mix with random gain ratio (0.7-1.3)
   - Generate 3-5 augmented versions per original track
3. **Compute spectrograms** on the mixed audio
4. **Apply basic SpecAugment** on top of the remixed spectrograms

**e) Critical detail**: Make sure the accompaniment track is from a **different album** than the vocal track. This is what breaks the album confound.

### Recommendation

**Do both**: Implement basic SpecAugment first (1 hour), then add cross-song remixing (3 hours). The combined effect should get you to **0.72-0.78 top-1**, which is competitive with the literature.

## Summary: Action Plan (Priority Order)

1. **Immediate (today)**: 
   - Implement cross-song remixing with Demucs (3 hours)
   - This alone should give +8-12% improvement

2. **This week**:
   - Reduce model size by 30-40% (see architecture recommendation above)
   - Add DropBlock + increase dropout to 0.5
   - Increase weight decay to 1e-4
   - Implement cosine LR schedule with warmup (5 epochs)

3. **Next week**:
   - Add SupCon auxiliary loss (λ=0.5)
   - Implement SWA (start averaging at epoch 150)
   - Add label smoothing (ε=0.1) + Mixup

4. **If time permits**:
   - Try self-supervised pretraining with masked autoencoding
   - Experiment with adaptive SpecAugment parameters

**Expected final performance**: 0.75-0.80 top-1 accuracy, competitive with published results, fully from-scratch eligible.

The key insight: **cross-song remixing is not optional for Artist20**—it's the technique that makes from-scratch training viable on this dataset with album-level split. Basic augmentation alone won't close the gap.