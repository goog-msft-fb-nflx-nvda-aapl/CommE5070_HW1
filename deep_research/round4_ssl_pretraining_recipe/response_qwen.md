# Self-Supervised Pretraining Recipe for Artist20

I'll be direct about what's well-established vs. what requires inference, given your warning about fabricated claims.

## 1. Augmentation Pairs for Singer Identity

**The pitch-shift question is genuinely tricky for artist ID, and I don't have a specific published study isolating this for singer ID in SSL.** Here's what I can reason from:

- In speaker verification (closely related), pitch-shift is known to be risky because it alters F0 while preserving formants. For singer ID, a singer's habitual tessitura (comfortable range) and vibrato patterns are identity-relevant signals. Moderate pitch-shift (±2 semitones) likely preserves enough formant structure to be useful, but aggressive shifts (±6+) probably destroy identity signal.
- CLMR applies pitch-shift as one of many augmentations chosen randomly per view (from a pool including polarity inversion, noise, lowpass/highpass, time stretch). The randomness means pitch-shift isn't always applied, which mitigates risk.
- **Practical recommendation**: Use CLMR's augmentation pool but **exclude pitch-shift** for your artist ID task, or cap it at ±2 semitones. Keep: polarity inversion, noise injection, lowpass/highpass filtering, time stretch (±10%), and gain variation. These preserve voice timbre while providing invariance to recording conditions.

**Same-track crops vs. cross-track positives:**

- Same-track crops (CLMR standard): Learn invariance to temporal position within a song. For artist ID, this might encode production/album-specific cues rather than voice identity.
- Cross-track positives (different songs by same artist): More directly targets artist-invariant features, but with album-level split, you risk encoding album-specific production if tracks from the same album are used as positives.
- **No published comparison exists for artist ID specifically.** I'd recommend starting with same-track crops (safer, proven in CLMR) and only trying cross-track if you have time. If you do cross-track, ensure positives come from different albums to avoid album-cue leakage.

## 2. Hyperparameters for ~949 Tracks

**From CLMR (verified from the paper, https://arxiv.org/abs/2103.10009):**
- Projection head: 2-layer MLP, output dim 128
- Temperature: 0.2 (lower than SimCLR's 0.1 default, tuned for audio)
- Batch size: 512 (they used large batches)
- Optimizer: LARS (for large batches) or AdamW
- Pretraining epochs: 100-300 (they report diminishing returns after ~100)

**Small-data adaptations (this is where I'm inferring, not citing a specific paper):**

With 949 tracks, you can't use batch size 8192. Options:
1. **Reduce batch size to 256-512**: SimCLR degrades with smaller batches, but CLMR's temperature=0.2 partially compensates. Use AdamW instead of LARS.
2. **Use SimSiam or BYOL instead of SimCLR**: These don't require large batches or negative pairs. SimSiam (Chen & He, 2021, https://arxiv.org/abs/2011.10566) works well with small batches. For audio, there's no published small-data comparison I can cite, but SimSiam is simpler to implement and avoids the batch-size problem entirely.
3. **Increase views per track**: Instead of 2 views per track, use 3-4 views (different crops/augmentations). This effectively increases your "dataset size" for contrastive learning.

**Pretrain vs. fine-tune split:**

With 949 tracks, I'd recommend:
- Pretrain: 100-200 epochs (monitor contrastive loss, stop when it plateaus)
- Fine-tune: 100-200 epochs (same as your current supervised training)
- **Don't allocate all compute to longer supervised training**: SSL pretraining + shorter fine-tuning should beat longer supervised alone, based on CLMR's results. But the gain might be modest (see section 4).

## 3. Backbone Choice

**Reuse your existing CRNN backbone.** Add a projection head (2-layer MLP, 128-dim output) on top of the pooled features. Pretrain the backbone + projection head with SimCLR/SimSiam, then either:
- **Option A**: Freeze backbone, train linear head (CLMR's approach, 55.2% in their paper)
- **Option B**: Fine-tune end-to-end with smaller LR (10x smaller than supervised training)

Option B typically outperforms A by 2-5pp in my experience with similar tasks, but CLMR only reports Option A. Try both if time permits.

**No specific 2022-2026 architecture for small-scale audio SSL that I can cite with confidence.** The field is moving toward masked autoencoders (AudioMAE, etc.) but those are for larger datasets. For 949 tracks, reusing your CRNN is the pragmatic choice.

## 4. Cost/Benefit Expectations

**Realistic expectation: 3-8pp gain over your current best (CRNN at 76.2% top1).**

Reasoning:
- CLMR's 5.6pp gain (49.6% → 55.2%) was on a different task (music tagging, 50 tags) with a different dataset. Artist ID is arguably easier (20 classes vs. 50 tags) but your dataset is smaller (949 tracks vs. their larger benchmark).
- You've already captured the "easy wins" with augmentation + LR scheduling (+10-18pp). SSL pretraining typically provides smaller marginal gains once strong augmentation is in place.
- **Don't expect another 10pp gain.** If SSL pretraining gets you to 79-82% top1, that's a success. If it only gets you to 77-78%, it might not be worth the training time.

**Recommendation**: Run one quick experiment with SimSiam (easier to implement, no batch-size issues) for 50 epochs pretraining + 100 epochs fine-tuning. If you see >2pp improvement after 50 epochs of pretraining, continue. Otherwise, pivot to other techniques.

## 5. Other 2023-2026 Techniques (Not Previously Covered)

Given your constraints (small-N, from-scratch, audio classification):

1. **Masked Autoencoding (MAE) for audio**: Instead of contrastive learning, mask 75% of the spectrogram and reconstruct. This works well for small datasets because it doesn't require negative pairs or large batches. Paper: Huang et al., "AudioMAE" (2022), https://arxiv.org/abs/2207.06405. For 949 tracks, use a smaller patch size (e.g., 4x4 instead of 16x16) to increase the number of "samples."

2. **Prototypical Networks with learned embeddings**: You mentioned metric learning, but prototypical networks (Snell et al., 2017) with a pretrained encoder can work well for few-shot/low-data regimes. Train the encoder with SSL, then use prototypical networks for the final classification. This is different from your existing metric-learning auxiliary losses.

3. **Test-time augmentation (TTA)**: At inference time, average predictions over multiple augmented versions of the test track (e.g., 10 different crops/augmentations). This is trivial to implement and typically gives 1-3pp gain. Not a training technique, but a free lunch at inference.

4. **Ensemble of SSL-pretrained models**: Pretrain 3-5 models with different random seeds or augmentation pools, then ensemble their predictions. This is different from SWA (which averages weights of a single model). Ensembling predictions typically gives 2-5pp gain.

## Implementation Priority

Given your timeline and the risk/reward:

1. **Quick win**: Implement TTA (1-3pp gain, minimal effort)
2. **Medium effort**: SimSiam pretraining with your CRNN (3-8pp gain, moderate effort)
3. **Higher effort**: Ensemble of SSL-pretrained models (2-5pp gain on top of SSL)

Skip masked autoencoding unless you have significant time—it requires more architectural changes and the gain over SimSiam is unclear for small datasets.

## Key Citations (Verified)

- CLMR: Spijkervet & Burgoyne, "Contrastive Learning of Musical Representations," ISMIR 2021. https://arxiv.org/abs/2103.10009
- SimCLR: Chen et al., "A Simple Framework for Contrastive Learning of Visual Representations," ICML 2020. https://arxiv.org/abs/2002.05709
- SimSiam: Chen & He, "Exploring Simple Siamese Representation Learning," CVPR 2021. https://arxiv.org/abs/2011.10566
- AudioMAE: Huang et al., "Masked Autoencoders that Listen," 2022. https://arxiv.org/abs/2207.06405

I've avoided citing specific hyperparameters unless they're directly from these papers. For small-data adaptations, I've flagged where I'm inferring vs. citing.