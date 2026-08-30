Here is a targeted, evidence-based synthesis addressing your five specific questions, grounded in recent literature (2021–2026) on small-data audio classification and singer identification. All recommendations are strictly compatible with your "from-scratch, no external data" constraint.

---

### 1. Attention Pooling vs. Simpler Pooling as a Function of Backbone Capacity
Your hypothesis is highly plausible and supported by the literature: the negative result you observed with attention pooling is almost certainly a **capacity artifact**, not an inherent flaw of attention pooling itself. 

Attention pooling introduces additional parameters (e.g., the scoring MLP/linear layer) and optimization complexity. On a narrow backbone (e.g., 32-hidden-dim GRU), the network lacks the representational bandwidth to learn meaningful, discriminative attention weights across time steps. Instead, it tends to overfit to spurious temporal correlations (e.g., track position or silence boundaries) rather than singer-invariant vocal features. Surveys on deep learning with data scarcity explicitly note that limited data requires careful matching of model capacity to dataset size; complex attention mechanisms on narrow networks often degrade performance due to optimization instability and insufficient parameters to regularize the attention weights [[195]]. 

Conversely, wider bidirectional RNNs (e.g., 256 hidden dim, 2 layers) provide the necessary representational bandwidth for attention pooling to successfully weight informative vocal segments (like sustained vowels or specific formant transitions) without collapsing into a trivial solution. Your prior submission’s success with this exact configuration strongly supports this capacity-dependent dynamic.

### 2. Chunk Length: 5s vs. 10s for Singer Identification
For singer identification specifically, **longer chunks (10s) systematically outperform shorter chunks (3s or 5s)**, and this trade-off is well-documented. 

Singer identity is encoded in long-range temporal dynamics (vibrato, phrasing, formant transitions) that shorter chunks may truncate. In the seminal Artist20 benchmark evaluations, segment length ablations show that song-level accuracy consistently improves as segment length increases. For example, Kuo et al. (AAAI 2021) demonstrated that for CRNN-based singer identification on the Artist20 dataset, song-level F1 scores improved from 0.73 at 3s, to 0.74 at 5s, and up to 0.79 at 10s [[176]]. 

While 10s chunks yield fewer independent training examples per epoch from your 949-track pool, the gain in temporal context outweighs the reduction in sample count. To mitigate the optimization risk of fewer steps, you can compensate by slightly increasing the batch size or using a slightly higher initial learning rate with your cosine annealing schedule to maintain gradient update frequency.

### 3. Latest From-Scratch Architectures/Techniques (2023–2026) for Small-Data Audio
Excluding the techniques you have already tested, recent literature emphasizes *structural efficiency* and *task-specific inductive biases* over brute-force capacity for small-data audio classification:

*   **Parallel MLP-Attention Architectures (e.g., Branchformer):** Recent work shows that decoupling local feature extraction (via MLPs/convolutions) from global attention in parallel branches is significantly more stable to train than standard sequential Conformers on limited data and short utterances [[197]]. This architecture reduces the risk of attention collapse, which is a common failure mode when training attention mechanisms from scratch on ~1,000 tracks.
*   **Depth-wise Convolution with Gaussian Smoothing:** As validated in singer identification specifically, injecting a depth-wise convolution with a Gaussian kernel into attention or non-local blocks smooths feature maps [[176]]. This acts as a powerful implicit regularizer that improves generalization on small datasets without requiring external data or complex contrastive losses.
*   **Spatio-Temporal Attention Pooling:** Recent adaptations for small-footprint audio classification replace standard global average pooling with spatio-temporal attention that explicitly models frame-to-frame correlations, showing greater robustness in limited-data regimes compared to static or last-state pooling [[193]].

### 4. FGNL vs. Classic Wang et al.-Style Non-Local ResNet
**Porting the prior submission’s classic non-local ResNet is not recommended.** Your current `fgnl` or a well-tuned `se_resnet` is structurally superior for this data scale.

The classic non-local block (Wang et al.) computes dense pairwise spatial-temporal affinities, which is extremely parameter-heavy and highly prone to overfitting on small datasets (~950 tracks) unless heavily regularized. The FGNL module (Kuo et al., AAAI 2021) was explicitly designed to address this exact limitation in singer identification. It extends non-local operations to explore correlations *across channels and layers* while using depth-wise convolutions to smooth features, achieving better performance with *fewer* parameters than a naive non-local + ResNet combination [[176]]. 

A channel-ramped ResNet (32→512) introduces a massive capacity jump that can easily memorize the 949 training tracks. Your `fgnl` already incorporates the modern, parameter-efficient evolution of the non-local idea that was specifically validated on the Artist20 dataset.

### 5. Ensemble Diversity: Architectural vs. Training-Recipe
For small-N classification tasks, **architectural diversity yields significantly higher marginal gains in ensemble performance than training-recipe diversity** (e.g., seeds, hyperparameters, or minor augmentations) once a baseline level of performance is reached.

Recent studies on ensemble diversity in small-data regimes demonstrate that combining models with different inductive biases (e.g., a CRNN capturing temporal dynamics, an SE-ResNet capturing channel-wise timbral features, and a non-local network capturing long-range dependencies) reduces variance and alleviates individual model biases more effectively than ensembling variants of the same architecture [[185]]. 

Hyperparameter ensembles of the same architecture tend to make correlated errors on small datasets because they converge to similar local minima in the loss landscape [[192]]. The prior submission’s +11pp top1 gain from a 3-way architecturally distinct ensemble aligns perfectly with this principle: diverse architectures learn orthogonal feature representations, which is crucial when the total number of training examples is too small to allow a single architecture to learn all invariant features robustly. 

**Actionable Recommendation:** Prioritize porting 1–2 architecturally distinct models (e.g., the faithful `crnn_nasrullah` with 10s chunks and Bi-GRU attention pooling, plus a parallel MLP-attention variant) over further hyperparameter ablations on your existing `sota_crnn` variants. This will maximize the orthogonality of your ensemble's errors.