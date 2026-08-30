## Part 1 — Error analysis methodology for this singer-identification task

### 1. What the Artist20 literature actually does — and does not — establish

The most important finding from the original Artist20 literature is that **the central error-analysis question is not simply “which singers sound alike?” but “is the model using vocal identity or accompaniment/production cues?”**

Hsieh et al. explicitly frame singer identification as vulnerable to accompaniment confounds: because singers tend to occur in characteristic genres/styles, a classifier can use non-vocal information that correlates with the artist label and then fail when that singer appears in a different musical context. Their experiments therefore manipulate the source of information using **vocal-only, accompaniment-remixed, and original mixtures**, rather than relying only on a confusion matrix. ([arXiv][1])

That is the closest thing I found to a *documented Artist20-specific error-analysis methodology*, and it is worth adopting conceptually.

The earlier Artist20 CRNN work by Nasrullah & Zhao likewise emphasizes the **producer/album effect**, showing a large gap between song-split and album-split performance and visualizing learned bottleneck embeddings by artist. Importantly, however, that paper does **not** provide a detailed musicological “X is confused with Y because of these vocal characteristics” analysis. ([ResearchGate][2])

Similarly, Ellis's original timbral/chroma work establishes that MFCC/timbral information and chroma carry partly complementary information for artist identification, but again does not give you a standard pair-by-pair musicological explanation procedure. ([コロンビア大学 電気工学部][3])

So I would **not** write that Hsieh et al. established a standard confusion-pair taxonomy such as “bass/baritone/tenor + genre + gender.” They did not. Their contribution is much more useful: **test whether predictions survive removal/replacement of accompaniment context.**

---

### 2. A concrete confusion-analysis procedure I would use for your 20 classes

For every important off-diagonal pair, turn the confusion matrix into a small case study.

A good ranking is:

$$
P(\hat y=j\mid y=i)
$$

for each true artist \(i\), rather than just counting raw errors. Then prioritize pairs that are either frequent or strongly asymmetric.

For example, instead of saying:

> “Tori Amos and Fleetwood Mac are often confused.”

write something like:

> “Among true Tori Amos tracks, X% were predicted as Fleetwood Mac, making this one of the model's largest asymmetric confusions. The confusion is concentrated in albums A/B and in tracks with greater vocal register overlap / stronger rock instrumentation.”

Then investigate the pair in **four layers**.

#### Layer A — Vocal properties

For each of the two artists, characterize the actual training/test material with measurable quantities rather than relying only on biographical descriptions:

* estimated vocal fundamental-frequency distribution / tessitura
* pitch range actually used in the evaluated tracks
* proportion of low-, mid-, and high-register frames
* spectral centroid / spectral slope / spectral contrast
* MFCC distribution
* harmonic-to-noise or related voice-quality measures, where reliable
* vibrato rate / extent if you have a robust pitch track
* vocal activity ratio

The key phrase here is **“actual distribution in your dataset,” not “Wikipedia says singer X is a tenor.”**

This is particularly important because a singer's physiological range is not the same thing as the range/tessitura used on a particular album.

#### Layer B — Musical context

Then ask whether the pair also shares:

* genre/subgenre
* approximate era
* tempo
* instrumentation
* harmonic/chord characteristics
* recording/production era
* studio effects
* arrangement density

This is not merely speculative. The Artist20 literature explicitly identifies **production and accompaniment style as a potential confound**, and MIR work has repeatedly treated studio production as a meaningful source of artist/genre leakage. ([arXiv][1])

#### Layer C — Source-separation control

This is the strongest analysis I would add because it is directly aligned with the Artist20 literature.

For a confusion pair or several high-confusion pairs, compare predictions on:

1. original mixture
2. vocal-separated audio
3. accompaniment-only audio, if practical

Hsieh et al. show why this is informative: they explicitly compare original, vocal-only, and remixed contexts and find that removing accompaniment can reduce performance, while remixing vocals with different accompaniment can improve generalization. ([Dihana][4])

A particularly compelling result would be:

> “The A→B confusion drops substantially on vocal-only inference, suggesting that part of the original confusion is driven by accompaniment/production rather than vocal identity.”

Conversely:

> “The A→B confusion persists on vocal-only audio, strengthening the interpretation that the model is responding to vocal characteristics shared by the two artists.”

That is much stronger than a paragraph based purely on genre labels.

#### Layer D — Album/context concentration

For each major confusion \(i\rightarrow j\), make a tiny table:

| True artist | Predicted as | Overall | Album 1 | Album 2 |  … |
| ----------- | ------------ | ------: | ------: | ------: | -: |
| Artist A    | Artist B     |     18% |      8% |     31% |  … |

If one album generates most of the A→B errors, investigate its production, vocal treatment, instrumentation and era.

This is especially relevant because the album split was introduced precisely to reduce leakage from album/producer characteristics. Nasrullah & Zhao demonstrate how dramatically performance can change when the split changes, and Hsieh et al. explicitly retain album-split evaluation for this reason. ([ResearchGate][2])

### Important methodological distinction

Don't treat **gender, solo-vs-band, genre, or era as explanations by themselves**.

They are candidate explanatory variables. A strong report would phrase:

> “This pair shares genre and register characteristics, which are plausible contributors…”

rather than:

> “The model confuses them because they are both mezzo-sopranos.”

The latter is not demonstrated by the experiment.

---

### 3. Connecting Task-1 feature importance to specific artist pairs

Yes — and this is one of the easiest ways to make Task 1 substantially more interesting.

Your current global permutation-importance ranking answers:

> “Which features does the model rely on overall?”

It does **not** answer:

> “Which features distinguish artist A from artist B?”

Permutation importance measures the degradation of the fitted model when a feature is shuffled; it is model- and dataset-specific rather than an intrinsic property of the feature. It also becomes difficult to interpret when features are highly correlated. ([scikit-learn][5])

I would therefore add a **pairwise feature-analysis matrix**.

For each important confusion pair \(A,B\):

1. Take only A and B.
2. Evaluate the same SVM/kNN/RF feature representation.
3. Compute pairwise permutation importance on held-out data.
4. Aggregate individual dimensions into feature families:

   * MFCC
   * chroma
   * spectral contrast
   * ZCR
   * tonnetz
   * etc.
5. Compare the pairwise importance with the **global** importance.

That gives you a very useful distinction:

> “MFCC was globally the most important feature family, but chroma became disproportionately important for distinguishing Queen from Radiohead.”

You can go one step further and calculate a simple **feature separability measure** for each artist pair, such as the held-out AUC of an individual feature or standardized effect size. That tells you whether the pairwise feature importance is supported by an observable distributional difference.

I would also **group correlated feature dimensions** rather than over-interpreting individual MFCC coefficients. Scikit-learn explicitly warns that correlated features can make permutation importance appear artificially small because another correlated feature can substitute for the permuted one. ([scikit-learn][5])

So a very nice report figure would be:

> **Artist-pair × feature-family heatmap**

where each cell is “pairwise permutation importance” or “pairwise performance drop.”

That turns Task 1 from a generic classifier bake-off into actual scientific analysis.

---

### 4. Statistical significance for your 231-track validation set

This is one area where I would make a fairly strong recommendation.

### For Top-1 accuracy: McNemar's test

Because every model is evaluated on the **same tracks**, predictions are paired. McNemar's test is designed exactly for comparing two paired classifiers' binary correctness outcomes. ([Rasbt][6])

For models A and B, construct:

* A correct, B wrong
* A wrong, B correct

and test whether those discordant counts are symmetric.

For very small discordant counts, use the **exact binomial version**, rather than relying on the asymptotic chi-square approximation. ([Rasbt][6])

### For your actual grading metric: paired bootstrap

Your grading quantity is

$$
U_i=I(\text{Top1 correct})+0.5I(\text{Top3 correct})
$$

per track.

That makes a **paired bootstrap of the per-track utility difference** particularly appropriate:

$$
\Delta=\frac{1}{N}\sum_i(U_i^{(A)}-U_i^{(B)})
$$

Then resample tracks with replacement, using the same sampled indices for both models, and report:

> Δ = +0.43 percentage points, 95% CI [−0.31, +1.12].

This is much more informative than saying “Model B is 0.4 pp better.”

Recent ML evaluation papers commonly use exactly this paired-bootstrap structure for model comparisons, including confidence intervals on performance differences. ([Frontiers][7])

### There is one Artist20-specific wrinkle

Your examples are not truly independent tracks in the statistical sense: **tracks from the same artist and album share context**.

Because your evaluation unit is album-split music, I would ideally report:

* track-level paired bootstrap, and
* a **clustered sensitivity analysis** resampling at the album or artist level.

The latter will give wider, more conservative intervals because it respects within-album dependence.

You do not need to overcomplicate the main report. A good sentence is:

> “Paired bootstrap confidence intervals were computed over tracks; because the album-split design introduces within-album dependence, an album-clustered bootstrap was additionally used as a sensitivity analysis.”

### For many pairwise comparisons

You have ~17–19 models, so testing every model against every other model creates a multiple-comparison problem. Don't dump 171 p-values into the report.

Pick a **predefined small set of comparisons**:

* best single model vs best ensemble
* best single model vs strongest baseline
* ensemble vs strongest constituent model
* maybe SOTA vs previous best

Then use Holm correction if you still have multiple hypotheses.

### About the “0.4–0.5 pp” differences

Your intuition is correct.

With \(N=231\),

$$
1/231 \approx 0.433\%
$$

so a +0.43 pp improvement is literally one additional correctly classified track.

That absolutely deserves a significance/stability analysis before being described as a meaningful improvement.

---

# Part 2 — Interpreting the Sia “Unstoppable” OOD result

There is an important correction before discussing the vocal analysis:

### “Unstoppable” is **not** from the *Fifty Shades of Grey* soundtrack

Sia's **“Unstoppable” is a track from her 2016 album *This Is Acting***. Sia's official discography page places it on *This Is Acting*, and AllMusic lists it as part of that album. ([sia.lnk.to][8])

The Sia songs associated with the *Fifty Shades* films are different:

* **“Salted Wound”** — *Fifty Shades of Grey*
* **“Helium”** — *Fifty Shades Darker* ([sia.lnk.to][9])

I would definitely fix this in the report.

### Musical characterization of “Unstoppable”

AllMusic classifies *This Is Acting* broadly as **Pop/Rock**, with pop and alternative-pop styles, and describes “Unstoppable” as a highly confident, diva-oriented hit-song performance. ([AllMusic][10])

Contemporary reception also repeatedly framed the song as an **empowerment/anthemic pop song**, with comparisons to big mainstream female pop vocalists such as Rihanna and Katy Perry. ([ウィキペディア][11])

So for the report I would call it:

> **mainstream electropop / pop anthem with a strongly theatrical, empowerment-oriented vocal delivery**

rather than “pop-soul” unless you have a specific source supporting that narrower label.

---

### Sia's vocal characteristics

There is a surprisingly important distinction between **descriptive criticism** and **scientifically measured vocal analysis**.

I found strong descriptive evidence that Sia is associated with:

* a distinctive, somewhat **raspy / smoky** timbre
* strong dynamic contrast
* powerful belting
* expressive strain/exertion
* an unusually theatrical delivery

For example, NPR's published profile of her “Chandelier” describes a low-register pre-chorus followed by a high-intensity belt characterized by exertion and rasp. A 2025 vocal-coach analysis of “Unstoppable” similarly emphasizes dynamic range, powerful choruses and a slightly raspy timbre. ([Ksut][12])

However, I **did not find a high-quality peer-reviewed note-by-note study establishing a definitive vocal range specifically for the studio recording of “Unstoppable.”**

There are many web-based range estimates, but their methodological quality is much weaker. For a graduate report, I would therefore avoid writing something like:

> “Sia sings from D3 to A5 on Unstoppable.”

unless you actually measure the recording yourself.

A much safer formulation is:

> “Sia is widely described as a high-intensity pop vocalist with a broad usable range, a raspy/smoky timbre, and a characteristic contrast between restrained low-register passages and forceful upper-register belting.”

That statement is much better supported.

---

## Which of your model predictions make musical sense?

There is an important caveat first:

### Your OOD input is not actually “Sia's isolated voice”

You said it is a **Voice Memos recording of someone singing along to “Unstoppable.”**

That means the classifier may hear some combination of:

1. the actual singer in the Voice Memos recording,
2. the original Sia recording,
3. accompaniment,
4. phone/microphone characteristics,
5. room acoustics,
6. mixing between the user's voice and the original track.

This is extremely important.

You therefore **cannot interpret the prediction as “the model thinks Sia sounds like Tori Amos”** without qualification.

It is more accurate to say:

> “The model mapped an unseen recording of a singer performing along with Sia's ‘Unstoppable’ to the following nearest known Artist20 classes.”

That distinction will make the report much more defensible.

---

### Tori Amos

Tori Amos is documented as a **mezzo-soprano** with a background spanning alternative rock, chamber pop, pop rock and electronic music. ([ウィキペディア][13])

There is therefore some plausible overlap in:

* female mid-register vocal classification,
* alternative/pop-rock vocal texture,
* expressive phrasing and dynamics.

But I **did not find a credible source explicitly documenting Sia–Tori Amos vocal similarity**.

So I would classify this as:

**Plausible acoustic similarity, but not a documented Sia-specific comparison.**

Also, Tori's characteristic vocal presentation is much more associated with singer-songwriter/alternative-rock performance and piano than with the heavily programmed mainstream electropop production of “Unstoppable.” That makes a pure **production-style explanation less obvious than for Madonna**.

---

### Queen / Freddie Mercury

This is the most interesting false-nearest-neighbor.

There is high-quality quantitative work on Mercury's voice. A 2017 acoustic study found:

* speaking fundamental frequency consistent with a baritone,
* a measured singing range of about **37 semitones from F#2 to G5** in the analyzed material,
* unusually rapid average vibrato around **7 Hz**,
* distinctive irregular vibrato,
* evidence of subharmonic phonation associated with his distinctive voice production. ([Taylor & Francis Online][14])

So Freddie is **not** a simple “same female vocal range” match.

That actually makes your model's Queen prediction more interesting: a possible explanation is similarity in **vocal intensity, spectral texture, harmonic richness, distortion/rasp-like qualities, or expressive dynamics**, rather than raw pitch range.

But there is **no source I found documenting Sia and Freddie Mercury as recognized vocal look-alikes**.

So I would write:

> “The Queen prediction is difficult to interpret as a direct vocal-range match; Freddie Mercury's documented voice spans baritone through high-register singing and is acoustically distinctive for its vibrato and nonlinear/subharmonic characteristics. The match may therefore reflect broader spectral or expressive similarity, although the present experiment cannot distinguish this from accompaniment or production effects.”

That is a much stronger scientific statement than trying to force a “Sia sounds like Freddie” claim.

---

### Fleetwood Mac / Stevie Nicks

This is one of the **most plausible** predictions on vocal-texture grounds.

Stevie Nicks is repeatedly described as having a **raspy**, emotionally expressive voice, and contemporary criticism emphasizes precisely those characteristics. ([ザ・ガーディアン][15])

That overlaps reasonably well with the documented descriptions of Sia's:

* rasp
* emotional intensity
* dramatic phrasing
* rock/pop crossover

However, there is one very important dataset-specific caveat:

> **“Fleetwood Mac” is not equivalent to “Stevie Nicks.”**

The Artist20 class represents the band, and Fleetwood Mac has multiple lead vocalists, notably Stevie Nicks, Christine McVie and Lindsey Buckingham.

So you can use Stevie Nicks as a *possible explanatory reference*, but you should not state that the classifier's Fleetwood Mac prediction proves that it identified “Stevie-like” vocals.

I'd rate this as:

**strongest plausible vocal-texture hypothesis among your female predictions, but not a documented Sia-specific similarity.**

---

### Madonna

Madonna is commonly classified as a **mezzo-soprano**, and descriptions of her vocal style emphasize pop/rock roots, aggression, rhythmic phrasing and highly stylized delivery. ([ウィキペディア][16])

Here I think **production/genre is at least as plausible as vocal similarity**.

“Unstoppable” is explicitly described as a diva-oriented mainstream pop hit, and the song's lineage has been compared with Rihanna and other mainstream female pop performers. ([AllMusic][10])

So a model trained on full mixtures could associate:

> female pop lead + large anthem + electronic production + compressed/processed vocal + strong chorus

with Madonna even if the underlying voice itself is not especially Madonna-like.

This is exactly the kind of hypothesis that your accompaniment-removal experiment could test.

---

### Roxette / Marie Fredriksson

This is another **very plausible voice-level hypothesis**.

The Guardian's obituary for Marie Fredriksson describes her as having approximately a **three-octave range**, mezzo-soprano technique and strongly emotional singing. ([ザ・ガーディアン][17])

Her voice is also frequently characterized as powerful and emotionally forceful in accounts of Roxette's work. ([AllMusic][18])

That makes Roxette a plausible nearest class because of the combination of:

* female mezzo-oriented register,
* wide usable range,
* emotional intensity,
* powerful pop-rock delivery,
* crossover between rock and mainstream pop.

Again, though, I found **no authoritative source saying “Sia sounds like Marie Fredriksson.”**

So the correct interpretation is:

> **The two voices have several independently documented properties that overlap, making the model's Roxette prediction acoustically plausible; it is not evidence of a published Sia–Fredriksson similarity.**

---

## My interpretation of the cross-model pattern

Your results are actually more interesting than a single top-3:

| Model         | Top result                          | What the pattern suggests                |
| ------------- | ----------------------------------- | ---------------------------------------- |
| CRNN-wide     | Tori Amos / Queen / Fleetwood Mac   | relatively distributed similarity        |
| CRNN          | Madonna / Fleetwood Mac / Roxette   | mainstream-pop + female-vocal similarity |
| ECAPA         | Fleetwood Mac / Roxette / Tori Amos | stronger speaker/timbre representation   |
| Confound-CRNN | Madonna / Radiohead / Prince        | stronger sensitivity to mixture/context  |

The fact that **Fleetwood Mac, Roxette and Tori Amos recur across different models** is more interesting scientifically than any individual prediction.

Those recurring classes could represent a **stable acoustic neighborhood** under different representations.

By contrast, Madonna appears prominently in models that may be more sensitive to overall mixture/production characteristics.

But with one OOD recording, this remains a hypothesis.

### The experiment I'd add

Run the *same* OOD recording through:

> **original mixture → vocal-isolated → accompaniment-isolated**

and compare the top-5 distribution.

If:

* Fleetwood Mac/Roxette/Tori remain after vocal isolation,
* while Madonna/Prince/Radiohead collapse,

you have evidence for a voice-driven similarity.

If the opposite happens, you have evidence for an accompaniment/production explanation.

That experiment maps almost directly onto the central issue identified by Hsieh et al. ([arXiv][1])

---

## What does the literature say about OOD singers generally?

There is a solid distinction between **open-set speaker recognition** and your current closed-set classifier.

In open-set speaker recognition, an unknown speaker is explicitly allowed to be rejected rather than being forced into one of the enrolled identities. That is a longstanding formulation of the problem. ([サイエンスダイレクト][19])

Your 20-way softmax classifier does **not** have that capability.

For an unseen singer, it must output:

$$
p(y=k\mid x),\quad k\in\{1,\dots,20\}
$$

even though the true label is not in the label set.

So the Sia result should be described as:

> **closed-set nearest-class behavior on OOD input**

rather than genuine OOD recognition.

### Can literature tell us whether it chooses timbre or production?

Not reliably enough to make a universal claim.

What the singer-ID literature *does* establish is that **both vocal information and accompaniment/production information can be discriminative**, and that the latter can be an unwanted confound. Hsieh et al. explicitly demonstrate this through vocal separation and context remixing. ([arXiv][1])

Therefore the most scientifically defensible conclusion is:

> **There is no basis for assuming that an OOD nearest-class prediction is primarily a vocal-timbre match. In a full-song classifier, it may reflect a mixture of vocal acoustics, arrangement, production, genre, recording conditions and the specific representation learned by the model.**

That is exactly why your source-separation OOD experiment would be so valuable.

Also, your Sia example illustrates a calibration issue: **13.7% / 12.9% / 12.4% is not evidence that the model “knows it doesn't know.”** A 20-way softmax always distributes probability across the enrolled classes. Open-set rejection requires an explicit mechanism or score. ([サイエンスダイレクト][19])

---

# Part 3 — What I would add to make the report substantially stronger

You already have a very large amount of experimentation. At this point, **more models are much less valuable than turning the existing experiments into evidence about what the models learned.**

My priority order would be:

## 1. Statistical rigor — highest priority

Do this.

For the main comparisons, report:

* Top-1
* Top-3
* grading utility \(Top1+0.5Top3\)
* paired bootstrap 95% CI for the difference
* exact McNemar p-value for Top-1
* multiplicity correction if making several formal comparisons

For your 231 tracks, this is especially important because one track is ~0.43 percentage points.

I would make the **paired bootstrap CI on the actual grading utility** the headline statistical analysis.

A recent 2026 singer-identification paper also reports confidence intervals explicitly for its held-out performance, demonstrating that this is a perfectly reasonable presentation convention in current singer-ID work. ([ACL Anthology][20])

---

## 2. Confound analysis — probably the most scientifically valuable addition

This is arguably more important than another architecture table.

Do:

**mixture vs vocal-only vs accompaniment-only**

on the models you already have.

Then report:

$$
\Delta_{\text{vocal}} =
Acc(\text{vocal-only})-Acc(\text{mixture})
$$

and, if possible,

$$
Acc(\text{accompaniment-only})
$$

If accompaniment-only produces surprisingly strong artist discrimination, that is a very powerful result: it directly demonstrates that Artist20 contains artist-associated production information that a model can exploit.

This is closely aligned with the motivation and experiments of Hsieh et al. ([arXiv][1])

---

## 3. Calibration analysis — definitely worth doing

I would include:

* reliability diagram
* ECE
* multiclass Brier score
* NLL
* average max probability
* entropy

and show these for the strongest single models and ensemble.

Neural networks can be substantially miscalibrated even when their accuracy is good, and temperature scaling is a standard post-hoc calibration method. ([Proceedings of Machine Learning Research][21])

Your Sia example makes calibration especially relevant because it gives you a nice qualitative demonstration:

> Model A is highly confident on an unknown singer; Model B distributes probability more evenly.

That makes the OOD demo analytically meaningful rather than merely entertaining.

One caution: **do not fit the calibration transform on the test set.** Use validation data only.

---

## 4. Per-album analysis — yes, but don't stop at accuracy

Definitely do it, but use it as a **confound diagnostic**, not just another table.

For each artist:

* per-album Top-1
* per-album Top-3
* dominant confusion pairs
* confidence / entropy

Then ask:

> Does the same artist's prediction quality collapse on a particular album or production era?

A particularly good figure would show:

> **Artist × Album accuracy heatmap**

followed by two or three discussion cases.

But be careful with the claim:

> “No album difference means the producer confound is solved.”

That does **not** logically follow.

A stronger test is whether performance remains stable under **vocal/accompaniment manipulations** or context remixing, because that's much closer to the causal question Hsieh et al. are addressing. ([arXiv][1])

---

## 5. Embedding geometry — worth doing, but quantitatively

Your t-SNE is useful as a visualization, but t-SNE alone is not a strong representation-quality metric.

I would add:

### Same-artist vs different-artist cosine distance

Plot two distributions:

$$
d_{\text{same}} = \cos^{-1}(sim(x_i,x_j))
$$

and

$$
d_{\text{different}}
$$

or simply cosine similarity distributions.

Then report:

* mean/median same-class similarity
* mean/median different-class similarity
* effect size
* overlap
* optionally silhouette score

The basic representation-learning objective is exactly to reduce intra-class distances and increase inter-class distances. ([サイエンスダイレクト][22])

A very good comparison would be:

> CRNN / wide-CRNN / pretrained embedding / ensemble embedding

with the same distance statistics.

That tells you whether an apparently better classifier is actually producing a better-separated representation, or merely exploiting decision-boundary effects.

---

## 6. A clean ablation-summary convention for the ~19 experiments

Do **not** put 19 unrelated rows in one giant table and expect the reader to understand them.

I would use a two-level presentation.

### Main table

Only show the major experimental families:

| Model / intervention | Top-1 | Top-3 | Grade metric | Δ vs baseline |
| -------------------- | ----: | ----: | -----------: | ------------: |
| Baseline CRNN        |       |       |              |               |
| Wider CRNN           |       |       |              |               |
| + augmentation       |       |       |              |               |
| + attention          |       |       |              |               |
| Best single model    |       |       |              |               |
| 7-model ensemble     |       |       |              |               |

### Secondary table / appendix

Put every ablation there with:

* exact architecture
* parameters
* input representation
* loss
* augmentation
* Top-1
* Top-3
* seed/run count

Then the body of the report tells the **story**, while the appendix proves completeness.

That is much more readable than a wall of 19 accuracy numbers.

---

# What I would consider the “ideal” report analysis package

Given everything you already have, I would aim for these **six pieces**:

**A. Model comparison:**
Top-1 / Top-3 / grading metric + paired bootstrap CIs.

**B. Error analysis:**
Top confusion pairs + per-artist and per-album breakdown + 2–4 detailed musical/acoustic case studies.

**C. Confound analysis:**
Mixture vs vocal-only vs accompaniment-only, ideally for both ID performance and OOD behavior.

**D. Task-1 feature analysis:**
Global permutation importance + **pairwise artist × feature-family importance**.

**E. Representation analysis:**
Same/different artist embedding-distance distributions + t-SNE as visualization.

**F. Uncertainty/OOD analysis:**
Calibration curves + entropy/confidence + the Sia example framed explicitly as **closed-set OOD nearest-class behavior**.

That would turn the report from:

> “We trained 17 models and the ensemble got 86.1%.”

into:

> “We established which architectures improve performance, whether the gains are statistically distinguishable, which artist pairs remain difficult, what acoustic/contextual properties characterize those errors, whether the learned representation separates artists, whether the system is calibrated, and whether its apparent behavior on an unseen singer is driven by vocal or accompaniment information.”

That second narrative is much closer to what I'd expect from a strong graduate-level MIR report.

### Most important correction to carry into the report

I would explicitly fix the soundtrack sentence to:

> **“Sia's ‘Unstoppable’ was released on her 2016 album *This Is Acting*; it was not part of the *Fifty Shades of Grey* soundtrack.”** ([sia.lnk.to][8])

And I would avoid claiming that the model “heard Sia and recognized her as similar to Tori Amos/Stevie Nicks/etc.” The experimentally defensible statement is that the **closed-set classifier assigned an OOD performance to those enrolled classes**, with the recurring predictions providing hypotheses about vocal and/or production similarity that can be tested with source-separation and counterfactual-context analysis.

[1]: https://arxiv.org/abs/2002.06817 "Addressing the confounds of accompaniments in singer identification"
[2]: https://www.researchgate.net/publication/336157135_Music_Artist_Classification_with_Convolutional_Recurrent_Neural_Networks?utm_source=chatgpt.com "(PDF) Music Artist Classification with Convolutional Recurrent Neural Networks"
[3]: https://www.ee.columbia.edu/~dpwe/LabROSA/projects/timbrechroma/?utm_source=chatgpt.com "Artist Identification of Music Audio by Timbral and Chroma Features in Matlab"
[4]: https://dihana.cps.unizar.es/proceedings/ICASSP/2020/pdfs/0000001.pdf?utm_source=chatgpt.com "ADDRESSING THE CONFOUNDS OF ACCOMPANIMENTS IN SINGER IDENTIFICATION"
[5]: https://scikit-learn.org/stable/modules/permutation_importance.html?utm_source=chatgpt.com "5.2. Permutation feature importance — scikit-learn 1.9.0 documentation"
[6]: https://rasbt.github.io/mlxtend/user_guide/evaluate/mcnemar/?utm_source=chatgpt.com "mcnemar: McNemar's test for classifier comparisons - mlxtend"
[7]: https://www.frontiersin.org/journals/immunology/articles/10.3389/fimmu.2026.1602846/full?utm_source=chatgpt.com "Frontiers | ProMMF_Kron: a multimodal deep learning model for immunotherapy response prediction in stomach adenocarcinoma"
[8]: https://sia.lnk.to/thisisacting?utm_source=chatgpt.com "Sia - This Is Acting (Deluxe Version)"
[9]: https://sia.lnk.to/listen?utm_source=chatgpt.com "Listen Now"
[10]: https://www.allmusic.com/album/this-is-acting-mw0002894912?utm_source=chatgpt.com "Sia: This Is Acting Tracks & Reviews | AllMusic"
[11]: https://en.wikipedia.org/wiki/Unstoppable_%28Sia_song%29?utm_source=chatgpt.com "Unstoppable (Sia song)"
[12]: https://www.ksut.org/music/2018-11-06/sia-is-the-21st-centurys-most-resilient-songwriter?utm_source=chatgpt.com "Sia Is The 21st Century's Most Resilient Songwriter | KSUT Public Radio"
[13]: https://en.wikipedia.org/wiki/Tori_Amos?utm_source=chatgpt.com "Tori Amos"
[14]: https://www.tandfonline.com/doi/full/10.3109/14015439.2016.1156737?utm_source=chatgpt.com "Freddie Mercury—acoustic analysis of speaking fundamental frequency, vibrato, and subharmonics: Logopedics Phoniatrics Vocology: Vol 42 , No 1 - Get Access"
[15]: https://www.theguardian.com/music/2011/jun/24/stevie-nicks-in-your-dreams?utm_source=chatgpt.com "Stevie Nicks: In Your Dreams – review | Pop and rock | The Guardian"
[16]: https://en.wikipedia.org/wiki/Madonna?utm_source=chatgpt.com "Madonna"
[17]: https://www.theguardian.com/music/2019/dec/13/marie-fredriksson-obituary?utm_source=chatgpt.com "Marie Fredriksson obituary | Pop and rock | The Guardian"
[18]: https://www.allmusic.com/album/pearls-of-passion-mw0000462822?utm_source=chatgpt.com "Roxette: Pearls of Passion Tracks & Reviews | AllMusic"
[19]: https://www.sciencedirect.com/science/article/pii/016763939190035R?utm_source=chatgpt.com "Experiments of automatic speaker recognition in open sets - ScienceDirect"
[20]: https://aclanthology.org/2026.eacl-srw.10/?utm_source=chatgpt.com "Quality-Aware Adversarial Ensemble for Singer Identification in 1960s Tamil Film Music - ACL Anthology"
[21]: https://proceedings.mlr.press/v70/guo17a.html?utm_source=chatgpt.com "On Calibration of Modern Neural Networks"
[22]: https://www.sciencedirect.com/science/article/abs/pii/S0893608018301850?utm_source=chatgpt.com "Survey and experimental study on metric learning methods - ScienceDirect"
