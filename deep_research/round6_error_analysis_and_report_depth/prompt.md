# Deep Research prompt — error analysis methodology, the Sia out-of-distribution result, and report depth

For relaying to Deep Research (a web-research-capable agent). Context for
whoever answers this:

## Problem statement

We built a singer/artist identification system on the **Artist20** dataset
for a graduate-course assignment: 20 artists, 949 training tracks, 16kHz
mono full songs, album-level train/val/test split. The **20 trained artists,
exactly as labeled in our data**, are:

    aerosmith, beatles, creedence_clearwater_revival, cure,
    dave_matthews_band, depeche_mode, fleetwood_mac, garth_brooks,
    green_day, led_zeppelin, madonna, metallica, prince, queen, radiohead,
    roxette, steely_dan, suzanne_vega, tori_amos, u2

Two graded parts: **Task 1** (traditional ML — hand-crafted features into
kNN/SVM/RandomForest; graded entirely on analysis quality, not accuracy)
and **Task 2** (deep learning from scratch — no pretrained encoder eligible
as the graded submission). Final grading is Report (50%) + Prediction (50%,
formula: top1 + 0.5×top3 on a 233-track held-out test set). We currently
have: a 9-classifier Task-1 comparison, 17 from-scratch Task-2 architectures/
ablations, 2 pretrained-encoder baselines, a zero-shot audio-LLM baseline,
a 7-model weighted ensemble as the graded Task-2 submission (val
top1=0.861, top3=0.913), and a mel-spectrogram + inference demo on a
user-supplied out-of-distribution clip (see part 2 below). Full results:
this project's `MATERIALS.md`/`EXPERIMENT_LOG.md` (not attached here, but
available if you need more context than what's summarized below).

We have three separate asks. Please answer them as clearly separated
sections, not blended together.

---

## Part 1: How should we do error analysis for this task specifically?

We already have, per model: confusion matrices, top1/top3 accuracy, and (for
a few models) per-artist accuracy deltas and pairwise ensemble-diversity
metrics (Cohen's kappa, disagreement rate, oracle-ensemble upper bound). We
have NOT yet done a systematic "why does the model confuse artist X with
artist Y" analysis tied to musicological or acoustic reasoning.

1. What's the standard/best-practice methodology for error analysis in
   artist/singer identification specifically (as opposed to generic
   multi-class image classification error analysis, which is a different
   literature)? We're looking for citable approaches — does Hsieh et al.
   (the paper this dataset's album-split methodology comes from, "Addressing
   the confounds of accompaniments in singer identification," ICASSP 2020)
   or any other Artist20 paper do a specific documented error-analysis
   procedure worth replicating?
2. Concretely, for a 20-class confusion matrix on ~950 training tracks: is
   there a standard way to explain *why* two artists get confused beyond
   "the numbers say so" — e.g. grouping by vocal range/register (bass/
   baritone/tenor/alto/soprano), genre/era, gender, band vs. solo act,
   production style (heavily processed vs. raw), or something else
   established in the MIR/speaker-verification literature? We'd like a
   concrete checklist we can actually apply to our own confusion matrices,
   not just a general concept.
3. Given Task 1's SVM/kNN/RandomForest features are hand-crafted and
   individually named (MFCC, chroma, spectral contrast, ZCR, tonnetz, etc.),
   is there a standard way to connect *feature importance* (e.g. permutation
   importance, already computed) back to *which pairs of artists* a given
   feature helps distinguish, to make the Task 1 analysis (which is what's
   actually graded, not its raw accuracy) more concrete and artist-specific
   rather than just a global importance ranking?
4. Is there a standard statistical-significance method appropriate for a
   231-track validation set — we've been comparing ensemble variants that
   differ by as little as +0.4-0.5 percentage points (each val track is
   worth ~0.43pp, so small differences may not be meaningful). Is a
   bootstrap confidence interval or McNemar's test the right tool here, and
   is there a citable convention for reporting this in a singer-ID or
   similar small-eval-set classification paper?

## Part 2: Describing the "Unstoppable" / Sia out-of-distribution result

As a bonus/qualitative probe (not part of the graded submission), we ran our
best from-scratch model on a Voice Memos recording of someone singing along
to Sia's "Unstoppable" — Sia is **not** one of the 20 trained artists, so
this is a genuine out-of-distribution input, and there is no "correct"
answer among the 20 labels. The model's top-3 guesses (softmax probability,
our strongest single from-scratch model, `sota_crnn_wide` — a CNN+GRU
architecture trained on mel-spectrograms) were:

    1. tori_amos      (13.7%)
    2. queen           (12.9%)
    3. fleetwood_mac   (12.4%)

(For reference, three other/earlier models we also ran this same clip
through gave different top-3s: `sota_crnn` — madonna 22.4%, fleetwood_mac
12.4%, roxette 10.8%; a frozen-ECAPA-TDNN speaker-embedding baseline —
fleetwood_mac 46.9%, roxette 23.2%, tori_amos 15.9%; `confound_crnn` —
madonna 45.8%, radiohead 17.5%, prince 11.4%.)

We want to write an informed, musicologically-grounded paragraph in our
report about *why* these particular artists (not others from the 20-artist
list above) came up as the nearest matches for Sia's voice on "Unstoppable,"
across these different models. Please research and tell us:

1. What are Sia's actual vocal characteristics on "Unstoppable" specifically
   (or her voice generally, if that's better documented) — vocal range/
   register, timbre descriptors (breathy, powerful, raspy, belting,
   vibrato use), genre/production style of this specific song (it's from
   the "Fifty Shades of Grey" soundtrack — is it typically described as pop,
   pop-soul, or something else)?
2. Of the specific artists our models picked (tori_amos, queen [i.e. Freddie
   Mercury's voice, presumably what the model is keying on], fleetwood_mac
   [likely Stevie Nicks], madonna, roxette [Marie Fredriksson]), which ones
   have a *documented, sourced* vocal/stylistic similarity to Sia (vocal
   range overlap, similarly powerful/belting delivery, similar emotional
   dynamic range, etc.) — and which picks look more like an artifact of
   *production style or genre* similarity (e.g. anthemic pop production)
   rather than voice similarity per se? We want to be able to say something
   more specific than "the model picked a female vocalist," ideally citing
   vocal-range/style comparisons if any exist (professional vocal analysis,
   music journalism, or academic MIR work comparing these specific artists).
3. Is there any published/documented general finding about what
   out-of-distribution singer classifiers tend to key on when given a voice
   outside their training set — e.g. do such systems more often converge on
   *vocal range/timbre* similarity or *production/genre* similarity when
   forced to pick a "nearest" answer? This would help us frame whether our
   models' picks are more likely "hearing" something real about Sia's voice
   or just latching onto surface production cues.

## Part 3: What else should we do for comprehensive report analysis?

Beyond error analysis and the Sia demo, what other analyses would
substantially strengthen a report like this, given the grading is 50% report
quality/completeness? We're not asking for more architectures to train (we
already have 17) — we're asking what *analysis* of what we already have
would be most valuable to add. Some candidate directions, tell us which are
worth pursuing and if there's anything we're missing entirely:

1. Statistical rigor on model comparisons (see Part 1, #4) — bootstrap CIs
   or significance tests on the reported ensemble improvements.
2. Calibration analysis — are our models' confidence scores well-calibrated
   (e.g. reliability diagrams, Brier score), especially relevant given the
   Sia demo's flat-vs-confident-but-wrong contrast between models.
3. Per-artist / per-album breakdown of accuracy, specifically checking
   whether errors cluster by album (a residual sign of the album/production
   confound this dataset's split is designed to prevent) even after our
   best models — is there a standard diagnostic for "did we actually solve
   the confound, or just get better at learning it more subtly"?
4. Embedding-space analysis beyond t-SNE (which we already have) — e.g.
   is there value in comparing embedding geometry (e.g. average pairwise
   cosine distance between same-artist vs. different-artist embeddings)
   across our different architectures to quantify representation quality
   more rigorously than accuracy alone?
5. Any standard "ablation summary" presentation convention from published
   MIR/audio classification papers worth following, given we have ~19
   measured architecture/ablation results to present clearly rather than as
   a wall of numbers.

Please prioritize concrete, actionable, and — where you make a specific
factual claim (a vocal-range comparison, a stated best practice, a cited
methodology) — sourced answers over general/plausible-sounding prose. Earlier
rounds of this research surfaced both outright-uncited numeric claims and at
least one real-but-mismatched citation (a real paper cited as precedent for
a task it doesn't actually address); we test or verify every claim before
acting on it rather than trusting or dismissing on priors, so an unsourced
or mismatched claim just costs us verification time rather than misleading
us — but a clearly sourced one saves us that time.
