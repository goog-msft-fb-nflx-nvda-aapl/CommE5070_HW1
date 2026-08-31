# Prompt for Claude web — building the final HTML slide-deck report

Copy everything below into a new Claude web conversation, and attach/upload
`CommE5070_HW1_report_materials.zip` (in the same folder as this file) to
that conversation.

---

## Task

Build `R13921031_report.html`: a presentation-slide-deck report in HTML,
**16:9 aspect ratio**, for a graduate course assignment (CommE5070, "Deep
Learning for Music Analysis and Generation," HW1 — Singer Classification on
the Artist20 dataset). This report is 50% of the assignment grade (the
other 50% is the test-set prediction score) and is graded on clarity and
completeness — per the assignment spec itself: "create a report that is
clear and can be understood without the need for oral explanations."

Everything you need is in the attached zip. Read `MATERIALS.md` first — it
is the curated, primary source, written specifically for this report-
assembly step, and already states clearly what's measured vs. what's
sourced-from-literature vs. what's flagged as unverified/discounted. Then
`README.md` and `Hw1_Singer_Classification_ASSIGNMENT_SPEC.md` for the
problem statement and grading rubric, `TA_discussion.md` for clarifications
that shape what counts as in-scope, and `EXPERIMENT_LOG.md` if you want the
full chronological narrative behind any specific decision (it's long — treat
it as a reference to dip into, not something to read end-to-end).

## Zip contents map

```
report_materials/
  MATERIALS.md                              <- primary source, read first
  README.md                                 <- problem statement, setup, model roster
  readme                                    <- TA-facing inference instructions (graded submission command)
  EXPERIMENT_LOG.md                         <- full chronological dev log (reference, not primary)
  TA_discussion.md                          <- TA clarifications (Task 1 grading, Task 2 from-scratch rule)
  Hw1_Singer_Classification_ASSIGNMENT_SPEC.md  <- the assignment doc itself
  requirements.txt
  data_index/labels.json                    <- the 20 trained artist names, exact spelling
  deep_research/                            <- prompts only (not full raw responses -
                                                already synthesized into MATERIALS.md);
                                                useful if you want to see exactly what
                                                question motivated a given experiment
  src/                                       <- full source code (models, training,
                                                evaluation, ensembling, all analysis
                                                scripts) - use this for architecture
                                                accuracy, not just MATERIALS.md's prose
  results/                                   <- every model's confusion matrix, t-SNE
                                                plot, training history, and every
                                                analysis script's output (JSON + PNG):
                                                calibration/reliability diagrams,
                                                embedding geometry, significance testing,
                                                pair-conditional feature importance,
                                                ensemble diversity, vocal-separation
                                                attribution, the own-voice demo's
                                                mel-spectrogram, etc. Checkpoints
                                                (*.pt) and cached feature arrays
                                                (*.npz) are intentionally excluded -
                                                not needed for a report and just bulk.
```

## Requirements for the report itself

1. **Explicit and self-contained.** Assume a reader who knows general ML but
   not this specific project's conventions. Every technical term that isn't
   universally standard (SpecAugment, ArcFace/AAM-Softmax margin loss,
   Attentive Statistics Pooling, ECE, McNemar's test, silhouette score,
   FGNL/non-local blocks, the album-level split and why it exists, chunk-
   level vs. song-level prediction, etc.) needs a short inline explanation
   the first time it's used — not just the term dropped in. Don't assume the
   reader has read `MATERIALS.md`.
2. **References must be properly annotated, not just listed.** Every
   architecture, technique, or dataset claim that traces to a specific paper
   or repo should carry an inline citation marker at the point it's used
   (e.g. "a bidirectional GRU with attention pooling [3]"), resolving to a
   numbered references slide at the end with full paper titles/venues/years
   and, where available, links. `MATERIALS.md`'s own "## Citations" section
   is the vetted core list — extend it with anything else you cite, but
   don't cite anything MATERIALS.md explicitly flagged as discounted/
   low-quality (a couple of round-6 deep-research claims were traced to a
   fan blog and a mismatched-topic paper and are called out as such — do
   not present those as if they were reliable sources).
3. **Illustrate, don't just describe, the architectures and data pipeline.**
   Model architecture diagrams (input → conv/GRU/attention/classifier
   stages, with tensor shapes at each stage where known) and the data-
   processing pipeline (raw audio → chunking → mel-spectrogram → 
   augmentation → model → chunk-level softmax → song-level mean-pooling)
   should be actual diagrams (inline SVG, or clean HTML/CSS box-and-arrow
   layouts), not paragraphs of prose. `src/models/*.py` has the exact layer
   configurations (channel counts, kernel sizes, GRU hidden dims, etc.) if
   you need precision beyond what `MATERIALS.md` states in prose — use it
   for accuracy rather than approximating from memory of the architecture's
   name alone. At minimum, illustrate: the winning single-model architecture
   (`sota_crnn_wide`), the winning ensemble's composition, the album-level
   train/val/test split concept (why it prevents the production confound),
   and the song-level chunk-aggregation evaluation convention.
4. **Every number must be sourced from the materials, not invented or
   remembered/approximated.** Pull exact figures from `MATERIALS.md`'s
   tables and the `results/*/summary.json` / `results/analysis/*.json`
   files, don't round or restate from a vague recollection. Where
   `MATERIALS.md` explicitly says something is uncertain, noisy, or not
   statistically significant (e.g. the incremental ensemble steps this
   session, or the negative permutation-importance estimates on small
   binary subsets), the slide must say so — do not smooth this into a
   cleaner-sounding claim than what was actually measured. Where a response
   from the deep-research rounds offered an interpretation that was flagged
   as unsourced, speculative, or a mismatched citation, either omit it or
   present it explicitly as "one AI research assistant's unverified claim,"
   never as an established fact.
5. **Structure**, matching the assignment's own required sections (see
   `Hw1_Singer_Classification_ASSIGNMENT_SPEC.md` for the exact grading
   breakdown — Report 50% / Prediction 50%, formula
   top1 + 0.5×top3): title/overview, problem statement, dataset, Task 1
   (traditional ML — remember the TA said its *accuracy doesn't count*,
   only the analysis quality, so the feature-ablation/permutation-importance
   work should get real slide space, not just the accuracy number), Task 2
   (deep learning from scratch — architectures tried, training recipe,
   results table, the winning ensemble and its trajectory across this
   session with the significance-testing caveat), error analysis
   (calibration, embedding geometry, pair-conditional feature importance,
   vocal-separation attribution), the own-voice demo (mel-spectrogram +
   the Sia/"Unstoppable" musicological discussion — `MATERIALS.md`'s
   "Visualizations" section has a report-ready paragraph for this already,
   with sourced vs. speculative claims clearly separated — preserve that
   distinction), baselines (pretrained-encoder and zero-shot audio-LLM
   comparisons, clearly marked as not-the-graded-submission per the TA
   rule), and references.
6. **Student ID**: R13921031 (used in filenames throughout the materials —
   `R13921031.json`, etc.).

Ask me (in this chat) if anything in the zip is ambiguous or missing before
guessing — the materials were prepared specifically for this handoff, so a
gap is more likely something to flag than something to fill in yourself.
