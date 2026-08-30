# Deep Research prompts and responses

Six rounds of externally-run research, used to inform architecture, data,
and training-recipe decisions in this project. Each round is a prompt we
wrote (`prompt.md`), relayed to one or more research assistants (Gemini,
Perplexity, ChatGPT, and/or Qwen — whichever we had access to for that
round), with their raw responses (`response_<engine>.md`) kept alongside it
for traceability. See `MATERIALS.md` at the repo root for how each round's
findings were synthesized into our actual decisions and experiments.

- **round1_sota_and_architecture_survey** — architectures beyond CRNN/non-
  local, augmentation/regularization strategies for the album-level confound,
  and singing-voice-specific SSL checkpoints worth considering.
- **round2_sota_context_and_perartifact_ablations** — Artist20 SOTA context
  for our results, plus specific diagnostic ablations per method we'd
  already implemented (feature-group ablation, encoder-vs-head analysis,
  ensemble diversity, vocal-separation error attribution).
- **round3_from_scratch_improvement** — literature sanity-check on our
  from-scratch training recipe, and the highest-leverage from-scratch-
  eligible techniques to try next (this round's answer — cross-song
  vocal/instrumental remixing — was implemented directly).
- **round4_ssl_pretraining_recipe** — a concrete, from-scratch-eligible
  self-supervised (SimCLR-style) contrastive pretraining recipe, since a
  pretrained encoder isn't eligible as the graded submission for this
  assignment.
- **round5_prior_year_gap_and_latest_literature** — problem statement,
  current pipeline/architectures, and a specific gap found while
  re-investigating our own prior-year submission for this assignment
  (attention pooling and per-sample mel normalization were only tested on a
  smaller backbone than the one that originally used them); asks for
  literature grounding on that gap plus the latest (2023-2026) ICASSP/
  Interspeech/AAAI/NeurIPS work on from-scratch, small-N singer/artist ID.
- **round6_error_analysis_and_report_depth** — three-part ask for the final
  report: (1) error-analysis methodology for singer/artist classification
  specifically, including whether small-eval-set (231 tracks) model
  comparisons need a significance test; (2) a musicologically-grounded
  explanation for why our models' out-of-distribution guesses for a Sia
  "Unstoppable" clip landed on the specific artists they did, given the
  20-artist label list; (3) what further analyses (beyond more training
  runs) would most strengthen the report. All four engines (ChatGPT, Gemini,
  Perplexity, Qwen) independently converged on paired-bootstrap/McNemar
  significance testing as top priority — implemented directly
  (`src/analysis_significance.py`) and found our incremental ensemble gains
  this session don't individually clear significance, only the overall
  ensemble-vs-single-model effect does. Also surfaced a citation-quality
  split on Part 2: ChatGPT caught our own prompt's factual error (the wrong
  Sia album), Perplexity's independent sourcing implicitly agreed, Gemini
  and Qwen both repeated it uncritically; Gemini separately sourced several
  "documented" vocal-similarity claims to a fan blog and a headphone-review
  site.

Some responses cite hyperparameters or claims without a traceable source;
where that happened, we noted it explicitly in `MATERIALS.md`/
`EXPERIMENT_LOG.md` and tested the claim rather than either trusting or
dismissing it outright.
