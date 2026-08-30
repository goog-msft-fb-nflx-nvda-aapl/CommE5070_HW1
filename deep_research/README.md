# Deep Research prompts and responses

Five rounds of externally-run research, used to inform architecture, data,
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

Some responses cite hyperparameters or claims without a traceable source;
where that happened, we noted it explicitly in `MATERIALS.md`/
`EXPERIMENT_LOG.md` and tested the claim rather than either trusting or
dismissing it outright.
