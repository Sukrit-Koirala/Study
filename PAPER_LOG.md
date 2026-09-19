# DIME Paper — Writing Log

Companion to [PAPER_CONTEXT.md](PAPER_CONTEXT.md). That file is the research context
dump (findings, numbers, open experimental issues) — treat it as close to read-only,
regenerate it if the underlying results change materially. **This file is the
writing-process log**: what's actually been drafted, decisions made about structure/
wording, and open TODOs specific to the paper document itself. Update this file (not
PAPER_CONTEXT.md) whenever a writing session ends, so the next session/computer/Claude
instance picks up where you left off instead of re-deriving state.

Sync via whatever you're already using for the rest of the repo (git). Both files live
at repo root so they travel together.

---

## Current status (2026-09-10)

Paper not yet started. No sections drafted.

## Section-by-section draft status

- [ ] Abstract — not started (write last)
- [~] Introduction / Motivation — draft v2 written in-chat (2026-09-10), not yet
      pasted into Notepad file. Covers: retrieval-memory cost problem, prior
      compression ceiling (~5x, He et al.), DIME's distributional-collapse pitch
      (~99-100x), rigor requirements (equal-budget/illusion-check/significance),
      5-bullet contributions list. **Writing fresh, NOT porting from old draft
      `final.pdf`** (decided 2026-09-10 — see Open decisions).
- [~] Related Work — draft written in-chat (2026-09-10), not yet pasted into Notepad
      file. Covers: Khandelwal 2020 (foundational), RETRO/Memorizing Transformers/
      Xu&Alon (framing), RetoMaton + Martins et al. (datastore efficiency), He et al.
      2021 (closest comparator, full mathematical distinction included — see
      Changelog), He et al./Xu et al. (adaptive retrieval precedent for Q-read).
      Citations are author-year placeholders — needs real BibTeX before this compiles
      anywhere but Notepad.
- [~] Method — draft written in-chat (2026-09-11/12), not yet pasted into Notepad
      file. Covers: kNN-LM background/notation, DIME construction (clustering →
      prototype+distribution per cluster), explicit note that building the raw
      datastore D is still a full one-time O(N) forward pass (compression only
      applies to what's retained/queried after, not construction compute), the
      generalized two-level mixing formula, optional beta Katz-backoff, multi-action
      Q-read (verified against code: `q_read_dense.py`'s `build_obs_features` is
      genuinely 11-dim, not 8 — checked `run_q_read_multi_action_baseline.py`, no
      subsetting happens, matches PAPER_CONTEXT §6 exactly).
- [~] Setup — draft written in-chat (2026-09-12), not yet pasted into Notepad file.
      Formalized: NLL metric, splits, tuning rule, equal-budget baselines (with
      |D'|=B stated explicitly), illusion-check ablations, significance testing with
      paired-difference notation at both position level and seed level.
- [ ] Results — not started (4/6 grid settings ready; WikiText-2 pending re-run)
- [ ] Discussion — not started
- [ ] Limitations — not started
- [ ] Conclusion — not started
- [ ] Math formalization appendix — in progress separately, see `MATH_FORMALIZATION.md`

## Open decisions (not yet made — flag when resolved)

- Target venue / page limit / template (unknown — affects how much of Tier 3
  enrichment and the rich ablation tables can be included vs. cut to appendix)
- Whether WikiText-2 ships in the final paper at all, or gets dropped if the re-run
  doesn't land in time (currently blocking on GPT_Module data re-extraction)
- Whether multi-action Q-read (currently TinyStories-only) is presented as a full
  result or a "future work" teaser, given it's not yet run on the 3x2 grid

## Post-paper ideas (not for this paper — parked for later)

- **Learned representation for clustering, instead of raw GPT-2 hidden states.**
  Currently all four construction methods (`state_object.py`:
  `minibatch_kmeans_partition`, `query_kmeans_partition`,
  `utility_weighted_partition`, `random_partition`) cluster directly on raw hidden
  states, full dimensionality, no PCA or any other transform. Idea (2026-09-10):
  train a small network to project hidden states into a space explicitly shaped for
  clustering — e.g. contrastively, pulling together points that share a next-token
  distribution — rather than relying on whatever geometry the frozen LM's residual
  stream happens to have. Related but distinct from Martins et al.'s contrastive
  "compact network" (that one reduces dimension for storage/speed within the
  existing voting mechanism; this one would reshape cluster geometry itself,
  upstream of DIME's distributional collapse). Risk: adds a training step and a new
  overfitting surface on `controller_train`. Explicitly deferred — not blocking the
  current paper, revisit after submission.

## Changelog

- 2026-09-10: File created. PAPER_CONTEXT.md read and confirmed current. No paper
  content drafted yet.
- 2026-09-10: Decided NOT to port Motivation/Method from the old draft
  (`C:\Users\Acer\Documents\DIME\final.pdf`) — writing both sections from scratch
  instead, despite PAPER_CONTEXT §1 noting they'd be reusable. Added Related Work
  and Conclusion as explicit sections (weren't broken out before). Drafting in
  Notepad for now, plain text, synced via the repo like everything else.
