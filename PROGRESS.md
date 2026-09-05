# DIME Recreation — Progress Notes

## What this project is

Recreating the DIME method (frozen-GPT-2 + compressed retrieval memory) from scratch,
by hand, for real understanding — not copying code from the reference implementation.
A working (but not-fully-understood) version already exists at `C:/Projects/TrackA/TrackA`
on this machine — that repo is for **read-only reference** ("what did the original do
here?"), never to be copied wholesale. Everything in *this* repo is being hand-written
and explained step by step.

## The roadmap (Phases A–F)

**Phase A — Foundation**
1. Frozen hidden-state extraction (toy, single example) — **DONE, verified**
2. Batched extraction over a real streamed dataset (TinyStories) — **DONE, verified**
3. Three-way split with no leakage (datastore / controller_train / val) — **NOT STARTED, next up**

**Phase B — Naive end-to-end memory (kNN-LM baseline)**
4. GPT-only NLL (reference number)
5. Full raw kNN baseline (every datastore example is its own retrievable unit)
6. Read-time mixing formula, fixed hyperparameters (`k`, `tau`, `alpha` — the classic
   kNN-LM hyperparameters, `alpha` = kNN-LM's `lambda`)

**Phase C — Compress the memory (the actual DIME idea)**
7. `random_partition` (sanity control)
8. `minibatch_kmeans` (default construction method)
9. `utility_weighted` (weight clusters by GPT's own NLL/entropy)
10. `query_kmeans` (cluster the query side, not the datastore side)

**Phase D — Learn the read policy**
11. Fixed-hyperparameter grid search
12. Q-read controller (fitted-Q MLP, `reward = NLL_GPT − NLL_action`)
13. Apply same Q-read code to both raw memory and DIME memory (fairness)

**Phase E — Comparison baselines at equal budget**
14. 6 equal-budget raw variants (`raw_random`, `raw_high_gpt_loss`, etc.)

**Phase F — Prove it's not an illusion**
15. State-object ablations (majority-token, top-k truncation, shuffled)
16. Efficiency analysis (entry-count, latency, bytes)
17. Rigor (significance testing, real byte measurements — not just formulas)

Everything past Phase F (more datasets/models) is replication, not new mechanism.

## Current file structure (`GPT_Module/`)

- **`gpt_class.py`** — `FrozenGPT2` class. Owns the frozen model only: `__init__`
  (loads tokenizer + model, `.eval()`, `.requires_grad_(False)`, moves to device),
  `encode_text(text)` (tokenize → `[1, L]` tensor), `forward(input_ids)` (one forward
  pass under `no_grad()`, returns last-layer `hidden` and `logits`). Deliberately dumb —
  no math, no batching logic, no dataset awareness. Works unchanged for batch size 1
  or N.

- **`helpers.py`** — pure tensor math, no model needed:
  - `shift_for_next_token(hidden, logits, input_ids)` — realigns hidden state at
    position `i` with the token at position `i+1` (its true target), by chopping the
    last position off `hidden`/`logits` and the first position off `input_ids`.
  - `true_token_stats(lgts_pred, y_target)` — softmax → gather the true token's
    probability → `-log()` for NLL.
  - `collect_chunks(dataset, tokenizer, seq_len, n_chunks_needed)` — streams a
    HuggingFace dataset, concatenates each story's tokens (+ `eos_token_id` as a
    boundary marker) onto a running buffer, slices off fixed-length chunks as the
    buffer fills. Chunks can straddle two stories — that's expected at this stage,
    it's exactly the leakage Phase A3 exists to fix.

- **`extract.py`** — orchestration layer (uses both the model and the math together):
  - `run_batch(gpt, chunks)` — stacks a `list[list[int]]` of equal-length chunks into
    one `[B, L]` tensor (on `gpt.device` — chunks are plain Python ints, so this must
    be explicit or it silently defaults to CPU and breaks on a GPU model), runs
    `gpt.forward()`, then `shift_for_next_token` / `true_token_stats`. Returns
    `h_pred, y_target, p_true, nll`.

- **`test_gpt.py`** — the manual test/proof script. Currently: loads TinyStories,
  collects 10 chunks of `seq_len=16`, runs `run_batch`, prints shapes + mean NLL +
  a per-position token/NLL table for chunk 0. Also still has the original A1 toy
  single-sentence test ("the cat sat on the mat") below it.

- **`requirements.txt`** (repo root) — `torch`, `transformers`, `datasets`, `numpy`,
  `pandas`, `tqdm`, `scikit-learn`. Covers the whole roadmap, not just Phase A.

## What's been verified (real output, not just "ran without crashing")

**A1 (toy sentence):** `input_ids.shape == [1, 6]`, per-position NLL table for
"the cat sat on the mat" — function words (`on`, `the`) get low NLL, content words
(`cat`, `sat`, `mat`) get high NLL because many alternatives were equally plausible
given the context available. Confirmed the shift direction is correct.

**A2 (batched real data):** `collect_chunks` produces chunks of exactly the requested
length. `run_batch` on 10 real TinyStories chunks (`seq_len=16`) gave
`h_pred.shape == [10, 15, 768]`, `nll.shape == [10, 15]`, mean NLL ≈ 3.86 (sane for
short-context windows). Decoded chunk-0 tokens matched the real story text exactly
("One day, a little girl named Lily found a needle in her room. She...") and the
same function-word-vs-content-word NLL pattern held. Confirms the whole pipeline
(chunking → batching → shift → stats) preserves correct token order and alignment.

Note: A2 was tested at toy scale (10 chunks, `seq_len=16`). Scaling to the real
`seq_len=128` and thousands of chunks later is just a parameter change, not a new
mechanism — don't re-litigate the design, just bump the numbers when building the
real extraction run.

## What's next: Phase A3 — leak-proof three-way split

**The problem:** `collect_chunks` chunks a token *stream*, so a single chunk can
straddle two different stories, and two chunks from the *same* story could later be
randomly assigned to different splits (e.g. one to `datastore`, one to `val`). That's
"group leakage" (same family as `GroupKFold` in scikit-learn) — the memory system
could "retrieve" a passage it's literally seen, from the same story it's being
evaluated on, making retrieval look artificially good.

**The fix:** decide per **story**, before any tokenizing happens, which of three
buckets it belongs to (e.g. 70% datastore / 15% controller_train / 15% val via one
random draw per story). Every chunk that story produces inherits that bucket.
Track `story_id` per chunk so leakage can be audited afterward (confirm no `story_id`
appears in more than one split's chunks).

**First small step (agreed, not yet started):** prove the random per-story assignment
logic alone, in isolation — loop over ~500 stories, draw one random number per story,
assign to `ds`/`ct`/`val` by the target fractions, count how many land in each bucket,
confirm the proportions roughly match 70/15/15. Don't tokenize or chunk anything yet
in this first step.

## Working conventions established in this project

- **Never declare a step "done" from code review alone** — always run it and look at
  real printed output (shapes, decoded tokens, NLL values) before moving on. Several
  bugs (a `__int__`/`__init__` typo, a `{}` set-literal instead of parens around a
  device conditional, an `ens_token_id`/`eos_token_id` typo) were only caught this way.
- **Separation of concerns across files:** `gpt_class.py` = model mechanics only.
  `helpers.py` = pure math/data functions, no model dependency. `extract.py` =
  orchestration that combines both. Keep new code in the file matching its actual
  job, not wherever's convenient in the moment.
- User prefers to type/apply code changes themselves after seeing an explained
  snippet, rather than having files edited directly — except for small, explicitly
  requested utility files (e.g. `requirements.txt`) or when explicitly asked to "just
  run/fix it."
- Reference repo `C:/Projects/TrackA/TrackA` is real, git-initialized, and has a
  `dime_paper/DIME_CHECKLIST.md` plus a `study/` folder (`DIME_STUDY_GUIDE.md`,
  `DIME_PROFESSOR_QA.md`, cheatsheet) — useful background reading, but this recreation
  project is intentionally not copying its code.
