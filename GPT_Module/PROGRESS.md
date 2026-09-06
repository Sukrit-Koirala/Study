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
3. Three-way split with no leakage (datastore / controller_train / val) — **DONE, verified at toy scale**

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

- **`dataset_helpers.py`** — per-story split logic, no model needed:
  - `assign_story_split(rand_val, frac_datastore=0.70, frac_controller=0.15)` — maps
    one random draw in `[0,1)` to a split name (`datastore` / `controller_train` /
    `val`) via fixed cutoffs. Dumb on purpose: doesn't know about stories or tokens.
  - `collect_chunks_split(dataset, tokenizer, seq_len, n_chunks_needed, ...)` — streams
    the dataset once; for each story draws one random number (via a local
    `random.Random(seed)`, not global `random.seed()`) to decide its split, then that
    story's tokens only ever flow into the buffer for that split — three separate
    buffers/chunk-lists/story-id-lists (one per split) instead of one shared buffer.
    Returns `chunks` (dict of split → list of chunks) and `chunk_story_ids` (dict of
    split → parallel list of the story_id each chunk came from, for leakage audits).
    `n_chunks_needed` is now a dict, e.g. `{"datastore": 350, "controller_train": 75,
    "val": 75}`, since each split needs a different target count.

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

## Phase A3 — leak-proof three-way split (DONE, verified at toy scale)

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

**Step 1 (done):** proved the random per-story assignment logic alone, in isolation —
looped over 500 dummy story ids, drew one random number per story via
`assign_story_split`, counted how many landed in each bucket: `{'datastore': 349,
'controller_train': 68, 'val': 83}`. Close enough to 350/75/75 (within ~1 standard
deviation for n=500 at p=0.15) — confirmed the split logic works before touching any
real tokenization.

**Step 2 (done):** wired per-story assignment into the real story-loading loop via
`collect_chunks_split` (see `dataset_helpers.py` above) — one pass over the dataset,
three separate buffers so a story's tokens only ever land in one split's buffer.
Tested with small targets (`{"datastore": 20, "controller_train": 5, "val": 5}`) and
ran the leakage audit (group `chunk_story_ids` by split, check no `story_id` appears
under more than one split): **`leaks found: []`** — confirmed clean, no story crosses
split boundaries.

**Not yet done:** running this at real scale (real `seq_len`, thousands of chunks per
split) — per the established pattern (see A2 note above), that's just a parameter
change, not a new mechanism, and will likely happen alongside building the real
datastore/controller_train/val for Phase B.

## Phase B step 4 — GPT-only NLL baseline (DONE, verified)

Ran the frozen GPT-2 alone (no memory/retrieval) over the `val` split
(`seq_len=128`, `n_chunks={"datastore": 350, "controller_train": 75, "val": 75}`) via
`run_gpt_baseline` (`extract.py`) + `save_results` (`results_io.py`), submitted as a
SLURM job (`submit_baseline.sh`) on an L40S node. Real output:

```
GPU: NVIDIA L40S
mean NLL on val: 2.798432154698415
```

Finished in under 2 minutes — plenty of headroom to scale up chunk counts later
without worrying about runtime. Result + per-chunk NLL saved to
`GPT_Module/results/gpt_only_baseline.json`. This is the reference number every later
memory variant (raw kNN, compressed DIME memory, etc.) must beat — if a
memory-augmented setup doesn't get *below* 2.798, the memory isn't helping.

Also established this run: **SLURM workflow** — code lives in a git repo, pushed
locally and `git pull`ed on the cluster at
`~/ondemand/upload_me/Research/Study/GPT_Module`; `sbatch submit_baseline.sh` from
that directory submits, `squeue -u sukrit.koirala` checks status, `logs/*.out`/`*.err`
hold output (must `mkdir -p logs` once before first submission, since SLURM needs the
output path to exist upfront and git doesn't track empty dirs).

## Phase B step 5 — Full raw kNN baseline (DONE, verified at toy scale)

Built the raw (uncompressed) retrievable memory: every datastore position is its own
`(key, value)` entry — key = hidden state, value = the true next token at that
position. Two new pieces:

- **`knn.py`** — pure retrieval math, no model needed: `build_datastore(keys, values)`
  wraps `sklearn.neighbors.NearestNeighbors` (brute-force/tree search — fine at current
  scale, revisit only if/when this becomes the bottleneck), `query_knn(index, values,
  query_keys, k)` returns `[M, k]` distances and retrieved token ids.
- **`extract.py` → `build_datastore_from_chunks(gpt, chunks, batch_size=256)`** —
  orchestration: runs `run_batch` over chunks in batches, flattens `[B, L, D]` hidden
  states and `[B, L]` target tokens down to `[B*L, D]` keys / `[B*L]` values (every
  position is an independent retrievable unit, chunk/position identity doesn't matter
  for the datastore).

**Isolation test (dummy vectors, done first):** 20 random 768-dim keys with known
values 0–19; queried with key #5 + tiny noise; got back value 5 as the closest match,
confirming the retrieval math alone is correct before touching real hidden states.

**Real-data test (`test_knn.py`, toy scale — `seq_len=32`,
`{"datastore": 40, "controller_train": 5, "val": 5}`):** datastore built from 1240 real
positions (`(1240, 768)` keys, `(1240,)` values). Queried with 10 real `val` hidden
states, decoded true next token vs. top-5 retrieved tokens. Real output confirmed the
mechanism is correctly wired, not just running-without-crashing:
- Exact matches: true token `' a'` retrieved `[' a', ' a', ' a', ...]`; true token `','`
  retrieved `[',', ...]` — nearby hidden states really do share next-token identity.
- Semantic-neighbor matches: true token `' time'` retrieved `[' little', ' big', ...]`;
  true token `' small'` retrieved `[' peaceful', ' big', ' little', ...]` — plausible
  same-slot fillers even without an exact match.
- Distances ranged ~5–24 — wide range expected at this scale (1240 entries can't
  densely cover a 50257-token vocabulary yet); should tighten once the real datastore
  has thousands of entries.

**Not yet done:** real-scale run (thousands of datastore entries, not 1240) — same
"just a parameter change" pattern as A2/A3, likely folded into building the actual
Phase B/C datastore later.

## Phase B step 6 — Read-time mixing formula (math DONE, verified in isolation)

Classic kNN-LM formula, added to `knn.py` next to the retrieval functions:
`mix_knn_and_lm(distances, retrieved_values, true_targets, p_lm_true, tau=1.0,
alpha=0.25)`. Per query position: (1) `softmax(-distance / tau)` turns the `k`
retrieved distances into a distribution over just those neighbors, (2) sum the weight
of whichever neighbors' value equals the true target → `p_knn_true` (0 if none
match), (3) `p_mixed = alpha * p_knn_true + (1 - alpha) * p_lm_true`, (4)
`nll_mixed = -log(p_mixed)`. `k`/`tau`/`alpha` are fixed hyperparameters for now —
tuning them via grid search is Phase D step 11, not now.

**Isolation test (dummy numbers, done):** two queries with identical GPT confidence
(`p_lm_true=0.3`, pure-LM NLL `1.204` both times) and identical retrieved distances,
but one where the nearest neighbor's value matched the true target and one where none
did. Real output: `nll_mixed = [0.750, 1.492]` — matched the predicted direction
exactly: retrieval agreeing pulled NLL *down* from 1.204, retrieval disagreeing pushed
it *up*. Confirms the formula behaves correctly, not just runs without crashing.

**Not yet done:** wiring this into real retrieval output (like `test_knn.py`'s
real-data test) to get an actual "raw kNN mean NLL" number on real `val` positions —
needed before this can be meaningfully compared to the GPT-only baseline (2.798), and
before the real-scale SLURM run that produces the final comparable Phase B step 5+6
result.

## Phase B steps 5+6 combined — real-scale raw kNN baseline (DONE, verified)

Combined real-scale datastore construction + the mixing formula into one script,
`run_knn_baseline.py`, submitted via `submit_knn_baseline.sh` on an L40S SLURM job
(same `seq_len=128`, `{"datastore": 350, "controller_train": 75, "val": 75}` split
config as the Phase B4 baseline, same default `seed=42` in `collect_chunks_split` so
the split is reproduced deterministically — confirmed by the pure-LM number matching
Phase B4's almost exactly). Fixed hyperparameters: `k=5`, `tau=1.0`, `alpha=0.25`
(untuned — grid search is Phase D step 11, not now). Real output:

```
datastore size: (49657, 768)
mean pure GPT NLL (same val positions): 2.7984323501586914
mean raw kNN-mixed NLL: 2.757404052302226
```

**Mixed NLL (2.757) is lower than pure GPT NLL (2.798)** — roughly a 1.5% relative
reduction, even at a ~49.7K-entry datastore and arbitrary, untuned hyperparameters.
This is real evidence the retrieval mechanism helps GPT prediction even in its
crudest (uncompressed) form — a positive signal for the whole DIME premise before any
compression is introduced. Saved to `GPT_Module/results/raw_knn_baseline.json`
(includes per-position mixed NLL for future significance testing per Phase F).

This raw-kNN number (2.757) is now the second reference line in `results/`, alongside
GPT-only (2.798): Phase C's whole point is to see whether a *compressed* memory can
match or beat 2.757 using far fewer stored entries than 49.7K, not just beat 2.798.

## Design note: what a "state object" actually is

Your own notes never defined this beyond Phase F's ablation list
(`majority-token, top-k truncation, shuffled`). Read (not copied — code, not
concept) the reference repo's `analyze_dime_efficiency.py` (a Phase-F-equivalent
script, much later than where this project is) to check what fields the original's
compressed state files actually store: `prototype_h` (one centroid hidden vector per
cluster — the compressed key), `top_k_token_ids` + `top_k_token_counts` (a **sparse**
distribution over next-tokens — only the top-K most frequent per cluster, not the
full vocab), `total_counts` (cluster size, for normalizing counts to probabilities),
and `state_entropy`/`state_purity` (metadata on how confident/dominated the cluster's
vote distribution is — used for later analysis, not core mixing math).

Decision: **build the full (non-truncated) observed token-count distribution per
cluster for now** — at current scale no cluster will have anywhere near 50257
distinct next-tokens, so "full" and "generously sparse" are practically identical.
Defer deliberate top-K truncation to when it actually matters: Phase F's own
"top-k truncation" ablation. Building it now would just be redone later.

## Folder-level split (agreed, not yet done)

Separate DIME-specific code (Phase C onward: `random_partition`, `minibatch_kmeans`,
`utility_weighted`, `query_kmeans`, the generalized distribution-based mixing formula,
DIME run/submit scripts) into a new sibling folder `Study/DIME/`, importing shared
pieces from `GPT_Module` (`FrozenGPT2`, `build_datastore_from_chunks`, `results_io`,
etc.) rather than duplicating them. `GPT_Module/` keeps model mechanics + the raw
kNN-LM baseline (Phases A/B) unchanged.

## Phase C step 7 — `random_partition` sanity control (DONE, verified)

Split DIME-specific code into its own sibling folder, `Study/DIME/` (reusing
`GPT_Module` pieces via `sys.path.insert`, not duplicating them — see "Folder-level
split" design note above, now done):

- **`DIME/state_object.py` → `random_partition(keys, values, n_clusters, seed=42)`** —
  randomly assigns each raw datastore entry to one of `n_clusters` groups, collapses
  each group into one compressed state object: `compressed_keys[c]` = mean of the
  group's key vectors (a centroid), `compressed_dists[c]` = a `Counter` of the group's
  raw target tokens (full observed distribution, no top-K truncation — that's
  deliberately deferred to Phase F per the design note above).
- **`DIME/mixing.py` → `mix_dime_and_lm(...)`** — generalizes `mix_knn_and_lm`: instead
  of checking "does the retrieved value equal the true token," it looks up "what
  fraction of this retrieved cluster's distribution was the true token" (raw kNN is
  the special case where every "cluster" has exactly one member — a point-mass
  distribution).
- **`DIME/run_dime_baseline.py`** — orchestration: builds the raw datastore (reusing
  `build_datastore_from_chunks`), compresses it via `random_partition`, feeds the
  compressed `(keys, Counters)` into `GPT_Module/knn.py`'s generic
  `build_datastore`/`query_knn` (confirming those work unchanged on object-dtype
  values, not just raw token ids), then mixes with `mix_dime_and_lm`.

**Isolation tests (done):** `random_partition` on 12 dummy entries/3 clusters — every
entry accounted for exactly once (`4+3+5=12`), structurally correct (caught and fixed
two bugs along the way: `compressed_dists = [0]` instead of `[]`, and a `sys.path`
`"."` vs `".."` typo in `mixing.py`). `mix_dime_and_lm` on hand-checkable dummy
Counters — partial-match case landed between the earlier full-match (0.750) and
no-match (1.492) results, exactly as predicted.

**Real-scale run (`seq_len=128`, same split as Phase B, `n_clusters=500` — ~100x
fewer stored entries than raw kNN's 49,657):**

```
mean pure GPT NLL (same val positions): 2.7984323501586914
mean random-partition DIME NLL: 2.994753195180113
```

**Random-partition DIME (2.995) is worse than both GPT-only (2.798) and raw kNN
(2.757)** — and this is the expected, correct result for a sanity control, not a bug.
The sanity-check print showed why: random clusters mix ~100 semantically unrelated
positions together, so each cluster's token distribution is a scattershot with no
coherent pattern (mostly count-1 entries across dozens of unrelated tokens), and
averaging unrelated key vectors into one centroid produces a much less informative key
(retrieval distances of 141-148, vs. raw kNN's 5-24). Blending in this diffuse,
near-noise signal at `alpha=0.25` dilutes GPT's own good guesses rather than
reinforcing them — same mechanism as the "disagreement" case in the isolation tests,
just happening almost every time instead of occasionally. This proves compression by
entry-count alone doesn't preserve retrieval's benefit — the clustering has to
actually group *similar* contexts together, which is exactly what step 8 needs to fix.

## Phase C step 8 — `minibatch_kmeans` (DONE, verified at real scale)

Added `minibatch_kmeans_partition(keys, values, n_clusters, seed=42)` to
`state_object.py` — same `(compressed_keys, compressed_dists)` return shape as
`random_partition`, but centroids come from `sklearn.cluster.MiniBatchKMeans.fit_predict`
(iteratively fitted, not just averaged-after-random-grouping) so entries with
*similar* hidden states land in the same cluster. Everything downstream
(`build_datastore`, `query_knn`, `mix_dime_and_lm`) reused unchanged.

**Isolation test (done):** three well-separated dummy 4D "blobs" (offset by 20, only
`scale=0.1` spread — trivially separable), each tagged with its own distinct token.
k-means recovered them *exactly*: three pure, single-token clusters (`{20: 5}`,
`{30: 5}`, `{10: 5}`) — unlike `random_partition`'s messy 50-token mixes, direct proof
clustering groups genuinely similar keys together.

**Real-scale run (`run_dime_kmeans_baseline.py` + `submit_dime_kmeans_baseline.sh`,
same split/seed, `n_clusters=500`):**

```
mean pure GPT NLL (same val positions): 2.7984323501586914
mean minibatch_kmeans DIME NLL: 2.7923980578792857
```

Comparison table so far:

| Method | Mean NLL |
|---|---|
| GPT-only (no retrieval) | 2.798 |
| random_partition (dumb control) | 2.995 (worse) |
| **minibatch_kmeans** | **2.792** (slightly better than GPT-only) |
| raw kNN (uncompressed, 49,657 entries) | 2.757 (best) |

Smart clustering recovered almost everything `random_partition` destroyed —
sanity-check retrieval distances dropped from ~141-148 (random) to 6.8-45.6 (k-means),
confirming centroids now sit near real local structure instead of averaging unrelated
noise. `minibatch_kmeans` even edges out GPT-only slightly, though it doesn't fully
match raw kNN — expected, since compressing ~49,657 positions into 500 clusters
(~99x compression) necessarily loses some position-level distinction that full-fidelity
retrieval keeps. Saved to `DIME/results/minibatch_kmeans_baseline.json`.

## Rule locked in before Phase D: `controller_train` is for tuning, `val` is not

`controller_train` has existed since Phase A3 but has never been used — every run so
far (Phase B, Phase C steps 7-8) only touched `datastore` and `val`, and every
hyperparameter (`k`, `tau`, `alpha`, `n_clusters`) was picked arbitrarily, never
searched for. That's fine so far, since no tuning against `val` has happened. But
Phase D step 11 ("fixed-hyperparameter grid search") changes that — and the grid
search **must** run against `controller_train`, never directly against `val`.
Picking whichever hyperparameter combination scores best on `val` and then reporting
that same `val` score as "the result" is the textbook leakage problem: it overfits
the hyperparameters to the exact number being reported, making every downstream
Phase E/F comparison meaningless. `val` gets touched exactly once — at the end, to
report the final number for whatever configuration won on `controller_train`.

Also worth noting on `utility_weighted` (Phase C step 9, below): weighting datastore
entries by GPT's own NLL doesn't leak `val` — the NLL comes from GPT's forward pass
on the *datastore* positions themselves, a property of each stored example, same
category of thing raw kNN/minibatch_kmeans already use (the entry's own true target
token). But it's not guaranteed to help: NLL conflates *reducible* uncertainty
(retrieval can fix — a recurring pattern GPT hasn't connected) with *irreducible*
uncertainty (retrieval can't fix — genuinely ambiguous/rare tokens), so weighting up
high-NLL positions might just amplify noise in some clusters. Go in expecting "maybe
helps, maybe doesn't, maybe slightly hurts" — any of those is a legitimate result, not
a sign of a bug.

## What's next: Phase C step 9 — `utility_weighted`

Weight clusters by GPT's own NLL/entropy at construction time (not just k-means
proximity) — the idea being some clusters carry more useful signal than others
(e.g. positions where GPT itself was uncertain may benefit more from retrieval than
positions it already predicts confidently). Needs deciding exactly how "utility"
factors into clustering or read-time weighting — a design decision, not yet spec'd
beyond the roadmap's one-line description.

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
