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

## Phase C step 9 — `utility_weighted` (DONE, verified at real scale)

Design (informed by reading — not copying — the reference repo's
`build_predictive_states.py`; also found and didn't replicate a dead-code
inconsistency there, where the fancier percentile-clipped/`utility_positive` weights
it computes are silently discarded in favor of plain `nll_gpt + floor`): same
clustering as `minibatch_kmeans` (identical `MiniBatchKMeans` step), but each raw
entry's contribution to its cluster's token distribution is weighted by GPT's own
NLL at that position (+ a `0.1` floor) instead of a flat count of 1 — added
`build_datastore_with_nll_from_chunks` (`extract.py`) to carry per-position NLL, and
`utility_weighted_partition` (`state_object.py`). `mixing.py` needed zero changes —
`mix_dime_and_lm` already does `count / total` generically on whatever's in a
`Counter`, so float weighted-sums work identically to integer counts.

**Critical-thinking pass (done before running, not after):** confirmed no `val`
leakage — the NLL weighting comes from the *datastore* split's own GPT forward pass,
a property of each stored example, same category of thing raw kNN already uses.
But flagged a real (non-leakage) concern: NLL conflates *reducible* uncertainty
(retrieval can fix) with *irreducible* uncertainty (retrieval can't — genuinely
ambiguous/rare tokens), so weighting up high-NLL positions might amplify noise
instead of signal. Went in explicitly expecting "maybe helps, maybe hurts" rather
than assuming improvement. Also used this pass to lock in the `controller_train`
vs. `val` tuning rule above, before Phase D needs it.

**Isolation test (done):** two positions in one cluster, same two distinct tokens,
one with NLL=0.1 and one with NLL=5.0 — weighted distribution came out exactly
`{10: 0.2, 20: 5.1}` (not the equal `{10:1, 20:1}` a flat count would give),
confirming the high-NLL position dominates as designed.

**Real-scale run (`run_dime_utility_baseline.py` + `submit_dime_utility_baseline.sh`,
same split/seed/`n_clusters=500`):**

```
mean pure GPT NLL (same val positions): 2.7984323501586914
mean utility_weighted DIME NLL: 2.805119776562979
```

**The concern from the critical-thinking pass was confirmed, not just theoretical:**
utility_weighted (2.805) came out *worse* than plain `minibatch_kmeans` (2.792), and
even slightly worse than doing nothing at all (GPT-only, 2.798). Full comparison
table:

| Method | Mean NLL |
|---|---|
| raw kNN (uncompressed, 49,657 entries) | 2.757 (best) |
| minibatch_kmeans | 2.792 |
| GPT-only (no retrieval) | 2.798 |
| utility_weighted | 2.805 |
| random_partition (dumb control) | 2.995 (worst) |

Real, meaningful negative result: raw-NLL weighting injected more noise (amplifying
genuinely-ambiguous positions) than useful signal (amplifying retrieval-fixable
positions) here. Saved to `DIME/results/utility_weighted_baseline.json`.

## Phase C step 10 — `query_kmeans` (DONE, verified at real scale)

Added `query_kmeans_partition(ds_keys, ds_values, ct_keys, n_clusters, seed=42)` to
`state_object.py`: fits `MiniBatchKMeans` on `controller_train`'s hidden states (not
the datastore's), then `.predict()`s which centroid each *datastore* entry is nearest
to — inverting which split drives the cluster geometry vs. step 8. First real use of
`controller_train` in the project (`val` stays untouched — assigning datastore
entries to CT-derived clusters and then evaluating on `val` isn't tuning against
`val`, just changing what data shapes the clusters).

**Critical question raised before running (and it was the right question to ask):**
does this method really achieve the same "49,657 → 500" compression claim as the
other three? Unlike `random_partition`/`minibatch_kmeans`/`utility_weighted` — whose
clusters are fit directly on the datastore, guaranteeing every cluster gets members —
`query_kmeans`'s centroids come from a *different* dataset slice (`controller_train`),
so some CT-derived clusters could plausibly end up with zero datastore members
assigned, making the real ("effective") compression lower than the nominal
`n_clusters`. This mirrors why the reference repo's state files track an
`assigned_count` field per cluster — the original authors clearly instrumented for
exactly this. Added a non-empty-cluster count to the real-scale script as an honesty
check before trusting the compression claim.

**Isolation test (done):** CT blobs (3, well-separated) define the clusters; DS blobs
are a *different* sample, slightly offset from CT's exact centers, tagged with
distinct tokens. Got back three pure single-token clusters (`{20:5}`, `{30:5}`,
`{10:5}`), confirming datastore entries get routed to the *right* externally-fit
cluster, not just to clusters containing identical points.

**Real-scale run (`run_dime_query_kmeans_baseline.py` +
`submit_dime_query_kmeans_baseline.sh`, same split/seed/`n_clusters=500`):**

```
non-empty clusters: 496 / 500
mean pure GPT NLL (same val positions): 2.7984323501586914
mean query_kmeans DIME NLL: 2.7940686816820333
```

**The honesty check came back reassuring** — only 4/500 clusters empty (~0.8%), so
the compression claim holds almost exactly as stated; the theoretical concern was
worth checking but wasn't a real problem here (CT and datastore hidden-state
distributions overlap well). Result (2.794) landed almost identical to
`minibatch_kmeans` (2.792) — expected, since both cluster by genuine similarity, just
on different (but overlapping) data. Full comparison table:

| Method | Mean NLL |
|---|---|
| raw kNN (uncompressed, 49,657 entries) | 2.757 (best) |
| minibatch_kmeans | 2.792 |
| query_kmeans | 2.794 |
| GPT-only (no retrieval) | 2.798 |
| utility_weighted | 2.805 |
| random_partition (dumb control) | 2.995 (worst) |

Saved to `DIME/results/query_kmeans_baseline.json`. **This completes Phase C (steps
7-10) — all four compression methods now have real, verified numbers.**

## Phase D step 11 — fixed-hyperparameter grid search (DONE, verified)

`DIME/grid_search.py` → `grid_search_hyperparams(distances, retrieved, true_targets,
p_lm_true, mix_fn, k_values, tau_values, alpha_values)` — queries once at the largest
`k` needed, slices smaller `k` from the same retrieved array (avoids re-running
`query_knn` per combo), loops `tau`×`alpha` combos, sorts by mean NLL. Works for both
raw kNN (`mix_knn_and_lm`) and compressed DIME (`mix_dime_and_lm`) since they share
the same call signature. `DIME/run_grid_search.py` + `submit_grid_search.sh` run the
search on `controller_train` for both raw kNN and `minibatch_kmeans`, then check the
winning config on `val` exactly once, per the locked-in tuning rule.

**Isolation test (done):** one query with a dominant, exactly-matching nearest
neighbor — grid search correctly picked the highest tested `alpha` (0.9), confirming
the loop/sort logic (not just that `mix_knn_and_lm` itself works, already known).

**Round 1** (`k=[3,5,10,20]`, `tau=[0.5,1,2,5]`, `alpha=[0.1,.25,.5,.75]`): winning
config `k=20, tau=2.0, alpha=0.1` for *both* systems — but `k=20` (max tested) and
`alpha=0.1` (min tested) both sat at the edge of their lists, a classic "haven't found
the true optimum, just the best of too narrow a range" warning sign (like testing
oven temperatures 350-425°F and finding 425°F best — you don't know if 450°F would've
won, because you never tried it). Checked the full top-5 lists to confirm this wasn't
noise: *every* top-5 entry for both systems had `alpha=0.1`, unambiguous evidence to
extend downward; DIME's `k` scores were flat across `3-20` (not trending, no real
edge effect there) while raw kNN's showed a real if modest lean toward larger `k`.

**Round 2** (`k=[10,20,30,50]`, `tau=[1,2,5,10]`, `alpha=[.01,.05,.1,.25]` — shifted
based on round 1's actual evidence, not blind widening): first submission accidentally
reran round 1's grid (cluster hadn't `git pull`ed the edit — caught by noticing the
result was suspiciously bit-identical to round 1, then confirmed by rerunning after
pulling). Real round 2 result:

```
raw kNN best on controller_train: {'k': 50, 'tau': 2.0, 'alpha': 0.1}  -> val mean NLL: 2.6940908318605463
DIME (minibatch_kmeans) best on controller_train: {'k': 20, 'tau': 2.0, 'alpha': 0.05}  -> val mean NLL: 2.7451901708278066
```

`alpha` and `tau` resolved to genuine interior optima for both systems this round.
`k=50` for raw kNN is *still* at the tested edge — but the gain from `k=20→50` was
small (val NLL `2.699→2.694`, ~0.17% relative), a diminishing-returns pattern
suggesting a plateau rather than still-climbing behavior. Called this the stopping
point rather than opening a third round chasing a shrinking marginal gain.

**Final tuned numbers vs. the fixed defaults used throughout Phases B/C:**

| | Untuned (fixed defaults) | Tuned (grid search) |
|---|---|---|
| raw kNN | 2.757 | **2.694** (k=50, tau=2.0, alpha=0.1) |
| minibatch_kmeans | 2.792 | **2.745** (k=20, tau=2.0, alpha=0.05) |

Both systems improved meaningfully from tuning — DIME's margin over GPT-only (2.798)
went from razor-thin (2.792, barely distinguishable) to solid (2.745). Saved to
`DIME/results/grid_search_results.json`.

## Phase D step 12 — Q-read controller (DONE, verified — genuine improvement)

Framed as a one-shot contextual bandit, not full RL (no state transitions — each
query is an independent decision): **state** = query features knowable *without*
the true label, **action** = binary (retrieve at the Phase-D-11-tuned config, or
don't), **reward** = `NLL_GPT − NLL_action` (computed offline from `controller_train`
labels, fine as a training target — labels are always used to supervise training;
the leakage risk is specifically about what goes into the *state*, not the reward).

**Caught before coding, not after:** GPT's own NLL/`p_true` requires already knowing
the true next token, so it can't be a state feature — at real decision time you don't
have the answer yet, that's what you're predicting. Used **entropy of GPT's full
distribution** instead (knowable without the label) plus nearest-neighbor distance
and the retrieved cluster's own entropy/purity — same class of concern as
`utility_weighted`'s leakage check, one level closer to an actual bug this time.

New pieces: `predictive_entropy` (`helpers.py`, entropy from logits, no label
needed), `run_batch_with_entropy` (`extract.py`, sibling to `run_batch` — didn't
touch `run_batch`'s signature since 7 existing scripts unpack it positionally),
`retrieval_purity_entropy` + `train_q_read_controller` (`DIME/q_read.py` — handles
both raw kNN's plain-token retrieved values and DIME's `Counter`s; MLP via
`sklearn.neural_network.MLPRegressor`, not hand-written — the thing under study is
DIME's mechanism, not general ML algorithms, same reasoning as using sklearn for
k-means/NearestNeighbors rather than re-deriving them).

**Isolation tests (done):** entropy on a peaked vs. flat 3-token distribution gave
`≈0` and `≈log(3)=1.0986` exactly. Retrieval entropy/purity on a 9:1 vs. 5:5 split
gave `0.325`/`0.9` and `log(2)=0.693`/`0.5` exactly. MLP trained on synthetic data
with a known linear relationship (`reward = 2*x0 - x1`) predicted `[2.19,-0.95,-1.12]`
vs. true `[2,-1,-1]` — correctly recovered the relationship.

**Real-scale run (`run_q_read_baseline.py` + `submit_q_read_baseline.sh`, tuned
`minibatch_kmeans` config `k=20,tau=2.0,alpha=0.05` from step 11):**

```
controller_train: 9652 examples, mean reward = 0.0443
mean NLL, GPT-only:           2.7984323501586914
mean NLL, always mixed:       2.7451901708278066
mean NLL, Q-read (adaptive):  2.7428829272220168
fraction of val queries where Q-read chose to retrieve: 0.8999787188763567
```

**Q-read (2.743) beats the single fixed tuned config (2.745)** — modest but genuine,
and exactly the result Phase D is chasing: a per-query decision beating one global
choice for everyone. It retrieved for ~90% of `val` queries and correctly identified
the other ~10% as cases where trusting GPT alone was better. Saved to
`DIME/results/q_read_baseline.json`.

## Phase D step 13 — same Q-read code on raw kNN (DONE, verified — completes Phase D)

`run_q_read_raw_baseline.py` + `submit_q_read_raw_baseline.sh` — identical pipeline
to step 12, swapped to the raw (uncompressed) datastore, `mix_knn_and_lm`, and raw
kNN's own tuned config from step 11 (`k=50, tau=2.0, alpha=0.1`). One expected
quirk noted going in: `retrieval_purity_entropy` returns a *constant*
`entropy=0, purity=1` for every raw kNN query (its retrieved values are always plain
token ids, never `Counter`s — no real distribution to measure), so those two
features carry no information here; the MLP leans on GPT's own entropy and
nearest-neighbor distance instead.

**Real result:**

```
controller_train: 9652 examples, mean reward = 0.0887
mean NLL, GPT-only:           2.7984323501586914
mean NLL, always mixed:       2.6940908318605463
mean NLL, Q-read (adaptive):  2.6893500226508866
fraction of val queries where Q-read chose to retrieve: 0.7317869050152515
```

**Q-read (2.689) beats always-mixed (2.694) again** — confirms the mechanism isn't
specific to compressed memory. Interesting difference from the DIME run: it retrieved
for only ~73% of queries here (vs. ~90% for `minibatch_kmeans`), i.e. it learned to
be *more* selective with raw kNN — plausible, since a single raw nearest-neighbor
match is noisier per-query than a compressed cluster's averaged vote, so skipping it
more often when the features look unfavorable pays off more. Saved to
`DIME/results/q_read_raw_baseline.json`.

**Full Phase D comparison table (both memory types, all three approaches):**

| | GPT-only | untuned fixed | tuned fixed (grid search) | Q-read (adaptive) |
|---|---|---|---|---|
| raw kNN | 2.798 | 2.757 | 2.694 | **2.689** |
| minibatch_kmeans | 2.798 | 2.792 | 2.745 | **2.743** |

Every step of Phase D improved both systems, and the ordering (raw kNN always ahead
of compressed, as expected since compression trades some fidelity for far fewer
stored entries) held at every stage. **This completes Phase D.**

## Phase E step 14 — 6 equal-budget raw variants (DONE, verified — completes Phase E)

`run_equal_budget_raw_variants.py` + `submit_equal_budget_raw_variants.sh` (written
directly — flagged as "just a diagnostic script," so skipped the type-it-yourself
step this time) — six criteria for selecting 500 raw datastore entries (same budget
as DIME's 500 clusters), all evaluated with raw kNN's tuned config from step 11
(`k=50,tau=2.0,alpha=0.1`, reused as-is rather than re-tuned per variant — a
deliberate simplification for a diagnostic pass):

- `raw_random` — uniform random 500 (baseline control)
- `raw_high_gpt_loss` / `raw_low_gpt_loss` — the 500 hardest / easiest positions by NLL
- `raw_high_entropy` / `raw_low_entropy` — the 500 with the most / least spread-out
  GPT belief overall
- `raw_kmeans_representative` — cluster into 500 groups like DIME does, but keep the
  single *real* raw entry closest to each centroid instead of merging into a
  synthetic prototype+distribution (diversity-driven selection, without compressing)

**Real result — one of the strongest findings in the project:**

```
raw_random                   n= 500  mean NLL = 2.8075
raw_high_gpt_loss            n= 500  mean NLL = 2.8606
raw_low_gpt_loss             n= 500  mean NLL = 2.9016
raw_high_entropy             n= 500  mean NLL = 2.8805
raw_low_entropy              n= 500  mean NLL = 2.9017
raw_kmeans_representative    n= 500  mean NLL = 2.7818
```

**Every single equal-budget raw variant is worse than GPT-only (2.798)** — none of
the six 500-entry raw selections beat doing nothing at all. **DIME's
`minibatch_kmeans` (2.745, also 500 entries) beats every one of them**, including
the best raw variant (`raw_kmeans_representative`, 2.782). Full table:

| Method | Entries | Mean NLL |
|---|---|---|
| raw kNN (full, tuned) | 49,657 | 2.694 (best overall) |
| minibatch_kmeans (DIME, tuned) | 500 | **2.745** |
| GPT-only (no retrieval) | — | 2.798 |
| raw_kmeans_representative | 500 | 2.782 |
| raw_random | 500 | 2.808 |
| raw_high_gpt_loss | 500 | 2.861 |
| raw_high_entropy | 500 | 2.881 |
| raw_low_gpt_loss | 500 | 2.902 |
| raw_low_entropy | 500 | 2.902 |

**Why this matters:** this directly answers the fairness question Phase E exists to
ask — is DIME's benefit just "500 well-chosen pointers," or something about
compression specifically? It's the latter. No selection criterion for 500 *individual*
raw examples comes close to DIME's 500 *compressed clusters*, because a single raw
point's true-next-token is one noisy sample, while a cluster's pooled distribution is
many samples' worth of averaged signal — broader, denser coverage per stored unit
than any single point can offer, however well-chosen. Also: biasing selection toward
hard (`high_gpt_loss`/`high_entropy`) or easy (`low_gpt_loss`/`low_entropy`) examples
both did *worse* than unbiased `raw_random` — the same lesson `utility_weighted`
taught in Phase C, resurfacing at the selection level. Saved to
`DIME/results/equal_budget_raw_variants.json`.

## Phase F step 15 — state-object ablations (DONE, verified — strong "not an illusion" result)

`run_state_object_ablations.py` + `submit_state_object_ablations.sh` (written
directly, diagnostic) — three ablations on the tuned `minibatch_kmeans` compressed
datastore (`k=20,tau=2.0,alpha=0.05`), reusing the same fitted `NearestNeighbors`
index across all of them (only the stored *values* array changes per ablation, not
the keys/geometry):

- **majority-token**: collapse each cluster's full `Counter` to just its single most
  common token (keeping its full count)
- **top-5 truncation**: keep only the 5 most frequent tokens per cluster, drop the rest
- **shuffled**: keep every centroid in its correct geometric position, but randomly
  reassign *which* cluster's distribution it reports — breaks the correspondence
  between "where a query sits" and "what actually follows there," the direct
  illusion-check

**Real result:**

```
full (unablated):      2.7451901708278066
majority-token:        2.764617674191417
top-5 truncation:      2.7471423633757466
shuffled (illusion check): 2.8349186427610755
```

| Variant | Mean NLL | vs. full |
|---|---|---|
| full (unablated) | 2.745 | — |
| top-5 truncation | 2.747 | +0.07% (basically free) |
| majority-token | 2.765 | +0.71% (real, modest loss) |
| shuffled | 2.835 | +3.27% — **worse than GPT-only (2.798)** |

**`shuffled` scoring worse than doing nothing at all is the key finding** — direct,
strong evidence DIME's benefit is real, not an artifact of the mixing formula or
GPT's own behavior: when a centroid's geometric position is right but its reported
distribution is wrong, the mixing formula confidently blends in actively-misleading
votes, actively hurting rather than just failing to help. If shuffling had barely
changed the result, that would have been the red flag; instead it confirms the
(location → content) correspondence is doing real work. Separately, `top-5
truncation ≈ full` is a strong efficiency signal for step 16 — sparse top-5 storage
loses almost nothing, a real lever for byte-level savings. `majority-token` sits
between the two, as expected: real degradation from a single-token reduction, but
nowhere near as damaging as breaking the correspondence entirely, and still clearly
better than GPT-only. Saved to `DIME/results/state_object_ablations.json`.

## Phase F step 16 — efficiency analysis (DONE, verified — real measurements, not formulas)

`run_efficiency_analysis.py` + `submit_efficiency_analysis.sh` — real, measured
numbers (`.nbytes` on the actual numpy arrays, `pickle.dumps` on the actual `Counter`
objects for a genuine serialized size, `time.perf_counter()` wall-clock timing with
warmup), not hypothetical formula-based estimates:

```
=== Entry count ===
raw kNN: 49657  DIME: 500  compression: 99.3x
=== Real measured bytes ===
raw kNN total:         152.94 MB
DIME full:             1.65 MB  (92.6x smaller than raw)
DIME top-5 truncation: 1.55 MB  (98.5x smaller than raw)
DIME majority-token:   1.54 MB  (99.1x smaller than raw)
=== Retrieval latency (ms per query) ===
raw kNN (k=50):               0.3516 ms/query
DIME minibatch_kmeans (k=20): 0.0060 ms/query
```

| Metric | Raw kNN | DIME (minibatch_kmeans) | Ratio |
|---|---|---|---|
| Entries | 49,657 | 500 | 99.3x fewer |
| Bytes (measured) | 152.94 MB | 1.65 MB (full dist.) | 92.6x smaller |
| Latency (ms/query) | 0.3516 | 0.0060 | 58.6x faster |

All three metrics reinforce the case DIME already made on NLL quality — a genuine
practical efficiency win, not just a modeling curiosity. **One honest nuance worth
keeping, not overclaiming:** the byte savings from truncating *values*
(full→top-5→majority: 1.65→1.55→1.54 MB) are tiny next to the savings from
clustering itself (152.94→1.65 MB) — because the 500 clusters' **keys alone**
already cost ~1.54 MB (`500×768×4 bytes`), dominating the total. Step 15's "top-5
truncation loses almost no quality" finding is real, but its *byte* payoff here is
modest, since there wasn't much value-side weight to trim — the real byte win
already happened when the datastore shrank from 49,657 keys to 500. This is exactly
what "real measurements, not formulas" is meant to catch: a formula-based guess could
have overstated how much value-truncation would save. Saved to
`DIME/results/efficiency_analysis.json`.

## Phase F step 17 — rigor / significance testing (DONE, verified — THE FINAL STEP)

`run_significance_testing.py` + `submit_significance_testing.sh` — rebuilt the two
most important comparisons fresh, on the same `val` split/seed so positions align
exactly for paired testing: **GPT-only vs. tuned `minibatch_kmeans`**, and
**Phase E's best raw equal-budget variant (`raw_kmeans_representative`) vs. tuned
`minibatch_kmeans`**. Ran both a paired t-test and a Wilcoxon signed-rank test
(non-parametric, robust to non-normal NLL differences) on each. Added `scipy` to
`requirements.txt` for this.

**Real result:**

```
mean NLL — GPT-only: 2.7984324
mean NLL — minibatch_kmeans (tuned): 2.7451901708278066
mean NLL — raw_kmeans_representative: 2.781779688831718

Test 1 (GPT-only vs DIME): mean_diff=0.0532, paired t-test p=2.97e-42, Wilcoxon p=0.0
Test 2 (best raw equal-budget vs DIME): mean_diff=0.0366, paired t-test p=1.55e-28, Wilcoxon p=0.0
```

**Both differences are statistically overwhelming** (p-values effectively zero,
Wilcoxon underflowing to exactly `0.0`) — with ~9,525 paired `val` positions, even
these modest-looking mean differences (0.053 and 0.037 NLL) are robust, consistent
effects, not noise from one lucky split. DIME's advantage over GPT-only *and* over
the best possible equal-budget raw alternative is real. Saved to
`DIME/results/significance_testing.json`.

# ROADMAP COMPLETE — Phases A through F all done and verified

Every phase (A: foundation, B: naive kNN-LM baseline, C: compression, D: learned read
policy, E: equal-budget fairness check, F: proof it's not an illusion) has real,
verified results. Per the roadmap's own framing: "Everything past Phase F (more
datasets/models) is replication, not new mechanism" — the core empirical question
this whole recreation project set out to answer is now answered.

## Final results summary (all on the same TinyStories/GPT-2/`seq_len=128` setup)

| Stage | Mean NLL | Note |
|---|---|---|
| GPT-only (no memory) | 2.798 | reference baseline |
| raw kNN, untuned (49,657 entries) | 2.757 | Phase B |
| raw kNN, tuned (grid search) | 2.694 | Phase D step 11 |
| raw kNN, Q-read (adaptive) | 2.689 | Phase D step 13 — best raw kNN result |
| random_partition (500, dumb control) | 2.995 | Phase C step 7 — worse than doing nothing |
| minibatch_kmeans, untuned (500) | 2.792 | Phase C step 8 |
| query_kmeans (500) | 2.794 | Phase C step 10 |
| utility_weighted (500) | 2.805 | Phase C step 9 — negative result, as predicted |
| minibatch_kmeans, tuned (500) | 2.745 | Phase D step 11 |
| minibatch_kmeans, Q-read (adaptive, 500) | 2.743 | Phase D step 12 — best DIME result |
| best raw variant at equal budget (`raw_kmeans_representative`, 500) | 2.782 | Phase E — still worse than DIME at the same budget |

**The core finding:** at an equal 500-entry storage budget, no way of selecting which
raw examples to keep beats DIME's compression (Phase E) — and this margin is
statistically significant (Phase F step 17), not an artifact of the mixing formula or
one lucky split (Phase F step 15's shuffle ablation directly confirmed the retrieved
content is doing real work). DIME also achieves this while using ~99x fewer entries,
~93x fewer measured bytes, and ~59x faster retrieval than raw kNN (Phase F step 16).

# POST-ROADMAP: generalizing beyond TinyStories/GPT-2-small

Professor (Austin) reviewed the project (2026-09-06 email): encouraging, worth
pursuing further, with two concrete asks — (1) formalize the mathematics of storage
savings/retrieval ratios/other metrics, (2) extend the datasets used for testing.
Ask (2) matches this project's own "everything past Phase F is replication" framing
exactly. Ask (1) (math formalization) is **not yet started** — deferred in favor of
generating more empirical data points first (the compression sweep, below, is a step
toward it). User also wants a bigger model, not just a bigger dataset.

## Caching infrastructure (new, in `GPT_Module/`)

Every prior script recomputed GPT forward passes from scratch — fine at TinyStories
scale, wasteful once bigger models/datasets are involved. New shared infrastructure:
- **`cache_io.py`** — `save_cache`/`load_cache`, `.npz` files holding `keys`, `values`,
  `p_lm_true`, `entropy` per split.
- **`extract_and_cache.py`** — parametrized by `--dataset`, `--model`, `--seq_len`,
  chunk counts; runs extraction once, saves to cache. `DATASET_REGISTRY` currently has
  `tinystories` and `wikitext103` (`wikitext`, config `wikitext-103-raw-v1` — the same
  dataset the reference repo itself used, confirmed from an earlier file-structure
  read, not a random pick).
- Verified at tiny scale first (`gpt2-medium` + `wikitext103`, 40/5/5 chunks) before
  committing to the real extraction — confirmed both the new model and the new
  dataset's chunking (WikiText-103 has many near-empty header/separator rows mixed
  with real content; harmless, just contributes a lone EOS token per empty row).
- Real extraction: `gpt2-medium` + `wikitext103`, `seq_len=128`,
  `n_datastore=3000, n_controller_train=500, n_val=500` chunks →
  **381,127 / 90,170 / 77,089 positions** (only ~2.5 minutes on the L40S).

## WikiText-103 / gpt2-medium replication (DONE — core finding holds, bigger margin)

Downstream analysis scripts now read from the cache instead of recomputing —
`run_wikitext_medium_analysis.py` combines GPT-only, raw kNN + DIME grid search,
`raw_kmeans_representative` (Phase E's best equal-budget raw variant), and both
significance tests in one script, `n_clusters=4000` (~95x compression, matching the
ratio already validated on TinyStories).

**Round 1** (`k∈[10,20,50,100], tau∈[.5,1,2,5], alpha∈[.01,.05,.1,.25]`): both raw kNN
and DIME hit `k=100` (max tested) as their boundary; raw kNN also hit `alpha=0.25`
(max). **Round 2** (`k∈[50,100,200,300], tau∈[.5,1,2,5], alpha∈[.05,.1,.25,.5]`,
shifted based on round 1's evidence): raw kNN → `k=300` (still boundary, but
`k=200→300` only gained `0.004` — diminishing returns, called this the stopping point,
same judgment as TinyStories), `tau=1.0`, `alpha=0.25` (now interior). DIME → `k=200`
(now interior, flanked both sides), `tau=2.0`, `alpha=0.1` (interior) — DIME's number
barely moved between rounds, it was already well-tuned. One resubmission accidentally
reran round 1's exact code (cluster hadn't `git pull`ed the edit) — caught via
noticing bit-for-bit identical output, a useful diagnostic pattern now established.

**Final numbers:**

| Method | Mean NLL |
|---|---|
| GPT-only | 4.049 |
| raw kNN (tuned: k=300,tau=1.0,alpha=0.25) | **3.659** |
| DIME minibatch_kmeans (tuned: k=200,tau=2.0,alpha=0.1) | 3.754 |
| raw_kmeans_representative (best equal-budget raw, n=3995) | 3.956 |

**Significance (paired t-test + Wilcoxon, ~77,089 positions):** GPT-only vs DIME
`p=2.97e-42`; best-raw-equal-budget vs DIME `p=1.55e-28`. Both p-values effectively
zero. **DIME beats the best equal-budget raw variant by 0.202 NLL here — ~5.5x the
margin seen on TinyStories (0.037)** — the core finding didn't just replicate, it got
stronger on a bigger model and a more complex dataset. Saved to
`DIME/results/wikitext103_gpt2_medium_replication.json`.

## Compression sweep — does DIME ever beat raw kNN's own tuned ceiling?

User's hypothesis: relaxing DIME's compression ratio (larger `n_clusters`, less
aggressive) might let DIME's pooled/averaged cluster votes beat raw kNN's noisier
per-query lookup — a real, testable bias-variance/shrinkage argument (distinct from
"DIME beats equal-budget raw," which is already proven). **Hard constraint noted
before running anything:** as `n_clusters → N` (the full 381,127), DIME's clusters
each hold exactly one raw point — DIME literally *becomes* raw kNN at that limit, so
it can only ever converge to *equality*, not exceed it, at the lossless end. The open
question is whether it crosses *above* raw kNN's line somewhere in the *middle*.

`run_compression_sweep.py`: computes raw kNN's own tuned ceiling once (shared
reference line), then sweeps DIME across `n_clusters ∈ {4000, 15000, 40000, 100000}`
(~95x down to ~3.8x compression), re-tuning `k/tau/alpha` at each point.

**Results so far (3 of 4 points; 4th, `n_clusters=100000`, ran long — see below):**

| n_clusters | Compression ratio | val NLL | Gap to raw ceiling (3.659) |
|---|---|---|---|
| 4,000 | 95.3x | 3.754 | 0.094 |
| 15,000 | 25.4x | 3.738 | 0.079 |
| 40,000 | 9.5x | 3.728 | 0.069 |
| 100,000 | 3.8x | *(pending)* | — |

Quality improves monotonically as compression eases, but at a **shrinking** rate
(`0.015` gap-closing 95x→25x, only `0.010` closing 25x→9.5x) — a diminishing-returns
curve. **User's conclusion (endorsed): aggressive compression (`n_clusters=4000`) is
the best cost/benefit tradeoff** — it captures almost all the practical benefit for a
bounded quality cost, while chasing milder compression buys shrinking quality gains
for much greater storage cost. Whether `n_clusters=100000` ever actually crosses
*below* raw kNN's `3.659` ceiling is still an open, unresolved question — the trend
so far suggests probably not, but this wasn't confirmed either way before the job's
status became uncertain (see below).

**Operational notes from this run, worth remembering:**
- Hit a **stdout buffering scare** — 20+ hours elapsed with nothing in the log beyond
  the bash `echo` startup lines, looked like a hang. Root cause: Python's `print()`
  fully block-buffers when stdout is redirected to a file (as under SLURM), unlike
  bash's own unbuffered `echo`. Every prior job had only ever been checked *after*
  finishing (buffers auto-flush at exit), so this was never visible before. **Fix
  going forward: add `python -u` or `PYTHONUNBUFFERED=1` to submit scripts.** The
  *reliable* way to check real progress mid-run is the incremental JSON checkpoint
  (`save_results` after each sweep point), not the buffered print log.
  `results/compression_sweep.json` confirmed genuine progress (3 points saved) while
  the log looked empty.
  `--time=24:00:00` was set; job was at ~21h elapsed with the largest, slowest point
  (`100,000` clusters — `MiniBatchKMeans` cost scales with `n_clusters`) still running.
  Last known status when last checked: still running, real progress via checkpoint,
  final 4th point's outcome unconfirmed as of this note.

## Reference-repo gap review (Austin's/reviewer's feedback) — 3 points to close

Comparing this recreation against "Track A" (the original reference implementation)
surfaced three gaps, tackled in reverse-priority order (user chose to start with the
biggest lift first):

- **Point 3 — Q-read is a simplified binary gate vs. Track A's genuine multi-action
  policy.** IN PROGRESS, see below.
- **Point 5 — no mechanism-diagnostics layer** (hit@k, `p_state(true)`, active-state
  fraction, oracle gap, top helpful/harmful states). NOT STARTED. Oracle gap will fall
  out of point 3's work "for free" once the reward matrix exists (it's just
  `reward_matrix.max(axis=1)` per query).
- **Point 6 — raw-baseline selection criteria only partially overlap** (missing
  `raw_token_rarity` and `raw_coverage`/greedy-farthest-first). NOT STARTED.
- Scale (this project's ~50k-equivalent vs. Track A's canonical 200k) — explicitly
  **not** being addressed for now, by user's own call; not a gap, a deliberate choice.

### Point 3 — genuine multi-action Q-read (Track A parity), IN PROGRESS

Read Track A's actual `train_q_state_read.py` (`QStateReadMLP`) in full for concept
understanding (not copying code). Real design, corrected from an earlier wrong guess:
actions aren't "which retrieved candidate to trust" — they're a **discretized grid of
entire read-policy configs** (`k × tau × alpha × beta`, 257 total with one `gpt_only`
action), so the controller picks a different *complete* config per query, not just a
binary gate on one fixed config. New mechanism, not in this project before: **`beta`**
— a Laplace-style shrinkage term blending a retrieved cluster's empirical token counts
with the **global corpus frequency** of the true token, `(count + beta*global_freq) /
(total + beta)` — regularizes small/unreliable clusters toward a safer global prior.
Directly relevant to the compression-sweep question above (shrinkage as a real
noise-reduction mechanism).

**Two deliberate deviations from Track A, on principled grounds, not oversights:**
- Track A's `alpha` means the *opposite* of ours (theirs: weight on GPT; ours: weight
  on retrieval). Kept **our** established convention — flipping now would silently
  invalidate every tuned number from the whole project so far.
- Track A's own observation features 0/1 (`nll_gpt`, `p_gpt_true`) require already
  knowing the true next token — the same leakage category caught and avoided for
  `utility_weighted` earlier, just closer to an actual bug this time (these would be
  uncomputable in real online generation, not just methodologically borderline). Used
  `predictive_entropy` (label-free) instead, consistent with this project's established
  rigor standard. Track A's own features 2/16/17 are hardcoded to zero in their code
  anyway ("unavailable") — dead placeholder slots, not replicated.

**A real performance problem, solved via GPU vectorization (`Study/DIME/q_read_dense.py`,
new file):** naively porting the existing `Counter`-based `mix_dime_and_lm` to loop
over the full 257-action grid would mean billions of Python-level dict lookups
(`256 actions × ~90,170 queries × up to 300 neighbors`) — plausibly days, not hours.
Fix: convert each cluster's `Counter` into a **dense, padded `[B, top_k]` representation**
(`counters_to_dense`) — this turns out to be exactly the representation Track A's own
state files use (`top_k_token_ids`/`top_k_token_counts`/`total_counts`), and is
independently justified by Phase F step 15's finding that top-K truncation loses
almost no quality. Dense arrays enable genuine batched GPU tensor ops instead of
per-query Python loops.

**Pieces built and verified (isolation-tested, several cross-validated bit-for-bit
against the already-trusted Counter-based path):**
- `build_action_grid` / `build_action_features` (`DIME/action_grid.py`) — 257 actions
  (`k∈{50,100,200,300}×tau∈{.5,1,2,5}×alpha∈{.05,.1,.25,.5}×beta∈{0,1,5,20}` + `gpt_only`),
  `[257,4]` feature matrix (dropped Track A's redundant `is_gpt_only`/`uses_state_read`
  flags — `k==0` already encodes that). Verified: 257 actions, shape `(257,4)`,
  `gpt_only` row `[0,1,1,0]`.
- `query_knn_indices` (`GPT_Module/knn.py`) — sibling to `query_knn`, returns raw
  neighbor indices instead of gathered values (needed for dense/GPU gather). Didn't
  modify `query_knn` itself — same "add a sibling, don't touch existing callers"
  pattern as `run_batch`/`run_batch_with_entropy`.
- `counters_to_dense` (`DIME/q_read_dense.py`) — verified via round-trip
  (`{5:8,2:2}→[[5,2,0,0]]`, counts/totals all matched).
- `mix_dime_and_lm` extended **in place** with optional `beta=0.0, global_freq=None`
  (safe — `beta=0` mathematically reduces to the exact original formula, so every
  existing caller is unaffected) + `build_global_freq` (both in `mixing.py`). Verified:
  `beta=0` reproduces the historical no-match NLL (`1.4917`) exactly; `beta=5` with a
  nonzero global frequency for the true token correctly *lowers* NLL (`1.4732`) by
  rescuing probability mass from the global prior even when no retrieved cluster
  empirically contained that token.
- `dense_p_state` / `dense_mix_nll` (`q_read_dense.py`) — GPU/PyTorch batched version
  of the same mixing math. Verified: **bit-for-bit match** against the Counter-based
  path on the same toy setup (`0.8592`, `1.4917`).
- `compute_reward_matrix_dense` (`q_read_dense.py`) — `[N,A]` reward matrix, looping
  over actions but each iteration now a batched GPU op instead of `N×k` dict lookups.
  Verified: `gpt_only` column exactly zero, other column matches
  `nll_gpt - [0.8592,1.4917]` within float32/rounding precision.
- `dense_state_entropy_purity` + `build_obs_features` (`q_read_dense.py`) — own leaner
  **11-feature** observation vector (vs. Track A's 22): `gpt_entropy` (from
  `predictive_entropy`, already sitting in the extraction cache — no new computation
  needed), `dist_top1/top4_mean/top8_mean/gap_1_2`, `entropy_top1/top4_mean`,
  `purity_top1/top4_mean`, `count_top1_log/top4_mean_log`. All verified exact-match
  against hand-computed values on a toy 4-cluster/8-neighbor setup. **Nice side
  effect of this design:** raw kNN can be represented in the exact same dense format
  as DIME (each raw entry = a degenerate size-1 "cluster"), so this whole pipeline
  works unchanged for raw kNN too — satisfies the original roadmap's step 13
  ("apply same Q-read code to both raw memory and DIME memory") with one shared
  implementation.

**Specified, not yet verified (next up):**
- `subsample_qa_pairs` — subsamples `(query,action)` pairs from the full `[N,A]`
  reward matrix into training data (matches Track A's own `max_q_samples`
  subsampling; `N×A` can be ~23M pairs at WikiText-103 scale, too many to train on
  directly). Reuses the existing `train_q_read_controller` (sklearn `MLPRegressor`)
  unchanged — it's already a generic `(X,y)→model` wrapper.
- `apply_q_controller` — scores all `A` actions per query, argmax, batched (mirrors
  Track A's own `batch_q` batching). Isolation test given (a `FakeModel` mechanics
  check, `chosen` should be `[1,1,1]`) but not yet confirmed with real output.

**First real end-to-end run (TinyStories/gpt2/minibatch_kmeans, `n_clusters=500`) —
a genuine, important negative result, not yet a success:**

```
mean_nll_gpt_only:              2.7984323501586914
mean_nll_q_read_multi_action:   2.795409126307937
mean_nll_oracle_multi_action:   2.53417504059039
best_fixed_action:              k50_t2.0_a0.05_b1.0  -> mean_nll 2.7469928663060403
```

| | Mean NLL |
|---|---|
| GPT-only | 2.798 |
| **Q-read multi-action (learned)** | **2.795** — barely better than doing nothing |
| best single fixed action, same 257-action grid | **2.747** — beats the learned policy |
| oracle (best action per query, hindsight) | 2.534 — huge unclaimed headroom |

**The learned multi-action policy currently underperforms a simple fixed baseline
picked from the same action grid**, and only barely beats GPT-only. `chosen_action_counts`
explains why: the policy collapsed to essentially two choices — `gpt_only` (2,941/14,097
≈ 21%) and a single action, `k300_t5.0_a0.05_b20.0` (10,257/14,097 ≈ 73%) — with the
other 255 actions getting single-digit-or-zero picks. Not genuine per-query
differentiation; close to a near-constant policy, and a weak one (barely beats
GPT-only despite "retrieving" 73% of the time). The oracle gap (`2.798 → 2.534`
possible vs. `2.795` actually achieved) shows the action grid itself contains real,
large headroom the current Q-MLP isn't capturing.

**Diagnosis, not yet fixed:** the `sklearn.MLPRegressor` call in
`train_q_read_controller` has no input/target normalization, no early stopping tuned
against a held-out slice — exactly the things Track A's own `train_q_mlp` did
carefully (z-scoring `X`/`y`, train/dev split with patience-based early stopping) that
this project's version doesn't yet have. Given the earlier design note that sklearn
was chosen deliberately over hand-rolled PyTorch (the MLP itself isn't the mechanism
under study), the fix should stay within sklearn's own tools —
`MLPRegressor(early_stopping=True, validation_fraction=..., n_iter_no_change=...)`
plus manual feature/target normalization before fitting — rather than abandoning that
earlier decision.

**Fix applied and confirmed working (2026-09-09):** added input/target normalization
to `train_q_read_controller` (`DIME/q_read.py`) via a `NormalizedMLPWrapper` — z-scores
`X`/`y` before fitting, transparently un-normalizes on `.predict()` so no caller needed
to change. Verified locally against the synthetic isolation test first (predictions
`[2.02,-1.07,-1.03]` vs. true `[2,-1,-1]` — actually tighter than the pre-fix version's
`[2.19,-0.95,-1.12]`). Re-ran the real TinyStories multi-action job:

```
mean NLL, GPT-only:                     2.7984323501586914
mean NLL, best single fixed action:     2.7469928663060403 (k50_t2.0_a0.05_b1.0)
mean NLL, Q-read (learned, per-query):  2.7307902168953184
mean NLL, oracle (perfect per-query):   2.53417504059039
```

**The learned multi-action policy now beats the fixed baseline** (2.731 vs. 2.747,
inverted from the broken run's 2.795 vs. 2.747) and action usage is healthy —
`gpt_only` still the single most common pick (~45%) but the rest spread across many
genuinely different actions instead of collapsing to one dominant choice. Oracle gap
narrowed from ~0.26 to ~0.20 (2.731 vs. 2.534) — real headroom still exists, but the
mechanism is now doing genuine, positive work. Point 3 is functionally done for
TinyStories; still needs re-running against WikiText-103/gpt2-medium for consistency
with everything else in this project.

**Not yet started:** wiring the (now-working) multi-action pipeline against the cached
WikiText-103/gpt2-medium data; points 5 and 6 from the gap review.

## Multi-seed variance (TinyStories) — DONE, the strongest rigor result in the project

Ran overnight (2026-09-09): `run_tinystories_multiseed.py` reuses the already-tuned
`seed=42` hyperparameters (no re-grid-search per seed — the point is testing
robustness of an already-established result, not re-discovering it) across 100 seeds
(`42`–`141`), rebuilding the full split/datastore/DIME-cluster/equal-budget-raw
pipeline fresh each time. `python -u` used this time — no buffering repeat of the
compression-sweep scare. Checkpointed after every seed.

```
gpt_only:                  mean=2.7946  std=0.0280  n=100
raw_knn_tuned:              mean=2.6943  std=0.0269  n=100
dime_tuned:                 mean=2.7449  std=0.0266  n=100
raw_kmeans_representative:  mean=2.8000  std=0.0292  n=100
```

Ran proper **seed-level** paired significance tests (distinct from — and stronger
than — the earlier position-level paired tests from Phase F17, which paired across
`val` positions within *one* seed's split; this pairs across 100 *independent* random
data splits, the classical, most convincing form of robustness evidence):

```
DIME beats GPT-only:                        100/100 seeds, mean diff=0.0497, paired-t p=5.23e-104
DIME beats raw_kmeans_representative:       100/100 seeds, mean diff=0.0551, paired-t p=3.19e-79
raw_knn_tuned beats DIME (expected, honest): 100/100 seeds, mean diff=0.0507, paired-t p=1.79e-101
```

**DIME won every single one of 100 independent random splits against both GPT-only
and the best equal-budget raw variant — and lost every single one against
uncompressed raw kNN, exactly as expected.** No inversions, no lucky/unlucky seeds in
either direction. This directly closes the single biggest rigor gap identified by
both the original DIME draft (which had "seed variation only for GPT-2 medium... error
bars are thin" as an explicit limitation) and this project's own gap-audit. Saved to
`DIME/results/tinystories_multiseed.json`.

**Still not done:** the equivalent multi-seed sweep for WikiText-103/gpt2-medium (this
was TinyStories-only, chosen for speed — ~100 seeds completed in ~45 minutes).

## TinyStories extras: Point 5, Point 6, and a TinyStories-scale compression sweep

Ran as one combined, per-section-checkpointed script (`run_tinystories_extras.py`) —
Point 5 diagnostics, Point 6's two new raw-baseline criteria, and a TinyStories
compression sweep, all against the same seed=42, `n_clusters=500` tuned datastore.

**Point 5 (mechanism diagnostics):**

```
hit@1: 0.6885   hit@4: 0.8166   hit@8: 0.8584
mean p_state(true) at nearest state: 0.1085
```

Sensible, monotonically increasing hit-rate as `k` grows. `p_state(true)≈0.11` means
the single nearest cluster puts ~11% of its mass on the true token on average —
meaningfully above naive chance, real signal.

**One real bug caught here, not yet re-confirmed:** the active-state-fraction
computation had a variable-unpacking-order mistake — `query_knn_indices` returns
`(distances, neighbor_idx)`, but the script did `nearest_idx_arr, _ = query_knn_indices(...)`,
silently assigning *distances* (continuous, effectively all-unique per query) to what
should have been discrete cluster indices bounded by `[0, 500)`. Symptom: printed
`"28.1020 (14051/500)"` — a fraction over 1.0, mathematically impossible, since you
can't have more active clusters than the 500 that exist. Fixed
(`_, nearest_idx_arr = query_knn_indices(...)`) and a small standalone recheck script
(`check_active_state_fraction.py`) written — cheap enough not to need rerunning the
whole combined job. **Not yet re-confirmed with real output** — top-10 helpful/harmful
cluster IDs printed in the original run may also need a sanity recheck since they
don't depend on this specific bug (they use `nll_delta` grouped by `nearest_idx_arr`
too — same buggy array!). Treat the "top helpful/harmful clusters" numbers from the
first run as suspect until rechecked alongside the fraction fix.

**Point 6 (two new raw-baseline selection criteria):**

```
raw_token_rarity:  n=500  mean NLL = 2.9020
raw_coverage:      n=500  mean NLL = 2.8313
```

Both lose to DIME (2.745) and to the earlier best raw variant (`raw_kmeans_representative`,
2.782) — reinforces that density-aware clustering selection beats either frequency-biased
(`rarity`) or naive-max-spread (`coverage`, greedy farthest-first) selection alone.

**Compression sweep (TinyStories scale, `n_clusters ∈ {500, 2000, 5000, 13000}`,
matching WikiText-103's ~95x/25x/9.5x/3.8x ratios but relative to TinyStories' smaller
~49,657-entry datastore):**

```
raw kNN tuned ceiling: 2.6941
n_clusters=   500  ratio= 99.3x  val NLL=2.7452  gap=0.0511
n_clusters=  2000  ratio= 24.8x  val NLL=2.7255  gap=0.0314
n_clusters=  5000  ratio=  9.9x  val NLL=2.7158  gap=0.0217
n_clusters= 13000  ratio=  3.8x  val NLL=2.7070  gap=0.0129
```

Same diminishing-returns shape as the WikiText-103 sweep. **Directly confirms the
earlier hypothesis about why the original DIME draft's much milder compression (4x-10x,
~4-10 raw points per cluster) came close to matching raw kNN**: at `13000` clusters here
(~3.8 raw points per cluster — very close to the old draft's own compression regime),
the gap (`0.0129`) is now *smaller than one standard deviation* (~0.027-0.029) from the
100-seed variance measurement — genuinely within noise territory. Saved to
`DIME/results/tinystories_extras.json`.

## TinyStories rich ablation table (8 variants, matching the old draft's structure)

`run_tinystories_rich_ablations.py` — launched, not yet returned results. Adds
`top64/top32/top16` (finer truncation than the earlier top-5-only ablation),
`global_unigram` (every cluster reports the same corpus-wide token distribution,
reusing `build_global_freq` — a sharper illusion-check than shuffling, since it
removes all cluster-specific content rather than just scrambling it),
`random_partition` (recomputed under this table's shared tuned config, not the old
untuned Phase C number), and two distinct shuffle controls: `shuffled_distribution`
(existing — correct prototypes, scrambled content) and `shuffled_prototype` (new —
correct distributions, scrambled keys, so retrieval finds essentially a random
cluster; this interpretation of "shuffled_prototype" wasn't fully specified by the old
draft, so it's this project's own best-reasoned construction, documented as such, not
copied). All 8 variants share the same tuned `(k=20,tau=2.0,alpha=0.05)` config for a
fair table.

## Git/ops note: generated caches must never be committed

Hit this for real: a `git add -A` on the cluster (after `extract_and_cache.py` had
generated `GPT_Module/cache/*.npz`, one over 1.3GB) got committed and rejected by
GitHub's 100MB file limit. Since the rejected push meant that commit never reached
`origin`, it was safe to rewrite locally (`git reset --soft origin/main` +
selectively un-stage just the cache directory + recommit) without any shared-history
risk — but it's a real trap to avoid going forward. `.gitignore` now excludes
`GPT_Module/cache/` and `*.npz` (logs/`results/*.json` stay tracked, by explicit
choice — only the regeneratable multi-hundred-MB caches are excluded).

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
- **When a script/function name differs from what's expected but the tool shows the
  file changed on disk, read the actual current file before assuming.** Twice now
  (`state_object.py`'s `compressed_dists = [0]` bug, `q_read_dense.py` missing
  `counters_to_dense` after a later piece was pasted in) the real bug was found only
  by reading the file fresh, not by reasoning about what "should" be there.
- **Checking a SLURM job's log file while it's still running is not the same as
  checking it after it finishes.** Python's `print()` fully block-buffers when stdout
  is redirected to a file (every SLURM job's case) — bash's own `echo` doesn't. Every
  job checked *after* completion looks fine regardless (buffers flush at exit), so
  this was invisible until the first time a still-running job's log was checked
  mid-flight (the compression sweep). The reliable progress signal for a long-running
  job is an incremental file checkpoint (`save_results` after each unit of work), not
  the stdout log. Fix going forward: add `python -u` / `PYTHONUNBUFFERED=1` to submit
  scripts that will be checked while still running.
- **Adding new *optional keyword arguments with backward-compatible defaults* to an
  existing function is safe and doesn't need a sibling function** — unlike changing a
  function's *positional return shape* (which breaks every caller doing tuple
  unpacking, hence `run_batch_with_entropy` being a separate function from
  `run_batch`). `mix_dime_and_lm` grew `beta=0.0, global_freq=None` in place because
  `beta=0` provably reduces to the original formula exactly.
- **When porting an idea from the reference repo, check for sign/convention
  mismatches before reusing any code, even when copying is otherwise off the table.**
  Track A's `alpha` means the opposite of this project's own `alpha`. Always keep this
  project's own established convention rather than the reference's, and flag the
  mismatch explicitly so it doesn't cause confusion later.
- **A Python `Counter`-based loop that's fine for one hyperparameter combo at a time
  can become a billions-of-iterations problem once generalized to a large action
  grid.** The fix was converting to a dense, padded array representation (which turned
  out to already be what the reference repo's own state files use) enabling batched
  GPU tensor ops — not just "add more compute."
