# DIME Paper — Full Context Dump

Generated 2026-09-10 for use in other Claude instances/sessions while writing the paper.
This file is self-contained — paste it into a fresh conversation and it has everything
needed to help draft Motivation, Method, Setup, Results, Discussion, and Limitations.

## 1. What this project is

Hand-built recreation of **DIME (Distributional Memory Entries)** — a compressed
kNN-LM retrieval memory for a frozen GPT-2 — from scratch, in `e:/TrackA/Study`, for
genuine understanding (not copying an existing implementation). This is a formal
recreation of an earlier, messier round of experiments already written up in an early
draft at `C:\Users\Acer\Documents\DIME\final.pdf`:

> **"DIME: Distributional Memory Entries for Compact kNN Language Modeling"**
> Sukrit Koirala & HitMan Gurung (co-author), Dakota State University, July 2026.

That draft's **Motivation and Method sections describe the general idea only (no
numbers)** and are likely reusable near-as-is as the paper skeleton. Its old empirical
numbers (4x-10x compression, WikiText-2, a different GPT-only baseline) do **not**
match this recreation's numbers (~99x compression, WikiText-103 as the primary
large-scale target) — different specific configs of the same idea, not directly
comparable. The old draft explicitly flagged as missing/limitations: equal-budget raw
baselines, proper paired significance testing, and thin/absent seed-variance error
bars — **this recreation directly addresses all three** (see sections 4-6 below).

Professor (Austin) reviewed this recreation and gave two concrete asks that shaped the
post-roadmap work: (1) formalize the mathematics of storage/retrieval metrics
(`MATH_FORMALIZATION.md`, in progress — see section 9), (2) extend beyond one
dataset/model (the 3×2 grid — see section 5).

## 2. The method, in one paragraph

Standard kNN-LM (Khandelwal et al.) builds a datastore of `(hidden_state, next_token)`
pairs from a frozen LM's forward pass over a corpus; at inference, retrieve the `k`
nearest neighbors to the current hidden state, turn their distances into weights via
`softmax(-distance/tau)`, and mix the resulting retrieval distribution with the LM's
own distribution: `p_mixed = alpha * p_retrieval + (1-alpha) * p_LM`. DIME's
compression: instead of storing every raw `(hidden_state, token)` pair, cluster the
datastore (k-means by default) into `B` clusters, and store one prototype hidden state
+ one aggregated next-token count distribution per cluster. Retrieval becomes
`(prototype, distribution)` pairs; mixing generalizes to `p_state = count(true_token in
cluster) / total_count(cluster)` instead of a point-mass match. A `beta` parameter
(Laplace-style shrinkage toward global corpus token frequency) is an optional
extension used only in the multi-action Q-read work (section 6). Compression ratio =
`N / B` (raw datastore size / cluster count) — this recreation predominantly uses
~99-100x compression (e.g., 500 clusters from ~49,657 raw entries on TinyStories, or
~3,800-4,000 clusters from ~381,127 raw entries on WikiText-103), which is far more
aggressive than the old draft's 4x-10x — this difference (**points-per-cluster**, see
section 7) explains several quantitative gaps between old and new results.

## 3. Experimental discipline (methods section material)

- **Leak-proof three-way split**: every story/article is randomly assigned *whole* to
  one of `datastore` (70%) / `controller_train` (15%) / `val` (15%) before any
  chunking, so no chunk from one story ever lands in two different splits. Verified
  with an explicit leakage audit early in the project (zero leaks found).
- **Hyperparameter tuning rule**: grid search over `(k, tau, alpha)` always runs
  against `controller_train`; `val` is touched exactly once, to report the final
  number for whichever config won on `controller_train`. This is enforced throughout
  — never report a `val` score for a hyperparameter combination chosen by looking at
  `val` itself.
- **Equal-budget fairness baselines**: for every DIME result at `B` clusters, compare
  against `B` *raw* (uncompressed) entries selected by the best available raw
  selection criterion (`raw_kmeans_representative`: cluster like DIME, but keep the
  real raw point nearest each centroid instead of synthesizing a prototype+
  distribution) — this isolates whether DIME's benefit is really about compression,
  not just "having 	B well-chosen pointers."
- **Illusion-check ablations**: shuffle which cluster's *content* (distribution) is
  reported at which cluster's *geometric position* (prototype). If the mixing formula
  or GPT's own behavior were doing all the work, shuffling should barely matter. If
  location↔content correspondence is what's really helping, shuffling should hurt —
  ideally past the point of even beating GPT-only.
- **Significance testing**: paired t-test + Wilcoxon signed-rank test on the paired
  per-position NLL differences (DIME vs. each baseline), same `val` positions for
  both. Later extended to **seed-level** paired tests (100 independent seeds,
  TinyStories only so far) — the more classical, stronger robustness claim.

## 4. Original roadmap results — TinyStories / gpt2-small (COMPLETE, all phases A-F)

`seq_len=128`, datastore 49,657 / val 14,097 (varies slightly run to run depending on
exact script; 9,525-14,097 range across different phases due to different chunk
counts used at different points — always noted per-result).

| Stage | Mean NLL | Note |
|---|---|---|
| GPT-only (no memory) | 2.798 | reference baseline |
| raw kNN, untuned (49,657 entries) | 2.757 | |
| raw kNN, tuned (grid search, k=50,tau=2.0,alpha=0.1) | 2.694 | |
| raw kNN, Q-read (adaptive, binary retrieve/don't) | 2.689 | best raw kNN result |
| random_partition (500 clusters, dumb control) | 2.995 | worse than doing nothing |
| minibatch_kmeans, untuned (500) | 2.792 | |
| query_kmeans (500, clusters fit on controller_train) | 2.794 | |
| utility_weighted (500, NLL-weighted clusters) | 2.805 | negative result, worse than GPT-only |
| **minibatch_kmeans, tuned (500, k=20,tau=2.0,alpha=0.05)** | **2.745** | |
| minibatch_kmeans, Q-read (binary adaptive, 500) | 2.743 | best binary-Q-read DIME result |
| minibatch_kmeans, multi-action Q-read (257 actions) | **2.731** | beats best fixed action (2.747); oracle ceiling 2.534 |
| best raw variant at equal 500-entry budget (`raw_kmeans_representative`) | 2.782 | still loses to DIME |

**Every one of 6 equal-budget raw-500 selection criteria loses to both DIME and
GPT-only** at TinyStories scale:
`raw_random`=2.808, `raw_high_gpt_loss`=2.861, `raw_low_gpt_loss`=2.902,
`raw_high_entropy`=2.881, `raw_low_entropy`=2.902, `raw_kmeans_representative`=2.782
(best of the six, still loses to DIME). Point 6 extras: `raw_token_rarity`=2.902,
`raw_coverage`(greedy farthest-first)=2.831 — also both lose to DIME.

**Significance (Phase F17):** GPT-only vs. DIME tuned: mean_diff=0.0532, paired-t
p=2.97e-42, Wilcoxon p≈0. Best-raw-equal-budget vs. DIME tuned: mean_diff=0.0366,
paired-t p=1.55e-28, Wilcoxon p≈0. (n=14,097 val positions in this run.)

**Illusion-check ablations (rich, 10-variant table, k=20/tau=2.0/alpha=0.05, n=500):**

| Variant | Mean NLL | vs. full |
|---|---|---|
| original (full distribution) | 2.7452 | — |
| top64 truncation | 2.7448 | slightly better |
| top32 truncation | 2.7445 | slightly better |
| top16 truncation | 2.7441 | slightly better (best of all truncations) |
| top5 truncation | 2.7471 | +0.07%, basically free |
| majority_token (collapse to single token) | 2.7646 | +0.71%, real modest loss |
| global_unigram (every cluster reports the same corpus-wide distribution) | 2.8168 | worse than GPT-only |
| random_partition (dumb clustering control) | 2.8175 | worse than GPT-only |
| shuffled_prototype (content correct, geometry shuffled) | 2.8317 | worse than GPT-only |
| shuffled_distribution (geometry correct, content shuffled) | 2.8349 | worse than GPT-only, worst of all |

**Key finding: `shuffled` variants land *worse than GPT-only alone* at TinyStories
scale** — strong, direct evidence the location↔content correspondence is doing real
work, not an artifact of the mixing formula. Top-16/32/64 truncation actually
*improves slightly* on the full distribution (filters long-tail noise) — a genuine
efficiency lever, real signal starts being lost only at top-5.

**Efficiency (real measured, not formula-based):** entries 49,657→500 (99.3x fewer),
bytes 152.94MB→1.65MB full/1.55MB top-5/1.54MB majority-token (92.6x/98.5x/99.1x
smaller), latency 0.352ms→0.006ms per query (59x faster). Note: byte savings from
value-truncation are small relative to the savings from clustering itself, since the
500 clusters' *keys alone* already cost ~1.54MB, dominating total DIME size.

**Point 5 diagnostics:** hit@1/4/8 = 0.689/0.817/0.858 (probability the true token
appears anywhere in the top-1/4/8 retrieved clusters' distributions), mean
p_state(true)=0.109, active-state fraction 499/500 (99.8% of clusters get used as a
top-1 match at least once across ~14K queries — healthy utilization, not a few
dominant clusters).

**Multi-seed variance (100 seeds, 42-141, THE strongest rigor result in the
project):** reusing already-tuned hyperparameters, full pipeline rebuilt fresh per
seed (different random story split each time). Results (n=100 each):
`gpt_only` 2.7946±0.0280, `raw_knn_tuned` 2.6943±0.0269, `dime_tuned` 2.7449±0.0266,
`raw_kmeans_representative` 2.8000±0.0292. **Seed-level paired tests: DIME beat
GPT-only in 100/100 seeds (p=5.23e-104), beat raw_kmeans_representative in 100/100
seeds (p=3.19e-79), lost to raw kNN in 100/100 seeds as expected (p=1.79e-101) — zero
inversions in either direction.** This is a stronger, more classical robustness claim
than the position-level paired tests above, and directly answers the old draft's own
flagged gap (thin/absent seed variance).

**Compression-ratio-vs-quality sweep** (does relaxing compression let DIME cross above
raw kNN's own tuned ceiling of 2.694?):

| n_clusters | compression ratio | DIME val NLL | gap to raw ceiling |
|---|---|---|---|
| 500 | 99.3x | 2.7452 | 0.0511 |
| 2,000 | 24.8x | 2.7255 | 0.0314 |
| 5,000 | 9.9x | 2.7158 | 0.0217 |
| 13,000 | 3.8x | 2.7070 | 0.0129 |

Diminishing-returns pattern: DIME closes the gap to raw kNN as compression relaxes,
but never crosses above it (mathematically it can only converge to equality as
`n_clusters → N`). At 13,000 clusters (~3.8 points/cluster — matching the OLD draft's
own compression regime almost exactly), the gap (0.0129) falls below the seed-noise
standard deviation from the multi-seed study (~0.027) — **direct confirmation of why
the old draft's much milder compression nearly matched raw kNN**, while this
recreation's much more aggressive ~99x compression shows a clearly bigger gap.

## 5. The 3×2 grid — {TinyStories, WikiText-103, WikiText-2} × {gpt2, gpt2-medium}

Decided to expand beyond the single TinyStories/gpt2-small + WikiText-103/gpt2-medium
pair before meeting Austin, to show the core finding holds across both dataset domain
and model size. Built parametrized scripts (`DIME/run_tier1_core.py`,
`run_tier2_rigor.py`, `run_tier3_extras.py`, `run_generic_multiseed.py`) reusable
across settings via `--dataset`/`--model` flags. **Tier framing: Tier 1 (core
compression claim) + Tier 2 (Q-read/efficiency/illusion-check) is what the paper
needs, for every setting. Tier 3 (remaining construction methods, remaining raw
baselines, rich ablation table, multi-action Q-read, Point 5 diagnostics) is
enrichment, not required.**

All settings below auto-scale `n_clusters` to `max(100, round(datastore_size/100))`
(kept at ~100x compression, consistent with the original TinyStories result) and grid
search over `k∈{10..300}, tau∈{0.5..10}, alpha∈{0.01..0.5}` (wide enough to
self-tune per setting without hand-picking).

### 5a. Tier 1 results (GPT-only, raw kNN tuned, DIME tuned, best equal-budget raw,
significance) — **5 of 6 settings valid and done**

| Setting | datastore N | n_clusters | GPT-only | raw kNN tuned | DIME tuned | raw_kmeans_rep | DIME vs GPT-only (p) | DIME vs raw_rep (p) |
|---|---|---|---|---|---|---|---|---|
| TinyStories/gpt2 (original) | 49,657 | 500 | 2.798 | 2.694 | **2.745** | 2.782 | 2.97e-42 | 1.55e-28 |
| TinyStories/gpt2-medium | 381,000 | 3,810 | 2.501 | 2.404 | **2.463** | 2.590 | 3.70e-126 | ≈0 |
| WikiText-103/gpt2 | 381,127 | 3,811 | 4.361 | 3.883 | **4.011** | 4.190 | ≈0 | ≈0 |
| WikiText-103/gpt2-medium | 381,127 | 3,811 | 4.049 | 3.659 | **3.754** | 3.956 | ≈0 | ≈0 |
| WikiText-2/gpt2 | **INVALID — see 5c** | | | | | | | |
| WikiText-2/gpt2-medium | **INVALID — see 5c, re-run in progress** | | | | | | | |

**Core finding replicates in all 4 valid new settings, all p≈0 or effectively so**:
DIME beats GPT-only and beats the best equal-budget raw alternative, at every tested
combination of domain (TinyStories vs. Wikipedia) and model size (gpt2 vs.
gpt2-medium).

### 5b. Tier 2 results (binary Q-read, efficiency, 3-variant illusion-check ablation)

| Setting | raw Q-read | DIME Q-read | entries ratio | bytes ratio | latency ratio | majority_token | top5 | shuffled |
|---|---|---|---|---|---|---|---|---|
| TinyStories/gpt2-medium | 2.379 | 2.440 | 100.0x | 95.4x | 94.2x | 2.472 | 2.456 | **2.588** |
| WikiText-103/gpt2 | 3.882 | 4.013 | 100.0x | 91.0x | 79.2x | 4.049 | 4.009 | **4.272** |
| WikiText-103/gpt2-medium | 3.657 | 3.752 | 100.0x | 93.1x | 68.6x | 3.770 | 3.739 | **3.985** |

Compare `shuffled` to that setting's own GPT-only baseline (table 5a):
- TinyStories/gpt2-medium: shuffled 2.588 **>** GPT-only 2.501 — shuffle destroys the
  advantage entirely, same pattern as original TinyStories/gpt2 result (2.835 > 2.798).
- WikiText-103/gpt2: shuffled 4.272 **<** GPT-only 4.361 — shuffle badly hurts DIME's
  own result (huge gap from tuned 4.011) but does *not* fully erase the advantage over
  plain GPT.
- WikiText-103/gpt2-medium: shuffled 3.985 **<** GPT-only 4.049 — same pattern, thin
  remaining margin.

**Cross-dataset divergence worth stating explicitly in Discussion, not smoothing
over: at TinyStories scale, shuffling fully erases DIME's advantage (worse than doing
nothing); at WikiText-103 scale (both model sizes), shuffling severely degrades DIME
but doesn't fully erase the advantage over GPT-only.** The illusion-check mechanism
(location↔content correspondence matters) is confirmed in every setting by the sheer
size of the shuffle-induced degradation relative to tuned DIME — what varies is only
whether that degradation is large enough to also cross back below GPT-only.

`top5` truncation matches or slightly beats the full distribution in every setting
tested so far (2.456 vs 2.463 dime-tuned at TinyStories/medium; 4.009 vs 4.011 at
WikiText-103/gpt2; 3.739 vs 3.754 at WikiText-103/medium) — consistent efficiency
lever across all settings.

### 5c. WikiText-2 — methodological issue found, re-run in progress (NOT YET LANDED)

**Important finding, keep this in the paper's Setup/Limitations section regardless of
final numbers:** WikiText-2 and WikiText-103 share the same article ordering (WT2 is
a genuine subset of WT103's article list — confirmed directly: both stream identical
text starting with "Valkyria Chronicles III"). The default extraction target
(3000/500/500 chunks ≈ 552K tokens) is well within WikiText-2's own ~2.09M-token
budget, so early WikiText-2 extractions (both gpt2 and gpt2-medium) accidentally only
consumed the *shared-prefix* region — content that's also verbatim in WikiText-103 —
making those results non-independent duplicates, not a genuine additional data point.
**Confirmed via direct stream inspection:** `load_dataset('wikitext',
'wikitext-2-raw-v1', split='train', streaming=True)` fully drained gives exactly
36,718 rows / 10,892,990 chars (~2.09M tokens) — the loader itself is correct, this
is a genuine property of the two datasets, not a pipeline bug.

**Fix in progress:** re-extracting both WikiText-2 settings with deliberately
oversized targets (`N_DS=N_CT=N_VAL=100000`, all unreachable) to force genuine full
corpus exhaustion (~2.09M tokens total, all 36,718 rows) instead of an early
stopping point. **Framing for the paper: this makes WikiText-2 a "total data
scarcity" data point (DIME's behavior when literally no more data could be added,
even if desired) — not a "smaller than WikiText-103's extraction" data point, since
full WikiText-2 (~2.09M tokens) is actually *larger* than the ~550K-token slice
currently used for TinyStories/WikiText-103.** Do not claim WikiText-2 tests "domain
diversity" either — content is not independent of WikiText-103's beginning.
**Check `DIME/results/wikitext2_gpt2_tier1.json`/`tier2.json` and the `-medium`
equivalents for real numbers once the re-run lands — do not use any pre-2026-09-10
WikiText-2 result files, they're from the invalid early-stopped extraction.**

## 6. Multi-action Q-read (WikiText-103/gpt2-medium partial, TinyStories complete)

Generalizes the binary Q-read gate to a genuine per-query policy over a 257-action
grid (`k∈{50,100,200,300} × tau∈{0.5,1,2,5} × alpha∈{0.05,0.1,0.25,0.5} × beta∈
{0,1,5,20}`, plus a `gpt_only` no-retrieval action). State features (label-free, no
leakage): GPT's own predictive entropy, nearest-neighbor distance(s), retrieved
cluster's entropy/purity, log cluster size — 11-dim observation vector. Reward =
`NLL_GPT - NLL_action`, fit via a normalized MLP (`NormalizedMLPWrapper` — z-scores
inputs/targets; without this the policy collapsed to ~2 dominant actions and barely
beat GPT-only). TinyStories result (full, verified): **learned policy 2.731, beats
best single fixed action 2.747, GPT-only 2.798, oracle (perfect hindsight) ceiling
2.534** — genuine improvement from adaptivity, not yet close to the oracle gap (real
headroom exists). A GPU-vectorized dense pipeline (`DIME/q_read_dense.py`) makes this
tractable at WikiText-scale (avoids billions of Python dict lookups). Not yet run for
the new 3×2 grid settings (Tier 3 item, optional).

## 7. Points-per-cluster — the recurring explanatory variable

This recreation compresses far more aggressively (~99-100x, ~95-100 raw points per
cluster) than the old draft (4x-10x, ~4-10 raw points per cluster). This single
variable explains two otherwise-puzzling cross-comparisons:
1. **Bigger gap to raw kNN here than in the old draft** — more aggressive compression
   discards more position-level distinction.
2. **Smaller-magnitude ablation degradations here than in the old draft's own
   ablation table** (e.g. majority_token: +0.019 here vs. +0.15 there, ~7.5x bigger
   in the old draft) — more raw points pooled per cluster makes the aggregate
   token-count statistics more robust to certain kinds of corruption (majority-token
   collapse, shuffling), even though it also means more information is discarded
   overall. Same underlying mechanism, opposite-direction consequence, both now
   documented with real numbers (section 4's compression sweep is the direct
   confirmation: at 13,000 clusters, ~3.8 points/cluster matching the old draft's
   regime, the gap to raw kNN's ceiling shrinks below one seed-noise standard
   deviation).

## 8. Cross-dataset findings that do NOT replicate cleanly — flag in Discussion

Do not present these as settled; state both directions honestly:
- **`utility_weighted` construction method**: worst DIME variant at TinyStories scale
  (2.805, worse than GPT-only) but the *best* variant at WikiText-103/gpt2-medium
  scale (3.743, edges out minibatch_kmeans's 3.754) — see `DIME/results/wikitext_extras.json`.
  Open question, not resolved — only one data point each direction so far.
- **`raw_random` (equal-budget baseline) vs. GPT-only**: loses to GPT-only at
  TinyStories scale (2.808 > 2.798) but *beats* GPT-only at WikiText-103/gpt2-medium
  scale (3.960 < 4.049). What *does* replicate: no raw variant beats DIME at equal
  budget, at either scale, and biased selection criteria (high/low loss, high/low
  entropy, rarity, coverage) always lose to unbiased `raw_random`.
- **Shuffled-ablation vs. GPT-only**: see section 5b — fully erases DIME's advantage
  at TinyStories scale, doesn't quite at WikiText-103 scale (both model sizes).

## 9. Math formalization — status (see `Study/MATH_FORMALIZATION.md`)

Done: byte/compression-ratio formulas (`M_raw = N(4D+8)`, `M_DIME = B(4D+8K)`,
cross-validated exactly against the real TinyStories measurement, 152.94MB), a
latency asymptotic argument. **Not yet done**: the mixing formula written up in clean
formal notation, a bias-variance semi-formal argument for why compression trades
position-fidelity for statistical robustness (motivated by section 7's
points-per-cluster finding), cross-validation of the byte formula against
WikiText-103's real numbers (data exists — `DIME/results/wikitext_extras.json`'s
efficiency section — just needs writing up).

## 10. File map (for looking up exact numbers/configs beyond what's summarized here)

- `Study/GPT_Module/PROGRESS.md` — the full, authoritative, chronological project log
  (1400+ lines). Read this for anything not covered above, especially exact bug
  histories and isolation-test details not relevant to the paper itself.
- `Study/MATH_FORMALIZATION.md` — math writeup in progress.
- `Study/DIME/results/*.json` — every raw result file. Tier 1/2 results for the new
  grid are named `{dataset}_{model}_tier1.json` / `_tier2.json`. Original roadmap
  results use older descriptive names (`significance_testing.json`,
  `tinystories_rich_ablations.json`, `efficiency_analysis.json`,
  `tinystories_multiseed.json`, `tinystories_extras.json`,
  `q_read_multi_action_baseline.json`, `wikitext103_gpt2_medium_replication.json`,
  `wikitext_extras.json`, `compression_sweep.json`).
- `Study/DIME/run_tier1_core.py` / `run_tier2_rigor.py` / `run_tier3_extras.py` /
  `run_generic_multiseed.py` — the reusable scripts behind the new grid, useful if
  you need to know exactly what a number measures.
- `C:\Users\Acer\Documents\DIME\final.pdf` — the old draft; Motivation/Method sections
  are reusable, numbers/Results section are not (different config, see section 1).

## 11. What's still outstanding as of 2026-09-10

- WikiText-2/gpt2 and WikiText-2/gpt2-medium: re-extraction with forced full-corpus
  consumption in progress, Tier 1+2 queued behind it. **Do not write the WikiText-2
  numbers into the paper until confirmed done.**
- Tier 3 (enrichment) not run for any of the 4 new grid settings — optional depth,
  not blocking.
- Multi-seed variance only exists for TinyStories/gpt2-small (100 seeds) — the other
  5 settings only have position-level paired significance tests, not seed-level. Say
  this precisely in the paper; don't imply seed-level robustness everywhere.
- Math formalization: mixing-formula notation + bias-variance writeup + WikiText-103
  byte-formula cross-validation still to be written (data exists, just needs prose).
