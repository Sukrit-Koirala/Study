# DIME-V2 Experiment C1 — Preflight Review (Revision 2)

Audience: an external research assistant with no repository access, who already read Revision 1 and raised 7 points. This revision addresses each one; it replaces Revision 1 in full.

## Revision summary (point-by-point)

1. **sklearn version / historical reproducibility.** `environment_metadata()` and `save_construction_artifact()` (new, lines 541–578) now record `python/numpy/sklearn/scipy/pandas` versions and `platform.platform()`, and persist the actual fitted `centroids`/`assignment` plus their SHA-256 hashes to `dime_v1_cluster_artifact.npz` / `construction_fingerprint.json` the moment this script re-derives clustering. No such artifact existed anywhere in the repository before this change (confirmed by search in Revision 1). This makes the **next** run comparable against a real saved ground truth; it does **not**, and cannot, retroactively recover the sklearn version used for the historical `tier1.json`/`run_manifest.json` artifacts — that remains genuinely unrecoverable and is stated as such, not glossed over.
2. **Routing distinction.** Inspected `DIME/mixing.py` directly (not inferred): `mix_dime_and_lm` loops over **all `k`** retrieved centroids per query with distance-weighted softmax mixing (`for j in range(k): ... weights[i,j] * p_state`) — genuine multi-centroid production inference. Separately, `build_splits_for` (R3 onward) reads **only `idx[:, 0]`**, the single nearest centroid, purely as a diagnostic "which cluster does this query belong to" grouping, never for prediction. The script now computes and reports **both**, as separate columns (`traffic_top1_*`, `traffic_top{k}_*`), with explicit normalization assertions for each (`assert_routing_invariants`, lines 520–539).
3. **Singleton clusters.** `population_stats` now reports `n_singleton` (count of `n_j==1`) as a field **separate from** `n_empty` (lines 141–168). A new synthetic check, `check_singleton_cluster_geometry`, verifies `cluster_geometry` on a single point returns finite, mathematically correct `R_j/mean_dist/rms/delta_j` (all equal to the one point's distance to the *fitted* centroid, which need not be the point itself) rather than being skipped or producing NaN; `dimensionality_stats` correctly returns `None` for `n<2` (PCA is genuinely undefined there) and that `None` is now kept **visible** in `dimensionality_results.csv` (with an explanatory `note` field) instead of being silently dropped, which Revision 1's sampling loop did.
4. **Spearman + stratified summaries.** `correlation_pair()` (lines 601–609) now returns `(pearson_r, spearman_r, n)` together, never one alone. `structural_relationships.csv` carries an explicit `scope` column: `all_nonempty_clusters` for every `n_j`/`R_j`/`sigma_j`/`V_j`/`rho_j`/entropy/traffic relationship, vs. `dimensionality_sample_n=40` for the two relationships involving `effective_rank` (which only exists for the bounded sample) — the two scopes are never mixed in one row. `stratified_population_summary()` (lines 580–599) buckets **all nonempty clusters** into 5 population quintiles and reports mean/median geometry and token statistics per quintile, written to `structural_summary.md`.
5. **Invariant checks.** `assert_real_data_invariants` (lines 493–518) is substantially strengthened: now also checks key/label alignment (`len(ds_keys)==len(ds_values)`), key dimensionality (`==768`), finite coordinates on keys/labels/centroids, `compressed_dists` length, and centroid array shape — not just the population-sum/assignment-range checks Revision 1 had. `assert_routing_invariants` (new) checks the routing index matrix's shape and that every index is in `[0, n_clusters)`, plus the resulting traffic vector's sum.
6. **Missing deliverables.** Revision 1 generated only 4 of 8 required files. The 4 missing ones — `structural_summary.md`, `construction_audit.md`, `validation_report.md`, `final_report.md` — are now implemented (`write_structural_summary_md`, `write_construction_audit_md`, `write_validation_report_md`, `write_final_report_md`) and called from `main()`. All 8 (plus the new `dime_v1_cluster_artifact.npz`/`construction_fingerprint.json`) are listed in Section C below.
7. **Execution guard.** Unchanged and still the first line of `main()` (`raise RuntimeError("PREFLIGHT HOLD...")`, line ~719). Only `run_synthetic_checks()` (now 10 checks, all on hand-built synthetic arrays — never a cache file) was run to validate this revision; verified again just before writing this document.

A full unified diff against Revision 1's script is available on request (575 lines); SHA-256 of the current file is given in Section B.

## A. Verified baseline configuration

Unchanged from Revision 1 except as noted; not repeated in full here to keep this revision compact. Key facts, still verified from source/artifact: datastore=381,127, controller_train=90,170, val=77,089, n_clusters=3,811 (`round(381127/100)`), `MiniBatchKMeans(n_clusters=3811, random_state=42)` with every other parameter left at the installed sklearn default, Euclidean distance, no normalization, float32 at fit time, centroids fit on the full datastore. **New in this revision**: the script itself now records and persists its own `python/numpy/sklearn/scipy/pandas` versions and `platform.platform()` on every real run (see point 1 above) — still local-dev-verified only as of this review (`sklearn 1.8.0`), not yet cluster-verified, since the real job has not run.

## B. Exact source-code references (new/changed since Revision 1)

**Routing mechanism, resolved** — `DIME/mixing.py`, lines 9–25 (exact, unmodified; the file this revision additionally inspected to resolve point 2):

```python
def mix_dime_and_lm(distances, retrieved_dists, true_targets, p_lm_true, tau=1.0, alpha=0.25, beta = 0.0, global_freq=None):
    """distances, retrieved_dists: [M, k] (retrieved_dists holds Counter objects).
    true_targets, p_lm_true: [M]."""
    M, k = distances.shape
    weights = softmax_np(-distances / tau, axis=1)  # [M, k]

    p_knn_true = np.zeros(M, dtype=np.float64)
    for i in range(M):
        true_tok = int(true_targets[i])
        global_p = global_freq[true_tok] if (beta > 0 and global_freq is not None) else 0.0
        for j in range(k):
            dist_counter = retrieved_dists[i, j]
            total = sum(dist_counter.values())
            denom = total + beta
            if denom > 0:
                count = dist_counter.get(true_tok, 0)
                p_state = (count + beta * global_p) / denom
                p_knn_true[i] += weights[i, j] * p_state
```

This is genuine multi-centroid mixing over all `k` retrieved centroids, confirming the production-inference routing mechanism. SHA-256 of `mixing.py`: `000785e75f827789b10a8c05fda43e10c09e1c7f31de246fc0a1e48bc38554b3`.

**New real-data invariant checks** — `run_dime_v2_c1_cluster_structure_audit.py`, lines 493–539 (both functions, in full):

```python
def assert_real_data_invariants(ds_keys, ds_values, populations, compressed_dists, assignment, dime_centroids, n_clusters):
    N = len(ds_values)
    assert ds_keys.shape[0] == len(ds_values), \
        f"key/label misalignment: {ds_keys.shape[0]} keys vs {len(ds_values)} labels"
    assert ds_keys.shape[1] == DIM, f"unexpected key dimensionality {ds_keys.shape[1]} (expected {DIM})"
    assert np.all(np.isfinite(ds_keys)), "datastore keys contain non-finite (NaN/inf) coordinates"
    assert np.all(np.isfinite(ds_values.astype(np.float64))), "datastore labels contain non-finite values"
    assert int(populations.sum()) == N, \
        f"cluster populations ({int(populations.sum())}) do not sum to datastore size ({N})"
    assert len(assignment) == N, "assignment length does not match datastore size"
    assert len(compressed_dists) == n_clusters, \
        f"compressed_dists has {len(compressed_dists)} entries, expected {n_clusters}"
    assert np.all((assignment >= 0) & (assignment < n_clusters)), \
        "assignment contains an out-of-range cluster id"
    assert dime_centroids.shape == (n_clusters, DIM), \
        f"centroid array shape {dime_centroids.shape} != expected ({n_clusters}, {DIM})"
    assert np.all(np.isfinite(dime_centroids)), "fitted centroids contain non-finite coordinates"
    for c, cd in enumerate(compressed_dists):
        token_sum = sum(cd.values())
        assert token_sum == populations[c], \
            f"cluster {c}: token counts sum to {token_sum}, population is {populations[c]}"
    assert np.all(np.isfinite(populations)), "non-finite population value"


def assert_routing_invariants(idx_matrix, n_clusters, k_expected, n_queries_expected):
    assert idx_matrix.shape == (n_queries_expected, k_expected), \
        f"routing index matrix shape {idx_matrix.shape} != expected ({n_queries_expected}, {k_expected})"
    assert np.all((idx_matrix >= 0) & (idx_matrix < n_clusters)), \
        "routing index matrix contains an out-of-range cluster id"
    traffic = routing_traffic(idx_matrix, n_clusters)
    expected_sum = float(k_expected)
    assert abs(float(traffic.sum()) - expected_sum) < 1e-6, \
        f"routing traffic does not sum to k={k_expected}: got {traffic.sum()}"
    return traffic
```

Full file to review in full (grew from ~420 to ~914 lines): `DIME/run_dime_v2_c1_cluster_structure_audit.py`.

**SHA-256 (full files, this revision)**:
- `38ec302a0474435751ac9ffd0fa3fadebfae5fc3823ebdd826de2ac0066d9487` `DIME/run_dime_v2_c1_cluster_structure_audit.py`
- `000785e75f827789b10a8c05fda43e10c09e1c7f31de246fc0a1e48bc38554b3` `DIME/mixing.py`
- (state_object.py / run_tier1_core.py / run_e1_wikitext103.py hashes unchanged from Revision 1, not re-quoted)

## C. Diagnostic implementation (updated)

Same file, `main()` still opens with the `RuntimeError` guard (unchanged). New/changed functions beyond Revision 1:

1. `environment_metadata()` / `save_construction_artifact(...)` (lines 541–578) — records package versions + platform, hashes and persists the fitted centroids/assignment.
2. `assert_real_data_invariants(...)` / `assert_routing_invariants(...)` (lines 493–539) — quoted in full above.
3. `correlation_pair(a, b)` (lines 601–609) — returns `(pearson_r, spearman_r, n)`.
4. `stratified_population_summary(cluster_df, n_strata=5)` (lines 580–599) — population-quintile stratified means/medians, all-nonempty-cluster scope.
5. `write_structural_summary_md`, `write_construction_audit_md`, `write_validation_report_md`, `write_final_report_md` — the 4 previously-missing deliverable writers.
6. `main()`'s Part 7 block now computes `idx_*_top1` (`= idx_*_topk[:, :1]`) and `idx_*_topk` (`k=20`) separately, running `assert_routing_invariants` on each before writing `traffic_top1_*` and `traffic_top{k}_*` as distinct `cluster_statistics.csv` columns.
7. Part 4/6's per-cluster loop now proceeds for every `n_j>=1` (singleton clusters included, tagged `is_singleton`), only `n_j==0` short-circuits to a population-only row. Part 5's sampling loop keeps a visible row (with `note`) when `dimensionality_stats` returns `None`, instead of dropping it.

**Full deliverable list produced by `main()` (all 8 required + 2 new reproducibility artifacts)**: `cluster_statistics.csv`, `structural_summary.md`, `construction_audit.md`, `dimensionality_results.csv`, `structural_relationships.csv`, `figures/` (4 PNGs), `validation_report.md`, `final_report.md`, plus `dime_v1_cluster_artifact.npz` and `construction_fingerprint.json` (new, addressing point 1).

## D. Planned calculations (addition)

- `pearson_r, spearman_r = correlation_pair(a, b)` — both computed over the same finite-value mask, reported together, never collapsed into one number (addresses point 4).
- Population-quintile stratified summary: `pd.qcut(n_j, 5)` over all nonempty clusters, reporting mean/median of `R_j, sigma_j, V_j, rho_j, H_j, delta_j` per quintile.
- `T_j^{(1)}` (top-1, diagnostic-grouping scope) and `T_j^{(k=20)}` (top-20, production-mixing scope) are now DISTINCT reported quantities, each separately asserted to sum to `1.0` and `k` respectively across clusters.

## E. Sanity checks (updated — now 10, all passed, all synthetic-only)

`population_sum_identity`, **`population_stats_singleton_vs_empty`** (new), `geometry_known_values`, **`singleton_cluster_geometry`** (new — hand-computed single-point case), `effective_rank_bounds`, `token_composition_known_values`, `routing_traffic_known_values`, **`routing_normalization`** (new — hand-built `[97,11,4]`-shaped synthetic routing matrix, confirms `sum(top1)=1.0`, `sum(topk)=k`), **`spearman_matches_scipy`** (new — monotonic-nonlinear `x` vs `x**3`, confirms Pearson and Spearman visibly differ, so they are not accidentally the same computation), `cluster_to_members_matches_manual`. Re-ran immediately before writing this document: 10/10 passed. The markdown-writer functions and the fingerprint/stratified-summary helpers were also dry-run on hand-built synthetic DataFrames/arrays (never a cache file) to confirm they do not crash; outputs inspected manually.

## F. Unresolved issues (updated)

1. **Still genuinely unresolved**: the sklearn version that produced the historical `tier1.json`/`run_manifest.json` artifacts cannot be recovered retroactively — no amount of code change fixes a measurement that was never taken. This revision only prevents the problem from recurring (every future run now saves its own fingerprint).
2. **Still genuinely unresolved**: no artifact existed before this revision, so there is still no historical ground truth to diff this run's centroids against — only forward-looking comparison is now possible.
3. Unchanged from Revision 1: the exact `extract_and_cache.py` CLI invocation for the base-`gpt2` WikiText-103 cache specifically was not found verbatim (only indirectly confirmed via two manifests).
4. **Partially addressed**: whether any real cluster is empty or a singleton will now be reported explicitly and separately (`n_empty`, `n_singleton` in `population_stats`) the moment the real job runs — still not verified from any existing artifact, since the real job has not run.
5. **Resolved**: the single-nearest vs. multi-centroid routing distinction is now fully attributed to specific source (`mixing.py` vs. `build_splits_for`) and reported as two separate column families with independent normalization checks.
6. Unchanged from Revision 1: `GPT2LMHeadModel.from_pretrained` defaults to float32; not independently cross-checked against the actual saved `.npz` dtype (will be logged when the real job runs, since `assert_real_data_invariants` now reports key dimensionality/finiteness explicitly, though not dtype by name — noting this as a residual gap worth a one-line addition if the reviewer wants it before approval).

## G. Execution status

**PREFLIGHT COMPLETE — DIAGNOSTIC EXPERIMENT NOT EXECUTED. AWAITING REVIEW AND APPROVAL.**
