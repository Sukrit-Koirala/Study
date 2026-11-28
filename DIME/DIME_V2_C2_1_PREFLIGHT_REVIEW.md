# DIME-V2 Experiment C2.1 — Preflight Review (Revision 1)

Audience: an external research assistant with no repository access. Revision 0 was reviewed and returned "REVISION REQUIRED" with 8 numbered points. This revision addresses each; it replaces Revision 0 in full.

**Hashing methodology**: the SHA-256 below is computed over the file's RAW BYTES as read from disk (`hashlib.sha256(open(path, "rb").read()).hexdigest()`), with NO line-ending normalization. The file is pure LF (0 CRLF sequences, 1,815 LF-terminated lines, 104,992 bytes) — confirmed by direct byte inspection, not assumed.

Full file (1,836 lines, after 3 final corrections below): `DIME/run_dime_v2_c2_1_local_retrieval_audit.py`. **SHA-256**: `e188a3ed9a32accac8f8585dd79ee911d9cbda163db7199d9c162f056f3f1fcd`

## Final corrections (post-Revision-1, pre-execution)

1. **Missing JS divergence between nearby-query output distributions.** Spec Part 5 required it alongside TV; Revision 1 computed only TV. Fixed: `analyze_cluster`'s pair loop now also computes `js_total = js_divergence_dict(vi["F"], vj["F"])`, aggregated into `mean_js_total` (cluster_behavior_statistics.csv) and the distance-stratified table, reported in `final_report.md`'s Q1 and Q6.
2. **Query-dependence failure cases selected by the wrong statistic.** `high_entropy_low_query_dependence`/`low_entropy_high_query_dependence` were scored using `mean_tv_total` (nearby-pair TV) while claiming to measure "query dependence," which spec Part 6.3 defines via `I_C`. Fixed: both cases are now scored using `I_C_val`; `mean_tv_total` is still recorded in the output as context, but is no longer the selection criterion.
3. **`frac_boundary_tie_diff_label` denominator mismatch.** The field was `n_boundary_ties_diff_label / len(val_info)` (all sampled queries) while `final_report.md`'s text described it as "of those" (i.e. conditional on landing on a boundary tie). Fixed: denominator is now `n_boundary_ties`, reported as `None` (not a misleading `0`) when a cluster had zero boundary ties.

Re-ran the full synthetic test suite (20/20 pass) and the end-to-end synthetic pilot (never real data) after all three fixes; confirmed `mean_js_total` is a genuinely distinct column from `mean_tv_total`, confirmed the query-dependence cases now carry a `summary_I_C_val` field, and confirmed `frac_boundary_tie_diff_label` reports `NaN` rather than `0.0` for clusters with zero ties.

## Change log (review point → corrected functions/fields)

| Point | Issue | Corrected functions | New/changed output fields |
|---|---|---|---|
| 1 | Centroid mixture weights never computed | `centroid_mixture_weights` (new), `routed_query_rows` (now threads distance+weight), `analyze_cluster` | `centroid_distance`/`centroid_weight` per query; `massweighted_mean_jaccard`, `massweighted_mean_tv_total`, `massweighted_mean_tv_vs_static`, `massweighted_mean_tv_vs_trainconst` |
| 2 | Ineffective synthetic tests | `check_candidate_change_without_token_change` (redesigned, dead `or True` removed), `check_train_val_baseline_construction_and_evaluation` (replaces superficial separation check), `check_centroid_mixture_weights_known_values`, `check_counterfactual_decomposition_known_values` (new) | — (test-only) |
| 3 | Query distance not part of stability analysis | `analyze_cluster` (margin-vs-displacement guarantee check, pair-reuse counter) | `margin_i_minus_2x_qqdist`, `stability_guaranteed`, `max_pair_reuse_count`, `frac_pairs_stability_guaranteed`; distance-stratified Q1 table in `final_report.md` |
| 4 | Failure cases not inspectable | `query_case_detail`, `build_pair_case_detail`, `build_single_query_case_detail` (new) | `failure_cases.md` now carries local candidate positions, global datastore row ids, labels, distances, weights for both query sides |
| 5 | Sampling documentation incomplete | `stratified_cluster_sample` (now returns cell map/population/inclusion probability), Part 7 relationship loop (new `stratified_only` scope) | `sampling_mode`, `inclusion_probability`, `is_stratified_include` columns; every `structural_relationships.csv` row duplicated under `stratified_only_*` scope |
| 6 | Incomplete analysis / placeholders | `write_final_report_md` (Q6/Q11 rewritten), `main()` (distance-stratified + Q11 stats blocks) | Q11 now reports real forward/reverse counterfactual agreement and split-half I_C correlation; `frac_missing_support_vs_trainconst`/`_vs_static` |
| 7 | Weak data invariants | `load_and_verify_c1_artifact` (fingerprint now REQUIRED, dtype validation), `main()` (dim/length/finite checks, traffic-vs-C1 comparison), `local_candidate_selection` boundary-tie detection in `analyze_cluster` | `frac_boundary_tie`, `frac_boundary_tie_diff_label`; `construction_consistency.md`'s traffic-comparison section |
| 8 | Bypassable execution guard | `DIME_MODE` (new module constant), `main()` guard rewritten | `DIMEV2C21_MODE` env var now mandatory; `dime_v2_c2_1_summary.json`'s `mode` field |

## A. Verified implementation (unchanged from Revision 0 unless noted)

- C1 artifact hashes (unchanged, still read from the real persisted files): centroids `e7adc93629e102ad7dc16fee9b1c0b1a29be599c44e1f5e42fedbf514d96957e`, assignment `9dae0c746a8f89dc6f6a3e96b4e0cead73d00965f474d7c3795bdffe6767f13e`.
- **New in this revision**: `load_and_verify_c1_artifact` now REQUIRES `construction_fingerprint.json` to exist (Revision 0 treated it as optional-if-present) and validates `centroids`/`assignment` dtypes against the fingerprint's own recorded dtypes, not just shape/range. `check_missing_c1_artifact_raises`/`check_missing_fingerprint_raises` (new) confirm both failure modes actually raise.
- **New**: `centroid_mixture_weights(distances, tau_route)` (point 1) — `softmax_np(-distances/tau_route, axis=1)`, the literal formula `mixing.py::mix_dime_and_lm` uses, asserted to sum to 1 for every query.
- **New**: `DIME_MODE` execution-mode guard (point 8) — see Part G.

## B. Mathematical definitions (new/changed since Revision 0)

- Centroid mixture weight: `w_ij = softmax_j(-d_ij/tau_route)` over the k_route retrieved centroids for query i — now actually computed and reported per routed query-cluster pair, not just the distances.
- Margin-vs-displacement stability guarantee (point 3): for a pair (q, q') in the same cluster, if `gamma_m(q) > 2*||q-q'||`, triangle inequality guarantees the top-m selected sets are identical. Implemented as `margin_i_minus_2x_qqdist = margin_i - 2*qq_dist`, `stability_guaranteed = margin_i_minus_2x_qqdist > 0`, and ACTIVELY ASSERTED (not just reported) against `identical_candidate_set` on every pair in `analyze_cluster` — a violation raises `AssertionError` naming the cluster.
- Boundary tie (point 7): `|margin| <= BOUNDARY_TIE_EPS * max(1, |d_m|)` with `BOUNDARY_TIE_EPS = 1e-9`; when true, compares the labels of the m-th and (m+1)-th nearest members (via a fresh `argsort` of the full distance array) for `boundary_tie_diff_label`.
- Mass-weighted aggregate (point 1): `agg_weighted(key, weight_key, records) = sum(value*weight)/sum(weight)`, computed as a SEPARATE column (`massweighted_*`) from the uniform-weight `mean_*` columns — never blended.
- Inclusion probability (point 5): for a stratified-sampled cluster, `(1/cell_population) * cell_keep_prob`, where `cell_population` is how many ELIGIBLE clusters share that cluster's quantile-bin cell and `cell_keep_prob = n_cells_kept/n_occupied_cells`. Explicitly stated as an approximation (ignores higher-order sampling effects), not an exact design-based weight. Deliberately-included unusual clusters get no inclusion probability (`None`/`NaN`), not an assumed one.

## C. Sampling protocol (changed: point 5)

Unchanged mechanics (independent quantile binning, seeded shuffle, cell-budget subsampling) — see Revision 0 history for the base design. New in this revision: `stratified_cluster_sample` returns, for every ELIGIBLE cluster, its cell tuple and that cell's population (not just for sampled ones), plus a per-sampled-cluster `inclusion_probability`. Every sampled cluster now carries an explicit `sampling_mode` ∈ {`stratified`, `unusual`, `both`} (point 5's "whether selection was probabilistic or deliberately unusual"). `structural_relationships.csv` reports every relationship pair TWICE: once over all sampled clusters, once restricted to `is_stratified_include` only (excludes unusual-but-not-stratified clusters) — confirmed in the pilot to produce MATERIALLY DIFFERENT correlation estimates (e.g. `entropy` vs `mean_tv_total`: Pearson r=0.49 all-sampled vs. r=0.06 stratified-only), demonstrating this was not a cosmetic fix.

## D. Implementation details (changed: points 1, 3, 4, 7)

- `routed_query_rows` now accepts the full distance and mixture-weight matrices and returns, per matching row, its rank, centroid distance, AND centroid mixture weight (previously only rank).
- `analyze_cluster`'s pair loop now computes, per pair: the stability-guarantee check (asserted), the directed-pair reuse counter (`Counter(nn)`, point 3's "directed-pair reuse" requirement — a query can be the nearest-neighbor target of several other sampled queries; `max_pair_reuse_count` reports the largest such count per cluster), and per-query boundary-tie detection.
- `query_case_detail`/`build_pair_case_detail`/`build_single_query_case_detail` (point 4) RE-DERIVE full detail (local candidate positions, global datastore row ids via `member_row_ids[sel_pos]`, candidate labels, distances, weights) for a specific extremal example identified from the summary CSVs — cheap, since it is one query against one cluster's (≤few-hundred-member) candidate set, not a new global search. Local positions and global row ids are reported under explicitly distinct field-name prefixes (`local_candidate_positions` vs. `global_datastore_row_ids`) so they cannot be confused. No text decoding is attempted; every case record states this explicitly rather than silently omitting it.
- Traffic-comparison invariant (point 7): recomputes `np.bincount(idx_val_topk.ravel(), ...)`/top-1 traffic and compares against C1's own recorded `traffic_top{K_ROUTE}_val`/`traffic_top1_val` columns with an EXPLAINED tolerance of `1e-6` — this re-runs the identical routing mechanism over the identical centroids/data C1 used, so any larger discrepancy is a real inconsistency, not noise. Skipped (not silently passed) with an explicit note whenever `K_ROUTE` is overridden for a pilot run and therefore doesn't match the column C1 saved.

## E. Leakage prevention (unchanged mechanics, now also test-verified)

`check_train_val_baseline_construction_and_evaluation` (point 2) replaces Revision 0's superficial index-disjointness check with a real functional test: builds a tiny cluster, computes `F_C(q)` for genuinely separate train/val query arrays (not index tricks on one array), builds `bar_F_train` ONLY from the train list, and verifies its value against a hand-computed elementwise average — confirming the construction is correct, not just that two hand-picked arrays happen to be disjoint.

## F. Validation

**20 synthetic checks, 20/20 passed** (isolated from any real cache; run first in `main()`, before either part of the guard is checked): `local_candidate_selection_known_values`, `margin_undefined_when_nc_le_m`, `local_weights_known_values`, `exact_local_distribution_shared_token`, `candidate_change_without_token_change` (REDESIGNED, point 2a), `label_changing_replacement`, `fixed_candidate_competition_change`, `tv_js_kl_known_values`, `static_vs_query_dependent`, `mean_distribution_forward_kl_identity`, `train_val_baseline_construction_and_evaluation` (NEW, point 2b), `centroid_mixture_weights_known_values` (NEW, point 1/2c), `counterfactual_decomposition_known_values` (NEW, point 2d), `selection_margin_stability_guarantee` (NEW, point 3), `boundary_tie_handling` (NEW, point 7), `c1_artifact_hash_verification`, `missing_c1_artifact_raises` (NEW, point 2g), `missing_fingerprint_raises` (NEW, point 7), `routing_duplicate_and_normalization`, `stratified_cluster_sample_covers_axes` (strengthened to also check cell-map/inclusion-probability outputs).

Every new hand-built test scenario's expected numbers were independently derived by hand before being encoded as assertions (e.g. `check_boundary_tie_handling`'s exact-tie margin=0 and excluded-member label; `check_selection_margin_stability_guarantee`'s margin=18 > 2×0.5; `check_counterfactual_decomposition_known_values`'s forward/reverse TV≈0.9866 agreement) — not reasoned about only, then run and confirmed to actually produce those values.

**End-to-end synthetic pilot, re-run after every fix in this revision** (never real data): same 40-cluster synthetic setup as Revision 0 (true singleton through 265-member cluster), now run under the mandatory `DIMEV2C21_MODE=synthetic_pilot`. Confirmed via direct execution:
- All 11 required deliverables + 6 figures produced; `mean_jaccard` vs `massweighted_mean_jaccard` genuinely differ per cluster (e.g. cluster 1: 0.530 vs 0.429) — the weighting is doing real work, not duplicating the unweighted column.
- `sampling_mode`/`inclusion_probability` populated correctly: unusual-only clusters show `inclusion_probability=NaN` (not a fabricated number); stratified/both clusters show real probabilities in (0, 1].
- `structural_relationships.csv`'s `stratified_only` scope rows differ substantially from `all_sampled_clusters` rows (see Part C) — the exact failure mode point 5 was raised to prevent.
- `failure_cases.md` now contains per-query global row ids/labels/distances/weights for both sides of a pair, with `text_decoding: "not attempted..."` stated explicitly.
- Re-ran the pilot a SECOND time with the real `K_ROUTE=20` (regenerating the synthetic C1 artifact's `traffic_top20_val` column accordingly) specifically to exercise the point-7 traffic-comparison path previously untested at `K_ROUTE=8`: PASSED with `max|diff|=5.55e-17` (float noise, well under the `1e-6` explained tolerance).
- Confirmed all 4 guard branches by direct execution, not just code review: (a) `DIMEV2C21_MODE` unset → raises naming both valid values; (b) `real_approved` without `DIMEV2C21_APPROVED=1` → raises; (c) `real_approved` pointed at a non-real `CACHE_DIR` → raises (closes the exact bypass the reviewer flagged); (d) `synthetic_pilot` pointed at the real `CACHE_DIR` → raises. All four leave zero filesystem trace (checked via `ls results/`).
- One cosmetic bug found and fixed during this revision's own pilot: `sampling_report.md`'s printed cell tuples showed `(np.int64(2), np.int64(2), ...)` (NumPy 2.x's scalar repr) instead of `(2, 2, ...)` — fixed by casting bin values to plain Python `int` before building the cell tuples in `stratified_cluster_sample`; re-ran the pilot after the fix to confirm.

## G. Computational plan (point 8 detail)

The guard is now a two-part EXPLICIT declaration, never inferred from `CACHE_DIR` alone:

```python
DIME_MODE = os.environ.get("DIMEV2C21_MODE")  # no default
# ...
if DIME_MODE not in ("synthetic_pilot", "real_approved"):
    raise RuntimeError(...)
if DIME_MODE == "real_approved":
    if not is_real_cache: raise RuntimeError(...)       # must point at the real path
    if os.environ.get("DIMEV2C21_APPROVED") != "1": raise RuntimeError(...)  # required regardless of path
else:  # synthetic_pilot
    if is_real_cache: raise RuntimeError(...)           # must NOT point at the real path
```

This closes the Revision 0 gap: previously, ANY `CACHE_DIR` other than the real default silently ran the full pipeline with no approval needed. Now, a run must explicitly declare which mode it is, and the declared mode is cross-checked (not decided) against the path. `DIMEV2C21_APPROVED=1` is required for `real_approved` unconditionally — there is no path that grants real execution without it. Resource/memory plan (datastore load, routing cost, per-cluster bounds) is unchanged from Revision 0.

## H. Known limitations (unchanged items renumbered + one new)

1. Query identifiers remain positional row indices (cache format has no row-id field).
2. Historical CRISP-R5 cluster-ID alignment remains unproven and unassumed.
3. The TV decomposition remains explicitly path-dependent, now with BOTH forward and reverse paths reported and their agreement quantified (Q11) rather than merely asserted non-additive.
4. Local within-cluster search still reuses `np.linalg.norm` rather than a per-cluster `NearestNeighbors` instance (unchanged from Revision 0; not required by any review point).
5. Sampling remains bounded; the `stratified_only` scope (point 5) is the closest approximation to a representativeness claim this script makes, and it is still explicitly NOT a full design-based estimator (inclusion probabilities are approximate, per Part B).
6. **New**: the stability-guarantee assertion (point 3) is a correctness check on THIS script's own arithmetic, not a claim about DIME-V1's production behavior — it would only ever fire on a real bug in distance/margin computation, and did not fire in either pilot run.

## I. Execution status

**C2.1 PREFLIGHT COMPLETE — REAL-DATA EXPERIMENT NOT EXECUTED. AWAITING EXTERNAL REVIEW.**
