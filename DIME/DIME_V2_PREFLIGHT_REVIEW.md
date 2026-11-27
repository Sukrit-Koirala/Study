# DIME-V2 Experiment C1 — Preflight Review (Revision 3)

Audience: an external research assistant with no repository access. Revision 2's overall design was approved, subject to 5 corrections. This revision addresses each; it replaces Revision 2 in full.

## Revision summary (point-by-point)

1. **Fingerprint hash/dtype mismatch (real bug, confirmed exactly as described).** `save_construction_artifact` (lines 705–744) previously hashed the IN-MEMORY float64 centroids / whatever-dtype assignment, then saved float32/int32 to disk — the recorded hash did not match the persisted bytes. Fixed: both arrays are now converted to their final storage dtype **first**, hashed, saved, then the written `.npz` is **reloaded** and its bytes re-hashed and asserted equal to the recorded hash. A new synthetic check, `check_fingerprint_matches_persisted_artifact` (lines 381–406), independently reloads the artifact and confirms the fingerprint's hash matches, and that it **differs** from a hash of the original pre-conversion bytes — demonstrating the fix changed real behavior. This check itself caught a second real bug on first run: `np.load`'s returned `NpzFile` kept its file handle open, which blocked `tempfile.TemporaryDirectory`'s cleanup on Windows (`PermissionError`). Fixed by using `np.load(...)` as a context manager (`with np.load(path) as reloaded:`) in both the production function and the test.
2. **1-D combined-rank sampler replaced with genuine 2-D stratification.** `stratified_dimensionality_sample` (lines 260–299) now bins population and dispersion **independently** into quantile edges (`DIM_POP_BINS=8 x DIM_DISP_BINS=5`, duplicate edges collapsed via `np.unique`), forms a grid, and samples one deterministic cluster (lowest id) per **occupied** cell — reporting every cell, including empty ones, in a new `dimensionality_sample_strata.csv`. The synthetic check (`check_2d_stratification_covers_both_dimensions`, lines 408–436) needed two iterations: my first version hand-placed 4 "corner" clusters among random filler and asserted all 4 would be individually selected — this was WRONG (quantile bins are count-based, not threshold-based, so a filler cluster can legitimately share a corner's cell and win the deterministic tie-break; that is correct sampler behavior, not a bug). Replaced with a test that constructs PERFECTLY RANK-ANTI-CORRELATED population/dispersion (so a 1-D combined score would be ~constant for every cluster) and asserts the 2-D sampler still independently covers all 4 population bins AND all 4 dispersion bins — verified empirically before committing it, not just reasoned about. `structural_relationships.csv`'s `scope` column (`all_nonempty_clusters` vs. `dimensionality_sample_n=40`) and `structural_summary.md`'s explicit "NOT summarized here as an all-cluster statistic" wording (added to the new Part 5 coverage section) ensure sample-only statistics are never presented as population estimates.
3. **Duplicate-index assertion added.** `assert_no_duplicate_indices_per_query` (lines 658–668), called from inside `assert_routing_invariants` (lines 671–686) BEFORE the traffic-sum check, verifies every query's top-k row contains `k` distinct cluster ids — because `routing_traffic`'s bincount-based statistic is only a true "fraction of queries containing cluster j" if no row has an internal duplicate. `check_duplicate_index_rejection` (lines 439–453) confirms the assertion passes on a clean matrix and raises `AssertionError` on a hand-built matrix with one duplicated index.
4. **Alignment-check wording clarified, limitation stated explicitly.** `assert_real_data_invariants`'s key/label check (lines ~600–625) is now documented as checking only **structural** alignment (equal row counts) — not semantic row-for-row provenance, which the cache format (`cache_io.py`) cannot support since it stores no row-id field. This limitation is now stated in the function's docstring, in `validation_report.md`'s generated text, and in Section F below — not implied to be proven.
5. **Dead SMOKE flag removed.** `SMOKE = os.environ.get("DIMEV2C1_SMOKE") == "1"` was read into a variable and printed/logged but never changed any actual behavior anywhere in the script. Removed entirely (constant, print statement, and summary-dict entry); a comment at its former location explains that pilot testing is done via the already-functional `DIMEV2C1_N_CLUSTERS` + `DIMEV2C1_CACHE_DIR` overrides against a small synthetic cache, exactly as this project's other scripts establish real scope reductions rather than a flag that merely claims to.

**Historical sklearn-version uncertainty remains an acknowledged, unresolved limitation** (Section F, items 1–2) — saving this run's own fingerprint prevents the problem recurring going forward; it does not and cannot retroactively resolve what version produced the historical `tier1.json`/`run_manifest.json` artifacts. Not claimed as resolved.

MiniBatchKMeans configuration (n_clusters=3811, seed=42), full top-20 routing, and the experiment's objectives are all unchanged from Revision 2.

## A. Verified baseline configuration

Unchanged from Revision 2; not repeated here to keep this revision compact (see Revision 2 history, or `construction_audit.md`'s generated output once the real job runs, for the full table).

## B. Exact source-code references (new/changed since Revision 2)

**Fixed fingerprint function** — `run_dime_v2_c1_cluster_structure_audit.py`, lines 705–744 (in full):

```python
def save_construction_artifact(dime_centroids, assignment, out_dir, n_clusters, seed, ds_keys_on_disk_dtype=None):
    centroids_final = np.ascontiguousarray(dime_centroids.astype(np.float32))
    assignment_final = np.ascontiguousarray(assignment.astype(np.int32))
    centroids_sha256 = hashlib.sha256(centroids_final.tobytes()).hexdigest()
    assignment_sha256 = hashlib.sha256(assignment_final.tobytes()).hexdigest()

    artifact_path = os.path.join(out_dir, "dime_v1_cluster_artifact.npz")
    np.savez_compressed(artifact_path, centroids=centroids_final, assignment=assignment_final)

    with np.load(artifact_path) as reloaded:
        reloaded_centroids = np.ascontiguousarray(reloaded["centroids"])
        reloaded_assignment = np.ascontiguousarray(reloaded["assignment"])
    reloaded_centroids_sha256 = hashlib.sha256(reloaded_centroids.tobytes()).hexdigest()
    reloaded_assignment_sha256 = hashlib.sha256(reloaded_assignment.tobytes()).hexdigest()
    assert reloaded_centroids_sha256 == centroids_sha256, \
        f"persisted centroids hash mismatch: recorded {centroids_sha256}, reloaded {reloaded_centroids_sha256}"
    assert reloaded_assignment_sha256 == assignment_sha256, \
        f"persisted assignment hash mismatch: recorded {assignment_sha256}, reloaded {reloaded_assignment_sha256}"
    # ... fingerprint dict assembly (environment, dtypes, shapes) and json.dump, unchanged in structure
    return fingerprint
```

**New routing check** — lines 658–686 (in full):

```python
def assert_no_duplicate_indices_per_query(idx_matrix):
    n_q, k = idx_matrix.shape
    unique_counts = np.array([len(np.unique(row)) for row in idx_matrix])
    assert np.all(unique_counts == k), \
        f"duplicate retrieved-centroid indices found in {int(np.sum(unique_counts != k))}/{n_q} queries"


def assert_routing_invariants(idx_matrix, n_clusters, k_expected, n_queries_expected):
    assert idx_matrix.shape == (n_queries_expected, k_expected), \
        f"routing index matrix shape {idx_matrix.shape} != expected ({n_queries_expected}, {k_expected})"
    assert np.all((idx_matrix >= 0) & (idx_matrix < n_clusters)), \
        "routing index matrix contains an out-of-range cluster id"
    assert_no_duplicate_indices_per_query(idx_matrix)
    traffic = routing_traffic(idx_matrix, n_clusters)
    expected_sum = float(k_expected)
    assert abs(float(traffic.sum()) - expected_sum) < 1e-6, \
        f"routing traffic does not sum to k={k_expected}: got {traffic.sum()}"
    return traffic
```

**New 2-D sampler** — lines 260–299 (signature and core binning logic; full function, including the cell-report loop, is in the file):

```python
def stratified_dimensionality_sample(populations, dispersions, n_pop_bins=DIM_POP_BINS, n_disp_bins=DIM_DISP_BINS):
    eligible = sorted(c for c in range(len(populations)) if populations[c] > 0 and c in dispersions)
    if len(eligible) == 0:
        return [], []
    pops = np.array([populations[c] for c in eligible], dtype=np.float64)
    disps = np.array([dispersions[c] for c in eligible], dtype=np.float64)
    pop_edges = np.unique(np.quantile(pops, np.linspace(0, 1, n_pop_bins + 1)))
    disp_edges = np.unique(np.quantile(disps, np.linspace(0, 1, n_disp_bins + 1)))
    # ... independent per-axis bin assignment, grid cell construction,
    # one deterministic (lowest-id) sample per OCCUPIED cell, every cell
    # (including empty ones) reported in cell_report
    return sorted(set(sampled)), cell_report
```

Full file to review in full (grew to 1,122 lines): `DIME/run_dime_v2_c1_cluster_structure_audit.py`.

**SHA-256**: `a764ee4d28b3e38c76eff9954ae651ddb093e648dbf4df95171683344cfad1a0` `DIME/run_dime_v2_c1_cluster_structure_audit.py`

## C. Diagnostic implementation (updated)

New/changed since Revision 2:

1. `save_construction_artifact` — hash-then-save-then-reload-then-verify, fixed dtype-before-hash ordering, `with np.load(...)` context manager.
2. `assert_no_duplicate_indices_per_query` (new) — called inside `assert_routing_invariants` before the traffic-sum check.
3. `stratified_dimensionality_sample` — replaced in full with genuine 2-D quantile-grid stratification; now returns `(sampled_cluster_ids, cell_report)` instead of a plain list. `main()`'s Part 5 updated to unpack this and write `dimensionality_sample_strata.csv`; `write_structural_summary_md` gained optional `sample_clusters`/`cell_report` parameters and a new coverage section.
4. `assert_real_data_invariants`'s docstring and assertion message, plus `write_validation_report_md`'s generated text, now explicitly state the key/label check is structural-only.
5. `SMOKE` constant, its print statement, and its summary-dict entry all removed.

**Deliverable list unchanged in count (8 required + 2 reproducibility artifacts) plus one new file**: `dimensionality_sample_strata.csv` (the full 2-D grid, including empty cells).

## D. Planned calculations (addition)

- Fingerprint hashes are now computed on, and verified against, the EXACT bytes written to `dime_v1_cluster_artifact.npz` (float32 centroids, int32 assignment) — never the pre-conversion in-memory arrays.
- 2-D stratified sample: independent quantile bins per axis, `DIM_POP_BINS=8 x DIM_DISP_BINS=5` grid (ceiling 40, actual count depends on occupied cells after duplicate-edge collapsing).
- Routing traffic's appearance-probability interpretation is now conditioned on `assert_no_duplicate_indices_per_query` passing first.

## E. Sanity checks (updated — now 13, all passed, all synthetic-only)

All 10 from Revision 2, plus: `duplicate_index_rejection` (new), `fingerprint_matches_persisted_artifact` (new — also caught and fixed the Windows file-handle bug above), `2d_stratification_covers_both_dimensions` (new — rewritten once after the first version's own flawed premise was caught by actually running it). Re-ran immediately before writing this document: **13/13 passed**.

## F. Unresolved issues (updated)

1. **Still genuinely unresolved, not claimed otherwise**: the sklearn version behind the historical `tier1.json`/`run_manifest.json` artifacts cannot be recovered retroactively.
2. **Still genuinely unresolved**: no historical ground-truth artifact existed before Revision 2, so only forward-looking fingerprint comparison is possible from here on.
3. Unchanged: the exact `extract_and_cache.py` CLI invocation for the base-`gpt2` WikiText-103 cache was not found verbatim.
4. Unchanged: real empty/singleton cluster counts will be reported the moment the real job runs; not yet verified from any artifact.
5. **New, per this revision's point 4**: the key/label alignment check verifies row-count equality only, not semantic row-for-row provenance — the cache format has no row-id field to check against. Stated explicitly, not implied to be proven.

## G. Execution status

**PREFLIGHT COMPLETE — DIAGNOSTIC EXPERIMENT NOT EXECUTED. AWAITING REVIEW AND APPROVAL.**
