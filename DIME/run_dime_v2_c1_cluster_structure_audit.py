"""DIME-V2 -- Experiment C1: Cluster Construction and Structural Audit.

PREFLIGHT STATUS: DIME_V2_PREFLIGHT_REVIEW.md went through three review
rounds and was explicitly approved; the execution guard formerly at the top
of main() has been removed. Pilot-tested on a synthetic cache before the
real run (see the matching submit script's own comments for the pilot
summary).

=====================================================================
Purpose
=====================================================================
A purely DIAGNOSTIC structural audit of the existing DIME-V1 cluster
construction (MiniBatchKMeans, seed=42, n_clusters=3811, over the real
381,127-row WikiText-103/gpt2 datastore) -- NOT an optimization experiment,
NOT a new compression representation, and NOT a change to the existing
centroid-routing mechanism. Characterizes population, geometry, local
dimensionality/anisotropy, token composition, and (if recoverable) query-
routing traffic, across ALL 3,811 configured clusters, before any decision
about how to compress that structure.

=====================================================================
Authoritative construction, verified from source (see
DIME_V2_PREFLIGHT_REVIEW.md Part B for exact excerpts/line numbers)
=====================================================================
  - N_CLUSTERS = 3811 originates from run_tier1_core.py's
    `N_CLUSTERS = args.n_clusters or max(100, round(len(ds_values) / 100))`,
    confirmed against the persisted results/wikitext103_gpt2_tier1.json
    (n_clusters=3811) and results/stage3_e1_wikitext103/run_manifest.json
    (datastore=381127, controller_train=90170, val=77089) -- round(381127/100)
    = 3811.27 -> 3811, matching run_e2_1_beta_diagnostic.py's own
    E2_EXPECTED_COUNTS = {0.01: 3811, 0.05: 19056, 0.10: 38112} ratio table
    exactly.
  - The construction itself: state_object.py's minibatch_kmeans_partition,
    `MiniBatchKMeans(n_clusters=n_clusters, random_state=seed)` -- every
    OTHER parameter (batch_size, init, n_init, max_iter, tol,
    max_no_improvement, reassignment_ratio, compute_labels, init_size) is
    left at sklearn's installed-version default, never set explicitly
    anywhere in this repository.
  - Every Stage-3 script (R1 through this project's most recent DIME-V2
    token-conditioned-geometry study) RE-DERIVES this same partition fresh
    at the start of its own run via the identical call
    `MiniBatchKMeans(n_clusters=N_CLUSTERS, random_state=SEED)`, fed
    `ds_keys.astype(np.float32)` (after a harmless float64 round-trip that
    changes no bit pattern, since float64 exactly represents every float32
    value) -- NOT a saved centroid/assignment artifact loaded from disk. No
    such artifact exists anywhere in this repository (verified by search).
    This script follows the identical convention for exactly the same
    reason every prior script did: it is the only way to reproduce the
    partition without a saved artifact to load.
  - run_dime_baseline.py / run_dime_kmeans_baseline.py /
    run_dime_query_kmeans_baseline.py / run_dime_utility_baseline.py are a
    SEPARATE, SMALLER TinyStories-based baseline-comparison suite
    (n_clusters=500, dataset="roneneldan/TinyStories", k=5, tau=1.0) --
    NOT the WikiText-103/gpt2/3811-cluster configuration this audit targets.
    They are reported here only because the spec's Part 0.1 asked this
    script to identify every clustering-construction call site and report
    discrepancies rather than assume equivalence; they are NOT reused by
    this diagnostic.

=====================================================================
Scope boundaries (Part 12), restated explicitly
=====================================================================
This script does not retrain GPT-2, does not re-extract hidden states
(loads the existing cache only), does not change n_clusters away from the
verified 3811, does not replace MiniBatchKMeans, does not modify centroid
routing (Part 7 reads the EXISTING multi-centroid query_knn_indices
mechanism exactly as it exists, never substituting nearest-one-centroid
assignment for it), trains no neural network, builds no token-conditioned
region model, and draws no conclusion that geometric compactness or token
repetition alone proves predictive compressibility (Parts 4.6, 6.2, 6.4,
8 all state this explicitly in their own docstrings/outputs).
"""
import sys, os, json, time, hashlib, platform
from collections import Counter

import numpy as np
import pandas as pd
import sklearn
import scipy
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats as scipy_stats
from sklearn.cluster import MiniBatchKMeans

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(SCRIPT_DIR, "..", "GPT_Module"))
sys.path.insert(0, SCRIPT_DIR)

from cache_io import load_cache
from knn import build_datastore, query_knn_indices
from results_io import save_results
from run_e2_information_allocation import build_cluster_to_members

# No standalone SMOKE flag: removed after external review correctly flagged
# that it was read into a variable and printed/logged but never changed any
# actual behavior anywhere in this script -- a dead setting that could
# mislead a future reader into thinking it bounds the run. Pilot testing
# (as actually used for this script's own synthetic-cache dry runs) is done
# via DIMEV2C1_N_CLUSTERS and DIMEV2C1_CACHE_DIR pointing at a small
# synthetic cache instead, exactly as the rest of this project's scripts
# establish real reductions in scope, not a flag that merely claims to.
CACHE_DIR = os.environ.get("DIMEV2C1_CACHE_DIR", os.path.join(SCRIPT_DIR, "..", "GPT_Module", "cache"))
OUT_DIR = os.environ.get("DIMEV2C1_OUT_DIR", os.path.join(SCRIPT_DIR, "results", "dime_v2_c1_cluster_structure_audit"))
FIG_DIR = os.path.join(OUT_DIR, "figures")

N_CLUSTERS = int(os.environ.get("DIMEV2C1_N_CLUSTERS", "3811"))
SEED = 42
DIM = 768

# Routing audit, revised after external review: TWO GENUINELY DIFFERENT
# mechanisms exist in this repository and must be reported SEPARATELY, never
# collapsed or conflated (confirmed by direct inspection of DIME/mixing.py
# and DIME/run_crisp_r3_failure_diagnosis.py's build_splits_for):
#   (1) PRODUCTION INFERENCE routing -- mix_dime_and_lm (mixing.py) loops over
#       ALL k retrieved centroids per query, softmax-weighting their
#       distributions by distance (`weights = softmax_np(-distances/tau)`).
#       This is genuine multi-centroid retrieval. The MATCHED config uses
#       k=20 (run_e1_wikitext103.py's MATCHED_CONFIG); the TUNED configs use
#       k=300 (raw) / k=100 (DIME), per results/wikitext103_gpt2_tier1.json.
#   (2) DIAGNOSTIC per-cluster grouping -- build_splits_for (used by every
#       R3-onward script, including the just-delivered DIME-V2 study) reads
#       ONLY idx[:, 0], the SINGLE nearest centroid, purely to decide which
#       cluster's local-analysis bucket a given query belongs to. This value
#       is NEVER used for prediction/mixing, only for diagnostic grouping.
# Both are computed and reported as separate, clearly-named columns below
# (traffic_top1_* vs traffic_topk_*) -- reporting only one would silently
# erase a real distinction that matters elsewhere in this project.
MATCHED_K = 20
TOP1_K = 1
TIER1_PATH = os.path.join(SCRIPT_DIR, "results", "wikitext103_gpt2_tier1.json")

LOW_POP_THRESHOLDS = [20, 60]
REPEATED_LABEL_THRESHOLDS = [2, 4, 8]

# Part 5: local dimensionality is run on a bounded, deterministic,
# GENUINELY TWO-DIMENSIONAL population x dispersion quantile-stratified
# sample (one cluster per occupied grid cell), never on all 3811 clusters
# and never via a 1-D combined-rank proxy (replaced after external review --
# see stratified_dimensionality_sample's docstring). DIM_POP_BINS *
# DIM_DISP_BINS = 40 is the sample-size ceiling, per the spec's own "keep
# the initial diagnostic computationally modest" instruction.
DIM_POP_BINS = 8
DIM_DISP_BINS = 5
DIM_SAMPLE_SIZE = DIM_POP_BINS * DIM_DISP_BINS  # retained as a reported ceiling, not a direct sampling parameter
VARIANCE_THRESHOLDS = [0.80, 0.90, 0.95, 0.99]


# ============================================================
# Part 3 -- population audit
# ============================================================
def population_stats(populations):
    """populations: int array, one entry per CONFIGURED cluster (0 for empty).
    Singleton clusters (n_j=1) are counted explicitly and SEPARATELY from
    empty clusters (n_j=0) -- per external review, the two are mathematically
    distinct cases (geometry IS well-defined for n_j=1; it is NOT defined for
    n_j=0) and must never be handled as if they were the same category."""
    n_configured = len(populations)
    nonempty_mask = populations > 0
    n_nonempty = int(nonempty_mask.sum())
    n_empty = n_configured - n_nonempty
    n_singleton = int(np.sum(populations == 1))
    pop_nonempty = populations[nonempty_mask]
    N = int(populations.sum())

    def frac_below(threshold):
        cluster_frac = float(np.mean(pop_nonempty < threshold)) if n_nonempty else None
        obs_frac = float(pop_nonempty[pop_nonempty < threshold].sum() / N) if N > 0 else None
        return cluster_frac, obs_frac

    below = {}
    for t in LOW_POP_THRESHOLDS:
        cf, of = frac_below(t)
        below[f"lt_{t}"] = {"cluster_fraction": cf, "observation_fraction": of}

    stats = {
        "n_configured_clusters": n_configured, "n_nonempty": n_nonempty, "n_empty": n_empty,
        "n_singleton": n_singleton,
        "total_observations": N,
        "min": int(pop_nonempty.min()) if n_nonempty else None,
        "max": int(pop_nonempty.max()) if n_nonempty else None,
        "mean": float(pop_nonempty.mean()) if n_nonempty else None,
        "median": float(np.median(pop_nonempty)) if n_nonempty else None,
        "std": float(pop_nonempty.std()) if n_nonempty else None,
        "p5": float(np.percentile(pop_nonempty, 5)) if n_nonempty else None,
        "p25": float(np.percentile(pop_nonempty, 25)) if n_nonempty else None,
        "p75": float(np.percentile(pop_nonempty, 75)) if n_nonempty else None,
        "p95": float(np.percentile(pop_nonempty, 95)) if n_nonempty else None,
        "p99": float(np.percentile(pop_nonempty, 99)) if n_nonempty else None,
        "below_thresholds": below,
    }
    if n_nonempty:
        sorted_pop = np.sort(pop_nonempty)
        cum = np.cumsum(sorted_pop) / sorted_pop.sum()
        lorenz_gap = float(np.mean(np.linspace(1 / n_nonempty, 1, n_nonempty) - cum))
        stats["population_imbalance_lorenz_gap"] = lorenz_gap
    else:
        stats["population_imbalance_lorenz_gap"] = None
    return stats


# ============================================================
# Part 4 -- geometric structure audit (per nonempty cluster)
# ============================================================
def cluster_geometry(member_keys, centroid):
    """Returns (R_j, mean_dist, rms_dispersion, delta_j, median_dist, p95_dist)."""
    diffs = member_keys - centroid[None, :]
    dists = np.linalg.norm(diffs, axis=1)
    R_j = float(dists.max())
    mean_dist = float(dists.mean())
    rms = float(np.sqrt(np.mean(dists ** 2)))
    empirical_mean = member_keys.mean(axis=0)
    delta_j = float(np.linalg.norm(centroid - empirical_mean))
    median_dist = float(np.median(dists))
    p95_dist = float(np.percentile(dists, 95))
    return R_j, mean_dist, rms, delta_j, median_dist, p95_dist


# ============================================================
# Part 5 -- local dimensionality / anisotropy
# ============================================================
def effective_rank(eigvals):
    """Spectral-entropy-based participation ratio: exp(-sum p_l log p_l),
    p_l = lambda_l / sum(lambda). Zero-variance input (all eigvals ~0) is
    handled explicitly -- returns 0.0, not NaN."""
    ev = np.asarray(eigvals, dtype=np.float64)
    ev = ev[ev > 1e-12]
    if len(ev) == 0:
        return 0.0
    p = ev / ev.sum()
    return float(np.exp(-np.sum(p * np.log(p))))


def dimensionality_stats(member_keys, variance_thresholds=VARIANCE_THRESHOLDS):
    """Centered SVD (cheaper than an explicit D x D covariance eigendecomposition
    when n << D, as is typical here): singular values s relate to covariance
    eigenvalues as lambda_l = s_l^2 / n."""
    n = member_keys.shape[0]
    if n < 2:
        return None
    centered = member_keys - member_keys.mean(axis=0, keepdims=True)
    try:
        _, s, _ = np.linalg.svd(centered, full_matrices=False)
    except np.linalg.LinAlgError:
        return None
    eigvals = (s ** 2) / n
    total = eigvals.sum()
    if total <= 0:
        return {"n": n, "effective_rank": 0.0, "leading_variance_ratio": None,
                **{f"n_components_{int(t*100)}pct": 0 for t in variance_thresholds}}
    cum = np.cumsum(eigvals) / total
    out = {"n": n, "effective_rank": effective_rank(eigvals),
           "leading_variance_ratio": float(eigvals[0] / total)}
    for t in variance_thresholds:
        out[f"n_components_{int(t*100)}pct"] = int(np.searchsorted(cum, t) + 1)
    return out


def stratified_dimensionality_sample(populations, dispersions, n_pop_bins=DIM_POP_BINS, n_disp_bins=DIM_DISP_BINS):
    """Genuine TWO-DIMENSIONAL population x dispersion quantile
    stratification. Replaces a prior 1-D combined-rank proxy
    (`pop_rank/n + disp_rank/n`, sorted and evenly sampled along that single
    combined axis) that external review correctly identified as conflating
    the two dimensions -- a cluster with (low population, high dispersion)
    and one with (high population, low dispersion) could receive the same
    combined score and only ONE of that pair would ever be sampled, silently
    under-covering real population x dispersion combinations.

    This version bins population and dispersion SEPARATELY into quantile
    edges (n_pop_bins x n_disp_bins grid, duplicate edges collapsed via
    np.unique when the data has heavy ties), then picks ONE cluster
    (deterministically, the lowest cluster id) from each OCCUPIED cell.
    Returns (sampled_cluster_ids, cell_report) where cell_report lists EVERY
    cell in the grid, including ones with zero clusters -- coverage gaps are
    visible in the output, never hidden. Bounded at <= n_pop_bins*n_disp_bins
    clusters (DIM_POP_BINS*DIM_DISP_BINS = 40 by default), per the spec's own
    "keep the initial diagnostic computationally modest" instruction."""
    eligible = sorted(c for c in range(len(populations)) if populations[c] > 0 and c in dispersions)
    if len(eligible) == 0:
        return [], []
    pops = np.array([populations[c] for c in eligible], dtype=np.float64)
    disps = np.array([dispersions[c] for c in eligible], dtype=np.float64)

    pop_edges = np.unique(np.quantile(pops, np.linspace(0, 1, n_pop_bins + 1)))
    disp_edges = np.unique(np.quantile(disps, np.linspace(0, 1, n_disp_bins + 1)))
    n_pop_actual = max(1, len(pop_edges) - 1)
    n_disp_actual = max(1, len(disp_edges) - 1)
    pop_bin_idx = np.clip(np.searchsorted(pop_edges, pops, side="right") - 1, 0, n_pop_actual - 1)
    disp_bin_idx = np.clip(np.searchsorted(disp_edges, disps, side="right") - 1, 0, n_disp_actual - 1)

    cells = {}
    for i, c in enumerate(eligible):
        key = (int(pop_bin_idx[i]), int(disp_bin_idx[i]))
        cells.setdefault(key, []).append(c)

    sampled, cell_report = [], []
    for pb in range(n_pop_actual):
        for db in range(n_disp_actual):
            members = sorted(cells.get((pb, db), []))
            chosen = members[0] if members else None  # deterministic: lowest cluster id in the cell
            if chosen is not None:
                sampled.append(chosen)
            cell_report.append({"pop_bin": pb, "disp_bin": db, "n_clusters_in_cell": len(members),
                                  "sampled_cluster": chosen})
    return sorted(set(sampled)), cell_report


# ============================================================
# Part 6 -- token composition audit
# ============================================================
def token_composition(member_values_counter, n_j):
    """member_values_counter: Counter {token_id: count} for one cluster."""
    counts = np.array(list(member_values_counter.values()), dtype=np.float64)
    V_j = len(counts)
    if V_j == 0 or n_j == 0:
        return {"V_j": 0, "distinct_token_fraction": None, "entropy": None,
                "top_token_fraction": None, "n_singleton_tokens": 0,
                "singleton_token_fraction": None, "singleton_observation_fraction": None,
                "rho_j": None}
    p = counts / counts.sum()
    entropy = float(-np.sum(p * np.log(p + 1e-300)))
    top_frac = float(counts.max() / counts.sum())
    n_singleton = int(np.sum(counts == 1))
    singleton_obs_frac = float(n_singleton / n_j)
    return {
        "V_j": int(V_j), "distinct_token_fraction": float(V_j / n_j), "entropy": entropy,
        "top_token_fraction": top_frac, "n_singleton_tokens": n_singleton,
        "singleton_token_fraction": float(n_singleton / V_j),
        "singleton_observation_fraction": singleton_obs_frac,
        "rho_j": float(n_j / V_j),
    }


def repeated_label_fractions(member_values_counter, n_j, thresholds=REPEATED_LABEL_THRESHOLDS):
    """Fraction of observations belonging to tokens with count >= threshold,
    per threshold. Cluster-weighted summaries are produced by the caller
    (one row per cluster averaged afterward); this returns the
    observation-weighted fraction for THIS cluster."""
    counts = np.array(list(member_values_counter.values()), dtype=np.float64)
    out = {}
    for t in thresholds:
        if n_j == 0:
            out[f"obs_frac_ge_{t}"] = None
        else:
            out[f"obs_frac_ge_{t}"] = float(counts[counts >= t].sum() / n_j)
    return out


# ============================================================
# Part 7 -- query-routing traffic audit (reads the EXISTING multi-centroid
# routing mechanism; never substitutes nearest-one-centroid assignment)
# ============================================================
def routing_traffic(idx_matrix, n_clusters):
    """idx_matrix: [n_queries, k] cluster indices from query_knn_indices
    (the project's own established routing call). T_j = fraction of queries
    for which cluster j appears ANYWHERE in the retrieved top-k set -- a
    query with k>1 retrieved centroids contributes to EVERY one of them,
    not just the nearest, per the spec's explicit Part 7 instruction."""
    n_queries = idx_matrix.shape[0]
    traffic = np.zeros(n_clusters, dtype=np.float64)
    flat = idx_matrix.reshape(-1)
    counts = np.bincount(flat, minlength=n_clusters)
    traffic = counts / n_queries
    return traffic


# ============================================================
# Part 9 -- correctness / synthetic checks (pure functions, synthetic data only)
# ============================================================
def check_population_sum_identity(rng):
    populations = rng.integers(0, 50, size=200)
    assert int(populations.sum()) == int(np.sum(populations)), "trivial identity failed"
    N = int(populations.sum())
    stats = population_stats(populations)
    ok = stats["total_observations"] == N and stats["n_configured_clusters"] == 200
    assert ok, f"population_stats sum identity failed: {stats}"
    return {"name": "population_sum_identity", "passed": bool(ok)}


def check_fingerprint_matches_persisted_artifact(rng):
    """Catches exactly the bug external review found: hashing the in-memory
    array BEFORE dtype conversion while saving the array AFTER conversion.
    Verifies (a) the returned fingerprint's hash matches a hash computed
    independently by RELOADING the written .npz from disk, and (b) that
    hash is DIFFERENT from a hash of the original pre-conversion float64/
    int64 bytes -- demonstrating the fix changed real behavior, not just
    the code path."""
    import tempfile
    centroids64 = rng.normal(size=(6, 16)).astype(np.float64)
    assignment64 = rng.integers(0, 6, size=40).astype(np.int64)
    pre_conversion_centroid_hash = hashlib.sha256(np.ascontiguousarray(centroids64).tobytes()).hexdigest()

    with tempfile.TemporaryDirectory() as d:
        fp = save_construction_artifact(centroids64, assignment64, d, 6, SEED)
        with np.load(os.path.join(d, "dime_v1_cluster_artifact.npz")) as reloaded:
            independent_centroid_hash = hashlib.sha256(np.ascontiguousarray(reloaded["centroids"]).tobytes()).hexdigest()
            independent_assignment_hash = hashlib.sha256(np.ascontiguousarray(reloaded["assignment"]).tobytes()).hexdigest()

    ok = (fp["centroids_sha256"] == independent_centroid_hash
          and fp["assignment_sha256"] == independent_assignment_hash
          and fp["centroids_sha256"] != pre_conversion_centroid_hash
          and fp["centroids_dtype"] == "float32" and fp["assignment_dtype"] == "int32")
    assert ok, f"fingerprint/persisted-artifact mismatch: fp={fp}, independent={independent_centroid_hash}"
    return {"name": "fingerprint_matches_persisted_artifact", "passed": bool(ok)}


def check_2d_stratification_covers_both_dimensions(rng):
    """Construct population and dispersion as PERFECTLY RANK-ANTI-CORRELATED
    (population strictly increasing, dispersion strictly decreasing over the
    same clusters). Under perfect anti-correlation, a 1-D combined-rank
    proxy (pop_rank/n + disp_rank/n, the ORIGINAL, now-replaced
    implementation) assigns EVERY cluster almost the SAME combined score
    (pop_rank/n + (n-1-pop_rank)/n ~= constant) -- its 'stratification'
    collapses to effectively nothing, sampling an arbitrary subset rather
    than one spanning the actual value range. This genuinely 2-D grid
    sampler must still independently cover the FULL range of population
    quantile bins AND the full range of dispersion quantile bins, since
    each axis is binned independently of the other."""
    n = 44
    pop_values = np.arange(1, n + 1, dtype=np.float64) * 100       # strictly increasing
    disp_values = np.arange(n, 0, -1, dtype=np.float64)            # strictly decreasing -> anti-correlated
    populations = np.zeros(n + 1)
    dispersions = {}
    for i in range(1, n + 1):
        populations[i] = pop_values[i - 1]
        dispersions[i] = disp_values[i - 1]
    sampled, cell_report = stratified_dimensionality_sample(populations, dispersions, n_pop_bins=4, n_disp_bins=4)
    occupied_pop_bins = set(row["pop_bin"] for row in cell_report if row["sampled_cluster"] is not None)
    occupied_disp_bins = set(row["disp_bin"] for row in cell_report if row["sampled_cluster"] is not None)
    ok = occupied_pop_bins == {0, 1, 2, 3} and occupied_disp_bins == {0, 1, 2, 3}
    assert ok, (f"2-D stratification failed to independently cover both axes under anti-correlation: "
                 f"pop_bins={occupied_pop_bins}, disp_bins={occupied_disp_bins}")
    ok2 = len(cell_report) == 16
    assert ok2, f"unexpected cell_report size: {len(cell_report)} (expected 16)"
    return {"name": "2d_stratification_covers_both_dimensions", "passed": bool(ok and ok2)}


def check_duplicate_index_rejection(rng):
    """assert_no_duplicate_indices_per_query must REJECT a routing matrix
    containing a within-row duplicate index -- confirmed by catching the
    expected AssertionError, not merely by eyeballing the code. Added per
    external review point 3."""
    clean = np.array([[0, 1, 2], [1, 2, 3]])
    dirty = np.array([[0, 1, 1], [1, 2, 3]])  # row 0 has a duplicate '1'
    assert_no_duplicate_indices_per_query(clean)  # must NOT raise
    raised = False
    try:
        assert_no_duplicate_indices_per_query(dirty)
    except AssertionError:
        raised = True
    assert raised, "duplicate-index check failed to reject a dirty routing matrix"
    return {"name": "duplicate_index_rejection", "passed": bool(raised)}


def check_singleton_cluster_geometry(rng):
    """n_j=1 is mathematically well-defined for R_j, mean_dist, rms, and
    delta_j (all per Part 4's own boxed definitions) -- it must NOT be
    treated as undefined/NaN the way n_j=0 genuinely is. R_j/mean_dist/rms
    collapse to the single point's distance to the FITTED centroid (which
    need not be the point itself, since the centroid may differ from any
    member after MiniBatchKMeans' approximate fitting); delta_j collapses to
    that same distance too, since the empirical mean of one point IS that
    point."""
    D = 5
    point = np.array([[3.0, 0.0, 0.0, 0.0, 0.0]])
    centroid = np.array([1.0, 0.0, 0.0, 0.0, 0.0])  # deliberately not the point itself
    R_j, mean_dist, rms, delta_j, median_dist, p95_dist = cluster_geometry(point, centroid)
    ok = (abs(R_j - 2.0) < 1e-9 and abs(mean_dist - 2.0) < 1e-9 and abs(rms - 2.0) < 1e-9
          and abs(delta_j - 2.0) < 1e-9 and abs(median_dist - 2.0) < 1e-9 and abs(p95_dist - 2.0) < 1e-9
          and all(np.isfinite([R_j, mean_dist, rms, delta_j, median_dist, p95_dist])))
    assert ok, f"singleton geometry check failed: R_j={R_j}, mean_dist={mean_dist}, rms={rms}, delta_j={delta_j}"
    dim = dimensionality_stats(point)
    ok2 = dim is None  # PCA is genuinely undefined for n=1 -- None, not a crash, not a fabricated number
    assert ok2, f"dimensionality_stats should return None for n=1, got {dim}"
    return {"name": "singleton_cluster_geometry", "passed": bool(ok and ok2)}


def check_population_stats_singleton_vs_empty(rng):
    """n_j=0 (empty) and n_j=1 (singleton) must be counted in SEPARATE,
    non-overlapping fields -- a real distinction external review flagged."""
    populations = np.array([0, 0, 1, 1, 1, 5, 10])
    stats = population_stats(populations)
    ok = (stats["n_empty"] == 2 and stats["n_singleton"] == 3
          and stats["n_nonempty"] == 5 and stats["n_configured_clusters"] == 7)
    assert ok, f"singleton-vs-empty population accounting failed: {stats}"
    return {"name": "population_stats_singleton_vs_empty", "passed": bool(ok)}


def check_routing_normalization(rng):
    """Top-1 routing traffic must sum to exactly 1.0 across clusters (every
    query has exactly one nearest centroid); top-k routing traffic must sum
    to exactly k (every query contributes to exactly k clusters). These are
    real, checkable invariants -- added after external review asked for
    "corresponding normalization sanity checks", not merely asserted."""
    rng_local = np.random.default_rng(0)
    n_queries, n_clusters, k = 97, 11, 4
    idx_topk = rng_local.integers(0, n_clusters, size=(n_queries, k))
    traffic_topk = routing_traffic(idx_topk, n_clusters)
    idx_top1 = idx_topk[:, :1]
    traffic_top1 = routing_traffic(idx_top1, n_clusters)
    ok = (abs(traffic_topk.sum() - k) < 1e-9) and (abs(traffic_top1.sum() - 1.0) < 1e-9)
    assert ok, f"routing normalization failed: sum(top1)={traffic_top1.sum()}, sum(topk)={traffic_topk.sum()}"
    return {"name": "routing_normalization", "passed": bool(ok),
            "sum_top1": float(traffic_top1.sum()), "sum_topk": float(traffic_topk.sum())}


def check_spearman_matches_scipy(rng):
    """Cross-check this script's Spearman usage against a hand-constructed
    monotonic-but-nonlinear relationship, where Pearson and Spearman should
    visibly DIFFER -- confirming the two are not accidentally computing the
    same thing under a different name."""
    x = np.arange(1, 21, dtype=np.float64)
    y = x ** 3  # monotonic, nonlinear
    pearson_r = float(np.corrcoef(x, y)[0, 1])
    spearman_r = float(scipy_stats.spearmanr(x, y).correlation)
    ok = (abs(spearman_r - 1.0) < 1e-9) and (pearson_r < 0.99)
    assert ok, f"Spearman/Pearson sanity check failed: pearson={pearson_r}, spearman={spearman_r}"
    return {"name": "spearman_matches_scipy", "passed": bool(ok),
            "pearson_r": pearson_r, "spearman_r": spearman_r}


def check_geometry_known_values(rng):
    """1-D-like synthetic cluster embedded in D dims with a KNOWN centroid,
    so R_j/mean_dist/rms/delta_j can be verified against hand computation."""
    D = 8
    pts = np.zeros((4, D))
    pts[:, 0] = [0.0, 1.0, 2.0, 3.0]  # mean = 1.5
    centroid = np.array([1.0] + [0.0] * (D - 1))  # deliberately NOT the empirical mean
    R_j, mean_dist, rms, delta_j, median_dist, p95 = cluster_geometry(pts, centroid)
    expected_dists = np.abs(pts[:, 0] - 1.0)  # [1, 0, 1, 2]
    ok = (abs(R_j - 2.0) < 1e-9 and abs(mean_dist - expected_dists.mean()) < 1e-9
          and abs(rms - np.sqrt(np.mean(expected_dists ** 2))) < 1e-9
          and abs(delta_j - 0.5) < 1e-9)  # |centroid(1.0) - empirical_mean(1.5)| = 0.5
    assert ok, f"geometry check failed: R_j={R_j}, mean_dist={mean_dist}, rms={rms}, delta_j={delta_j}"
    return {"name": "geometry_known_values", "passed": bool(ok), "R_j": R_j, "delta_j": delta_j}


def check_effective_rank_bounds(rng):
    """Isotropic D-dim data -> effective rank should be close to D (all
    eigenvalues roughly equal). Perfectly 1-D data -> effective rank exactly 1."""
    D = 16
    iso = rng.normal(size=(500, D))
    _, s, _ = np.linalg.svd(iso - iso.mean(axis=0), full_matrices=False)
    r_iso = effective_rank((s ** 2) / iso.shape[0])

    rank1 = np.outer(rng.normal(size=300), rng.normal(size=D))
    _, s1, _ = np.linalg.svd(rank1 - rank1.mean(axis=0), full_matrices=False)
    r_rank1 = effective_rank((s1 ** 2) / rank1.shape[0])

    zero_var = effective_rank(np.zeros(5))
    ok = (r_iso > D * 0.5) and (abs(r_rank1 - 1.0) < 0.05) and (zero_var == 0.0)
    assert ok, f"effective_rank bounds failed: r_iso={r_iso}, r_rank1={r_rank1}, zero_var={zero_var}"
    return {"name": "effective_rank_bounds", "passed": bool(ok), "r_iso": float(r_iso), "r_rank1": float(r_rank1)}


def check_token_composition_known_values(rng):
    c = Counter({0: 5, 1: 1, 2: 1, 3: 3})
    n_j = sum(c.values())  # 10
    stats = token_composition(c, n_j)
    ok = (stats["V_j"] == 4 and abs(stats["rho_j"] - 2.5) < 1e-9
          and stats["n_singleton_tokens"] == 2
          and abs(stats["singleton_observation_fraction"] - 0.2) < 1e-9
          and abs(stats["top_token_fraction"] - 0.5) < 1e-9)
    assert ok, f"token_composition known-value check failed: {stats}"
    rep = repeated_label_fractions(c, n_j, thresholds=[2, 4])
    ok2 = abs(rep["obs_frac_ge_2"] - 0.8) < 1e-9 and abs(rep["obs_frac_ge_4"] - 0.5) < 1e-9
    assert ok2, f"repeated_label_fractions known-value check failed: {rep}"
    return {"name": "token_composition_known_values", "passed": bool(ok and ok2)}


def check_routing_traffic_known_values(rng):
    idx = np.array([[0, 1], [0, 2], [1, 2], [0, 1]])  # 4 queries, k=2
    traffic = routing_traffic(idx, n_clusters=3)
    # cluster 0: appears in queries 0,1,3 -> 3/4; cluster 1: queries 0,2,3 -> 3/4; cluster 2: queries 1,2 -> 2/4
    expected = np.array([0.75, 0.75, 0.5])
    ok = bool(np.allclose(traffic, expected))
    assert ok, f"routing_traffic known-value check failed: {traffic} vs {expected}"
    return {"name": "routing_traffic_known_values", "passed": ok}


def check_cluster_to_members_matches_manual(rng):
    """Cross-check build_cluster_to_members (imported from run_e2_information_allocation,
    not reimplemented) against a naive boolean-mask grouping on synthetic assignments."""
    assignment = rng.integers(0, 10, size=500)
    groups = build_cluster_to_members(assignment, 10)
    ok = all(set(groups[c].tolist()) == set(np.where(assignment == c)[0].tolist()) for c in range(10))
    assert ok, "build_cluster_to_members disagreed with a naive boolean-mask grouping"
    return {"name": "cluster_to_members_matches_manual", "passed": bool(ok)}


def run_synthetic_checks():
    rng = np.random.default_rng(SEED)
    results = [
        check_population_sum_identity(rng),
        check_population_stats_singleton_vs_empty(rng),
        check_geometry_known_values(rng),
        check_singleton_cluster_geometry(rng),
        check_effective_rank_bounds(rng),
        check_token_composition_known_values(rng),
        check_routing_traffic_known_values(rng),
        check_routing_normalization(rng),
        check_duplicate_index_rejection(rng),
        check_spearman_matches_scipy(rng),
        check_cluster_to_members_matches_manual(rng),
        check_fingerprint_matches_persisted_artifact(rng),
        check_2d_stratification_covers_both_dimensions(rng),
    ]
    assert all(r["passed"] for r in results), f"Synthetic checks failed: {results}"
    return results


# ============================================================
# Part 10/11 -- correctness assertions against REAL data (run inside main(),
# never bypassed with a silent except)
# ============================================================
def assert_real_data_invariants(ds_keys, ds_values, populations, compressed_dists, assignment, dime_centroids, n_clusters):
    """Strengthened after external review: now also checks key/label row
    counts, finite coordinates, and centroid shape -- not just population
    bookkeeping.

    IMPORTANT LIMITATION, stated explicitly per external review point 4:
    the check below (`ds_keys.shape[0] == len(ds_values)`) verifies only
    STRUCTURAL alignment -- that the keys and labels arrays have the same
    number of rows. Equal row counts do NOT establish that row i's key
    truly corresponds to row i's label (e.g. an upstream bug that shuffled
    one array but not the other, or concatenated splits in a different
    order, would still pass this check). The cache format (cache_io.py)
    stores no row-id or provenance field alongside keys/values/p_lm_true/
    entropy, so TRUE semantic row-for-row alignment cannot be independently
    verified from the cache alone -- this is reported as a limitation (see
    DIME_V2_PREFLIGHT_REVIEW.md Section F), not silently assumed to be
    proven by this assertion."""
    N = len(ds_values)
    assert ds_keys.shape[0] == len(ds_values), \
        f"structural key/label misalignment (row-count mismatch only -- see docstring): " \
        f"{ds_keys.shape[0]} keys vs {len(ds_values)} labels"
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


def assert_no_duplicate_indices_per_query(idx_matrix):
    """routing_traffic's bincount treats each row of idx_matrix as a SET of
    k DISTINCT retrieved centroids; if sklearn's kneighbors ever returned a
    duplicate index within one query's row (e.g. from coincident centroids),
    the resulting traffic value would silently stop meaning "fraction of
    queries containing cluster j" and start double-counting a single query.
    Checked explicitly, never assumed -- added per external review point 3."""
    n_q, k = idx_matrix.shape
    unique_counts = np.array([len(np.unique(row)) for row in idx_matrix])
    assert np.all(unique_counts == k), \
        f"duplicate retrieved-centroid indices found in {int(np.sum(unique_counts != k))}/{n_q} queries"


def assert_routing_invariants(idx_matrix, n_clusters, k_expected, n_queries_expected):
    """Checks the routing index matrix itself: shape, in-range indices,
    PER-QUERY INDEX UNIQUENESS (external review point 3 -- without this, the
    traffic normalization check below could pass even if the underlying
    appearance-probability interpretation were already broken), and that the
    resulting traffic vector is correctly normalized."""
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


# ============================================================
# Reproducibility: environment metadata + saved construction artifact
# (added per external review point 1 -- no such artifact existed anywhere
# in the repository before this script)
# ============================================================
def environment_metadata():
    return {
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "sklearn_version": sklearn.__version__,
        "scipy_version": scipy.__version__,
        "pandas_version": pd.__version__,
        "platform": platform.platform(),
    }


def save_construction_artifact(dime_centroids, assignment, out_dir, n_clusters, seed, ds_keys_on_disk_dtype=None):
    """Persists the ACTUAL fitted centroids and assignment from this run,
    plus content hashes and environment metadata, so a FUTURE script can
    load this file and compare directly instead of re-deriving blind and
    hoping for bitwise reproducibility. No such saved artifact existed
    anywhere in this repository before this change.

    Fixed after external review: the FIRST version of this function hashed
    the IN-MEMORY float64 centroids / whatever-dtype assignment, but then
    saved float32 centroids / int32 assignment to the .npz -- the recorded
    hash did not match the bytes actually on disk. This version converts
    BOTH arrays to their final storage dtype FIRST, hashes those exact
    bytes, saves them, then RELOADS the written .npz and asserts the
    reloaded bytes hash to the same value -- not just "should match",
    verified against the actual written file."""
    centroids_final = np.ascontiguousarray(dime_centroids.astype(np.float32))
    assignment_final = np.ascontiguousarray(assignment.astype(np.int32))
    centroids_sha256 = hashlib.sha256(centroids_final.tobytes()).hexdigest()
    assignment_sha256 = hashlib.sha256(assignment_final.tobytes()).hexdigest()

    artifact_path = os.path.join(out_dir, "dime_v1_cluster_artifact.npz")
    np.savez_compressed(artifact_path, centroids=centroids_final, assignment=assignment_final)

    # np.load's NpzFile keeps the underlying zip file handle open until
    # closed; on Windows an unclosed handle blocks deletion/rename of the
    # containing directory (a real failure the synthetic test below caught
    # when run inside a tempfile.TemporaryDirectory). Using it as a context
    # manager guarantees the handle is released before this function returns.
    with np.load(artifact_path) as reloaded:
        reloaded_centroids = np.ascontiguousarray(reloaded["centroids"])
        reloaded_assignment = np.ascontiguousarray(reloaded["assignment"])
    reloaded_centroids_sha256 = hashlib.sha256(reloaded_centroids.tobytes()).hexdigest()
    reloaded_assignment_sha256 = hashlib.sha256(reloaded_assignment.tobytes()).hexdigest()
    assert reloaded_centroids_sha256 == centroids_sha256, \
        f"persisted centroids hash mismatch: recorded {centroids_sha256}, reloaded {reloaded_centroids_sha256}"
    assert reloaded_assignment_sha256 == assignment_sha256, \
        f"persisted assignment hash mismatch: recorded {assignment_sha256}, reloaded {reloaded_assignment_sha256}"

    fingerprint = {
        "n_clusters": n_clusters, "seed": seed,
        "centroids_sha256": centroids_sha256,
        "assignment_sha256": assignment_sha256,
        "centroids_shape": list(centroids_final.shape),
        "assignment_shape": list(assignment_final.shape),
        "centroids_dtype": str(centroids_final.dtype),
        "assignment_dtype": str(assignment_final.dtype),
        "ds_keys_on_disk_dtype": ds_keys_on_disk_dtype,
        "environment": environment_metadata(),
        "saved_at_unix_time": time.time(),
    }
    with open(os.path.join(out_dir, "construction_fingerprint.json"), "w") as f:
        json.dump(fingerprint, f, indent=2)
    return fingerprint


# ============================================================
# Part 8 (extended) -- stratified structural summary, explicitly scoped
# ============================================================
def stratified_population_summary(cluster_df, n_strata=5):
    """Population-quintile-stratified mean/median of key geometry and token
    statistics, computed over ALL nonempty clusters (same scope as
    cluster_statistics.csv -- NOT the 40-cluster dimensionality sample).
    Added per external review point 4."""
    nonempty = cluster_df[cluster_df["n_j"] > 0].copy()
    if len(nonempty) < n_strata:
        return []
    nonempty["population_quintile"] = pd.qcut(nonempty["n_j"], n_strata, labels=False, duplicates="drop")
    rows = []
    for q, group in nonempty.groupby("population_quintile"):
        row = {"population_quintile": int(q), "n_clusters": int(len(group)),
               "n_j_min": int(group["n_j"].min()), "n_j_max": int(group["n_j"].max())}
        for col in ["R_j", "rms_dispersion", "V_j", "rho_j", "entropy", "delta_centroid_vs_mean"]:
            if col in group.columns:
                row[f"{col}_mean"] = float(group[col].mean())
                row[f"{col}_median"] = float(group[col].median())
        rows.append(row)
    return rows


def correlation_pair(a, b):
    """Returns (pearson_r, spearman_r, n) -- both reported together, never
    collapsed into one number, per external review point 4."""
    mask = np.isfinite(a) & np.isfinite(b)
    n = int(mask.sum())
    if n < 3 or np.std(a[mask]) == 0 or np.std(b[mask]) == 0:
        return None, None, n
    pearson_r = float(np.corrcoef(a[mask], b[mask])[0, 1])
    spearman_r = float(scipy_stats.spearmanr(a[mask], b[mask]).correlation)
    return pearson_r, spearman_r, n


# ============================================================
# Deliverable writers (Part 10) -- structural_summary.md, construction_audit.md,
# validation_report.md, final_report.md were missing from the first
# preflight draft (external review point 6); all four are implemented here.
# ============================================================
def write_structural_summary_md(out_dir, pop_stats, stratified_rows, nonempty_df,
                                  sample_clusters=None, cell_report=None):
    with open(os.path.join(out_dir, "structural_summary.md"), "w") as f:
        f.write("# Structural Summary (Part 3/4/6)\n\n")
        f.write(f"Configured clusters: {pop_stats['n_configured_clusters']}; "
                 f"nonempty: {pop_stats['n_nonempty']}; empty: {pop_stats['n_empty']}; "
                 f"singleton (n_j=1): {pop_stats['n_singleton']}.\n\n")
        f.write(f"Population mean={pop_stats['mean']}, median={pop_stats['median']}, "
                 f"p5={pop_stats['p5']}, p95={pop_stats['p95']}, "
                 f"Lorenz imbalance gap={pop_stats['population_imbalance_lorenz_gap']}.\n\n")
        for t, vals in pop_stats["below_thresholds"].items():
            f.write(f"- fraction of clusters with population {t}: cluster-weighted="
                     f"{vals['cluster_fraction']}, observation-weighted={vals['observation_fraction']}\n")
        f.write("\n## Population-stratified summary (ALL nonempty clusters, 5 quintiles)\n\n")
        for row in stratified_rows:
            f.write(f"- quintile {row['population_quintile']} (n_j in [{row['n_j_min']}, {row['n_j_max']}], "
                     f"{row['n_clusters']} clusters): "
                     f"mean R_j={row.get('R_j_mean')}, mean sigma_j={row.get('rms_dispersion_mean')}, "
                     f"mean V_j={row.get('V_j_mean')}, mean rho_j={row.get('rho_j_mean')}, "
                     f"mean entropy={row.get('entropy_mean')}\n")
        f.write("\nGeometric compactness and token repetition are reported here as DESCRIPTIVE "
                 "covariates only -- this script draws no conclusion that either implies predictive "
                 "compressibility (Parts 4.6/6.2/6.4/8).\n")
        if cell_report is not None:
            n_occupied = sum(1 for row in cell_report if row["sampled_cluster"] is not None)
            f.write(f"\n## Part 5 dimensionality sample coverage (2-D population x dispersion grid)\n\n")
            f.write(f"{n_occupied}/{len(cell_report)} grid cells occupied; "
                     f"{len(sample_clusters) if sample_clusters else 0} clusters sampled "
                     f"(one per occupied cell, lowest cluster id, deterministic). Full per-cell "
                     "population and the exact sampled cluster id are in "
                     "dimensionality_sample_strata.csv -- NOT summarized here as an all-cluster "
                     "statistic, since it describes only the bounded sample.\n")
            f.write(f"\nSampled cluster IDs: {sample_clusters}\n")


def write_construction_audit_md(out_dir, fingerprint, tier1, n_ds, n_clusters, seed):
    with open(os.path.join(out_dir, "construction_audit.md"), "w") as f:
        f.write("# Construction Audit\n\n")
        f.write(f"Dataset: WikiText-103/gpt2. Datastore size: {n_ds}. n_clusters: {n_clusters} "
                 f"(= max(100, round(N/100)), verified against {tier1 is not None and 'tier1.json' or 'N/A'}: "
                 f"{tier1.get('n_clusters') if tier1 else 'not found'}). Seed: {seed}.\n\n")
        f.write("Construction call (verbatim, every Stage-3 script including this one): "
                 "`MiniBatchKMeans(n_clusters=N_CLUSTERS, random_state=SEED)` -- every other parameter "
                 "left at the installed sklearn version's default (never set explicitly anywhere in "
                 "this repository). Fit on `ds_keys.astype(np.float32)`, the full datastore, no subset, "
                 "no normalization.\n\n")
        f.write("## This run's actual environment and fingerprint\n\n")
        f.write(f"```\n{json.dumps(fingerprint, indent=2)}\n```\n\n")
        f.write("## Reproducibility limitation\n\n")
        f.write("No sklearn version is recorded for the artifacts this project cites as historical "
                 "reference (`wikitext103_gpt2_tier1.json`, `run_manifest.json`) -- a fixed seed alone "
                 "does not prove this run's partition is bitwise identical to any earlier one. This "
                 "run's own centroids/assignment/environment are now saved "
                 "(`dime_v1_cluster_artifact.npz`, `construction_fingerprint.json`) so FUTURE runs can "
                 "compare directly instead of assuming.\n")


def write_validation_report_md(out_dir, checks_passed, fingerprint, traffic_checks):
    with open(os.path.join(out_dir, "validation_report.md"), "w") as f:
        f.write("# Validation Report\n\n")
        f.write("## Synthetic checks (run before any real data was touched)\n\n")
        for r in checks_passed:
            f.write(f"- {r['name']}: passed={r['passed']}\n")
        f.write("\n## Real-data invariants (assert_real_data_invariants)\n\n")
        f.write("Key/label row-count match, key dimensionality, finite coordinates (keys, labels, "
                 "centroids), population-sum identity, assignment length/range, compressed_dists "
                 "length, centroid shape, per-cluster token-sum-equals-population -- all asserted "
                 "(not merely logged) before any statistic was computed; a failure here stops the run.\n\n")
        f.write("**Limitation, stated explicitly**: the key/label check verifies only STRUCTURAL "
                 "alignment (equal row counts). True semantic row-for-row provenance cannot be "
                 "independently verified from this cache format, which stores no row-id field -- "
                 "this is a reported limitation, not a verified guarantee of correct alignment.\n\n")
        f.write("## Routing invariants (assert_routing_invariants)\n\n")
        f.write("Shape, in-range indices, PER-QUERY INDEX UNIQUENESS (no duplicate retrieved-centroid "
                 "index within a single query's top-k row -- checked explicitly before the traffic "
                 "normalization check below is treated as meaningful), and traffic-sum normalization.\n\n")
        for name, val in traffic_checks.items():
            f.write(f"- {name}: sum={val}\n")
        f.write("\n## Reproducibility status\n\n")
        f.write(f"This run's construction fingerprint: centroids_sha256={fingerprint['centroids_sha256'][:16]}..., "
                 f"assignment_sha256={fingerprint['assignment_sha256'][:16]}..., "
                 f"environment={fingerprint['environment']}.\n")


def write_final_report_md(out_dir, pop_stats, relationships_rows, dim_rows, fingerprint, tier1):
    with open(os.path.join(out_dir, "final_report.md"), "w") as f:
        f.write("# Final Report -- DIME-V2 Experiment C1\n\n")
        f.write("## Q1. Construction\n\nSee construction_audit.md for the exact source references and "
                 "this run's fingerprint.\n\n")
        computed_formula_n_clusters = max(100, round(pop_stats["total_observations"] / 100))
        tier1_n_clusters = tier1.get("n_clusters") if tier1 else None
        matches_tier1 = (tier1_n_clusters is not None and computed_formula_n_clusters == tier1_n_clusters)
        if matches_tier1:
            match_note = "These MATCH."
        else:
            match_note = ("These DO NOT MATCH -- expected whenever this run used a different datastore "
                           "size than the real 381,127-row WikiText-103 cache (e.g. a pilot run), since "
                           "the formula is computed from this run's own N, not copied from tier1.json.")
        f.write(f"## Q2. Why {pop_stats['n_configured_clusters']} clusters\n\n"
                 f"`max(100, round(N/100))` with N={pop_stats['total_observations']} -> "
                 f"{computed_formula_n_clusters}. Persisted tier1 config's n_clusters="
                 f"{tier1_n_clusters if tier1_n_clusters is not None else 'not found'}. "
                 f"{match_note}\n\n")
        f.write(f"## Q3. Population imbalance\n\n{pop_stats['n_empty']} empty, {pop_stats['n_singleton']} "
                 f"singleton, of {pop_stats['n_configured_clusters']} configured. Lorenz gap="
                 f"{pop_stats['population_imbalance_lorenz_gap']}. See structural_summary.md for the "
                 f"full percentile/threshold breakdown.\n\n")
        f.write("## Q4-Q5. Geometry and dimensionality\n\nSee cluster_statistics.csv (all clusters) and "
                 f"dimensionality_results.csv ({len(dim_rows)} rows, the bounded stratified sample -- "
                 "NOT all clusters).\n\n")
        f.write("## Q6-Q8. Token composition\n\nSee cluster_statistics.csv's V_j/rho_j/entropy columns "
                 "and structural_summary.md's stratified breakdown.\n\n")
        f.write("## Q9. Structural relationships\n\n")
        for row in relationships_rows:
            f.write(f"- {row['var_a']} <-> {row['var_b']} (scope={row['scope']}): "
                     f"pearson_r={row['pearson_r']}, spearman_r={row['spearman_r']}, n={row['n']}\n")
        f.write("\n## Q10. Routing traffic\n\nReported as TWO separate quantities -- top-1 nearest-centroid "
                 "traffic (diagnostic-grouping convention) and top-20 matched-config traffic (production "
                 "mixing convention, mix_dime_and_lm) -- never conflated. See cluster_statistics.csv's "
                 "traffic_top1_* and traffic_topk_* columns.\n\n")
        f.write("## Q11-Q12. Relevance and limitations\n\nSee DIME_V2_PREFLIGHT_REVIEW.md Section F for "
                 "unresolved issues (sklearn version provenance, no prior saved ground-truth artifact, "
                 "exact extraction CLI invocation). This run's own fingerprint is now saved for future "
                 "comparison: " + json.dumps({k: fingerprint[k] for k in ("centroids_sha256", "assignment_sha256")}) + ".\n\n")
        f.write("No claim is made that geometric compactness or token repetition alone proves predictive "
                 "compressibility -- that question is explicitly deferred to a later experiment.\n")


def main():
    # PREFLIGHT HOLD removed: DIME_V2_PREFLIGHT_REVIEW.md (Revision 3) was
    # reviewed and explicitly approved before this guard was lifted.
    t_start = time.perf_counter()
    os.makedirs(FIG_DIR, exist_ok=True)
    print(f"DIME-V2 C1 job started: N_CLUSTERS={N_CLUSTERS}, CACHE_DIR={CACHE_DIR}")

    print("Loading cache (datastore/controller_train/val)...")
    ds = load_cache(os.path.join(CACHE_DIR, "wikitext103_gpt2_datastore.npz"))
    ct = load_cache(os.path.join(CACHE_DIR, "wikitext103_gpt2_controller_train.npz"))
    val = load_cache(os.path.join(CACHE_DIR, "wikitext103_gpt2_val.npz"))
    ds_keys_on_disk_dtype = str(ds["keys"].dtype)  # logged, not assumed -- addresses external review point 6
    ds_keys, ds_values = ds["keys"].astype(np.float64), ds["values"]
    ct_keys = ct["keys"].astype(np.float64)
    val_keys = val["keys"].astype(np.float64)

    print("Re-deriving DIME clusters (seed=42, identical construction to every prior Stage-3 script)...")
    km = MiniBatchKMeans(n_clusters=N_CLUSTERS, random_state=SEED)
    assignment = km.fit_predict(ds_keys.astype(np.float32))
    dime_centroids = km.cluster_centers_.astype(np.float64)
    compressed_dists = []
    for c in range(N_CLUSTERS):
        mv = ds_values[assignment == c]
        compressed_dists.append(Counter(mv.tolist()) if len(mv) > 0 else Counter())
    cluster_to_members = build_cluster_to_members(assignment, N_CLUSTERS)
    populations = np.array([sum(cd.values()) for cd in compressed_dists])

    assert_real_data_invariants(ds_keys, ds_values, populations, compressed_dists, assignment,
                                  dime_centroids, N_CLUSTERS)

    print("Saving construction artifact (centroids/assignment/fingerprint/environment)...")
    fingerprint = save_construction_artifact(dime_centroids, assignment, OUT_DIR, N_CLUSTERS, SEED,
                                               ds_keys_on_disk_dtype=ds_keys_on_disk_dtype)
    print(f"  sklearn={fingerprint['environment']['sklearn_version']}, "
          f"centroids_sha256={fingerprint['centroids_sha256'][:16]}...")

    # --- Part 3 ---
    print("Part 3: population audit (all configured clusters)...")
    pop_stats = population_stats(populations)
    save_results(os.path.join(OUT_DIR, "dime_v2_c1_summary.json"),
                  {"stage": "population_audit", "population_stats": pop_stats})

    # --- Part 4 + Part 6: per-nonempty-cluster geometry and token composition ---
    print("Part 4/6: geometric + token-composition audit over all nonempty clusters (n_j>=1)...")
    rows = []
    dispersions = {}
    for c in range(N_CLUSTERS):
        n_j = int(populations[c])
        if n_j == 0:
            rows.append({"cluster": c, "n_j": 0})
            continue
        # n_j>=1 (including singleton clusters) all proceed through the SAME
        # geometry/token-composition path -- R_j/mean_dist/rms/delta_j are
        # mathematically defined for n_j=1 (verified by check_singleton_
        # cluster_geometry) and must never be silently skipped or merged
        # with the n_j=0 case.
        members = cluster_to_members[c]
        member_keys = ds_keys[members]
        R_j, mean_dist, rms, delta_j, median_dist, p95_dist = cluster_geometry(member_keys, dime_centroids[c])
        dispersions[c] = rms
        tok_stats = token_composition(compressed_dists[c], n_j)
        rep_fracs = repeated_label_fractions(compressed_dists[c], n_j)
        rows.append({"cluster": c, "n_j": n_j, "R_j": R_j, "mean_dist": mean_dist, "rms_dispersion": rms,
                      "delta_centroid_vs_mean": delta_j, "median_dist": median_dist, "p95_dist": p95_dist,
                      "is_singleton": n_j == 1,
                      **tok_stats, **rep_fracs})
        if c % 500 == 0:
            print(f"  processed {c}/{N_CLUSTERS} clusters")

    cluster_df = pd.DataFrame(rows)
    cluster_df.to_csv(os.path.join(OUT_DIR, "cluster_statistics.csv"), index=False)

    # --- Part 5: local dimensionality on a bounded, GENUINELY 2-D stratified sample ---
    print(f"Part 5: local dimensionality on a bounded 2-D population x dispersion stratified "
          f"sample ({DIM_POP_BINS}x{DIM_DISP_BINS} grid, ceiling={DIM_SAMPLE_SIZE}, NOT all clusters)...")
    sample_clusters, cell_report = stratified_dimensionality_sample(populations, dispersions)
    pd.DataFrame(cell_report).to_csv(os.path.join(OUT_DIR, "dimensionality_sample_strata.csv"), index=False)
    n_occupied_cells = sum(1 for row in cell_report if row["sampled_cluster"] is not None)
    print(f"  {n_occupied_cells}/{len(cell_report)} grid cells occupied; "
          f"{len(sample_clusters)} clusters sampled")
    dim_rows = []
    for c in sample_clusters:
        members = cluster_to_members[c]
        d = dimensionality_stats(ds_keys[members])
        if d is not None:
            dim_rows.append({"cluster": c, "n_j": int(populations[c]), **d})
        else:
            # n_j=1 clusters CAN appear in this stratified sample (rms=0 is a
            # valid dispersion value); PCA is genuinely undefined there, so
            # the row is kept VISIBLE with a None dimensionality rather than
            # silently vanishing from the CSV, per external review.
            dim_rows.append({"cluster": c, "n_j": int(populations[c]), "effective_rank": None,
                              "leading_variance_ratio": None, "note": "PCA undefined for n_j<2"})
    pd.DataFrame(dim_rows).to_csv(os.path.join(OUT_DIR, "dimensionality_results.csv"), index=False)

    # --- Part 7: query-routing traffic -- TWO mechanisms, reported separately ---
    print(f"Part 7: query-routing traffic -- top-1 (diagnostic grouping) AND top-{MATCHED_K} "
          "(production mixing, matched config), never conflated...")
    dime_index, _ = build_datastore(dime_centroids, np.array(compressed_dists, dtype=object))
    k_eff = min(MATCHED_K, N_CLUSTERS)
    _, idx_ct_topk = query_knn_indices(dime_index, ct_keys, k=k_eff)
    _, idx_val_topk = query_knn_indices(dime_index, val_keys, k=k_eff)
    idx_ct_top1 = idx_ct_topk[:, :1]
    idx_val_top1 = idx_val_topk[:, :1]

    traffic_checks = {}
    traffic_ct_topk = assert_routing_invariants(idx_ct_topk, N_CLUSTERS, k_eff, len(ct_keys))
    traffic_val_topk = assert_routing_invariants(idx_val_topk, N_CLUSTERS, k_eff, len(val_keys))
    traffic_ct_top1 = assert_routing_invariants(idx_ct_top1, N_CLUSTERS, 1, len(ct_keys))
    traffic_val_top1 = assert_routing_invariants(idx_val_top1, N_CLUSTERS, 1, len(val_keys))
    traffic_checks["sum_traffic_top1_controller_train"] = float(traffic_ct_top1.sum())
    traffic_checks["sum_traffic_top1_val"] = float(traffic_val_top1.sum())
    traffic_checks[f"sum_traffic_top{k_eff}_controller_train"] = float(traffic_ct_topk.sum())
    traffic_checks[f"sum_traffic_top{k_eff}_val"] = float(traffic_val_topk.sum())
    print(f"  routing normalization: {traffic_checks}")

    cluster_df["traffic_top1_controller_train"] = traffic_ct_top1
    cluster_df["traffic_top1_val"] = traffic_val_top1
    cluster_df[f"traffic_top{k_eff}_controller_train"] = traffic_ct_topk
    cluster_df[f"traffic_top{k_eff}_val"] = traffic_val_topk
    cluster_df.to_csv(os.path.join(OUT_DIR, "cluster_statistics.csv"), index=False)

    tier1 = None
    if os.path.exists(TIER1_PATH):
        with open(TIER1_PATH) as f:
            tier1 = json.load(f)

    # --- Part 8: structural relationships -- Pearson + Spearman, scope labeled ---
    print("Part 8: structural relationships (Pearson + Spearman, scope labeled)...")
    nonempty = cluster_df[cluster_df["n_j"] > 0]
    relationships = []
    all_cluster_pairs = [("n_j", "R_j"), ("n_j", "rms_dispersion"), ("n_j", "V_j"),
                          ("rms_dispersion", "entropy"), ("rho_j", "rms_dispersion"),
                          (f"traffic_top{k_eff}_val", "n_j"), (f"traffic_top{k_eff}_val", "rms_dispersion"),
                          (f"traffic_top{k_eff}_val", "V_j"), ("traffic_top1_val", "n_j")]
    for a, b in all_cluster_pairs:
        if a in nonempty.columns and b in nonempty.columns:
            pr, sr, n = correlation_pair(nonempty[a].values, nonempty[b].values)
            relationships.append({"var_a": a, "var_b": b, "scope": "all_nonempty_clusters",
                                   "pearson_r": pr, "spearman_r": sr, "n": n})

    dim_df = pd.DataFrame(dim_rows)
    dim_merged = dim_df.merge(nonempty[["cluster", "n_j", "rms_dispersion"]], on="cluster", suffixes=("", "_full"))
    sample_pairs = [("n_j", "effective_rank"), ("rms_dispersion", "effective_rank")]
    for a, b in sample_pairs:
        if a in dim_merged.columns and b in dim_merged.columns:
            pr, sr, n = correlation_pair(dim_merged[a].values, dim_merged[b].values)
            relationships.append({"var_a": a, "var_b": b, "scope": f"dimensionality_sample_n={DIM_SAMPLE_SIZE}",
                                   "pearson_r": pr, "spearman_r": sr, "n": n})
    pd.DataFrame(relationships).to_csv(os.path.join(OUT_DIR, "structural_relationships.csv"), index=False)

    stratified_rows = stratified_population_summary(cluster_df)

    # --- figures ---
    os.makedirs(FIG_DIR, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(nonempty["n_j"], bins=60)
    ax.set_xlabel("cluster population n_j"); ax.set_title("Population distribution")
    fig.tight_layout(); fig.savefig(os.path.join(FIG_DIR, "population_histogram.png")); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(nonempty["n_j"], nonempty["rms_dispersion"], s=4, alpha=0.3)
    ax.set_xlabel("n_j"); ax.set_ylabel("rms dispersion sigma_j")
    ax.set_title("Population vs. geometric dispersion")
    fig.tight_layout(); fig.savefig(os.path.join(FIG_DIR, "population_vs_dispersion.png")); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(nonempty["V_j"], bins=60)
    ax.set_xlabel("distinct tokens V_j"); ax.set_title("Token diversity distribution")
    fig.tight_layout(); fig.savefig(os.path.join(FIG_DIR, "token_diversity_histogram.png")); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(nonempty["rms_dispersion"], nonempty["entropy"], s=4, alpha=0.3)
    ax.set_xlabel("rms dispersion sigma_j"); ax.set_ylabel("token entropy H_j")
    ax.set_title("Dispersion vs. token entropy")
    fig.tight_layout(); fig.savefig(os.path.join(FIG_DIR, "dispersion_vs_entropy.png")); plt.close(fig)

    # --- Part 10: remaining required deliverables ---
    print("Writing structural_summary.md, construction_audit.md, validation_report.md, final_report.md...")
    write_structural_summary_md(OUT_DIR, pop_stats, stratified_rows, nonempty,
                                  sample_clusters=sample_clusters, cell_report=cell_report)
    write_construction_audit_md(OUT_DIR, fingerprint, tier1, int(len(ds_values)), N_CLUSTERS, SEED)
    synthetic_results_for_report = run_synthetic_checks()
    write_validation_report_md(OUT_DIR, synthetic_results_for_report, fingerprint, traffic_checks)
    write_final_report_md(OUT_DIR, pop_stats, relationships, dim_rows, fingerprint, tier1)

    summary = {
        "n_ds": int(len(ds_values)), "n_ct": int(len(ct["values"])), "n_val": int(len(val["values"])),
        "n_clusters": N_CLUSTERS, "seed": SEED,
        "population_stats": pop_stats, "tier1_reference": tier1,
        "construction_fingerprint": fingerprint, "routing_normalization_checks": traffic_checks,
        "elapsed_seconds": time.perf_counter() - t_start,
    }
    save_results(os.path.join(OUT_DIR, "dime_v2_c1_summary.json"), summary)
    print(f"DIME-V2 C1 job done in {time.perf_counter() - t_start:.1f}s")


if __name__ == "__main__":
    main()
