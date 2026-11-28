"""
DIME-V2 -- Experiment C2.1: Exact Local Retrieval Behavior Audit

PREFLIGHT STATUS: this file implements the C2.1 diagnostic described in the
spec pasted for this experiment. This is Revision 1, addressing 8 external
review points on the Revision-0 preflight (see
DIME_V2_C2_1_PREFLIGHT_REVIEW.md for the full change log).

Per Part 0 / Part 13 of that spec, the real-data pipeline inside main() is
gated behind an EXPLICIT two-part guard (see DIME_MODE / the guard check
near the top of main()):
  - DIMEV2C21_MODE must be set to either "synthetic_pilot" or
    "real_approved" -- there is no default and no inference from CACHE_DIR
    alone (Revision 0's "a different cache path implies synthetic" heuristic
    was flagged by external review as bypassable and has been removed).
  - "real_approved" additionally requires CACHE_DIR to resolve to the real
    production cache path AND DIMEV2C21_APPROVED=1 (intended to be set only
    in the SLURM submit script, only after external review and explicit
    approval are received) -- this requirement holds regardless of path.
  - "synthetic_pilot" additionally requires CACHE_DIR to NOT resolve to the
    real production path -- refused otherwise, so this mode can never
    silently touch real data.
Synthetic-only unit tests (run_synthetic_checks()) always run first, before
either part of the guard is even checked, and never touch any real cache or
the real C1 artifact.

Scope, reused verbatim from the spec and from direct source inspection of
this repository (see DIME_V2_C2_1_PREFLIGHT_REVIEW.md for exact line-number
citations):
  - The C1 partition (MiniBatchKMeans, n_clusters=3811, seed=42) is NEVER
    retrained here. This script loads C1's persisted
    dime_v1_cluster_artifact.npz directly, verifies its SHA-256 fingerprint
    against BOTH the externally supplied expected hashes AND C1's own
    construction_fingerprint.json, and sets the reloaded arrays read-only
    (ndarray.setflags(write=False)) as a concrete guarantee they are never
    mutated by anything downstream in this file.
  - Centroid-routing semantics are reused exactly from
    run_e1_wikitext103.py's MATCHED_CONFIG (k=20, tau=2.0) via
    GPT_Module/knn.py's build_datastore/query_knn_indices -- the SAME
    mechanism mix_dime_and_lm (DIME/mixing.py) uses for production
    inference, as opposed to the single-nearest-centroid diagnostic
    grouping build_splits_for uses elsewhere. Both a top-20-routed query
    population Q_C^(20) (primary) and a top-1-nearest-centroid population
    Q_C^(1) (secondary, reported separately, never merged) are computed.
  - Per-cluster static histograms p_C(y) are the same quantity
    state_object.py's minibatch_kmeans_partition / mixing.py's
    mix_dime_and_lm use internally (count(y)/n_C, no smoothing) -- reused
    here as the DIME-V1 baseline this diagnostic compares exact local
    retrieval against, never redefined.
  - C1's own per-cluster structural covariates (population, dispersion,
    token diversity/entropy, routing traffic) are read directly from
    cluster_statistics.csv, never recomputed, per spec 0.2.

This is NOT a compression algorithm and does not fit, select, or evaluate
any compressed representation -- see Part 12 of the spec for the explicit
list of things C2.1 must not do.
"""
import sys, os, json, time, hashlib, platform
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
import sklearn
import scipy
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats as scipy_stats
from scipy.spatial.distance import cdist

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(SCRIPT_DIR, "..", "GPT_Module"))
sys.path.insert(0, SCRIPT_DIR)

from cache_io import load_cache
from knn import build_datastore, query_knn_indices, softmax_np
from results_io import save_results
from run_e2_information_allocation import build_cluster_to_members
from e1_metrics import js_divergence_dict
from run_crisp_subspace_projection import total_variation, top1_agreement

# ============================================================
# Paths / configuration
# ============================================================
REAL_CACHE_DIR_DEFAULT = os.path.join(SCRIPT_DIR, "..", "GPT_Module", "cache")
CACHE_DIR = os.environ.get("DIMEV2C21_CACHE_DIR", REAL_CACHE_DIR_DEFAULT)

# Execution mode (Revision 1, review point 8): a different CACHE_DIR must
# NOT by itself imply "this is synthetic, skip approval" -- that was the
# Revision-0 design and the external reviewer correctly flagged it as a
# bypassable heuristic. The operator must now EXPLICITLY declare which mode
# this run is, and the declared mode is cross-checked against CACHE_DIR
# (defense in depth, not the sole decision):
#   - "synthetic_pilot": CACHE_DIR must NOT be the real production path --
#     refuses to run if it is, so this mode can never silently touch real data.
#   - "real_approved": CACHE_DIR must BE the real production path, AND
#     DIMEV2C21_APPROVED must be "1" -- the approval flag is required
#     regardless of path; there is no path that grants real execution
#     without it.
# Synthetic-only unit tests (run_synthetic_checks()) are unaffected by any of
# this -- they never read DIMEV2C21_MODE, never touch a cache, and always run.
DIME_MODE = os.environ.get("DIMEV2C21_MODE")
OUT_DIR = os.environ.get("DIMEV2C21_OUT_DIR", os.path.join(SCRIPT_DIR, "results", "dime_v2_c2_1_local_retrieval_audit"))
FIG_DIR = os.path.join(OUT_DIR, "figures")

C1_DIR = os.environ.get("DIMEV2C21_C1_DIR", os.path.join(SCRIPT_DIR, "results", "dime_v2_c1_cluster_structure_audit"))
C1_ARTIFACT_PATH = os.path.join(C1_DIR, "dime_v1_cluster_artifact.npz")
C1_FINGERPRINT_PATH = os.path.join(C1_DIR, "construction_fingerprint.json")
C1_CLUSTER_STATS_PATH = os.path.join(C1_DIR, "cluster_statistics.csv")

N_CLUSTERS = int(os.environ.get("DIMEV2C21_N_CLUSTERS", "3811"))
SEED = 42
DIM = 768

# Centroid routing (production mixing convention, run_e1_wikitext103.MATCHED_CONFIG).
K_ROUTE = int(os.environ.get("DIMEV2C21_K_ROUTE", "20"))
TAU_ROUTE = 2.0

# Local within-cluster candidate selection (diagnostic choice, spec Part 1):
# m is the number of ORIGINAL observations selected within a routed cluster,
# a different quantity from k_route (number of centroids retrieved
# globally) -- kept as separate named constants throughout, never conflated.
M_LOCAL = int(os.environ.get("DIMEV2C21_M_LOCAL", "20"))
TAU_LOCAL = 2.0

# C1's own real, persisted fingerprint (verified against C1's
# construction_fingerprint.json earlier this session -- see preflight review
# Part A). Override-able so a synthetic pilot can verify against its own
# synthetic artifact's real hashes instead of these production values.
EXPECTED_CENTROIDS_SHA256 = os.environ.get(
    "DIMEV2C21_EXPECTED_CENTROIDS_SHA256",
    "e7adc93629e102ad7dc16fee9b1c0b1a29be599c44e1f5e42fedbf514d96957e")
EXPECTED_ASSIGNMENT_SHA256 = os.environ.get(
    "DIMEV2C21_EXPECTED_ASSIGNMENT_SHA256",
    "9dae0c746a8f89dc6f6a3e96b4e0cead73d00965f474d7c3795bdffe6767f13e")

# Sampling (spec Part 2.2/2.3): bounded cluster set, independent quantile
# stratification across 4 structural axes read from C1's cluster_statistics.csv.
N_SAMPLE_CLUSTERS = int(os.environ.get("DIMEV2C21_N_SAMPLE_CLUSTERS", "60"))  # spec range: 40-80
N_BINS_PER_AXIS = int(os.environ.get("DIMEV2C21_N_BINS_PER_AXIS", "3"))
N_UNUSUAL_CLUSTERS = int(os.environ.get("DIMEV2C21_N_UNUSUAL_CLUSTERS", "8"))
STRATIFICATION_AXES = ["n_j", "rms_dispersion", "distinct_token_fraction", f"traffic_top{K_ROUTE}_val"]

N_QUERY_VAL_TARGET = int(os.environ.get("DIMEV2C21_N_QUERY_VAL_TARGET", "100"))  # spec range: ~80-128
N_QUERY_CT_TARGET = int(os.environ.get("DIMEV2C21_N_QUERY_CT_TARGET", "100"))

TIER1_PATH = os.path.join(SCRIPT_DIR, "results", "wikitext103_gpt2_tier1.json")


# ============================================================
# Part 0.2 -- load and verify the persisted C1 partition (never retrain)
# ============================================================
def sha256_of_array(arr):
    return hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()


def load_and_verify_c1_artifact(artifact_path, fingerprint_path, expected_centroids_sha256,
                                  expected_assignment_sha256, n_clusters_expected, n_ds_expected=None):
    """Loads C1's persisted centroids/assignment directly and verifies byte-level
    hashes BEFORE any further calculation, per spec 0.2. Raises (STOPs) rather
    than substituting a new clustering run if the artifact is missing,
    mismatched, or corrupted -- this function must never call MiniBatchKMeans.
    Reloaded arrays are set read-only (setflags(write=False)) so nothing
    downstream can mutate C1's construction, even by accident.

    Revision 1 (review point 7): the fingerprint JSON is now REQUIRED to
    exist and match (Revision 0 treated it as optional-if-present, which the
    external reviewer correctly flagged as too weak); centroid/assignment
    dtypes are validated against the fingerprint's own recorded dtypes, not
    just shape/range."""
    if not os.path.exists(artifact_path):
        raise RuntimeError(
            f"C1 construction artifact not found at {artifact_path}. Per spec Part 0.2: "
            "STOP and report this; do not substitute a new clustering run.")
    if not os.path.exists(fingerprint_path):
        raise RuntimeError(
            f"C1 construction_fingerprint.json not found at {fingerprint_path}. Per review point 7, "
            "this file is REQUIRED (not optional) -- STOP and report this; do not proceed without it.")
    with open(fingerprint_path) as f:
        fingerprint_on_disk = json.load(f)

    with np.load(artifact_path) as data:
        centroids = np.ascontiguousarray(data["centroids"])
        assignment = np.ascontiguousarray(data["assignment"])
    centroids_sha256 = sha256_of_array(centroids)
    assignment_sha256 = sha256_of_array(assignment)

    if centroids_sha256 != expected_centroids_sha256 or assignment_sha256 != expected_assignment_sha256:
        raise RuntimeError(
            "C1 artifact hash mismatch against the externally supplied expected hashes -- "
            f"centroids: got {centroids_sha256}, expected {expected_centroids_sha256}; "
            f"assignment: got {assignment_sha256}, expected {expected_assignment_sha256}. "
            "STOP: do not proceed, do not substitute a new clustering run.")

    if (fingerprint_on_disk.get("centroids_sha256") != centroids_sha256
            or fingerprint_on_disk.get("assignment_sha256") != assignment_sha256):
        raise RuntimeError(
            "Reloaded C1 artifact's hash does not match its own persisted "
            "construction_fingerprint.json. STOP: the artifact or the fingerprint file "
            "may be corrupted or mismatched; do not proceed.")

    expected_centroids_dtype = fingerprint_on_disk.get("centroids_dtype")
    expected_assignment_dtype = fingerprint_on_disk.get("assignment_dtype")
    if expected_centroids_dtype is not None and str(centroids.dtype) != expected_centroids_dtype:
        raise RuntimeError(
            f"C1 centroids dtype {centroids.dtype} does not match the fingerprint's recorded "
            f"{expected_centroids_dtype}. STOP.")
    if expected_assignment_dtype is not None and str(assignment.dtype) != expected_assignment_dtype:
        raise RuntimeError(
            f"C1 assignment dtype {assignment.dtype} does not match the fingerprint's recorded "
            f"{expected_assignment_dtype}. STOP.")

    if centroids.shape[0] != n_clusters_expected or centroids.shape[1] != DIM:
        raise RuntimeError(
            f"C1 centroids shape {centroids.shape} does not match expected "
            f"({n_clusters_expected}, {DIM}). STOP.")
    if not np.all(np.isfinite(centroids)):
        raise RuntimeError("C1 centroids contain a non-finite value (NaN/inf). STOP.")
    if assignment.max(initial=-1) >= n_clusters_expected or assignment.min(initial=0) < 0:
        raise RuntimeError("C1 assignment contains an out-of-range cluster id. STOP.")
    if n_ds_expected is not None and assignment.shape[0] != n_ds_expected:
        raise RuntimeError(
            f"C1 assignment length {assignment.shape[0]} does not match the current "
            f"datastore size {n_ds_expected}. STOP: the datastore may have changed since C1 ran.")

    centroids.setflags(write=False)
    assignment.setflags(write=False)
    return centroids, assignment, fingerprint_on_disk, centroids_sha256, assignment_sha256


# ============================================================
# Part 0.3 / Part 2 -- centroid routing (production mixing convention)
# ============================================================
def compute_routing(dime_centroids, compressed_dists, ct_keys, val_keys, k_route):
    """Reuses the EXACT mechanism run_e1_wikitext103.py uses for production
    DIME inference (build_datastore over centroids, query_knn_indices for the
    top-k retrieved centroids) -- not a re-derivation, a direct reuse."""
    dime_index, _ = build_datastore(dime_centroids, np.array(compressed_dists, dtype=object))
    d_ct, idx_ct = query_knn_indices(dime_index, ct_keys, k=k_route)
    d_val, idx_val = query_knn_indices(dime_index, val_keys, k=k_route)
    return d_ct, idx_ct, d_val, idx_val


def centroid_mixture_weights(distances, tau_route):
    """Review point 1: Revision 0 computed centroid distances/indices but
    never turned them into the actual DIME matched-configuration centroid
    softmax -- this is that missing step, reusing GPT_Module/knn.py's
    softmax_np exactly as mixing.py::mix_dime_and_lm does
    (weights = softmax_np(-distances/tau, axis=1)). Asserts every query's
    row sums to 1."""
    w = softmax_np(-distances / tau_route, axis=1)
    assert np.allclose(w.sum(axis=1), 1.0, atol=1e-9), \
        "centroid mixture weights do not sum to 1 for every query"
    return w


def assert_no_duplicate_indices_per_query(idx_matrix):
    unique_counts = np.array([len(np.unique(row)) for row in idx_matrix])
    k = idx_matrix.shape[1]
    assert np.all(unique_counts == k), \
        f"duplicate retrieved-centroid indices found in {int(np.sum(unique_counts != k))} queries"


def routed_query_rows(idx_topk, d_topk, w_topk, cluster_id):
    """All query row indices whose top-k routed set contains cluster_id,
    plus each such row's rank, centroid distance, and centroid mixture
    weight AT that rank (review point 1 -- these are now actually threaded
    through, not just computed and discarded)."""
    mask = idx_topk == cluster_id
    rows = np.nonzero(mask.any(axis=1))[0]
    if len(rows) == 0:
        empty = np.array([], dtype=int)
        return empty, empty, np.array([]), np.array([])
    ranks = np.argmax(mask[rows], axis=1)
    distances = d_topk[rows, ranks]
    weights = w_topk[rows, ranks]
    return rows, ranks, distances, weights


# ============================================================
# Part 1 -- exact local candidate selection / competition / distribution
# ============================================================
def local_candidate_selection(member_keys, query_key, m):
    """Returns (sel_pos, sel_d, margin, all_d). sel_pos indexes INTO
    member_keys (positions 0..n_c-1, not original datastore row ids).
    margin = d_(m+1) - d_(m) is reported only when n_c > m; for n_c <= m the
    hard top-m boundary does not exist and margin is None, never 0 (spec
    Part 3.1)."""
    n_c = member_keys.shape[0]
    d = np.linalg.norm(member_keys - query_key[None, :], axis=1)
    m_c = min(m, n_c)
    order = np.argsort(d, kind="stable")
    sel_pos = order[:m_c]
    sel_d = d[sel_pos]
    if n_c > m:
        margin = float(d[order[m]] - d[order[m - 1]])
    else:
        margin = None
    return sel_pos, sel_d, margin, d


def local_weights(sel_d, tau_local):
    """Numerically stable softmax over LOCAL candidate distances (unsquared
    Euclidean). m_c=1 handled explicitly (weight exactly 1.0, not a softmax
    over a single value that happens to also be 1.0 -- same result, stated
    explicitly per spec 4.1)."""
    if len(sel_d) == 1:
        w = np.array([1.0])
    else:
        w = softmax_np(-sel_d / tau_local, axis=-1)
    assert np.all(np.isfinite(w)) and np.all(w >= 0) and abs(float(w.sum()) - 1.0) < 1e-9, \
        "local competition weights failed the finite/nonnegative/sum-to-one invariant"
    return w


def exact_local_distribution(sel_values, weights):
    """F_C^(m)(q): sparse {token_id: prob} dict, sums to 1 by construction
    (weights already sum to 1); multiple selected observations sharing a
    token correctly accumulate into the same entry."""
    d = {}
    for v, w in zip(sel_values.tolist(), weights.tolist()):
        v = int(v)
        d[v] = d.get(v, 0.0) + float(w)
    return d


def weight_concentration_stats(weights):
    m_c = len(weights)
    if m_c == 1:
        return {"H_pi": 0.0, "N_eff": 1.0, "max_weight": 1.0, "top3_mass": 1.0, "H_pi_over_logm": None}
    H_pi = float(-np.sum(weights * np.log(np.clip(weights, 1e-300, None))))
    N_eff = float(1.0 / np.sum(weights ** 2))
    sorted_w = np.sort(weights)[::-1]
    logm = float(np.log(m_c))
    return {
        "H_pi": H_pi, "N_eff": N_eff,
        "max_weight": float(sorted_w[0]), "top3_mass": float(sorted_w[:min(3, m_c)].sum()),
        "H_pi_over_logm": (H_pi / logm) if logm > 0 else None,
    }


def local_distribution_fixed_candidates(member_keys, member_values, sel_pos, query_key, tau_local):
    """G(q; S): evaluates a FIXED candidate set S (positions into member_keys,
    selected at some other query) at a new query -- isolates competition-
    weight change from candidate-identity change (spec Part 4.2/4.3)."""
    d = np.linalg.norm(member_keys[sel_pos] - query_key[None, :], axis=1)
    w = local_weights(d, tau_local)
    F = exact_local_distribution(member_values[sel_pos], w)
    return F, w, d


def normalize_counter(counter):
    total = sum(counter.values())
    if total <= 0:
        return {}
    return {int(t): v / total for t, v in counter.items()}


def entropy_of_dict(d):
    return float(-sum(p * np.log(p) for p in d.values() if p > 0))


def kl_divergence_dict(p, q_ref):
    """KL(p || q_ref) in nats. Returns None (not a clipped/smoothed finite
    value) whenever p has support where q_ref is exactly zero -- spec 6.2
    explicitly forbids silently clipping or hiding this; those cases must be
    reported as missing-support, not as a finite number."""
    for t, pt in p.items():
        if pt > 0 and q_ref.get(t, 0.0) <= 0:
            return None
    s = 0.0
    for t, pt in p.items():
        if pt > 0:
            s += pt * np.log(pt / q_ref[t])
    return float(s)


def mean_distribution(dist_list):
    """bar_F = elementwise mean of a list of sparse {token: prob} dicts, each
    already summing to 1 -- the result also sums to 1. Used both for the
    train-derived constant baseline (6.2) and the in-sample val-sample mean
    used in the mutual-information estimate (6.3)."""
    if not dist_list:
        return {}
    acc = defaultdict(float)
    for d in dist_list:
        for t, p in d.items():
            acc[t] += p
    n = len(dist_list)
    return {t: v / n for t, v in acc.items()}


# ============================================================
# Part 2.2 -- cluster sampling (structural covariates only, from C1's CSV)
# ============================================================
def stratified_cluster_sample(cluster_stats_df, axes, n_bins_per_axis, n_to_sample, seed):
    """Independent per-axis quantile binning over `axes` (read from C1's
    cluster_statistics.csv, never recomputed), forming a grid; ONE cluster
    per occupied cell, chosen from a seeded SHUFFLE (not always the lowest
    id, per spec 2.2) rather than a deterministic tie-break. If more
    occupied cells exist than the sampling budget, a reproducible random
    SUBSET OF CELLS (not of clusters within a cell) is kept, preserving
    spread across the grid rather than clustering the cut near any one
    region.

    Review point 5: now also returns, for every ELIGIBLE cluster (not just
    sampled ones), its cell assignment and that cell's population, plus an
    approximate per-sampled-cluster inclusion probability
    (1/cell_population * cells_kept/n_occupied_cells), so sampling weight
    can be reported rather than assumed uniform."""
    eligible = cluster_stats_df[cluster_stats_df["n_j"] > 0].copy()
    if len(eligible) == 0:
        empty_coverage = {"n_eligible": 0, "n_occupied_cells": 0, "n_total_possible_cells": n_bins_per_axis ** len(axes),
                           "n_sampled": 0, "axes": axes, "n_bins_per_axis": n_bins_per_axis}
        return [], empty_coverage, {}, {}, {}

    bin_cols = {}
    for ax in axes:
        vals = eligible[ax].values.astype(np.float64)
        edges = np.unique(np.quantile(vals, np.linspace(0, 1, n_bins_per_axis + 1)))
        inner_edges = edges[1:-1]
        bins = np.digitize(vals, inner_edges, right=True) if len(inner_edges) else np.zeros(len(vals), dtype=int)
        bin_cols[ax] = bins
    # Cast to plain Python int: zip() over numpy arrays yields np.int64
    # scalars, whose repr (as of NumPy 2.x) is "np.int64(2)" -- would leak
    # into sampling_report.md's printed cell tuples otherwise.
    eligible["_cell"] = [tuple(int(v) for v in row) for row in zip(*(bin_cols[ax] for ax in axes))]

    cluster_cell_map = {int(row["cluster"]): row["_cell"] for _, row in eligible.iterrows()}
    cell_population = Counter(eligible["_cell"].tolist())

    shuffled = eligible.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    cell_to_cluster = {}
    for _, row in shuffled.iterrows():
        cell = row["_cell"]
        if cell not in cell_to_cluster:
            cell_to_cluster[cell] = int(row["cluster"])

    occupied_cells = sorted(cell_to_cluster.keys())
    rng = np.random.default_rng(seed)
    if len(occupied_cells) > n_to_sample:
        chosen = np.sort(rng.choice(len(occupied_cells), size=n_to_sample, replace=False))
        occupied_cells_kept = [occupied_cells[i] for i in chosen]
    else:
        occupied_cells_kept = occupied_cells
    sampled = sorted(cell_to_cluster[c] for c in occupied_cells_kept)

    cell_keep_prob = (len(occupied_cells_kept) / len(occupied_cells)) if occupied_cells else 0.0
    inclusion_prob = {
        cid: (1.0 / cell_population[cluster_cell_map[cid]]) * cell_keep_prob
        for cid in sampled
    }

    coverage = {
        "n_eligible": int(len(eligible)), "n_occupied_cells": int(len(occupied_cells)),
        "n_total_possible_cells": int(n_bins_per_axis ** len(axes)),
        "n_cells_kept": int(len(occupied_cells_kept)), "n_sampled": int(len(sampled)),
        "axes": axes, "n_bins_per_axis": n_bins_per_axis, "cell_keep_prob": cell_keep_prob,
    }
    return sampled, coverage, cluster_cell_map, dict(cell_population), inclusion_prob


def unusual_cluster_candidates(cluster_stats_df, n_unusual):
    """Deliberately included extreme structural cases (spec 2.2), labeled
    SEPARATELY from the randomly stratified sample -- selection is by a
    predeclared list of simple extremal criteria over C1's own covariates,
    never by any C2 outcome variable."""
    nonempty = cluster_stats_df[cluster_stats_df["n_j"] > 0]
    picks = {}
    picks["max_population"] = int(nonempty.loc[nonempty["n_j"].idxmax(), "cluster"])
    picks["max_dispersion"] = int(nonempty.loc[nonempty["rms_dispersion"].idxmax(), "cluster"])
    picks["max_singleton_token_fraction"] = int(nonempty.loc[nonempty["singleton_token_fraction"].idxmax(), "cluster"])
    picks[f"max_traffic_top{K_ROUTE}_val"] = int(nonempty.loc[nonempty[f"traffic_top{K_ROUTE}_val"].idxmax(), "cluster"])
    non_singleton = nonempty[nonempty["n_j"] > 1]
    if len(non_singleton):
        picks["min_population_above_singleton"] = int(non_singleton.loc[non_singleton["n_j"].idxmin(), "cluster"])
    picks["max_V_j"] = int(nonempty.loc[nonempty["V_j"].idxmax(), "cluster"])
    picks["max_top_token_fraction"] = int(nonempty.loc[nonempty["top_token_fraction"].idxmax(), "cluster"])
    picks["min_entropy"] = int(nonempty.loc[nonempty["entropy"].idxmin(), "cluster"])

    seen, reasons = [], {}
    for reason, cid in picks.items():
        if cid not in seen:
            seen.append(cid)
            reasons[cid] = reason
    return seen[:n_unusual], reasons


# ============================================================
# Part 9 -- synthetic correctness checks (isolated from real data)
# ============================================================
def check_local_candidate_selection_known_values():
    keys = np.array([[0.0], [1.0], [2.0], [5.0], [10.0]])
    q = np.array([0.0])
    sel_pos, sel_d, margin, all_d = local_candidate_selection(keys, q, m=3)
    assert list(sel_pos) == [0, 1, 2], sel_pos
    assert np.allclose(sel_d, [0.0, 1.0, 2.0]), sel_d
    assert margin is not None and abs(margin - (5.0 - 2.0)) < 1e-9, margin


def check_margin_undefined_when_nc_le_m():
    keys = np.array([[0.0], [1.0]])
    q = np.array([0.0])
    sel_pos, sel_d, margin, all_d = local_candidate_selection(keys, q, m=5)
    assert len(sel_pos) == 2 and margin is None, (sel_pos, margin)


def check_local_weights_known_values():
    w = local_weights(np.array([0.0, 2.0]), tau_local=2.0)
    expected = np.exp([0.0, -1.0]) / np.exp([0.0, -1.0]).sum()
    assert np.allclose(w, expected), (w, expected)
    w1 = local_weights(np.array([7.3]), tau_local=2.0)
    assert np.allclose(w1, [1.0]), w1


def check_exact_local_distribution_shared_token():
    values = np.array([5, 5, 7])
    w = np.array([0.5, 0.3, 0.2])
    F = exact_local_distribution(values, w)
    assert abs(F[5] - 0.8) < 1e-9 and abs(F[7] - 0.2) < 1e-9, F
    assert abs(sum(F.values()) - 1.0) < 1e-9


def check_candidate_change_without_token_change():
    # Review point 2a: Revision 0's version used two members 0.1 apart and
    # queried from their midpoint, which created an exact TIE (both at
    # distance 0.05) -- the candidate identity change was never actually
    # provable, which is why the old assertion needed "or True" to avoid
    # failing. Replaced with a DETERMINISTIC scenario: two members far apart
    # (distance 10, no tie possible under floating point), each
    # unambiguously nearest for one of two well-separated queries, both
    # carrying the SAME label (9) -- the selected observation id provably
    # changes, while the resulting token distribution must stay identical.
    member_keys = np.array([[0.0], [10.0], [100.0]])
    member_values = np.array([9, 9, 3])
    q1 = np.array([0.0])
    q2 = np.array([10.0])
    sel1, d1, _, _ = local_candidate_selection(member_keys, q1, m=1)
    sel2, d2, _, _ = local_candidate_selection(member_keys, q2, m=1)
    assert set(sel1.tolist()) != set(sel2.tolist()), "candidate identity should provably change here"
    assert member_values[sel1][0] == member_values[sel2][0] == 9
    F1 = exact_local_distribution(member_values[sel1], local_weights(d1, TAU_LOCAL))
    F2 = exact_local_distribution(member_values[sel2], local_weights(d2, TAU_LOCAL))
    assert total_variation(F1, F2) < 1e-9, (F1, F2)


def check_label_changing_replacement():
    member_keys = np.array([[0.0], [0.2], [10.0]])
    member_values = np.array([1, 2, 1])  # replacing idx0 (label1) with idx1 (label2) changes the label
    q1 = np.array([0.0])
    q2 = np.array([0.2])
    sel1, d1, _, _ = local_candidate_selection(member_keys, q1, m=1)
    sel2, d2, _, _ = local_candidate_selection(member_keys, q2, m=1)
    F1 = exact_local_distribution(member_values[sel1], local_weights(d1, TAU_LOCAL))
    F2 = exact_local_distribution(member_values[sel2], local_weights(d2, TAU_LOCAL))
    assert set(member_values[sel1].tolist()) != set(member_values[sel2].tolist())
    assert total_variation(F1, F2) > 0.5, (F1, F2)


def check_fixed_candidate_competition_change():
    member_keys = np.array([[0.0], [4.0]])
    member_values = np.array([1, 2])
    sel_pos = np.array([0, 1])
    F_q, w_q, d_q = local_distribution_fixed_candidates(member_keys, member_values, sel_pos, np.array([0.0]), tau_local=2.0)
    F_qp, w_qp, d_qp = local_distribution_fixed_candidates(member_keys, member_values, sel_pos, np.array([4.0]), tau_local=2.0)
    # Manually: at q=0, distances=[0,4] -> weights favor label1; at q'=4, distances=[4,0] -> favor label2.
    assert F_q[1] > F_q[2] and F_qp[2] > F_qp[1], (F_q, F_qp)
    manual_w_q = softmax_np(-np.array([0.0, 4.0]) / 2.0, axis=-1)
    assert np.allclose(w_q, manual_w_q), (w_q, manual_w_q)


def check_tv_js_kl_known_values():
    p = {1: 0.5, 2: 0.5}
    q = {1: 1.0}
    assert abs(total_variation(p, q) - 0.5) < 1e-9, total_variation(p, q)
    assert js_divergence_dict(p, p) < 1e-12
    kl_pq = kl_divergence_dict(p, q)
    assert kl_pq is None, kl_pq  # p has mass on token 2 where q is exactly zero -- must be None, not clipped
    kl_qp = kl_divergence_dict(q, p)
    assert kl_qp is not None and abs(kl_qp - np.log(2.0)) < 1e-9, kl_qp


def check_static_vs_query_dependent():
    member_values = np.array([1, 1, 1, 2])
    static_hist = normalize_counter(Counter(member_values.tolist()))
    assert abs(static_hist[1] - 0.75) < 1e-9 and abs(static_hist[2] - 0.25) < 1e-9
    F_q = {1: 0.1, 2: 0.9}  # a hand-built query-dependent distribution that strongly disagrees with the static histogram
    assert total_variation(F_q, static_hist) > 0.5


def check_mean_distribution_forward_kl_identity():
    rng = np.random.default_rng(0)
    dists = []
    for _ in range(20):
        probs = rng.dirichlet([1.0, 1.0, 1.0])
        dists.append({1: float(probs[0]), 2: float(probs[1]), 3: float(probs[2])})
    bar_F = mean_distribution(dists)
    assert abs(sum(bar_F.values()) - 1.0) < 1e-9
    H_bar = entropy_of_dict(bar_F)
    mean_H = float(np.mean([entropy_of_dict(d) for d in dists]))
    I_direct = H_bar - mean_H
    I_via_kl = float(np.mean([kl_divergence_dict(d, bar_F) for d in dists]))
    assert abs(I_direct - I_via_kl) < 1e-9, (I_direct, I_via_kl)
    assert I_direct >= -1e-9  # mutual information is non-negative


def check_train_val_baseline_construction_and_evaluation():
    # Review point 2b: Revision 0's "separation" test only checked that two
    # hand-picked index arrays were disjoint -- it never exercised the
    # actual construction of bar_F_train or its evaluation against held-out
    # queries. This version builds a real tiny cluster, computes F_C(q) for
    # a TRAIN set and a disjoint VAL set from genuinely different query
    # arrays (train_keys/val_keys, not index tricks on one array), builds
    # bar_F_train ONLY from the train F_C(q)'s, and checks its evaluation
    # against the held-out val F_C(q)'s is arithmetically correct against a
    # hand-computed expectation.
    member_keys = np.array([[0.0], [1.0], [2.0], [3.0]])
    member_values = np.array([1, 1, 2, 2])
    train_keys = [np.array([0.0]), np.array([3.0])]  # exercises both ends of the cluster
    val_keys_ = [np.array([1.0])]

    def F_for(q):
        sel_pos, sel_d, _, _ = local_candidate_selection(member_keys, q, m=2)
        w = local_weights(sel_d, TAU_LOCAL)
        return exact_local_distribution(member_values[sel_pos], w)

    train_F_list = [F_for(q) for q in train_keys]
    val_F_list = [F_for(q) for q in val_keys_]
    bar_F_train = mean_distribution(train_F_list)

    # Hand-computed: mean_distribution must be the exact elementwise average
    # of the two TRAIN F_C(q) dicts, using none of the val data.
    expected = {}
    for d in train_F_list:
        for t, p in d.items():
            expected[t] = expected.get(t, 0.0) + p / len(train_F_list)
    assert set(expected.keys()) == set(bar_F_train.keys())
    for t in expected:
        assert abs(expected[t] - bar_F_train[t]) < 1e-9

    tv_holdout = total_variation(val_F_list[0], bar_F_train)
    assert 0.0 <= tv_holdout <= 1.0
    # The held-out evaluation must use a query array NEVER passed into mean_distribution.
    assert id(val_keys_) != id(train_keys) and val_keys_[0].tolist() not in [k.tolist() for k in train_keys]


def check_centroid_mixture_weights_known_values():
    # Review point 1/2c: mixture-weight formula must match a hand-computed
    # softmax, and every query's row must sum to exactly 1.
    distances = np.array([[0.0, 2.0, 4.0], [1.0, 1.0, 1.0]])
    w = centroid_mixture_weights(distances, tau_route=2.0)
    expected_row0 = np.exp([0.0, -1.0, -2.0]) / np.exp([0.0, -1.0, -2.0]).sum()
    assert np.allclose(w[0], expected_row0), (w[0], expected_row0)
    assert np.allclose(w[1], [1 / 3, 1 / 3, 1 / 3]), w[1]
    assert np.allclose(w.sum(axis=1), 1.0)


def check_counterfactual_decomposition_known_values():
    # Review point 2d: hand-computed forward decomposition
    # (tv_weight_change, tv_selection_change) plus a forward/reverse
    # consistency check, independent of analyze_cluster's own code path.
    member_keys = np.array([[0.0], [10.0]])
    member_values = np.array([1, 2])
    sel_i = np.array([0, 1])  # candidates at query i = both members (m=2, n_c=2, trivial)
    q_i, q_j = np.array([0.0]), np.array([10.0])

    F_i, w_i, _ = local_distribution_fixed_candidates(member_keys, member_values, sel_i, q_i, TAU_LOCAL)
    sel_j, d_j, _, _ = local_candidate_selection(member_keys, q_j, m=2)
    F_j = exact_local_distribution(member_values[sel_j], local_weights(d_j, TAU_LOCAL))

    # G(q_j; S_i): same candidate SET (both members, trivial selection), evaluated at q_j.
    G_qj_Si, _, _ = local_distribution_fixed_candidates(member_keys, member_values, sel_i, q_j, TAU_LOCAL)
    tv_weight_change = total_variation(F_i, G_qj_Si)
    tv_selection_change = total_variation(G_qj_Si, F_j)  # selection set is identical (n_c<=m), so this must be ~0
    assert tv_selection_change < 1e-9, tv_selection_change
    assert tv_weight_change > 0.3, tv_weight_change  # weights swing hard from favoring label1 to favoring label2

    # Reverse path: should be symmetric here since the scenario itself is symmetric.
    G_qi_Sj, _, _ = local_distribution_fixed_candidates(member_keys, member_values, sel_j, q_i, TAU_LOCAL)
    tv_weight_change_rev = total_variation(F_j, G_qi_Sj)
    assert abs(tv_weight_change - tv_weight_change_rev) < 1e-9, (tv_weight_change, tv_weight_change_rev)


def check_selection_margin_stability_guarantee():
    # Review point 3: gamma_m(q) > 2*||q-q'|| => the top-m selected set at q
    # and q' must be IDENTICAL (triangle inequality). Construct a cluster
    # with a known large margin at q, then a nearby q' close enough that the
    # guarantee applies, and verify the sets are in fact equal.
    member_keys = np.array([[0.0], [1.0], [2.0], [20.0], [21.0]])
    member_values = np.array([1, 1, 1, 2, 2])
    q = np.array([0.0])
    sel_q, d_q, margin, _ = local_candidate_selection(member_keys, q, m=3)
    assert margin is not None and margin > 10  # d_(4)=20, d_(3)=2 -> margin=18
    q_prime = np.array([0.5])  # ||q-q'||=0.5, 2*0.5=1.0 < margin=18 -> guarantee applies
    sel_qprime, _, _, _ = local_candidate_selection(member_keys, q_prime, m=3)
    assert set(sel_q.tolist()) == set(sel_qprime.tolist()), (sel_q, sel_qprime)


def check_boundary_tie_handling():
    # Review point 7: an EXACT tie at the m/m+1 boundary with DIFFERENT
    # labels at the tied positions -- tie-breaking must be deterministic
    # (stable argsort on ascending member-array position) and reproducible
    # across repeated calls, and the margin must be reported as exactly 0
    # (not skipped, not silently treated as "no boundary").
    member_keys = np.array([[0.0], [5.0], [5.0], [9.0]])
    member_values = np.array([1, 7, 8, 2])  # positions 1 and 2 are the exact tie, different labels
    q = np.array([0.0])
    sel1, d1, margin1, all_d1 = local_candidate_selection(member_keys, q, m=2)
    sel2, d2, margin2, all_d2 = local_candidate_selection(member_keys, q, m=2)
    assert margin1 is not None and abs(margin1) < 1e-12, margin1  # exact tie -> margin exactly 0
    assert list(sel1) == list(sel2), "tie-breaking must be deterministic across repeated calls"
    assert list(sel1) == [0, 1]  # stable sort keeps the tie's lower-position member (idx1) over idx2
    assert member_values[sel1[-1]] != member_values[2]  # the excluded tied member has a DIFFERENT label


def check_missing_c1_artifact_raises():
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        raised = False
        try:
            load_and_verify_c1_artifact(os.path.join(d, "missing.npz"), os.path.join(d, "missing_fp.json"),
                                          "a" * 64, "b" * 64, n_clusters_expected=4)
        except RuntimeError:
            raised = True
        assert raised, "missing artifact file should raise"


def check_missing_fingerprint_raises():
    import tempfile
    rng = np.random.default_rng(3)
    centroids = rng.standard_normal((4, DIM)).astype(np.float32)
    assignment = rng.integers(0, 4, size=20).astype(np.int32)
    with tempfile.TemporaryDirectory() as d:
        artifact_path = os.path.join(d, "artifact.npz")
        np.savez_compressed(artifact_path, centroids=centroids, assignment=assignment)
        c_sha, a_sha = sha256_of_array(centroids), sha256_of_array(assignment)
        raised = False
        try:
            load_and_verify_c1_artifact(artifact_path, os.path.join(d, "missing_fp.json"), c_sha, a_sha,
                                          n_clusters_expected=4)
        except RuntimeError:
            raised = True
        assert raised, "missing fingerprint JSON should raise (review point 7 -- now required, not optional)"


def check_c1_artifact_hash_verification():
    import tempfile
    rng = np.random.default_rng(1)
    centroids = rng.standard_normal((4, DIM)).astype(np.float32)
    assignment = rng.integers(0, 4, size=50).astype(np.int32)
    with tempfile.TemporaryDirectory() as d:
        artifact_path = os.path.join(d, "artifact.npz")
        np.savez_compressed(artifact_path, centroids=centroids, assignment=assignment)
        c_sha = sha256_of_array(centroids)
        a_sha = sha256_of_array(assignment)
        fp_path = os.path.join(d, "fp.json")
        with open(fp_path, "w") as f:
            json.dump({"centroids_sha256": c_sha, "assignment_sha256": a_sha}, f)

        loaded_c, loaded_a, fp, got_c_sha, got_a_sha = load_and_verify_c1_artifact(
            artifact_path, fp_path, c_sha, a_sha, n_clusters_expected=4, n_ds_expected=50)
        assert got_c_sha == c_sha and got_a_sha == a_sha
        assert not loaded_c.flags.writeable and not loaded_a.flags.writeable

        raised = False
        try:
            load_and_verify_c1_artifact(artifact_path, fp_path, "0" * 64, a_sha, n_clusters_expected=4, n_ds_expected=50)
        except RuntimeError:
            raised = True
        assert raised, "tampered expected hash did not raise"


def check_routing_duplicate_and_normalization():
    idx_ok = np.array([[0, 1, 2], [2, 1, 0]])
    assert_no_duplicate_indices_per_query(idx_ok)  # should not raise
    idx_bad = np.array([[0, 0, 2]])
    raised = False
    try:
        assert_no_duplicate_indices_per_query(idx_bad)
    except AssertionError:
        raised = True
    assert raised


def check_stratified_cluster_sample_covers_axes():
    rng = np.random.default_rng(2)
    n = 400
    df = pd.DataFrame({
        "cluster": np.arange(n),
        "n_j": rng.integers(1, 500, size=n),
        "rms_dispersion": rng.uniform(1, 30, size=n),
        "distinct_token_fraction": rng.uniform(0, 1, size=n),
        f"traffic_top{K_ROUTE}_val": rng.uniform(0, 0.01, size=n),
    })
    sampled, coverage, cluster_cell_map, cell_population, inclusion_prob = stratified_cluster_sample(
        df, STRATIFICATION_AXES, n_bins_per_axis=3, n_to_sample=30, seed=42)
    assert len(sampled) == len(set(sampled)) and len(sampled) <= 30
    assert coverage["n_occupied_cells"] > 10, coverage  # with random continuous data, most cells should be occupied
    assert all(cid in cluster_cell_map for cid in sampled)
    assert all(0.0 < inclusion_prob[cid] <= 1.0 for cid in sampled), inclusion_prob
    assert all(cell_population[cluster_cell_map[cid]] >= 1 for cid in sampled)


def run_synthetic_checks():
    checks = {
        "local_candidate_selection_known_values": check_local_candidate_selection_known_values,
        "margin_undefined_when_nc_le_m": check_margin_undefined_when_nc_le_m,
        "local_weights_known_values": check_local_weights_known_values,
        "exact_local_distribution_shared_token": check_exact_local_distribution_shared_token,
        "candidate_change_without_token_change": check_candidate_change_without_token_change,
        "label_changing_replacement": check_label_changing_replacement,
        "fixed_candidate_competition_change": check_fixed_candidate_competition_change,
        "tv_js_kl_known_values": check_tv_js_kl_known_values,
        "static_vs_query_dependent": check_static_vs_query_dependent,
        "mean_distribution_forward_kl_identity": check_mean_distribution_forward_kl_identity,
        "train_val_baseline_construction_and_evaluation": check_train_val_baseline_construction_and_evaluation,
        "centroid_mixture_weights_known_values": check_centroid_mixture_weights_known_values,
        "counterfactual_decomposition_known_values": check_counterfactual_decomposition_known_values,
        "selection_margin_stability_guarantee": check_selection_margin_stability_guarantee,
        "boundary_tie_handling": check_boundary_tie_handling,
        "c1_artifact_hash_verification": check_c1_artifact_hash_verification,
        "missing_c1_artifact_raises": check_missing_c1_artifact_raises,
        "missing_fingerprint_raises": check_missing_fingerprint_raises,
        "routing_duplicate_and_normalization": check_routing_duplicate_and_normalization,
        "stratified_cluster_sample_covers_axes": check_stratified_cluster_sample_covers_axes,
    }
    results = {}
    for name, fn in checks.items():
        try:
            fn()
            results[name] = True
        except Exception as e:
            results[name] = False
            print(f"  SYNTHETIC CHECK FAILED: {name}: {e}")
    return results


# ============================================================
# Deliverable writers (Part 10)
# ============================================================
def environment_metadata():
    return {
        "python_version": platform.python_version(), "numpy_version": np.__version__,
        "sklearn_version": sklearn.__version__, "scipy_version": scipy.__version__,
        "pandas_version": pd.__version__, "platform": platform.platform(),
    }


def write_construction_consistency_md(out_dir, c1_centroids_sha, c1_assignment_sha, c1_fingerprint_on_disk,
                                        n_ds, n_clusters, k_route, tau_route, m_local, tau_local,
                                        traffic_comparison_note, pilot_note):
    path = os.path.join(out_dir, "construction_consistency.md")
    with open(path, "w") as f:
        f.write("# C2.1 Construction Consistency\n\n")
        f.write(pilot_note)
        f.write(
            "## C1 artifact verification\n\n"
            f"- Loaded from: `{C1_ARTIFACT_PATH}`\n"
            f"- Centroids SHA-256 (verified against externally supplied expected hash): `{c1_centroids_sha}`\n"
            f"- Assignment SHA-256 (verified against externally supplied expected hash): `{c1_assignment_sha}`\n"
            f"- C1's own construction_fingerprint.json REQUIRED and matching (review point 7): "
            f"{c1_fingerprint_on_disk is not None}\n"
            f"- Centroid/assignment dtypes validated against the fingerprint's own recorded dtypes: True\n"
            f"- Centroids/assignment loaded read-only (setflags(write=False)): True\n"
            f"- Datastore row count at load time: {n_ds}\n"
            f"- n_clusters: {n_clusters}\n\n"
            "## Routing semantics reused (not re-derived)\n\n"
            f"- Centroid routing: `GPT_Module/knn.py`'s `build_datastore`/`query_knn_indices`, the exact mechanism "
            f"`mixing.py`'s `mix_dime_and_lm` uses for production inference. k_route={k_route}, tau_route={tau_route}.\n"
            "- Centroid MIXTURE WEIGHTS (review point 1): `centroid_mixture_weights` applies "
            "`softmax_np(-distances/tau_route, axis=1)` to the routing distances -- the actual DIME "
            "matched-configuration softmax, not just the raw distances/indices Revision 0 stopped at. "
            "Verified to sum to 1 for every query (asserted at computation time and in "
            "`check_centroid_mixture_weights_known_values`).\n"
            f"- Local within-cluster candidate selection: m_local={m_local}, tau_local={tau_local} (a separate, "
            "diagnostic-only choice -- see module docstring for why these are never conflated with k_route/tau_route).\n\n"
            "## Tie-breaking convention (review point 7)\n\n"
            "Local top-m selection uses `np.argsort(d, kind=\"stable\")`: ties are broken by ascending "
            "position within the cluster's member array, which itself preserves ascending original "
            "datastore row order (`build_cluster_to_members` uses a stable sort). This is deterministic "
            "and reproducible across repeated calls (`check_boundary_tie_handling`). Boundary ties at the "
            "m/m+1 split, and whether the tied pair carries different labels, are detected per query and "
            "aggregated into `cluster_behavior_statistics.csv`'s `frac_boundary_tie`/"
            "`frac_boundary_tie_diff_label` columns.\n\n"
            f"## Recomputed routing traffic vs. C1's recorded traffic (review point 7)\n\n{traffic_comparison_note}\n\n"
            "## Source references\n\n"
            "- `DIME/state_object.py` lines 39-56 (`minibatch_kmeans_partition`) -- authoritative construction, "
            "NOT called here; C1's persisted artifact is loaded instead.\n"
            "- `DIME/run_e1_wikitext103.py` line 58 (`MATCHED_CONFIG = {'k': 20, 'tau': 2.0, 'alpha': 0.25}`) -- "
            "source of k_route/tau_route.\n"
            "- `DIME/mixing.py` (`mix_dime_and_lm`) -- confirms the static per-cluster histogram this diagnostic "
            "compares against is `count(y)/n_C`, no smoothing (beta=0 in the matched config), and the "
            "`weights = softmax_np(-distances/tau)` formula `centroid_mixture_weights` reuses.\n"
            "- `GPT_Module/cache_io.py` -- confirms the cache format has no row-id field; query identifiers used "
            "throughout this diagnostic are POSITIONAL row indices into the controller_train/val cache arrays, "
            "structural identifiers only, not independently verifiable semantic provenance.\n"
        )
    return path


def write_sampling_report_md(out_dir, coverage, sampled_df, unusual_reasons, exclusions,
                               cluster_cell_map, cell_population, inclusion_prob, pilot_note):
    path = os.path.join(out_dir, "sampling_report.md")
    with open(path, "w") as f:
        f.write("# C2.1 Sampling Report\n\n")
        f.write(pilot_note)
        f.write(
            "## Cluster sampling\n\n"
            f"- Stratification axes (from C1's `cluster_statistics.csv`, never recomputed): {coverage['axes']}\n"
            f"- Bins per axis: {coverage['n_bins_per_axis']} "
            f"(total possible cells: {coverage['n_total_possible_cells']})\n"
            f"- Eligible (nonempty) clusters: {coverage['n_eligible']}\n"
            f"- Occupied cells: {coverage['n_occupied_cells']}; cells kept after budget subsampling: "
            f"{coverage.get('n_cells_kept')} (cell_keep_prob={coverage.get('cell_keep_prob')})\n"
            f"- Clusters sampled via stratification: {coverage['n_sampled']}\n"
            f"- Random seed: {SEED} (shuffle before per-cell representative selection -- NOT always the lowest id)\n\n"
            "## Unusual/extreme clusters (labeled separately, not part of the stratified count)\n\n"
        )
        for cid, reason in unusual_reasons.items():
            f.write(f"- cluster {cid}: {reason}\n")
        f.write("\n## Selected cluster IDs (sorted, union of stratified + unusual)\n\n")
        f.write(", ".join(str(c) for c in sorted(sampled_df["cluster"].tolist())))
        f.write(
            "\n\n## Per-cluster stratum, cell population, and approximate inclusion probability\n\n"
            "Review point 5: `cell` is this cluster's bin-tuple over the stratification axes; "
            "`cell_population` is how many ELIGIBLE clusters shared that cell (not just the one chosen); "
            "`inclusion_probability` is `(1/cell_population) * cell_keep_prob` for clusters reached via "
            "the random stratified mechanism (`sampling_mode=stratified` or `both`) -- an approximation "
            "(it ignores that a cluster could in principle occupy a cell with unusual marginal properties), "
            "not an exact design-based weight. Deliberately-included unusual clusters "
            "(`sampling_mode=unusual`) have NO inclusion probability -- they were not randomly selected, "
            "and must not be pooled into population-level/representativeness estimates. "
            "`structural_relationships.csv` reports a `stratified_only` scope for exactly this reason.\n\n"
        )
        rows = []
        for _, r in sampled_df.iterrows():
            cid = int(r["cluster"])
            cell = cluster_cell_map.get(cid)
            rows.append({
                "cluster": cid, "sampling_mode": r.get("sampling_mode", ""),
                "cell": str(cell) if cell is not None else "N/A (unusual-only)",
                "cell_population": cell_population.get(cell, "N/A") if cell is not None else "N/A",
                "inclusion_probability": inclusion_prob.get(cid, None),
            })
        f.write(pd.DataFrame(rows).to_string(index=False))
        f.write("\n\n## Per-cluster query availability and weighting convention\n\n")
        f.write(
            "Primary within-cluster descriptive analysis treats sampled routed val query-cluster pairs "
            "UNIFORMLY (unbiased within-cluster sampling of Q_C^(20)); centroid mixture-weighted aggregates "
            "(`massweighted_*` columns) are reported separately in `cluster_behavior_statistics.csv` and "
            "never averaged together with the uniform-weight (`mean_*`) aggregates under the same column name.\n\n"
        )
        cols = ["cluster", "n_j", "n_val_available", "n_val_sampled", "n_ct_available", "n_ct_sampled", "is_trivial_selection"]
        f.write(sampled_df[cols].to_string(index=False))
        f.write("\n\n## Exclusions\n\n")
        f.write(f"{exclusions}\n")
    return path


def write_validation_report_md(out_dir, synthetic_results, real_data_notes, pilot_note):
    path = os.path.join(out_dir, "validation_report.md")
    n_pass = sum(1 for v in synthetic_results.values() if v)
    with open(path, "w") as f:
        f.write("# C2.1 Validation Report\n\n")
        f.write(pilot_note)
        f.write(f"## Synthetic checks: {n_pass}/{len(synthetic_results)} passed\n\n")
        for name, ok in synthetic_results.items():
            f.write(f"- [{'x' if ok else ' '}] {name}\n")
        f.write("\n## Real-data invariants checked at runtime\n\n")
        for note in real_data_notes:
            f.write(f"- {note}\n")
        f.write(
            "\n## Known limitations\n\n"
            "- Query identifiers are positional row indices into the controller_train/val cache arrays "
            "(the cache format has no row-id field) -- structural, not independently verifiable semantic "
            "provenance, same documented limitation as C1.\n"
            "- Historical alignment of these cluster IDs with older CRISP-R5 cluster IDs is unproven and "
            "not assumed anywhere in this script.\n"
            "- The counterfactual TV decomposition (tv_weight_change, tv_selection_change) is explicitly "
            "path-dependent and NOT asserted to sum exactly to tv_total.\n"
            "- Sampling is bounded to a stratified subset of clusters; aggregate statistics are NOT "
            "extrapolated to all 3,811 clusters without accounting for inclusion probabilities -- see "
            "`sampling_report.md` for per-cluster inclusion probabilities where calculable.\n"
            "- Review point 3: nearby query PAIRS are a deterministic 1-nearest-neighbor pairing, not an "
            "independent sample -- a single query can be the nearest neighbor of several other sampled "
            "queries simultaneously (directed-pair reuse). `cluster_behavior_statistics.csv`'s "
            "`max_pair_reuse_count` reports the largest such reuse count per cluster; pair-level statistics "
            "should be read as a set of possibly-correlated observations, not i.i.d. draws.\n"
            "- Review point 3: the margin-vs-displacement stability guarantee "
            "(`gamma_m(q) > 2*||q-q'|| => selected sets are equal`) is actively ASSERTED, not just reported, "
            "on every pair in `analyze_cluster` -- a violation would abort the run with an AssertionError "
            "naming the offending cluster.\n"
        )
    return path


def _format_failure_value(v):
    # Pandas' per-row .to_dict() upcasts whole-number fields (cluster,
    # query_row_i, ...) to float64 whenever the same column holds NaN
    # elsewhere (e.g. margin is None for trivial-selection clusters) --
    # cosmetic only, but printing "28.0" for a cluster id is misleading.
    if isinstance(v, float) and np.isfinite(v) and float(v).is_integer():
        return str(int(v))
    return str(v)


def write_failure_cases_md(out_dir, cases, pilot_note):
    path = os.path.join(out_dir, "failure_cases.md")
    with open(path, "w") as f:
        f.write("# C2.1 Failure Cases\n\n")
        f.write(pilot_note)
        for title, record in cases.items():
            f.write(f"## {title}\n\n")
            if record is None:
                f.write("No qualifying example found in the sampled data.\n\n")
                continue
            for k, v in record.items():
                f.write(f"- **{k}**: {_format_failure_value(v)}\n")
            f.write("\n")
    return path


def write_final_report_md(out_dir, cluster_behavior_df, relationships_rows, n_pair_records, n_query_records,
                            distance_stratified_rows, q11_stats, pilot_note):
    path = os.path.join(out_dir, "final_report.md")
    with open(path, "w") as f:
        f.write("# C2.1 Final Report -- Exact Local Retrieval Behavior Audit\n\n")
        f.write(pilot_note)
        f.write(f"Sampled clusters: {len(cluster_behavior_df)}. Query-cluster pairs analyzed: {n_query_records}. "
                f"Nearby-query pairs analyzed: {n_pair_records}.\n\n")

        def fmt(col):
            if col not in cluster_behavior_df.columns:
                return "N/A"
            s = cluster_behavior_df[col].dropna()
            return f"mean={s.mean():.4f}, median={s.median():.4f}, n={len(s)}" if len(s) else "N/A"

        f.write(
            "## Q1. Candidate-set stability across nearby queries\n\n"
            f"Mean Jaccard similarity of selected sets between nearby query pairs: {fmt('mean_jaccard')} "
            f"(centroid-mass-weighted: {fmt('massweighted_mean_jaccard')}). "
            f"Selection margin (only where n_C > m): {fmt('mean_margin')}. "
            "Review point 3 -- stability by query-query distance (all sampled pairs, pooled across clusters, "
            "quartiles of query_query_distance):\n\n"
        )
        for row in distance_stratified_rows:
            f.write(f"- {row['bin']} (n={row['n']}, distance in [{row['d_min']:.3f}, {row['d_max']:.3f}]): "
                    f"mean_jaccard={row['mean_jaccard']}, mean_tv_total={row['mean_tv_total']}, "
                    f"mean_js_total={row['mean_js_total']}, "
                    f"frac_stability_guaranteed={row['frac_stability_guaranteed']}\n")
        f.write(
            "\nNote: these pairs are a deterministic 1-nearest-neighbor pairing, not an independent sample "
            "(see `validation_report.md`'s pair-reuse caveat); the stratification is descriptive.\n\n"
            "## Q2. Candidate changes vs. label changes\n\n"
            f"Fraction of nearby pairs whose candidate-set change altered the represented token labels: "
            f"{fmt('frac_label_changing_pairs')}.\n\n"
            "## Q3. Competition weight concentration\n\n"
            f"Mean effective number of contributing candidates (N_eff): {fmt('mean_N_eff')}. "
            f"Mean top-3 weight mass: {fmt('mean_top3_mass')}. "
            f"Mean centroid mixture weight for the routed query-cluster pair itself: {fmt('mean_centroid_weight')}.\n\n"
            "## Q4/Q5. Competition-change vs. selection-change contributions to predictive variation\n\n"
            f"Mean TV from competition-weight change alone (candidates fixed): {fmt('mean_tv_weight_change')}. "
            f"Mean TV from candidate-selection change alone (query fixed): {fmt('mean_tv_selection_change')}. "
            f"Mean total predictive TV between nearby queries: {fmt('mean_tv_total')} "
            f"(centroid-mass-weighted: {fmt('massweighted_mean_tv_total')}). "
            "These two components are NOT asserted to sum to the total (the decomposition is path-dependent); "
            f"see Q11 for forward/reverse-path agreement.\n\n"
            "## Q6. Exact local token-distribution variation across the routed-query population\n\n"
            f"Pairwise local variation: mean TV between F_C(q) at nearby queries = {fmt('mean_tv_total')}; "
            f"mean JS divergence = {fmt('mean_js_total')}. "
            f"Query-dependence statistic: I_C (val, in-sample) = {fmt('I_C_val')} "
            "(mutual information in nats between the query and the retrieved token, under the sampled "
            "routed-query population; split-half sensitivity in Q11).\n\n"
            "## Q7/Q8. Static histogram vs. query-distribution-adapted constant baseline\n\n"
            f"Mean TV(F_C(q_val), static histogram p_C): {fmt('mean_tv_vs_static')} "
            f"(centroid-mass-weighted: {fmt('massweighted_mean_tv_vs_static')}; "
            f"missing-support fraction: {fmt('frac_missing_support_vs_static')}, expected ~0 since every "
            "candidate label is necessarily a member of the cluster's own static histogram). "
            f"Mean TV(F_C(q_val), train-derived constant bar_F_train): {fmt('mean_tv_vs_trainconst')} "
            f"(centroid-mass-weighted: {fmt('massweighted_mean_tv_vs_trainconst')}; "
            f"missing-support fraction: {fmt('frac_missing_support_vs_trainconst')} -- nonzero here is "
            "expected whenever a held-out val query's candidates include a label the controller_train "
            "sample for that cluster never selected).\n\n"
            "## Q9. C1 structural correlates of C2 behavior\n\n"
        )
        for row in relationships_rows:
            f.write(f"- {row['var_a']} vs {row['var_b']} ({row['scope']}): "
                    f"pearson_r={row['pearson_r']}, spearman_r={row['spearman_r']}, n={row['n']}\n")
        f.write(
            "\n## Q10. Clusters with high geometric/candidate variability but low predictive variability\n\n"
            "See `failure_cases.md` for predeclared-criterion examples drawn from the sampled data; this is "
            "NOT asserted to characterize all 3,811 clusters.\n\n"
            "## Q11. Observations robust across query splits, analysis choices, and numerical settings\n\n"
            f"Forward/reverse counterfactual-path agreement (review points 2/6): mean absolute difference "
            f"between tv_weight_change and its reverse-path counterpart = "
            f"{q11_stats.get('mean_abs_diff_weight_change_fwd_rev')}; between tv_selection_change and its "
            f"reverse = {q11_stats.get('mean_abs_diff_selection_change_fwd_rev')} "
            f"(n={q11_stats.get('n_pairs_for_fwd_rev')} pairs). Small values indicate the decomposition is "
            "not an artifact of which query was treated as the anchor.\n\n"
            f"Split-half I_C sensitivity: mean absolute difference between I_C_val_half1 and I_C_val_half2 "
            f"across sampled clusters = {q11_stats.get('mean_abs_diff_ic_halves')} "
            f"(n={q11_stats.get('n_clusters_for_ic_halves')} clusters with >=2 queries in each half); "
            f"Pearson correlation between the two halves = {q11_stats.get('corr_ic_halves')}. A high "
            "correlation / low mean difference indicates I_C is not dominated by small-sample noise at "
            "this sample size; a low correlation means I_C estimates at this sample size should be read "
            "as noisy, not as precise per-cluster values.\n\n"
            f"Boundary-tie prevalence (review point 7): mean fraction of sampled queries landing on an exact "
            f"local top-m boundary tie across clusters = {fmt('frac_boundary_tie')}; of those, mean fraction "
            f"where the tied pair carries different labels = {fmt('frac_boundary_tie_diff_label')} -- "
            "nonzero here means the stable-sort tie-breaking convention is doing real (not merely "
            "cosmetic) work in determining F_C(q) for some queries.\n\n"
            "## Q12. What C2.1 alone does NOT justify\n\n"
            "C2.1 measures exact local retrieval behavior under one diagnostic (m_local, tau_local) choice, "
            "over a STRATIFIED SAMPLE of clusters, not the full population -- and the deliberately-included "
            "unusual clusters (see `sampling_report.md`) have no inclusion probability and must not be "
            "pooled into population-level estimates (Q9's `stratified_only` scope exists for this reason). "
            "It does NOT measure distortion-versus-storage (that is C3), does NOT establish which structural "
            "property CAUSES compression difficulty (that is C4), and query-dependent variation measured "
            "here is NOT, by itself, proof that a cluster's exact retrieval behavior is hard to compress.\n"
        )
    return path


# ============================================================
# Per-cluster analysis (Parts 3-6)
# ============================================================
BOUNDARY_TIE_EPS = 1e-9


def analyze_cluster(c, member_row_ids, ds_keys, ds_values, val_keys, ct_keys,
                     idx_val_topk, d_val_topk, w_val_topk, idx_ct_topk, d_ct_topk, w_ct_topk,
                     static_hist, n_query_val_target, n_query_ct_target, m_local, tau_local, seed):
    member_keys = ds_keys[member_row_ids]
    member_values = ds_values[member_row_ids]
    n_c = len(member_row_ids)
    is_trivial = n_c <= m_local

    rows_val_all, ranks_val_all, dist_val_all, w_val_all = routed_query_rows(idx_val_topk, d_val_topk, w_val_topk, c)
    rows_ct_all, ranks_ct_all, dist_ct_all, w_ct_all = routed_query_rows(idx_ct_topk, d_ct_topk, w_ct_topk, c)
    rows_val_top1_all = np.nonzero(idx_val_topk[:, 0] == c)[0]  # secondary diagnostic population, never merged below

    sub_rng = np.random.default_rng([seed, c])
    n_val_sampled = min(n_query_val_target, len(rows_val_all))
    if len(rows_val_all) > 0:
        sel = sub_rng.choice(len(rows_val_all), size=n_val_sampled, replace=False)
        rows_val, ranks_val = rows_val_all[sel], ranks_val_all[sel]
        centroid_dist_val, centroid_w_val = dist_val_all[sel], w_val_all[sel]
    else:
        rows_val, ranks_val = np.array([], dtype=int), np.array([], dtype=int)
        centroid_dist_val, centroid_w_val = np.array([]), np.array([])

    n_ct_sampled = min(n_query_ct_target, len(rows_ct_all))
    if len(rows_ct_all) > 0:
        sel_ct = sub_rng.choice(len(rows_ct_all), size=n_ct_sampled, replace=False)
        rows_ct = rows_ct_all[sel_ct]
    else:
        rows_ct = np.array([], dtype=int)

    val_info = []
    for i, r in enumerate(rows_val):
        q = val_keys[r]
        sel_pos, sel_d, margin, all_d = local_candidate_selection(member_keys, q, m_local)
        w = local_weights(sel_d, tau_local)
        F = exact_local_distribution(member_values[sel_pos], w)
        conc = weight_concentration_stats(w)
        med_d = float(np.median(all_d))
        norm_margin = (margin / med_d) if (margin is not None and med_d > 0) else None

        # Review point 7 -- inspect the local top-m boundary for ties, and
        # whether a tied pair straddling the boundary carries different
        # labels (the case where tie-breaking convention actually matters).
        # Tie-breaking convention: np.argsort(..., kind="stable") breaks ties
        # by ascending position in member_keys, which itself preserves
        # ascending original datastore row order (build_cluster_to_members
        # uses a stable sort) -- deterministic and reproducible, stated here
        # and in construction_consistency.md.
        boundary_tie, boundary_tie_diff_label = False, None
        if margin is not None:
            boundary_tie = abs(margin) <= BOUNDARY_TIE_EPS * max(1.0, abs(float(all_d[sel_pos[-1]])))
            if boundary_tie:
                order_full = np.argsort(all_d, kind="stable")
                label_m = int(member_values[order_full[m_local - 1]])
                label_m1 = int(member_values[order_full[m_local]])
                boundary_tie_diff_label = label_m != label_m1

        val_info.append({
            "row": int(r), "rank": int(ranks_val[i]), "sel_pos": sel_pos, "w": w, "F": F, "conc": conc,
            "margin": margin, "norm_margin": norm_margin,
            "centroid_distance": float(centroid_dist_val[i]), "centroid_weight": float(centroid_w_val[i]),
            "boundary_tie": boundary_tie, "boundary_tie_diff_label": boundary_tie_diff_label,
        })

    ct_F_list = []
    for r in rows_ct:
        q = ct_keys[r]
        sel_pos, sel_d, margin, all_d = local_candidate_selection(member_keys, q, m_local)
        w = local_weights(sel_d, tau_local)
        ct_F_list.append(exact_local_distribution(member_values[sel_pos], w))
    bar_F_train = mean_distribution(ct_F_list)  # 6.2's training-derived constant -- NEVER built from val rows

    query_records, static_records = [], []
    for v in val_info:
        F = v["F"]
        ent = entropy_of_dict(F)
        sorted_F = sorted(F.items(), key=lambda kv: -kv[1])
        query_records.append({
            "cluster": c, "query_row": v["row"], "split": "val", "rank_in_topk": v["rank"],
            "centroid_distance": v["centroid_distance"], "centroid_weight": v["centroid_weight"],
            "n_c": n_c, "m_c": len(v["sel_pos"]), "is_trivial_selection": is_trivial,
            "margin": v["margin"], "normalized_margin": v["norm_margin"],
            "boundary_tie": v["boundary_tie"], "boundary_tie_diff_label": v["boundary_tie_diff_label"],
            "H_pi": v["conc"]["H_pi"], "N_eff": v["conc"]["N_eff"], "max_weight": v["conc"]["max_weight"],
            "top3_mass": v["conc"]["top3_mass"], "H_pi_over_logm": v["conc"]["H_pi_over_logm"],
            "entropy_F": ent, "max_prob_F": sorted_F[0][1] if sorted_F else None,
            "top_token_F": sorted_F[0][0] if sorted_F else None, "n_nonzero_F": len(F),
        })
        tv_static = total_variation(F, static_hist)
        js_static = js_divergence_dict(F, static_hist)
        kl_static = kl_divergence_dict(F, static_hist)
        tv_train = total_variation(F, bar_F_train) if bar_F_train else None
        js_train = js_divergence_dict(F, bar_F_train) if bar_F_train else None
        kl_train = kl_divergence_dict(F, bar_F_train) if bar_F_train else None
        static_records.append({
            "cluster": c, "query_row": v["row"], "centroid_weight": v["centroid_weight"],
            "tv_vs_static": tv_static, "js_vs_static": js_static,
            "kl_vs_static": kl_static, "kl_vs_static_missing_support": kl_static is None,
            "tv_vs_trainconst": tv_train, "js_vs_trainconst": js_train, "kl_vs_trainconst": kl_train,
            "kl_vs_trainconst_missing_support": (tv_train is not None and kl_train is None),
            "n_ct_available_for_trainconst": len(ct_F_list),
        })

    pair_records = []
    reuse_counter = Counter()
    if len(val_info) >= 2:
        sub_keys = val_keys[np.array([v["row"] for v in val_info])]
        pdist = cdist(sub_keys, sub_keys)
        np.fill_diagonal(pdist, np.inf)
        nn = np.argmin(pdist, axis=1)
        reuse_counter = Counter(int(j) for j in nn)  # review point 3: directed-pair reuse (in-degree of the 1-NN digraph)
        for i in range(len(val_info)):
            j = int(nn[i])
            vi, vj = val_info[i], val_info[j]
            qq_dist = float(pdist[i, j]) if np.isfinite(pdist[i, j]) else None
            sel_i, sel_j = vi["sel_pos"], vj["sel_pos"]
            set_i, set_j = set(sel_i.tolist()), set(sel_j.tolist())
            jaccard = len(set_i & set_j) / len(set_i | set_j) if (set_i or set_j) else 1.0
            n_replaced = len(set_i ^ set_j) // 2
            identical = int(set_i == set_j)

            labels_i = normalize_counter(Counter(member_values[sel_i].tolist()))
            labels_j = normalize_counter(Counter(member_values[sel_j].tolist()))
            tv_label_composition = total_variation(labels_i, labels_j)
            label_changing = int(tv_label_composition > 1e-9)

            G_qp_Si, _, _ = local_distribution_fixed_candidates(member_keys, member_values, sel_i, val_keys[vj["row"]], tau_local)
            tv_weight_change = total_variation(vi["F"], G_qp_Si)
            tv_selection_change = total_variation(G_qp_Si, vj["F"])
            tv_total = total_variation(vi["F"], vj["F"])

            G_q_Sj, _, _ = local_distribution_fixed_candidates(member_keys, member_values, sel_j, val_keys[vi["row"]], tau_local)
            tv_weight_change_rev = total_variation(vj["F"], G_q_Sj)
            tv_selection_change_rev = total_variation(G_q_Sj, vi["F"])

            # Review point 3: the margin-vs-query-displacement stability
            # guarantee. If gamma_m(q) > 2*||q-q'||, the triangle inequality
            # guarantees NO member outside q's top-m can enter q''s top-m and
            # vice versa, so the selected sets MUST be identical. This is
            # checked (not just reported) below -- a violation would mean a
            # real bug in candidate selection or distance computation.
            margin_minus_2qq = (vi["margin"] - 2.0 * qq_dist) if (vi["margin"] is not None and qq_dist is not None) else None
            stability_guaranteed = (margin_minus_2qq is not None) and (margin_minus_2qq > 0)
            if stability_guaranteed:
                assert identical == 1, (
                    f"stability guarantee violated: cluster {c}, margin_i={vi['margin']}, "
                    f"qq_dist={qq_dist}, but selected sets differ"
                )

            pair_records.append({
                "cluster": c, "query_row_i": vi["row"], "query_row_j": vj["row"],
                "query_query_distance": qq_dist,
                "jaccard": jaccard, "n_replaced": n_replaced, "identical_candidate_set": identical,
                "tv_label_composition": tv_label_composition, "label_changing": label_changing,
                "tv_weight_change": tv_weight_change, "tv_selection_change": tv_selection_change,
                "tv_total": tv_total,
                # Correction (external review): spec Part 5 explicitly requires JS divergence
                # between nearby-query output distributions, not just TV -- Revision 1 omitted it.
                "js_total": js_divergence_dict(vi["F"], vj["F"]),
                "top1_agreement": top1_agreement(vi["F"], vj["F"]),
                "tv_weight_change_reverse": tv_weight_change_rev, "tv_selection_change_reverse": tv_selection_change_rev,
                "margin_i": vi["margin"], "margin_j": vj["margin"],
                "margin_i_minus_2x_qqdist": margin_minus_2qq, "stability_guaranteed": stability_guaranteed,
                "centroid_weight_i": vi["centroid_weight"],
            })

    def mi_estimate(F_list):
        if len(F_list) < 2:
            return None
        bar = mean_distribution(F_list)
        return entropy_of_dict(bar) - float(np.mean([entropy_of_dict(F) for F in F_list]))

    F_list_val = [v["F"] for v in val_info]
    I_C_val = mi_estimate(F_list_val)
    half = len(F_list_val) // 2
    I_C_half1 = mi_estimate(F_list_val[:half]) if half >= 2 else None
    I_C_half2 = mi_estimate(F_list_val[half:]) if (len(F_list_val) - half) >= 2 else None

    def agg(key, records):
        vals = [r[key] for r in records if r.get(key) is not None]
        return float(np.mean(vals)) if vals else None

    def agg_weighted(key, weight_key, records):
        # Review point 1: centroid-mass-weighted summaries, clearly SEPARATE
        # (different column names) from the uniform-weight "mean_*" ones --
        # never silently blended under the same name.
        pairs = [(r[key], r[weight_key]) for r in records if r.get(key) is not None and r.get(weight_key) is not None]
        if not pairs:
            return None
        vals, weights = zip(*pairs)
        weights = np.array(weights)
        if weights.sum() <= 0:
            return None
        return float(np.sum(np.array(vals) * weights) / weights.sum())

    n_boundary_ties = sum(1 for v in val_info if v["boundary_tie"])
    n_boundary_ties_diff_label = sum(1 for v in val_info if v["boundary_tie_diff_label"])
    max_pair_reuse_count = max(reuse_counter.values()) if reuse_counter else 0

    behavior_row = {
        "cluster": c, "n_j": n_c, "m_c_effective": min(m_local, n_c), "is_trivial_selection": is_trivial,
        "n_val_available": int(len(rows_val_all)), "n_val_sampled": int(len(rows_val)),
        "n_ct_available": int(len(rows_ct_all)), "n_ct_sampled": int(len(rows_ct)),
        "n_val_top1_available": int(len(rows_val_top1_all)),
        "mean_margin": agg("margin", query_records), "mean_normalized_margin": agg("normalized_margin", query_records),
        "frac_boundary_tie": (n_boundary_ties / len(val_info)) if val_info else None,
        # Correction (external review): this must be CONDITIONAL on the
        # query actually landing on a boundary tie (denominator =
        # n_boundary_ties), matching the write-up's "of those" wording --
        # dividing by len(val_info) instead would silently answer a
        # different question (fraction of ALL queries, not of ties).
        # None (not 0) when there were no ties to begin with -- "0 out of 0
        # ties disagree" is undefined, not "no disagreement".
        "frac_boundary_tie_diff_label": (n_boundary_ties_diff_label / n_boundary_ties) if n_boundary_ties else None,
        "mean_H_pi": agg("H_pi", query_records), "mean_N_eff": agg("N_eff", query_records),
        "mean_max_weight": agg("max_weight", query_records), "mean_top3_mass": agg("top3_mass", query_records),
        "mean_entropy_F": agg("entropy_F", query_records),
        "mean_centroid_weight": agg("centroid_weight", query_records),
        "mean_jaccard": agg("jaccard", pair_records), "mean_n_replaced": agg("n_replaced", pair_records),
        "frac_identical_candidate_sets": agg("identical_candidate_set", pair_records),
        "frac_label_changing_pairs": agg("label_changing", pair_records),
        "mean_tv_weight_change": agg("tv_weight_change", pair_records),
        "mean_tv_selection_change": agg("tv_selection_change", pair_records),
        "mean_tv_total": agg("tv_total", pair_records), "mean_js_total": agg("js_total", pair_records),
        "mean_top1_agreement": agg("top1_agreement", pair_records),
        "max_pair_reuse_count": max_pair_reuse_count,
        "frac_pairs_stability_guaranteed": agg("stability_guaranteed", pair_records),
        "massweighted_mean_jaccard": agg_weighted("jaccard", "centroid_weight_i", pair_records),
        "massweighted_mean_tv_total": agg_weighted("tv_total", "centroid_weight_i", pair_records),
        "mean_tv_vs_static": agg("tv_vs_static", static_records), "mean_tv_vs_trainconst": agg("tv_vs_trainconst", static_records),
        "mean_js_vs_static": agg("js_vs_static", static_records), "mean_js_vs_trainconst": agg("js_vs_trainconst", static_records),
        "massweighted_mean_tv_vs_static": agg_weighted("tv_vs_static", "centroid_weight", static_records),
        "massweighted_mean_tv_vs_trainconst": agg_weighted("tv_vs_trainconst", "centroid_weight", static_records),
        "frac_missing_support_vs_static": agg("kl_vs_static_missing_support", static_records),
        "frac_missing_support_vs_trainconst": agg("kl_vs_trainconst_missing_support", static_records),
        "I_C_val": I_C_val, "I_C_val_half1": I_C_half1, "I_C_val_half2": I_C_half2,
        "tv_static_vs_trainconst_structural": total_variation(static_hist, bar_F_train) if bar_F_train else None,
    }
    return query_records, pair_records, static_records, behavior_row


# ============================================================
# Part 8 -- qualitative failure-case detail reconstruction (review point 4)
# ============================================================
def query_case_detail(row, ds_keys, ds_values, member_row_ids, query_key, m_local, tau_local):
    """Recomputes full, inspectable detail for ONE query against ONE
    cluster's members: LOCAL candidate positions (0..n_c-1, within
    member_row_ids) are reported SEPARATELY from GLOBAL datastore row ids
    (member_row_ids[sel_pos]) -- never conflated. No text decoding is
    attempted (this diagnostic has no tokenizer/text-provenance mapping
    loaded); only token ids are reported."""
    member_keys = ds_keys[member_row_ids]
    member_values = ds_values[member_row_ids]
    sel_pos, sel_d, margin, _ = local_candidate_selection(member_keys, query_key, m_local)
    w = local_weights(sel_d, tau_local)
    global_rows = member_row_ids[sel_pos]
    return {
        "query_row": int(row),
        "local_candidate_positions": sel_pos.tolist(),
        "global_datastore_row_ids": global_rows.tolist(),
        "candidate_labels": ds_values[global_rows].tolist(),
        "candidate_distances": sel_d.tolist(),
        "candidate_weights": w.tolist(),
        "margin": margin,
    }


def build_pair_case_detail(cluster, row_i, row_j, ds_keys, ds_values, member_row_ids,
                             val_keys, m_local, tau_local, c1_stats_row):
    detail_i = query_case_detail(row_i, ds_keys, ds_values, member_row_ids, val_keys[row_i], m_local, tau_local)
    detail_j = query_case_detail(row_j, ds_keys, ds_values, member_row_ids, val_keys[row_j], m_local, tau_local)
    out = {"cluster": int(cluster), "text_decoding": "not attempted -- no tokenizer/text provenance available"}
    for k, v in detail_i.items():
        out[f"i_{k}"] = v
    for k, v in detail_j.items():
        out[f"j_{k}"] = v
    if c1_stats_row is not None:
        for col in ("n_j", "rms_dispersion", "V_j", "distinct_token_fraction", "entropy",
                    "top_token_fraction", "singleton_token_fraction"):
            if col in c1_stats_row.index:
                out[f"c1_{col}"] = c1_stats_row[col]
    return out


def build_single_query_case_detail(cluster, row, ds_keys, ds_values, member_row_ids,
                                     query_array, m_local, tau_local, c1_stats_row):
    out = {"cluster": int(cluster), "text_decoding": "not attempted -- no tokenizer/text provenance available"}
    out.update(query_case_detail(row, ds_keys, ds_values, member_row_ids, query_array[row], m_local, tau_local))
    if c1_stats_row is not None:
        for col in ("n_j", "rms_dispersion", "V_j", "distinct_token_fraction", "entropy",
                    "top_token_fraction", "singleton_token_fraction"):
            if col in c1_stats_row.index:
                out[f"c1_{col}"] = c1_stats_row[col]
    return out


# ============================================================
# main
# ============================================================
def main():
    t_start = time.perf_counter()
    real_data_notes = []

    print("Running synthetic-only unit tests first (never touches real cache)...")
    synthetic_results = run_synthetic_checks()
    n_pass = sum(1 for v in synthetic_results.values() if v)
    print(f"Synthetic checks: {n_pass}/{len(synthetic_results)} passed")
    assert all(synthetic_results.values()), f"synthetic check(s) failed: {synthetic_results}"

    # Checked BEFORE any directory is created or any file is written -- a
    # blocked run must leave no trace (not even an empty results folder) in
    # the real OUT_DIR path. Per review point 8, the mode is an EXPLICIT
    # declaration, cross-checked against (never inferred solely from) CACHE_DIR.
    is_real_cache = os.path.abspath(CACHE_DIR) == os.path.abspath(REAL_CACHE_DIR_DEFAULT)
    if DIME_MODE not in ("synthetic_pilot", "real_approved"):
        raise RuntimeError(
            "DIMEV2C21_MODE must be explicitly set to 'synthetic_pilot' or 'real_approved' -- there is "
            "no default and no inference from CACHE_DIR alone. Got: "
            f"{DIME_MODE!r}. Per Part 0 / Part 13's mandatory stopping rule, this diagnostic must not "
            "run against real data until external review and explicit approval are given."
        )
    if DIME_MODE == "real_approved":
        if not is_real_cache:
            raise RuntimeError(
                "DIMEV2C21_MODE=real_approved requires CACHE_DIR to resolve to the real production "
                f"cache path ({REAL_CACHE_DIR_DEFAULT}); got {CACHE_DIR}. real_approved mode may not be "
                "pointed at any other directory."
            )
        if os.environ.get("DIMEV2C21_APPROVED") != "1":
            raise RuntimeError(
                "DIMEV2C21_MODE=real_approved also requires DIMEV2C21_APPROVED=1, set ONLY after "
                "external review and explicit approval (intended to be set in the SLURM submit script, "
                "not by hand here). This requirement holds regardless of CACHE_DIR."
            )
    else:  # synthetic_pilot
        if is_real_cache:
            raise RuntimeError(
                "DIMEV2C21_MODE=synthetic_pilot must not point CACHE_DIR at the real production cache "
                f"path ({REAL_CACHE_DIR_DEFAULT}) -- a synthetic-pilot declaration that resolves to the "
                "real path is refused rather than silently promoted to a real run."
            )
    os.makedirs(FIG_DIR, exist_ok=True)
    pilot_note = "" if DIME_MODE == "real_approved" else (
        "**PILOT/SYNTHETIC RUN** -- CACHE_DIR does not resolve to the real production cache "
        f"({CACHE_DIR}); every number in this file comes from synthetic data and exists only to verify "
        "the pipeline runs end-to-end, not as a real finding.\n\n"
    )

    print(f"C2.1 job started: N_CLUSTERS={N_CLUSTERS}, CACHE_DIR={CACHE_DIR}, C1_DIR={C1_DIR}")
    print("Loading and verifying the persisted C1 construction artifact (never retraining)...")
    dime_centroids, assignment, c1_fingerprint, c1_centroids_sha, c1_assignment_sha = load_and_verify_c1_artifact(
        C1_ARTIFACT_PATH, C1_FINGERPRINT_PATH, EXPECTED_CENTROIDS_SHA256, EXPECTED_ASSIGNMENT_SHA256,
        n_clusters_expected=N_CLUSTERS)
    real_data_notes.append(
        f"C1 artifact centroids/assignment hash verified: {c1_centroids_sha[:16]}.../{c1_assignment_sha[:16]}...")
    dime_centroids = np.ascontiguousarray(dime_centroids.astype(np.float64))

    print("Loading cache (datastore/controller_train/val)...")
    ds = load_cache(os.path.join(CACHE_DIR, "wikitext103_gpt2_datastore.npz"))
    ct = load_cache(os.path.join(CACHE_DIR, "wikitext103_gpt2_controller_train.npz"))
    val = load_cache(os.path.join(CACHE_DIR, "wikitext103_gpt2_val.npz"))
    ds_keys, ds_values = ds["keys"].astype(np.float64), ds["values"]
    ct_keys = ct["keys"].astype(np.float64)
    val_keys = val["keys"].astype(np.float64)

    if assignment.shape[0] != len(ds_values):
        raise RuntimeError(
            f"C1 assignment length ({assignment.shape[0]}) does not match the current datastore size "
            f"({len(ds_values)}). STOP: do not proceed, do not substitute a new clustering run.")
    real_data_notes.append(f"Datastore row count matches C1 assignment length: {len(ds_values)}")

    # Review point 7 -- strengthened invariants: dims, array lengths, finite vectors.
    for name, keys, values in (("datastore", ds_keys, ds_values), ("controller_train", ct_keys, ct["values"]),
                                ("val", val_keys, val["values"])):
        if keys.shape[1] != DIM:
            raise RuntimeError(f"{name} keys have dimension {keys.shape[1]}, expected {DIM}. STOP.")
        if keys.shape[0] != len(values):
            raise RuntimeError(f"{name} keys/values length mismatch: {keys.shape[0]} vs {len(values)}. STOP.")
        if not np.all(np.isfinite(keys)):
            raise RuntimeError(f"{name} keys contain a non-finite value (NaN/inf). STOP.")
    real_data_notes.append(f"Datastore/controller_train/val keys all dimension {DIM}, length-matched to their "
                            "values, and finite.")

    print("Building cluster_to_members and per-cluster static histograms from the PERSISTED assignment "
          "(a deterministic grouping of that assignment, never a new clustering)...")
    cluster_to_members = build_cluster_to_members(assignment, N_CLUSTERS)
    compressed_dists = []
    for c in range(N_CLUSTERS):
        mv = ds_values[cluster_to_members[c]]
        compressed_dists.append(Counter(mv.tolist()) if len(mv) > 0 else Counter())
    populations = np.array([sum(cd.values()) for cd in compressed_dists])

    if not os.path.exists(C1_CLUSTER_STATS_PATH):
        raise RuntimeError(f"C1 cluster_statistics.csv not found at {C1_CLUSTER_STATS_PATH}. STOP.")
    c1_stats = pd.read_csv(C1_CLUSTER_STATS_PATH)
    if len(c1_stats) != N_CLUSTERS:
        raise RuntimeError(f"C1 cluster_statistics.csv has {len(c1_stats)} rows, expected {N_CLUSTERS}. STOP.")
    if np.any(np.abs(c1_stats["n_j"].values - populations) > 0):
        raise RuntimeError(
            "Population recomputed from the loaded C1 assignment does not match C1's own "
            "cluster_statistics.csv n_j column -- the assignment or the CSV may be stale/mismatched. STOP.")
    real_data_notes.append("Recomputed per-cluster population from the loaded assignment matches C1's "
                            "cluster_statistics.csv n_j column exactly.")

    print(f"Computing centroid routing (top-{K_ROUTE}, matched config tau={TAU_ROUTE})...")
    d_ct_topk, idx_ct_topk, d_val_topk, idx_val_topk = compute_routing(
        dime_centroids, compressed_dists, ct_keys, val_keys, K_ROUTE)
    assert_no_duplicate_indices_per_query(idx_ct_topk)
    assert_no_duplicate_indices_per_query(idx_val_topk)
    real_data_notes.append(f"Routing index matrices {idx_ct_topk.shape}/{idx_val_topk.shape} contain no "
                            "duplicate centroid ids per query.")

    # Review point 1 -- the centroid MIXTURE WEIGHTS themselves (not just
    # distances/indices), the exact DIME matched-configuration softmax.
    print("Computing centroid mixture weights (matched-configuration softmax)...")
    w_ct_topk = centroid_mixture_weights(d_ct_topk, TAU_ROUTE)
    w_val_topk = centroid_mixture_weights(d_val_topk, TAU_ROUTE)
    real_data_notes.append("Centroid mixture weights verified to sum to 1 for every controller_train/val query.")

    # Review point 7 -- compare recomputed routing appearance counts against
    # C1's own recorded traffic, allowing only an EXPLAINED numerical
    # difference (this script re-runs the identical routing mechanism over
    # the identical centroids/data C1 used, so any discrepancy beyond float
    # round-off is a real inconsistency, not noise).
    traffic_tol = 1e-6
    if K_ROUTE == 20 and f"traffic_top{K_ROUTE}_val" in c1_stats.columns:
        recomputed_traffic_val = np.bincount(idx_val_topk.ravel(), minlength=N_CLUSTERS) / idx_val_topk.shape[0]
        recomputed_traffic_val_top1 = np.bincount(idx_val_topk[:, 0], minlength=N_CLUSTERS) / idx_val_topk.shape[0]
        max_diff_topk = float(np.max(np.abs(recomputed_traffic_val - c1_stats[f"traffic_top{K_ROUTE}_val"].values)))
        max_diff_top1 = float(np.max(np.abs(recomputed_traffic_val_top1 - c1_stats["traffic_top1_val"].values)))
        if max_diff_topk > traffic_tol or max_diff_top1 > traffic_tol:
            raise RuntimeError(
                f"Recomputed routing traffic differs from C1's recorded traffic by more than the explained "
                f"tolerance ({traffic_tol}): top-{K_ROUTE} max_diff={max_diff_topk}, top-1 max_diff={max_diff_top1}. "
                "STOP -- this re-runs the identical mechanism over the identical centroids/assignment, so any "
                "larger discrepancy indicates a real inconsistency (e.g. the val cache changed since C1 ran).")
        traffic_comparison_note = (
            f"max |diff| vs C1's recorded `traffic_top{K_ROUTE}_val`: {max_diff_topk:.2e}; "
            f"vs `traffic_top1_val`: {max_diff_top1:.2e} (tolerance {traffic_tol:.0e}) -- PASSED.")
        real_data_notes.append(f"Recomputed routing traffic matches C1's recorded traffic within {traffic_tol}.")
    else:
        traffic_comparison_note = (
            f"Skipped: K_ROUTE={K_ROUTE} does not match the column C1 saved (`traffic_top20_val`), as is "
            "expected whenever K_ROUTE is overridden for a pilot run -- not applicable, not a failure.")

    print("Sampling clusters (stratified over C1 structural covariates + deliberate unusual cases)...")
    stratified_ids, coverage, cluster_cell_map, cell_population, inclusion_prob = stratified_cluster_sample(
        c1_stats, STRATIFICATION_AXES, N_BINS_PER_AXIS, N_SAMPLE_CLUSTERS, SEED)
    unusual_ids, unusual_reasons = unusual_cluster_candidates(c1_stats, N_UNUSUAL_CLUSTERS)
    stratified_set, unusual_set = set(stratified_ids), set(unusual_ids)
    sampled_clusters = sorted(stratified_set | unusual_set)
    print(f"  {len(stratified_ids)} stratified + {len(unusual_ids)} unusual -> {len(sampled_clusters)} total")

    all_query_records, all_pair_records, all_static_records, behavior_rows = [], [], [], []
    for c in sampled_clusters:
        static_hist = normalize_counter(compressed_dists[c])
        qr, pr, sr, br = analyze_cluster(
            c, cluster_to_members[c], ds_keys, ds_values, val_keys, ct_keys,
            idx_val_topk, d_val_topk, w_val_topk, idx_ct_topk, d_ct_topk, w_ct_topk, static_hist,
            N_QUERY_VAL_TARGET, N_QUERY_CT_TARGET, M_LOCAL, TAU_LOCAL, SEED)
        in_stratified, in_unusual = c in stratified_set, c in unusual_set
        br["is_unusual_include"] = in_unusual
        br["is_stratified_include"] = in_stratified
        br["sampling_mode"] = "both" if (in_stratified and in_unusual) else ("stratified" if in_stratified else "unusual")
        br["unusual_reason"] = unusual_reasons.get(c, "")
        br["inclusion_probability"] = inclusion_prob.get(c)
        all_query_records.extend(qr); all_pair_records.extend(pr); all_static_records.extend(sr)
        behavior_rows.append(br)

    query_df = pd.DataFrame(all_query_records)
    pair_df = pd.DataFrame(all_pair_records)
    static_df = pd.DataFrame(all_static_records)
    behavior_df = pd.DataFrame(behavior_rows)

    join_cols = ["cluster", "rms_dispersion", "V_j", "distinct_token_fraction", "entropy",
                 "top_token_fraction", "singleton_token_fraction", "obs_frac_ge_2",
                 f"traffic_top{K_ROUTE}_val", "traffic_top1_val"]
    join_cols = [col for col in join_cols if col in c1_stats.columns]
    behavior_df = behavior_df.merge(c1_stats[join_cols], on="cluster", how="left", suffixes=("", "_c1"))

    query_df.to_csv(os.path.join(OUT_DIR, "local_query_statistics.csv"), index=False)
    pair_df.to_csv(os.path.join(OUT_DIR, "local_pair_statistics.csv"), index=False)
    static_df.to_csv(os.path.join(OUT_DIR, "static_baseline_comparison.csv"), index=False)
    behavior_df.to_csv(os.path.join(OUT_DIR, "cluster_behavior_statistics.csv"), index=False)

    print("Computing structural relationships (Pearson + Spearman)...")
    relationship_pairs = [
        ("n_j", "mean_jaccard", "population_and_selection_stability"),
        ("rms_dispersion", "mean_margin", "dispersion_and_selection_margins"),
        ("entropy", "mean_tv_total", "token_entropy_and_output_variation"),
        ("distinct_token_fraction", "frac_label_changing_pairs", "distinct_tokens_and_label_changing_replacements"),
        ("singleton_token_fraction", "mean_tv_total", "singleton_prevalence_and_output_sensitivity"),
        ("obs_frac_ge_2", "mean_tv_total", "token_repetition_and_output_variation"),
        (f"traffic_top{K_ROUTE}_val", "I_C_val", "centroid_traffic_and_query_dependence"),
        ("top_token_fraction", "mean_tv_vs_static", "dominant_token_fraction_and_static_fidelity"),
    ]
    def correlate(sub, a, b):
        a_vals = pd.to_numeric(sub[a], errors="coerce").values
        b_vals = pd.to_numeric(sub[b], errors="coerce").values
        mask = np.isfinite(a_vals) & np.isfinite(b_vals)
        n = int(mask.sum())
        if n >= 3 and np.std(a_vals[mask]) > 0 and np.std(b_vals[mask]) > 0:
            return float(np.corrcoef(a_vals[mask], b_vals[mask])[0, 1]), \
                   float(scipy_stats.spearmanr(a_vals[mask], b_vals[mask]).correlation), n
        return None, None, n

    relationships = []
    # Review point 5: every relationship is reported for BOTH the full
    # sampled set (which includes deliberately-chosen unusual clusters) AND
    # a "stratified_only" subset (excludes unusual-but-not-stratified
    # clusters) -- unusual clusters were chosen BECAUSE they are extremal,
    # so including them in a "representativeness" claim would bias the
    # correlation; stratified_only is the scope that approximates random
    # sampling (see inclusion_probability in sampling_report.md).
    for a, b, label in relationship_pairs:
        if a not in behavior_df.columns or b not in behavior_df.columns:
            continue
        base_mask = ~behavior_df["is_trivial_selection"] if b == "mean_margin" else pd.Series(True, index=behavior_df.index)
        base_scope_name = "sampled_clusters_with_n_j_gt_m" if b == "mean_margin" else "all_sampled_clusters"
        sub_all = behavior_df[base_mask]
        pr, sr, n = correlate(sub_all, a, b)
        relationships.append({"var_a": a, "var_b": b, "label": label, "scope": base_scope_name,
                               "pearson_r": pr, "spearman_r": sr, "n": n})

        sub_strat = behavior_df[base_mask & behavior_df["is_stratified_include"]]
        pr2, sr2, n2 = correlate(sub_strat, a, b)
        relationships.append({"var_a": a, "var_b": b, "label": label, "scope": f"stratified_only_{base_scope_name}",
                               "pearson_r": pr2, "spearman_r": sr2, "n": n2})
    pd.DataFrame(relationships).to_csv(os.path.join(OUT_DIR, "structural_relationships.csv"), index=False)

    print("Selecting failure-case examples (with full qualitative detail, review point 4)...")

    def c1_row_for(cluster):
        rows = c1_stats[c1_stats["cluster"] == cluster]
        return rows.iloc[0] if len(rows) else None

    def pair_case_from_row(row):
        detail = build_pair_case_detail(
            int(row["cluster"]), int(row["query_row_i"]), int(row["query_row_j"]),
            ds_keys, ds_values, cluster_to_members[int(row["cluster"])], val_keys, M_LOCAL, TAU_LOCAL,
            c1_row_for(int(row["cluster"])))
        detail["summary_jaccard"] = row["jaccard"]; detail["summary_tv_total"] = row["tv_total"]
        detail["summary_tv_weight_change"] = row["tv_weight_change"]
        detail["summary_tv_selection_change"] = row["tv_selection_change"]
        return detail

    def single_case_from_cluster(cluster, query_array_name, query_array, fallback_row=None):
        row = fallback_row if fallback_row is not None else 0
        return build_single_query_case_detail(
            cluster, int(row), ds_keys, ds_values, cluster_to_members[cluster], query_array,
            M_LOCAL, TAU_LOCAL, c1_row_for(cluster))

    cases = {}
    if len(pair_df):
        cand = pair_df[(pair_df["jaccard"] < 0.3) & (pair_df["tv_total"] < 0.05)]
        cases["large_candidate_change_small_predictive_change"] = (
            pair_case_from_row(cand.loc[(1 - cand["jaccard"] - cand["tv_total"]).idxmax()]) if len(cand) else None)
        cand = pair_df[(pair_df["jaccard"] > 0.8) & (pair_df["tv_total"] > 0.3)]
        cases["small_candidate_change_large_predictive_change"] = (
            pair_case_from_row(cand.loc[cand["tv_total"].idxmax()]) if len(cand) else None)
        cand = pair_df[pair_df["identical_candidate_set"] == 1]
        cases["stable_candidates_meaningful_competition_change"] = (
            pair_case_from_row(cand.loc[cand["tv_weight_change"].idxmax()]) if len(cand) else None)
    if len(static_df):
        worst = static_df.loc[static_df["tv_vs_static"].idxmax()]
        detail = build_single_query_case_detail(int(worst["cluster"]), int(worst["query_row"]), ds_keys, ds_values,
                                                   cluster_to_members[int(worst["cluster"])], val_keys, M_LOCAL,
                                                   TAU_LOCAL, c1_row_for(int(worst["cluster"])))
        detail["summary_tv_vs_static"] = worst["tv_vs_static"]
        cases["large_static_histogram_error"] = detail
    if len(behavior_df) and "entropy" in behavior_df.columns:
        # Correction (external review): query dependence is DEFINED via I_C
        # (spec Part 6.3), not nearby-pair TV -- selecting these cases by
        # mean_tv_total while describing them as "query dependence" was a
        # mismatch between the selection criterion and the label. Score by
        # I_C_val instead, which is what the case names actually claim.
        valid = behavior_df.dropna(subset=["entropy", "I_C_val"])
        if len(valid):
            for case_name, score in (
                ("high_entropy_low_query_dependence", valid["entropy"] - 3.0 * valid["I_C_val"]),
                ("low_entropy_high_query_dependence", valid["I_C_val"] - 3.0 * valid["entropy"]),
            ):
                best = valid.loc[score.idxmax()]
                cluster = int(best["cluster"])
                rows_c = query_df[query_df["cluster"] == cluster]
                detail = (single_case_from_cluster(cluster, "val", val_keys, fallback_row=int(rows_c.iloc[0]["query_row"]))
                           if len(rows_c) else {"cluster": cluster, "note": "no sampled query rows available"})
                detail["summary_entropy"] = best["entropy"]; detail["summary_I_C_val"] = best["I_C_val"]
                detail["summary_mean_tv_total"] = best.get("mean_tv_total")  # context only, not the selection criterion
                cases[case_name] = detail
    write_failure_cases_md(OUT_DIR, cases, pilot_note)

    # Review point 3: distance-stratified Q1 summary, pooled across all sampled pairs.
    distance_stratified_rows = []
    if len(pair_df) and pair_df["query_query_distance"].notna().sum() >= 4:
        valid_pairs = pair_df.dropna(subset=["query_query_distance"])
        edges = np.quantile(valid_pairs["query_query_distance"].values, [0, 0.25, 0.5, 0.75, 1.0])
        for i in range(4):
            lo, hi = edges[i], edges[i + 1]
            mask = (valid_pairs["query_query_distance"] >= lo) & (valid_pairs["query_query_distance"] <= hi)
            bucket = valid_pairs[mask]
            if len(bucket) == 0:
                continue
            distance_stratified_rows.append({
                "bin": f"Q{i+1}", "n": int(len(bucket)), "d_min": float(lo), "d_max": float(hi),
                "mean_jaccard": round(float(bucket["jaccard"].mean()), 4),
                "mean_tv_total": round(float(bucket["tv_total"].mean()), 4),
                "mean_js_total": round(float(bucket["js_total"].mean()), 4),
                "frac_stability_guaranteed": round(float(bucket["stability_guaranteed"].mean()), 4),
            })

    # Review point 6/Q11: forward/reverse counterfactual-path agreement, and
    # split-half I_C sensitivity -- real computed numbers, not placeholders.
    q11_stats = {}
    if len(pair_df):
        q11_stats["mean_abs_diff_weight_change_fwd_rev"] = round(
            float((pair_df["tv_weight_change"] - pair_df["tv_weight_change_reverse"]).abs().mean()), 6)
        q11_stats["mean_abs_diff_selection_change_fwd_rev"] = round(
            float((pair_df["tv_selection_change"] - pair_df["tv_selection_change_reverse"]).abs().mean()), 6)
        q11_stats["n_pairs_for_fwd_rev"] = int(len(pair_df))
    if len(behavior_df):
        halves = behavior_df.dropna(subset=["I_C_val_half1", "I_C_val_half2"])
        if len(halves) >= 2:
            q11_stats["mean_abs_diff_ic_halves"] = round(
                float((halves["I_C_val_half1"] - halves["I_C_val_half2"]).abs().mean()), 6)
            q11_stats["n_clusters_for_ic_halves"] = int(len(halves))
            if halves["I_C_val_half1"].std() > 0 and halves["I_C_val_half2"].std() > 0:
                q11_stats["corr_ic_halves"] = round(
                    float(np.corrcoef(halves["I_C_val_half1"], halves["I_C_val_half2"])[0, 1]), 4)
            else:
                q11_stats["corr_ic_halves"] = None
        else:
            q11_stats["mean_abs_diff_ic_halves"] = None
            q11_stats["n_clusters_for_ic_halves"] = int(len(halves))
            q11_stats["corr_ic_halves"] = None

    if len(query_df):
        fig, ax = plt.subplots(figsize=(6, 4))
        margins = query_df["margin"].dropna()
        if len(margins):
            ax.hist(margins, bins=40)
        ax.set_xlabel("selection margin gamma (n_C > m only)"); ax.set_title("Selection margin distribution")
        fig.tight_layout(); fig.savefig(os.path.join(FIG_DIR, "margin_histogram.png")); plt.close(fig)

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(query_df["N_eff"].dropna(), bins=40)
        ax.set_xlabel("N_eff"); ax.set_title("Competition weight concentration")
        fig.tight_layout(); fig.savefig(os.path.join(FIG_DIR, "weight_concentration_histogram.png")); plt.close(fig)

    if len(pair_df):
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.scatter(pair_df["jaccard"], pair_df["tv_total"], s=4, alpha=0.3)
        ax.set_xlabel("candidate-set Jaccard similarity"); ax.set_ylabel("total predictive TV")
        ax.set_title("Candidate stability vs. predictive variation")
        fig.tight_layout(); fig.savefig(os.path.join(FIG_DIR, "jaccard_vs_tv_total_scatter.png")); plt.close(fig)

        fig, ax = plt.subplots(figsize=(6, 6))
        sc = ax.scatter(pair_df["tv_weight_change"], pair_df["tv_selection_change"], s=4, alpha=0.3,
                         c=pair_df["tv_total"], cmap="viridis")
        ax.set_xlabel("TV from competition-weight change (candidates fixed)")
        ax.set_ylabel("TV from candidate-selection change (query fixed)")
        ax.set_title("Competition vs. selection contribution (not additive)")
        fig.colorbar(sc, ax=ax, label="tv_total")
        fig.tight_layout(); fig.savefig(os.path.join(FIG_DIR, "competition_vs_selection_scatter.png")); plt.close(fig)

    if len(static_df):
        fig, ax = plt.subplots(figsize=(6, 4))
        diff = (static_df["tv_vs_static"] - static_df["tv_vs_trainconst"]).dropna()
        if len(diff):
            ax.hist(diff, bins=40)
        ax.set_xlabel("TV(static histogram) - TV(train-derived constant)  [>0: constant baseline is better]")
        ax.set_title("Static histogram vs. query-distribution-adapted constant")
        fig.tight_layout(); fig.savefig(os.path.join(FIG_DIR, "static_vs_trainconst_fidelity.png")); plt.close(fig)

    if len(behavior_df) and f"traffic_top{K_ROUTE}_val" in behavior_df.columns:
        fig, ax = plt.subplots(figsize=(6, 6))
        valid = behavior_df.dropna(subset=["I_C_val", f"traffic_top{K_ROUTE}_val"])
        if len(valid):
            ax.scatter(valid[f"traffic_top{K_ROUTE}_val"], valid["I_C_val"], s=10)
        ax.set_xlabel(f"centroid routing traffic (top-{K_ROUTE}, val)"); ax.set_ylabel("I_C (val in-sample)")
        ax.set_title("Query dependence vs. centroid routing traffic")
        fig.tight_layout(); fig.savefig(os.path.join(FIG_DIR, "query_dependence_vs_traffic.png")); plt.close(fig)

    write_construction_consistency_md(OUT_DIR, c1_centroids_sha, c1_assignment_sha, c1_fingerprint,
                                        len(ds_values), N_CLUSTERS, K_ROUTE, TAU_ROUTE, M_LOCAL, TAU_LOCAL,
                                        traffic_comparison_note, pilot_note)
    exclusions = f"{coverage['n_eligible'] - coverage.get('n_sampled', 0)} eligible clusters not selected by stratification."
    write_sampling_report_md(OUT_DIR, coverage, behavior_df, unusual_reasons, exclusions,
                               cluster_cell_map, cell_population, inclusion_prob, pilot_note)
    write_validation_report_md(OUT_DIR, synthetic_results, real_data_notes, pilot_note)
    write_final_report_md(OUT_DIR, behavior_df, relationships, len(pair_df), len(query_df),
                            distance_stratified_rows, q11_stats, pilot_note)

    summary = {
        "n_ds": int(len(ds_values)), "n_ct": int(len(ct["values"])), "n_val": int(len(val["values"])),
        "n_clusters": N_CLUSTERS, "seed": SEED, "k_route": K_ROUTE, "tau_route": TAU_ROUTE,
        "m_local": M_LOCAL, "tau_local": TAU_LOCAL,
        "n_sampled_clusters": len(sampled_clusters), "sampling_coverage": coverage,
        "c1_centroids_sha256": c1_centroids_sha, "c1_assignment_sha256": c1_assignment_sha,
        "synthetic_checks_passed": n_pass, "synthetic_checks_total": len(synthetic_results),
        "elapsed_seconds": time.perf_counter() - t_start, "mode": DIME_MODE, "is_real_cache": is_real_cache,
    }
    save_results(os.path.join(OUT_DIR, "dime_v2_c2_1_summary.json"), summary)
    print(f"Done in {summary['elapsed_seconds']:.1f}s")


if __name__ == "__main__":
    main()
