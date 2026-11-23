"""DIME-V2 -- Token-Conditioned Geometric Compressibility Study.

This is a DIAGNOSTIC RESEARCH STUDY, not a new compressed retrieval
algorithm and not part of the R-series or Y-series (per the spec's own
explicit framing). It asks: within a DIME cluster, do the centroid-relative
residuals of observations sharing a token label occupy a geometrically
COMPACT, few-component structure -- and if so, is exploiting that structure
(vs. an equal-storage LABEL-AGNOSTIC geometric compression) actually worth
it for retrieval-relevant behavior (candidate selection AND candidate
competition/weights), not just for raw geometric reconstruction error?

Writes only to results/dime_v2_token_conditioned_geometry_study/ (no
"stage3_" prefix, per the spec's own Section 16 naming), touching no other
result directory.

=====================================================================
Mathematical starting point (Section 3), confirmed numerically (S0 below)
=====================================================================
For cluster C_j with centroid c_j: define query residual u = q - c_j and
observation residual v_i = h_i - c_j. Euclidean distance is translation
invariant, so ||q - h_i|| = ||u - v_i|| EXACTLY. This means every exact
retrieval quantity already established in this project (R1's
exact_evidence, the per-cluster-function series' local_m_distribution) can
be computed identically from either absolute coordinates or
centroid-relative residuals -- verified once numerically (S0), not assumed.
Centering matters NOT for computing exact targets (which this script still
computes via the established absolute-coordinate functions, reusing them
unmodified) but for the actual OBJECT OF STUDY: prototype-fitting is done
in centroid-relative residual space, V_{j,y} = {v_i : y_i = y}, because that
is the "token-conditioned geometric structure" the spec asks about --
geometry relative to the cluster's own frame, not the datastore's global
frame.

Token-conditioned local top-m evidence (as a fraction of a cluster's full
population -- Section 3): s_{j,y}^(m)(u) = sum over the m nearest same-
cluster neighbors of exp(-||u-v_i||/tau), restricted to the top-m SELECTION
(hard nearest-m across ALL labels in the cluster, not per-label) --
IDENTICAL definition to run_y1_per_cluster_function.py's local_m_distribution,
reused directly rather than reimplemented.

=====================================================================
Scope cuts (stated explicitly, not silently narrowed)
=====================================================================
  - No GPT-2 forward pass anywhere in this script. Every research question
    here concerns the GEOMETRY of the datastore residuals and the
    predictive fidelity of a reconstructed RETRIEVAL distribution F_j(y),
    never GPT-2's own next-token distribution or NLL -- FrozenGPT2 is not
    imported. This is a genuine, deliberate scope reduction from the GPU-
    heavy R6-style pipeline, not an oversight.
  - tau = 2.0 throughout (project's established matched-config temperature).
  - PRIMARY_M = 20 for the local top-m target (project's established
    matched-config outer k); M_DIAGNOSTIC = [1, 8] reported as secondary
    sensitivity, not swept exhaustively.
  - Cluster sample: TARGET_N_CLUSTERS, population >= MIN_CLUSTER_POP,
    stratified by rho_j = n_j/|Y_j| (low/mid/high) -- a property of the
    cluster itself, never of any method's accuracy, so no performance-based
    cherry-picking.
  - Per cluster, deep geometric/predictive analysis (Phases A2, B, C, D, E,
    F) is bounded to the TOP_N_TOKENS_PER_CLUSTER most populous tokens with
    population >= MIN_TOKEN_POP_FOR_GEOMETRY -- Phase A1 (composition)
    alone covers every token cheaply (it is just counting).
  - Component grid r in COMPONENT_GRID = [1, 2, 4, 8] per token (bounded,
    not an unbounded search for r_min).
  - Phase C2/C3 are CONSOLIDATED into one label-agnostic baseline
    (ordinary k-means over all cluster members' residuals, ignoring label,
    at the SAME total component count as the label-conditioned approach),
    stated explicitly per the spec's own allowance that these are
    conceptually the same construction (ordinary vector quantization).
    Token information for this baseline is retained via a per-prototype
    token-count histogram, whose storage is explicitly accounted for
    (label_agnostic_bytes) -- it is NOT free, and is reported separately
    from the matched-component-count comparison.
  - Phase D implements BOTH of the spec's own explicitly distinguished
    approximation rules (Section 8/D3), never conflating them:
      (i) ALL-MEMBER weighted sum: s_y^proto-all(u) = sum_l w_{y,l} *
          exp(-||u-z_{y,l}||/tau) -- reduces EXACTLY to R1's exact_evidence
          when r equals each token's own population (every observation is
          its own weight-1 prototype); verified numerically (S-reduction).
      (ii) TOP-m TRUNCATED: pool all prototypes (across tokens for the
          label-conditioned case; the single shared codebook for the
          label-agnostic case), select the m prototypes CLOSEST BY RAW
          DISTANCE (not weight-adjusted) as the truncated candidate set,
          then weight each selected prototype's softmax contribution by its
          own multiplicity w_l (or per-token histogram fraction for the
          agnostic case). This is ONE documented, explicit rule among
          several defensible choices -- stated here, never silently
          substituted for rule (i).
  - Phase B's "token region" analysis uses explicit MEASURABLE PROXIES
    (support/dominance region SHAPE in 768-dim is not claimed to be fully
    recoverable from finite samples, per the spec's own disclaimer): a
    k=1-vs-k=2 within-token k-means inertia-reduction ratio as a
    multimodality proxy, plus direct support/dominance counts over sampled
    held-out queries.
  - Phase E (predictive importance) and Phase F (capacity allocation) are
    both bounded, not exhaustive: Phase E reports per-prototype selection
    frequency/weight/leave-one-out TV impact and a single frequency-vs-
    importance correlation; Phase F compares exactly 4 named allocation
    rules (uniform, frequency-based, geometry-based, retrieval-aware) at
    ONE fixed total budget, not a budget sweep.

All five required synthetic tests (S1-S5), plus S0 (the centering identity)
and an exact-reduction sanity check, run first and MUST pass before any
real cache is touched -- per the project's established discipline and this
spec's own explicit instruction.
"""
import sys, os, json, time
from collections import Counter

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.cluster import MiniBatchKMeans, KMeans

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(SCRIPT_DIR, "..", "GPT_Module"))
sys.path.insert(0, SCRIPT_DIR)

from cache_io import load_cache
from knn import build_datastore, query_knn_indices
from results_io import save_results
from e1_metrics import js_divergence_dict
from run_e2_information_allocation import build_cluster_to_members
from run_crisp_subspace_projection import (
    exact_evidence, normalize_rows, dict_from_dist, total_variation,
)
from run_crisp_r2_query_sensitivity import mean_pairwise_tv, cluster_level_bootstrap_ci
from run_crisp_r3_failure_diagnosis import build_splits_for
from run_crisp_r4_histogram_corrections import mean_tv
from run_y1_per_cluster_function import local_m_distribution

SMOKE = os.environ.get("DIMEV2_SMOKE") == "1"
CACHE_DIR = os.environ.get("DIMEV2_CACHE_DIR", os.path.join(SCRIPT_DIR, "..", "GPT_Module", "cache"))
OUT_DIR = os.environ.get("DIMEV2_OUT_DIR", os.path.join(SCRIPT_DIR, "results", "dime_v2_token_conditioned_geometry_study"))
PLOT_DIR = os.path.join(OUT_DIR, "plots")
os.makedirs(PLOT_DIR, exist_ok=True)

N_CLUSTERS = int(os.environ.get("DIMEV2_N_CLUSTERS", "3811"))
SEED = 42
DIM, DTYPE_BYTES = 768, 4
OUTER_K = 20

TAU = 2.0
PRIMARY_M = 20
M_DIAGNOSTIC = [1, 8]

MIN_CLUSTER_POP = 60
TARGET_N_CLUSTERS = 10
TOP_N_TOKENS_PER_CLUSTER = 15
MIN_TOKEN_POP_FOR_GEOMETRY = 5

COMPONENT_GRID = [1, 2, 4, 8]
EPS_GRID = [0.5, 0.25, 0.1]

MAX_FIT, MAX_TUNE, MAX_TEST = 150, 50, 100
N_QUERY_PAIRS_PER_CLUSTER = 40
N_CLUSTER_BOOTSTRAP = 2000
RANDOM_BASELINE_REPEATS = 5

CAPACITY_BUDGET_PER_TOKEN = 2  # average components/token for Phase F's fixed R_total
MAX_IMPORTANCE_PROTOTYPES = 50  # bound on Phase E's leave-one-out sweep (see run_cluster)


# ============================================================
# Core geometry helpers (centroid-relative residual space)
# ============================================================
def within_group_dispersion(residuals):
    n = residuals.shape[0]
    if n < 2:
        return None
    mean_pt = residuals.mean(axis=0)
    centered = residuals - mean_pt[None, :]
    dist_to_mean = np.linalg.norm(centered, axis=1)
    try:
        _, s, _ = np.linalg.svd(centered, full_matrices=False)
        var = s ** 2
        total = var.sum()
        eff_dim = int(np.searchsorted(np.cumsum(var) / total, 0.90) + 1) if total > 0 else 0
    except np.linalg.LinAlgError:
        eff_dim = None
    return {"n": int(n), "mean_dist_to_centroid": float(dist_to_mean.mean()),
            "std_dist_to_centroid": float(dist_to_mean.std()), "effective_dim_90pct": eff_dim}


def random_group_dispersion_baseline(member_keys, centroid, group_size, rng, n_repeats=RANDOM_BASELINE_REPEATS):
    n_mem = member_keys.shape[0]
    if group_size >= n_mem or group_size < 2:
        return None
    vals = []
    for _ in range(n_repeats):
        idx = rng.choice(n_mem, size=group_size, replace=False)
        stats = within_group_dispersion(member_keys[idx] - centroid[None, :])
        if stats is not None:
            vals.append(stats["mean_dist_to_centroid"])
    return float(np.mean(vals)) if vals else None


def multimodality_proxy(residuals, seed):
    n = residuals.shape[0]
    if n < 4:
        return None
    inertia1 = KMeans(n_clusters=1, random_state=seed, n_init=1).fit(residuals).inertia_
    inertia2 = KMeans(n_clusters=2, random_state=seed, n_init=5).fit(residuals).inertia_
    reduction_ratio = 0.0 if inertia1 <= 0 else float(1.0 - (inertia2 / inertia1))
    return {"inertia1": float(inertia1), "inertia2": float(inertia2), "reduction_ratio": reduction_ratio}


def fit_prototypes(residuals, r, seed):
    """k-means on residuals (centroid-relative); returns (centers[r_eff,D],
    weights[r_eff], labels[n]). r_eff = min(r, n) -- never more components
    than observations."""
    n = residuals.shape[0]
    r_eff = max(1, min(r, n))
    if r_eff == 1:
        center = residuals.mean(axis=0, keepdims=True)
        labels = np.zeros(n, dtype=int)
        return center, np.array([float(n)]), labels
    km = KMeans(n_clusters=r_eff, random_state=seed, n_init=5).fit(residuals)
    weights = np.bincount(km.labels_, minlength=r_eff).astype(np.float64)
    return km.cluster_centers_, weights, km.labels_


def mean_squared_distortion(residuals, centers, labels):
    assigned = centers[labels]
    return float(np.mean(np.sum((residuals - assigned) ** 2, axis=1)))


def r_min_for_eps(distortion_by_r, eps):
    """Smallest r in COMPONENT_GRID whose distortion is <= eps * distortion(r=1).
    None if no grid point achieves it (report r=max grid point's distortion instead)."""
    if 1 not in distortion_by_r or distortion_by_r[1] is None or distortion_by_r[1] <= 0:
        return None
    target = eps * distortion_by_r[1]
    for r in sorted(distortion_by_r.keys()):
        d = distortion_by_r[r]
        if d is not None and d <= target:
            return r
    return None


def label_conditioned_bytes(n_prototypes_total):
    """Per prototype: 768-dim float32 location + 1 weight (4B) + 1 token id (4B)."""
    return n_prototypes_total * (DIM * DTYPE_BYTES + 4 + 4)


def label_agnostic_bytes(prototype_histograms):
    """Per prototype: location + total weight (4B), plus 8 bytes (token_id+count)
    per DISTINCT token assigned to that prototype -- the token-retention cost
    the spec requires to be accounted for, never treated as free."""
    total = 0
    for hist in prototype_histograms:
        total += DIM * DTYPE_BYTES + 4
        total += len(hist) * 8
    return total


def agnostic_prototype_histograms(member_values, labels, r_eff):
    hist = [Counter() for _ in range(r_eff)]
    for lbl, val in zip(labels, member_values):
        hist[int(lbl)][int(val)] += 1
    return hist


# ============================================================
# Phase D -- reconstruction rules (i) all-member weighted sum, (ii) top-m truncated
# ============================================================
def reconstruct_all_member_conditioned(query_residuals, token_ids, proto_by_token, tau):
    n_q = query_residuals.shape[0]
    evidence = np.zeros((n_q, len(token_ids)), dtype=np.float64)
    for ti, tok in enumerate(token_ids):
        centers, weights = proto_by_token[int(tok)]
        d = np.linalg.norm(query_residuals[:, None, :] - centers[None, :, :], axis=2)
        evidence[:, ti] = (np.exp(-d / tau) * weights[None, :]).sum(axis=1)
    return evidence


def reconstruct_all_member_agnostic(query_residuals, token_ids, agnostic_centers, agnostic_hists, tau):
    n_q = query_residuals.shape[0]
    d = np.linalg.norm(query_residuals[:, None, :] - agnostic_centers[None, :, :], axis=2)
    k = np.exp(-d / tau)
    evidence = np.zeros((n_q, len(token_ids)), dtype=np.float64)
    token_to_col = {int(t): i for i, t in enumerate(token_ids)}
    for l, hist in enumerate(agnostic_hists):
        for tok, cnt in hist.items():
            if tok in token_to_col:
                evidence[:, token_to_col[tok]] += k[:, l] * cnt
    return evidence


def exact_topm_candidate_tokens(query_keys, member_keys, member_values, m):
    d_all = np.linalg.norm(member_keys[None, :, :] - query_keys[:, None, :], axis=2)
    n_q, n_mem = d_all.shape
    m_eff = min(m, n_mem)
    nn_idx = np.argpartition(d_all, m_eff - 1, axis=1)[:, :m_eff]
    return [set(int(v) for v in member_values[nn_idx[qi]]) for qi in range(n_q)]


def reconstruct_topm_truncated(query_residuals, all_centers, per_proto_payload, m, tau, mode, token_ids,
                                track_importance=False):
    """mode='conditioned': per_proto_payload[l] = (token, weight).
    mode='agnostic': per_proto_payload[l] = Counter hist.
    Returns (evidence[n_q,T], candidate_sets[list of set], importance) where
    importance (only if track_importance) is {proto_idx: {"freq": int, "weight": float}}."""
    n_q = query_residuals.shape[0]
    r_total = all_centers.shape[0]
    m_eff = min(m, r_total)
    d = np.linalg.norm(query_residuals[:, None, :] - all_centers[None, :, :], axis=2)
    nn_idx = np.argpartition(d, m_eff - 1, axis=1)[:, :m_eff]
    token_to_col = {int(t): i for i, t in enumerate(token_ids)}
    evidence = np.zeros((n_q, len(token_ids)), dtype=np.float64)
    candidate_sets = []
    importance = {} if track_importance else None
    for qi in range(n_q):
        sel = nn_idx[qi]
        d_sel = d[qi, sel]
        k_sel = np.exp(-d_sel / tau)
        cand_tokens = set()
        for pos, proto_idx in enumerate(sel):
            proto_idx = int(proto_idx)
            if track_importance:
                rec = importance.setdefault(proto_idx, {"freq": 0, "weight": 0.0})
                rec["freq"] += 1
            if mode == "conditioned":
                tok, w = per_proto_payload[proto_idx]
                contrib = k_sel[pos] * w
                if tok in token_to_col:
                    evidence[qi, token_to_col[tok]] += contrib
                    cand_tokens.add(tok)
                if track_importance:
                    importance[proto_idx]["weight"] += float(contrib)
            else:
                hist = per_proto_payload[proto_idx]
                proto_total = 0.0
                for tok, cnt in hist.items():
                    contrib = k_sel[pos] * cnt
                    proto_total += contrib
                    if tok in token_to_col:
                        evidence[qi, token_to_col[tok]] += contrib
                        cand_tokens.add(tok)
                if track_importance:
                    importance[proto_idx]["weight"] += float(proto_total)
        candidate_sets.append(cand_tokens)
    return evidence, candidate_sets, importance


def jaccard(set_a, set_b):
    if not set_a and not set_b:
        return 1.0
    union = set_a | set_b
    if not union:
        return 1.0
    return len(set_a & set_b) / len(union)


# ============================================================
# Synthetic correctness tests (S0-S5), must pass before touching real cache
# ============================================================
def check_s0_centering_identity(rng):
    c = rng.normal(size=768)
    q = rng.normal(size=768)
    h = rng.normal(size=(50, 768))
    d_abs = np.linalg.norm(h - q[None, :], axis=1)
    d_rel = np.linalg.norm((h - c[None, :]) - (q - c)[None, :], axis=1)
    ok = np.allclose(d_abs, d_rel, atol=1e-9)
    assert ok, "S0 FAILED: centroid-relative distance does not match absolute distance"
    return {"name": "S0_centering_identity", "passed": bool(ok)}


def check_exact_reduction(rng):
    """Rule (i) with r = population per token (every observation its own
    weight-1 prototype) must reduce EXACTLY to R1's exact_evidence."""
    D = 16
    n_per_label = 12
    members0 = rng.normal(loc=0.0, scale=1.0, size=(n_per_label, D))
    members1 = rng.normal(loc=3.0, scale=1.0, size=(n_per_label, D))
    member_keys = np.vstack([members0, members1])
    member_values = np.array([0] * n_per_label + [1] * n_per_label)
    centroid = member_keys.mean(axis=0)
    query_keys = rng.normal(size=(10, D))
    token_ids, exact_ev = exact_evidence(query_keys, member_keys, member_values, TAU)
    exact_dist, _ = normalize_rows(exact_ev)

    proto_by_token = {}
    for tok in token_ids:
        res = member_keys[member_values == tok] - centroid[None, :]
        proto_by_token[int(tok)] = (res, np.ones(res.shape[0]))
    q_res = query_keys - centroid[None, :]
    recon_ev = reconstruct_all_member_conditioned(q_res, token_ids, proto_by_token, TAU)
    recon_dist, _ = normalize_rows(recon_ev)
    max_err = float(np.max(np.abs(exact_dist - recon_dist)))
    ok = max_err < 1e-8
    assert ok, f"exact-reduction check FAILED: max_err={max_err}"
    return {"name": "exact_reduction", "passed": bool(ok), "max_err": max_err}


def check_s1_one_compact_region(rng):
    """Single token, single tight Gaussian blob: r=1 prototype should
    reconstruct the all-member distribution almost exactly."""
    D = 16
    pop_a = rng.normal(loc=0.0, scale=0.3, size=(80, D))
    pop_b = rng.normal(loc=4.0, scale=0.3, size=(80, D))
    member_keys = np.vstack([pop_a, pop_b])
    member_values = np.array([0] * 80 + [1] * 80)
    centroid = member_keys.mean(axis=0)
    query_keys = rng.normal(loc=2.0, scale=1.5, size=(30, D))
    token_ids, exact_ev = exact_evidence(query_keys, member_keys, member_values, TAU)
    exact_dist, _ = normalize_rows(exact_ev)
    q_res = query_keys - centroid[None, :]
    proto_by_token = {}
    for tok in token_ids:
        res = member_keys[member_values == tok] - centroid[None, :]
        centers, weights, _ = fit_prototypes(res, 1, SEED)
        proto_by_token[int(tok)] = (centers, weights)
    recon_ev = reconstruct_all_member_conditioned(q_res, token_ids, proto_by_token, TAU)
    recon_dist, _ = normalize_rows(recon_ev)
    tv = mean_tv(exact_dist, recon_dist)
    ok = tv < 0.05
    assert ok, f"S1 FAILED: r=1 single-blob TV too high ({tv})"
    return {"name": "S1_one_compact_region", "passed": bool(ok), "tv": float(tv)}


def check_s2_multiple_regions(rng):
    """One token with TWO well-separated sub-blobs: r=1 should reconstruct
    poorly, r=2 should recover near-exact, and the multimodality proxy
    should flag it."""
    D = 16
    blob_a = rng.normal(loc=0.0, scale=0.2, size=(60, D))
    blob_b = rng.normal(loc=6.0, scale=0.2, size=(60, D))
    tok0_members = np.vstack([blob_a, blob_b])
    tok1_members = rng.normal(loc=3.0, scale=0.3, size=(120, D))
    member_keys = np.vstack([tok0_members, tok1_members])
    member_values = np.array([0] * 120 + [1] * 120)
    centroid = member_keys.mean(axis=0)
    query_keys = np.vstack([rng.normal(loc=0.0, scale=0.5, size=(15, D)),
                             rng.normal(loc=6.0, scale=0.5, size=(15, D))])
    token_ids, exact_ev = exact_evidence(query_keys, member_keys, member_values, TAU)
    exact_dist, _ = normalize_rows(exact_ev)
    q_res = query_keys - centroid[None, :]

    res0 = member_keys[member_values == 0] - centroid[None, :]
    res1 = member_keys[member_values == 1] - centroid[None, :]

    def recon_with_r(r0):
        c0, w0, _ = fit_prototypes(res0, r0, SEED)
        c1, w1, _ = fit_prototypes(res1, COMPONENT_GRID[-1], SEED)
        proto_by_token = {0: (c0, w0), 1: (c1, w1)}
        ev = reconstruct_all_member_conditioned(q_res, token_ids, proto_by_token, TAU)
        d, _ = normalize_rows(ev)
        return mean_tv(exact_dist, d)

    tv_r1 = recon_with_r(1)
    tv_r2 = recon_with_r(2)
    mm = multimodality_proxy(res0, SEED)
    ok = (tv_r1 > 0.05) and (tv_r2 < tv_r1) and (mm["reduction_ratio"] > 0.3)
    assert ok, f"S2 FAILED: tv_r1={tv_r1}, tv_r2={tv_r2}, multimodality={mm}"
    return {"name": "S2_multiple_regions", "passed": bool(ok), "tv_r1": float(tv_r1),
            "tv_r2": float(tv_r2), "multimodality_reduction_ratio": float(mm["reduction_ratio"])}


def check_s3_interleaved_labels(rng):
    """Two tokens whose residuals fully overlap spatially: each token's OWN
    geometry is still compact (r=1 reconstructs each token's marginal
    well), even though the two support regions coincide -- geometric
    compactness and predictive separability are different properties."""
    D = 16
    pop_a = rng.normal(loc=0.0, scale=0.4, size=(100, D))
    pop_b = rng.normal(loc=0.0, scale=0.4, size=(100, D))  # same location, different label
    member_keys = np.vstack([pop_a, pop_b])
    member_values = np.array([0] * 100 + [1] * 100)
    centroid = member_keys.mean(axis=0)
    query_keys = rng.normal(loc=0.0, scale=0.6, size=(30, D))
    token_ids, exact_ev = exact_evidence(query_keys, member_keys, member_values, TAU)
    exact_dist, _ = normalize_rows(exact_ev)
    q_res = query_keys - centroid[None, :]
    proto_by_token = {}
    for tok in token_ids:
        res = member_keys[member_values == tok] - centroid[None, :]
        c, w, _ = fit_prototypes(res, 1, SEED)
        proto_by_token[int(tok)] = (c, w)
    recon_ev = reconstruct_all_member_conditioned(q_res, token_ids, proto_by_token, TAU)
    recon_dist, _ = normalize_rows(recon_ev)
    tv = mean_tv(exact_dist, recon_dist)
    near_uniform_split = bool(np.all(np.abs(exact_dist - 0.5) < 0.1))
    ok = (tv < 0.1) and near_uniform_split
    assert ok, f"S3 FAILED: tv={tv}, near_uniform_split={near_uniform_split}"
    return {"name": "S3_interleaved_labels", "passed": bool(ok), "tv": float(tv)}


def check_s4_rare_but_important(rng):
    """A rare token (few members, small aggregate weight) sitting in a
    region where it is nonetheless competitive for nearby queries: a
    frequency-based allocation that starves it should measurably hurt
    reconstruction near it, vs. a rule that preserves at least one
    component for it."""
    D = 16
    common = rng.normal(loc=0.0, scale=1.0, size=(180, D))
    rare = rng.normal(loc=1.5, scale=0.15, size=(6, D))
    member_keys = np.vstack([common, rare])
    member_values = np.array([0] * 180 + [1] * 6)
    centroid = member_keys.mean(axis=0)
    query_keys = rng.normal(loc=1.5, scale=0.3, size=(20, D))  # queries NEAR the rare token
    token_ids, exact_ev = exact_evidence(query_keys, member_keys, member_values, TAU)
    exact_dist, _ = normalize_rows(exact_ev)
    q_res = query_keys - centroid[None, :]
    res0 = member_keys[member_values == 0] - centroid[None, :]
    res1 = member_keys[member_values == 1] - centroid[None, :]

    # Starved allocation: rare token gets 0 components (dropped entirely).
    c0, w0, _ = fit_prototypes(res0, 4, SEED)
    ev_starved = reconstruct_all_member_conditioned(q_res, token_ids, {0: (c0, w0), 1: (np.zeros((1, D)), np.zeros(1))}, TAU)
    dist_starved, _ = normalize_rows(ev_starved)
    tv_starved = mean_tv(exact_dist, dist_starved)

    # Preserved allocation: rare token keeps 1 component.
    c1, w1, _ = fit_prototypes(res1, 1, SEED)
    ev_preserved = reconstruct_all_member_conditioned(q_res, token_ids, {0: (c0, w0), 1: (c1, w1)}, TAU)
    dist_preserved, _ = normalize_rows(ev_preserved)
    tv_preserved = mean_tv(exact_dist, dist_preserved)

    ok = tv_preserved < tv_starved
    assert ok, f"S4 FAILED: tv_starved={tv_starved}, tv_preserved={tv_preserved}"
    return {"name": "S4_rare_but_important", "passed": bool(ok),
            "tv_starved": float(tv_starved), "tv_preserved": float(tv_preserved)}


def check_s5_candidate_vs_weight(rng):
    """Construct a case where a prototype swap preserves the CANDIDATE SET
    (which tokens show up in top-m) but changes the COMPETITION WEIGHTS
    (the softmax distribution over that set) -- verifying the two fidelity
    metrics are genuinely distinct, not collapsible into one number."""
    D = 4
    # token 0: three points spread along one axis (0, 2, 4); token 1: one
    # point (2.5) deliberately close to token 0's MEAN so the two tokens
    # compete closely (keeps the overall split away from a 0/1 extreme,
    # where absolute TV would be numerically tiny regardless of any
    # approximation error). The query sits right next to token 0's nearest
    # member (0), so token 0's exact evidence is dominated by that single
    # close point -- a detail an r=1 MEAN prototype for token 0 cannot see
    # (it collapses to the centroid at 2, farther from the query than the
    # true nearest member). Both tokens remain in the candidate set under
    # both representations (m = full population, so neither target
    # truncates anyone out) while the competition WEIGHT shifts measurably
    # (verified numerically below, not merely asserted by construction).
    member_keys = np.array([
        [0.0, 0.0, 0.0, 0.0],
        [2.0, 0.0, 0.0, 0.0],
        [4.0, 0.0, 0.0, 0.0],
        [2.5, 0.0, 0.0, 0.0],
    ])
    member_values = np.array([0, 0, 0, 1])
    query = np.array([[0.2, 0.0, 0.0, 0.0]])
    token_ids, exact_ev = exact_evidence(query, member_keys, member_values, TAU)
    exact_dist, _ = normalize_rows(exact_ev)
    candidate_exact = exact_topm_candidate_tokens(query, member_keys, member_values, m=len(member_values))[0]

    centroid = member_keys.mean(axis=0)
    q_res = query - centroid[None, :]
    res0 = member_keys[member_values == 0] - centroid[None, :]
    res1 = member_keys[member_values == 1] - centroid[None, :]
    c0, w0, _ = fit_prototypes(res0, 1, SEED)
    c1, w1, _ = fit_prototypes(res1, 1, SEED)
    ev_proto = reconstruct_all_member_conditioned(q_res, token_ids, {0: (c0, w0), 1: (c1, w1)}, TAU)
    dist_proto, _ = normalize_rows(ev_proto)
    # Candidate set from the prototype side: both tokens have nonzero prototype mass -> both appear.
    candidate_proto = set(int(t) for t, p in zip(token_ids, dist_proto[0]) if p > 1e-12)
    candidate_recall = jaccard(candidate_exact, candidate_proto)
    weight_tv = mean_tv(exact_dist, dist_proto)
    ok = (candidate_recall > 0.99) and (weight_tv > 0.01)
    assert ok, f"S5 FAILED: candidate_recall={candidate_recall}, weight_tv={weight_tv}"
    return {"name": "S5_candidate_vs_weight", "passed": bool(ok),
            "candidate_recall": float(candidate_recall), "weight_tv": float(weight_tv)}


def run_synthetic_checks():
    rng = np.random.default_rng(SEED)
    results = [
        check_s0_centering_identity(rng),
        check_exact_reduction(rng),
        check_s1_one_compact_region(rng),
        check_s2_multiple_regions(rng),
        check_s3_interleaved_labels(rng),
        check_s4_rare_but_important(rng),
        check_s5_candidate_vs_weight(rng),
    ]
    assert all(r["passed"] for r in results), f"Synthetic checks failed: {results}"
    return results


# ============================================================
# Query-change diagnostic (established project pattern)
# ============================================================
def query_change_errors(dist_a, dist_b, static_hist, rng, n_pairs):
    n = dist_a.shape[0]
    if n < 2:
        return None
    n_pairs = min(n_pairs, n * (n - 1) // 2)
    pi = rng.integers(0, n, size=n_pairs * 2)
    pj = rng.integers(0, n, size=n_pairs * 2)
    valid = pi != pj
    pi, pj = pi[valid][:n_pairs], pj[valid][:n_pairs]
    if len(pi) == 0:
        return None
    delta_exact = dist_a[pi] - dist_a[pj]
    delta_model = dist_b[pi] - dist_b[pj]
    e_delta = np.abs(delta_exact - delta_model).sum(axis=1)
    e_delta_static_baseline = np.abs(delta_exact).sum(axis=1)  # static hist never changes -> delta=0
    return {"mean_e_delta_model": float(np.mean(e_delta)),
            "mean_e_delta_static_baseline": float(np.mean(e_delta_static_baseline)),
            "beats_static_baseline_frac": float(np.mean(e_delta < e_delta_static_baseline))}


# ============================================================
# Per-cluster pipeline
# ============================================================
def run_cluster(c, member_keys, member_values, fit_keys, tune_keys, test_keys, rng):
    centroid = member_keys.mean(axis=0)
    n_j = len(member_values)
    token_counts = Counter(member_values.tolist())
    tokens_sorted = [t for t, _ in token_counts.most_common()]
    n_distinct = len(tokens_sorted)
    rho_j = n_j / max(1, n_distinct)
    label_counts = np.array(list(token_counts.values()), dtype=np.float64)
    p = label_counts / label_counts.sum()
    entropy = float(-(p * np.log(p + 1e-300)).sum())
    singleton_frac = float(np.mean(label_counts == 1))
    top_label_mass_frac = float(label_counts.max() / label_counts.sum())

    composition = {"cluster": int(c), "n_j": int(n_j), "n_distinct_tokens": int(n_distinct),
                    "rho_j": float(rho_j), "entropy": entropy, "singleton_frac": singleton_frac,
                    "top_label_mass_frac": top_label_mass_frac}

    geometry_tokens = [t for t in tokens_sorted if token_counts[t] >= MIN_TOKEN_POP_FOR_GEOMETRY][:TOP_N_TOKENS_PER_CLUSTER]

    token_geom_rows, region_rows = [], []
    for tok in geometry_tokens:
        res = member_keys[member_values == tok] - centroid[None, :]
        stats = within_group_dispersion(res)
        if stats is None:
            continue
        rand_baseline = random_group_dispersion_baseline(member_keys, centroid, stats["n"], rng)
        compactness_ratio = (stats["mean_dist_to_centroid"] / rand_baseline) if rand_baseline else None
        mm = multimodality_proxy(res, SEED + int(tok)) if stats["n"] >= 2 * MIN_TOKEN_POP_FOR_GEOMETRY else None
        token_geom_rows.append({"cluster": int(c), "token": int(tok), "n": stats["n"],
                                 "mean_dist_to_centroid": stats["mean_dist_to_centroid"],
                                 "effective_dim_90pct": stats["effective_dim_90pct"],
                                 "random_baseline_mean_dist": rand_baseline,
                                 "compactness_ratio": compactness_ratio,
                                 "multimodality_reduction_ratio": mm["reduction_ratio"] if mm else None})

    if len(fit_keys) < 5 or len(tune_keys) < 3 or len(test_keys) < 3:
        return composition, token_geom_rows, [], [], [], [], [], None

    token_ids, test_ev = exact_evidence(test_keys, member_keys, member_values, TAU)
    exact_all_dist, _ = normalize_rows(test_ev)
    exact_topm_ids, exact_topm_dist = local_m_distribution(test_keys, member_keys, member_values, PRIMARY_M, TAU)
    assert np.array_equal(token_ids, exact_topm_ids)
    exact_candidate_sets = exact_topm_candidate_tokens(test_keys, member_keys, member_values, PRIMARY_M)
    test_res = test_keys - centroid[None, :]

    # Phase B: support/dominance region diagnostics over the exact all-member test-split distribution
    for ti, tok in enumerate(token_ids):
        support_count = int(np.sum(exact_all_dist[:, ti] > 1e-12))
        dominance_count = int(np.sum(np.argmax(exact_all_dist, axis=1) == ti))
        if token_counts.get(int(tok), 0) >= MIN_TOKEN_POP_FOR_GEOMETRY:
            region_rows.append({"cluster": int(c), "token": int(tok), "n_test_queries": int(len(test_keys)),
                                 "support_count": support_count, "dominance_count": dominance_count})

    # Phase C/D: geometric compression + retrieval fidelity, label-conditioned vs label-agnostic
    primary_r_by_token, proto_by_token_primary, distortion_rows = {}, {}, []
    cond_distortion_vals = []
    for tok in geometry_tokens:
        res = member_keys[member_values == tok] - centroid[None, :]
        n_tok = res.shape[0]
        distortion_by_r = {}
        fits_by_r = {}
        for r in COMPONENT_GRID:
            if r > n_tok:
                continue
            centers, weights, labels = fit_prototypes(res, r, SEED)
            distortion_by_r[r] = mean_squared_distortion(res, centers, labels)
            fits_by_r[r] = (centers, weights)
        r_choice = None
        for eps in EPS_GRID:
            r_choice = r_min_for_eps(distortion_by_r, eps)
            if r_choice is not None:
                break
        r_choice = r_choice or max(distortion_by_r.keys())
        primary_r_by_token[int(tok)] = r_choice
        centers, weights = fits_by_r[r_choice]
        proto_by_token_primary[int(tok)] = (centers, weights)
        cond_distortion_vals.append(distortion_by_r[r_choice])
        distortion_rows.append({"cluster": int(c), "token": int(tok), "n": n_tok,
                                 "r_choice": int(r_choice),
                                 **{f"distortion_r{r}": distortion_by_r.get(r) for r in COMPONENT_GRID}})

    r_total_conditioned = sum(primary_r_by_token.values())
    # Observation-weighted (not token-weighted) mean, so a populous token does not
    # get the same weight as a near-threshold one -- matches agn_distortion's scope below.
    cond_pops = [int(np.sum(member_values == tok)) for tok in geometry_tokens]
    cond_distortion = (float(np.dot(cond_distortion_vals, cond_pops)) / sum(cond_pops)) if cond_pops else None

    # geometry_tokens is a BOUNDED subset (top TOP_N_TOKENS_PER_CLUSTER by population,
    # population >= MIN_TOKEN_POP_FOR_GEOMETRY). Every token in token_ids must still get
    # SOME representation in both approaches, or the reconstruction functions below
    # (which iterate over the full token_ids) would KeyError on any token outside this
    # subset -- a real bug caught via self-review before this script was first run.
    # Fix: tokens outside geometry_tokens get an EXACT per-member fallback (their own
    # full population as weight-1 "prototypes") in BOTH approaches -- this is exact
    # (zero distortion, matches exact_evidence exactly per the verified reduction
    # identity) and cheap, since these are by definition low-population tokens. The
    # matched-storage comparison (r_total_conditioned, bytes_*, distortion) stays
    # scoped to ONLY the geometry_tokens subset -- the actual object of this study.
    geometry_token_set = set(int(t) for t in geometry_tokens)
    non_geometry_mask = np.array([int(v) not in geometry_token_set for v in member_values])
    geom_mask = ~non_geometry_mask

    for tok in token_ids:
        tok_i = int(tok)
        if tok_i not in proto_by_token_primary:
            res = member_keys[member_values == tok_i] - centroid[None, :]
            proto_by_token_primary[tok_i] = (res, np.ones(res.shape[0]))

    # Label-agnostic baseline: fit the matched-budget codebook ONLY over
    # geometry_tokens' own members (the same scope cond_distortion uses), then append
    # exact per-member fallback entries for every non-geometry member so the full
    # token_ids set is still covered for reconstruction.
    geom_res = member_keys[geom_mask] - centroid[None, :]
    geom_values = member_values[geom_mask]
    if geom_res.shape[0] > 0:
        agn_centers_g, agn_weights_g, agn_labels_g = fit_prototypes(geom_res, r_total_conditioned, SEED)
        agn_hists_g = agnostic_prototype_histograms(geom_values, agn_labels_g, agn_centers_g.shape[0])
        agn_distortion = mean_squared_distortion(geom_res, agn_centers_g, agn_labels_g)
    else:
        agn_centers_g = np.zeros((0, member_keys.shape[1]))
        agn_hists_g = []
        agn_distortion = None

    non_geom_res = member_keys[non_geometry_mask] - centroid[None, :]
    non_geom_values = member_values[non_geometry_mask]
    non_geom_hists = [Counter({int(v): 1}) for v in non_geom_values]
    agn_centers = np.vstack([agn_centers_g, non_geom_res]) if non_geom_res.shape[0] else agn_centers_g
    agn_hists = agn_hists_g + non_geom_hists

    bytes_conditioned = label_conditioned_bytes(r_total_conditioned)
    bytes_agnostic = label_agnostic_bytes(agn_hists_g)  # matched-scope cost only, non-geometry fallback excluded from both

    # Rule (i): all-member weighted sum
    recon_all_cond = reconstruct_all_member_conditioned(test_res, token_ids, proto_by_token_primary, TAU)
    dist_all_cond, _ = normalize_rows(recon_all_cond)
    recon_all_agn = reconstruct_all_member_agnostic(test_res, token_ids, agn_centers, agn_hists, TAU)
    dist_all_agn, _ = normalize_rows(recon_all_agn)
    tv_all_cond = mean_tv(exact_all_dist, dist_all_cond)
    tv_all_agn = mean_tv(exact_all_dist, dist_all_agn)
    js_all_cond = float(np.mean([js_divergence_dict(dict_from_dist(token_ids, exact_all_dist[i]),
                                                      dict_from_dist(token_ids, dist_all_cond[i]))
                                   for i in range(len(test_keys))]))
    js_all_agn = float(np.mean([js_divergence_dict(dict_from_dist(token_ids, exact_all_dist[i]),
                                                     dict_from_dist(token_ids, dist_all_agn[i]))
                                  for i in range(len(test_keys))]))

    # Rule (ii): top-m truncated -- pool spans EVERY token in token_ids (not just
    # geometry_tokens), matching the agnostic pool's coverage (which also includes
    # exact non-geometry fallback entries), so neither side gets an unfair candidate-
    # availability advantage.
    cond_centers_pool = np.vstack([proto_by_token_primary[int(t)][0] for t in token_ids])
    cond_payload_pool = []
    for t in token_ids:
        centers, weights = proto_by_token_primary[int(t)]
        for w in weights:
            cond_payload_pool.append((int(t), float(w)))
    recon_topm_cond, cand_topm_cond, importance_cond = reconstruct_topm_truncated(
        test_res, cond_centers_pool, cond_payload_pool, PRIMARY_M, TAU, "conditioned", token_ids, track_importance=True)
    dist_topm_cond, _ = normalize_rows(recon_topm_cond)
    recon_topm_agn, cand_topm_agn, _ = reconstruct_topm_truncated(
        test_res, agn_centers, agn_hists, PRIMARY_M, TAU, "agnostic", token_ids)
    dist_topm_agn, _ = normalize_rows(recon_topm_agn)

    tv_topm_cond = mean_tv(exact_topm_dist, dist_topm_cond)
    tv_topm_agn = mean_tv(exact_topm_dist, dist_topm_agn)
    candidate_recall_cond = float(np.mean([jaccard(exact_candidate_sets[i], cand_topm_cond[i]) for i in range(len(test_keys))]))
    candidate_recall_agn = float(np.mean([jaccard(exact_candidate_sets[i], cand_topm_agn[i]) for i in range(len(test_keys))]))

    retrieval_row = {
        "cluster": int(c), "n_j": int(n_j), "n_geometry_tokens": len(geometry_tokens),
        "r_total_conditioned": int(r_total_conditioned),
        "cond_mean_distortion": cond_distortion, "agn_distortion": agn_distortion,
        "bytes_conditioned": int(bytes_conditioned), "bytes_agnostic": int(bytes_agnostic),
        "tv_all_member_conditioned": tv_all_cond, "tv_all_member_agnostic": tv_all_agn,
        "js_all_member_conditioned": js_all_cond, "js_all_member_agnostic": js_all_agn,
        "tv_topm_conditioned": tv_topm_cond, "tv_topm_agnostic": tv_topm_agn,
        "candidate_recall_topm_conditioned": candidate_recall_cond,
        "candidate_recall_topm_agnostic": candidate_recall_agn,
    }

    # Phase E: predictive importance of label-conditioned prototypes. Bounded to the
    # MAX_IMPORTANCE_PROTOTYPES most-frequently-selected prototypes -- the number of
    # distinct prototypes actually selected across all test queries is unbounded in
    # principle (up to PRIMARY_M * n_test_queries), and each one costs a full
    # reconstruct_topm_truncated call; an explicit cap keeps this tractable without
    # silently dropping the highest-traffic (most informative) prototypes.
    importance_rows = []
    weight_vals, impact_vals = [], []
    importance_items = sorted((importance_cond or {}).items(), key=lambda kv: -kv[1]["freq"])[:MAX_IMPORTANCE_PROTOTYPES]
    for proto_idx, rec in importance_items:
        tok, base_w = cond_payload_pool[proto_idx]
        reduced_pool = [(t, w if i != proto_idx else 0.0) for i, (t, w) in enumerate(cond_payload_pool)]
        recon_loo, _, _ = reconstruct_topm_truncated(test_res, cond_centers_pool, reduced_pool, PRIMARY_M, TAU,
                                                       "conditioned", token_ids)
        dist_loo, _ = normalize_rows(recon_loo)
        loo_tv_impact = mean_tv(dist_topm_cond, dist_loo)
        importance_rows.append({"cluster": int(c), "proto_idx": int(proto_idx), "token": int(tok),
                                 "weight": float(base_w), "selection_freq": int(rec["freq"]),
                                 "total_softmax_weight": float(rec["weight"]), "loo_tv_impact": float(loo_tv_impact)})
        if base_w > 0:
            weight_vals.append(np.log(base_w))
            impact_vals.append(loo_tv_impact)
    freq_importance_corr = float(np.corrcoef(weight_vals, impact_vals)[0, 1]) if len(weight_vals) >= 3 and np.std(weight_vals) > 0 else None
    retrieval_row["freq_importance_correlation"] = freq_importance_corr

    # Phase F: capacity allocation at a fixed total budget
    r_total_budget = max(len(geometry_tokens), CAPACITY_BUDGET_PER_TOKEN * len(geometry_tokens))
    importance_by_token = {}
    for row in importance_rows:
        importance_by_token.setdefault(row["token"], 0.0)
        importance_by_token[row["token"]] += row["total_softmax_weight"]
    allocation_rows = []
    for rule in ["uniform", "frequency", "geometry", "retrieval_aware"]:
        if rule == "uniform":
            weights_for_alloc = {int(t): 1.0 for t in geometry_tokens}
        elif rule == "frequency":
            weights_for_alloc = {int(t): float(token_counts[t]) for t in geometry_tokens}
        elif rule == "geometry":
            disp = {row["token"]: row["mean_dist_to_centroid"] for row in token_geom_rows}
            weights_for_alloc = {int(t): disp.get(int(t), 1.0) for t in geometry_tokens}
        else:
            weights_for_alloc = {int(t): importance_by_token.get(int(t), 1e-6) + 1e-6 for t in geometry_tokens}
        total_w = sum(weights_for_alloc.values()) or 1.0
        alloc_r = {}
        for t in geometry_tokens:
            frac = weights_for_alloc[int(t)] / total_w
            alloc_r[int(t)] = max(1, int(round(frac * r_total_budget)))
        # Non-geometry tokens reuse the same exact per-member fallback as the primary
        # comparison above (identical under every rule, since Phase F only reallocates
        # budget WITHIN geometry_tokens) -- without this, reconstruct_all_member_conditioned
        # would KeyError on any token outside geometry_tokens, the same bug fixed above.
        proto_by_token_alloc = {t: proto_by_token_primary[t] for t in proto_by_token_primary
                                  if t not in geometry_token_set}
        for tok in geometry_tokens:
            res = member_keys[member_values == tok] - centroid[None, :]
            r = min(alloc_r[int(tok)], res.shape[0])
            centers, weights, _ = fit_prototypes(res, r, SEED)
            proto_by_token_alloc[int(tok)] = (centers, weights)
        recon = reconstruct_all_member_conditioned(test_res, token_ids, proto_by_token_alloc, TAU)
        dist_alloc, _ = normalize_rows(recon)
        tv_alloc = mean_tv(exact_all_dist, dist_alloc)
        allocation_rows.append({"cluster": int(c), "rule": rule, "r_total_actual": sum(
            min(alloc_r[int(tok)], (member_values == tok).sum()) for tok in geometry_tokens),
            "tv_all_member": tv_alloc})

    # Query-change preservation
    static_hist = np.zeros(len(token_ids))
    for ti, tok in enumerate(token_ids):
        static_hist[ti] = token_counts.get(int(tok), 0)
    static_hist = static_hist / static_hist.sum()
    qc_cond = query_change_errors(exact_all_dist, dist_all_cond, static_hist, rng, N_QUERY_PAIRS_PER_CLUSTER)
    qc_agn = query_change_errors(exact_all_dist, dist_all_agn, static_hist, rng, N_QUERY_PAIRS_PER_CLUSTER)
    query_change_row = {"cluster": int(c), "conditioned": qc_cond, "agnostic": qc_agn}

    return composition, token_geom_rows, region_rows, distortion_rows, [retrieval_row], importance_rows, allocation_rows, query_change_row


# ============================================================
# Deliverables
# ============================================================
def classify_outcome(agg):
    tv_all = agg["tv_all_member_cond_minus_agn"]
    tv_topm = agg["tv_topm_cond_minus_agn"]
    cand = agg["candidate_recall_cond_minus_agn"]
    dist = agg["distortion_cond_minus_agn"]

    def sig_better(d):
        return d["mean"] is not None and d["ci_95_high"] is not None and d["ci_95_high"] < 0

    def sig_worse(d):
        return d["mean"] is not None and d["ci_95_low"] is not None and d["ci_95_low"] > 0

    geom_compressible = sig_better(dist)
    all_member_better = sig_better(tv_all)
    topm_better = sig_better(tv_topm)
    candidate_worse = sig_worse(cand)

    if geom_compressible and all_member_better and topm_better:
        return ("A", "Token-conditioned geometry is compressible AND predictively superior to the "
                      "label-agnostic baseline at matched storage on both the all-member and top-m "
                      "truncated targets -- worth exploiting.")
    if geom_compressible and not all_member_better and not topm_better:
        return ("B", "Token-conditioned residuals ARE geometrically more compact than the label-agnostic "
                      "baseline, but this does not translate into better predictive fidelity -- geometric "
                      "compactness alone does not imply retrieval advantage.")
    if not geom_compressible:
        return ("C", "Token-conditioned residuals are NOT measurably more compact than the label-agnostic "
                      "baseline at matched storage -- no clear geometric structure to exploit.")
    if all_member_better and (not topm_better or candidate_worse):
        return ("E", "Token-conditioned geometry preserves ALL-MEMBER (soft aggregate) fidelity but fails "
                      "on the top-m truncated / candidate-selection target specifically -- structure exists "
                      "for soft aggregation, not for hard truncated retrieval.")
    return ("D", "Mixed / cluster-dependent result -- no single comparison direction holds robustly across "
                  "the sampled clusters at the population level.")


def main():
    t_start = time.perf_counter()
    print(f"DIME-V2 job started: {time.strftime('%Y-%m-%d %H:%M:%S')}. SMOKE={SMOKE}, N_CLUSTERS={N_CLUSTERS}")

    print("Running synthetic correctness checks (S0, exact-reduction, S1-S5) BEFORE touching any real cache...")
    synthetic_results = run_synthetic_checks()
    print(f"  all {len(synthetic_results)} synthetic checks passed")
    save_results(os.path.join(OUT_DIR, "dime_v2_summary.json"),
                 {"synthetic_checks": synthetic_results, "smoke": SMOKE, "stage": "synthetic_checks_passed"})

    print("Loading cache (datastore/controller_train/val)...")
    ds = load_cache(os.path.join(CACHE_DIR, "wikitext103_gpt2_datastore.npz"))
    ct = load_cache(os.path.join(CACHE_DIR, "wikitext103_gpt2_controller_train.npz"))
    val = load_cache(os.path.join(CACHE_DIR, "wikitext103_gpt2_val.npz"))
    ds_keys, ds_values = ds["keys"].astype(np.float64), ds["values"]
    ct_keys, ct_values = ct["keys"].astype(np.float64), ct["values"]
    val_keys, val_values = val["keys"].astype(np.float64), val["values"]

    print("Re-deriving DIME clusters (seed=42, identical to every prior script in this series)...")
    km = MiniBatchKMeans(n_clusters=N_CLUSTERS, random_state=SEED)
    assignment = km.fit_predict(ds_keys.astype(np.float32))
    dime_centroids = km.cluster_centers_.astype(np.float64)
    compressed_dists = []
    for c in range(N_CLUSTERS):
        mv = ds_values[assignment == c]
        compressed_dists.append(Counter(mv.tolist()) if len(mv) > 0 else Counter())
    cluster_to_members = build_cluster_to_members(assignment, N_CLUSTERS)
    populations = np.array([sum(cd.values()) for cd in compressed_dists])

    dime_index, dime_values = build_datastore(dime_centroids, np.array(compressed_dists, dtype=object))
    outer_k_eff = min(OUTER_K, N_CLUSTERS)
    _, idx_dime_ct = query_knn_indices(dime_index, ct_keys, k=outer_k_eff)
    _, idx_dime_val = query_knn_indices(dime_index, val_keys, k=outer_k_eff)

    eligible = [c for c in range(N_CLUSTERS) if populations[c] >= MIN_CLUSTER_POP and len(compressed_dists[c]) >= 2]
    print(f"  {len(eligible)}/{N_CLUSTERS} clusters eligible (population >= {MIN_CLUSTER_POP}, >=2 distinct tokens)")
    if len(eligible) == 0:
        raise RuntimeError("No eligible clusters -- check MIN_CLUSTER_POP against the real population distribution.")
    rho = {c: populations[c] / len(compressed_dists[c]) for c in eligible}
    order_by_rho = sorted(eligible, key=lambda c: rho[c])
    n_target = min(TARGET_N_CLUSTERS, len(order_by_rho))
    positions = np.linspace(0, len(order_by_rho) - 1, n_target).astype(int)
    sample_clusters = sorted(set(int(order_by_rho[p]) for p in positions))
    print(f"  sampled {len(sample_clusters)} clusters (rho_j-stratified, deterministic): {sample_clusters}")

    splits = build_splits_for(sample_clusters, idx_dime_ct, idx_dime_val)

    all_composition, all_token_geom, all_region, all_distortion = [], [], [], []
    all_retrieval, all_importance, all_allocation, all_query_change = [], [], [], []
    for c in sample_clusters:
        members = cluster_to_members[c]
        member_keys, member_values = ds_keys[members], ds_values[members]
        sp = splits[c]
        fit_keys, tune_keys, test_keys = ct_keys[sp["fit"]], ct_keys[sp["tune"]], val_keys[sp["test"]]
        local_rng = np.random.default_rng(SEED + c)
        (composition, token_geom_rows, region_rows, distortion_rows,
         retrieval_rows, importance_rows, allocation_rows, query_change_row) = run_cluster(
            c, member_keys, member_values, fit_keys, tune_keys, test_keys, local_rng)
        all_composition.append(composition)
        all_token_geom.extend(token_geom_rows)
        all_region.extend(region_rows)
        all_distortion.extend(distortion_rows)
        all_retrieval.extend(retrieval_rows)
        all_importance.extend(importance_rows)
        all_allocation.extend(allocation_rows)
        if query_change_row is not None:
            all_query_change.append(query_change_row)
        print(f"  cluster {c}: n_j={composition['n_j']}, rho_j={composition['rho_j']:.2f}, "
              f"n_geometry_tokens={len(token_geom_rows)}, retrieval_rows={len(retrieval_rows)}")

    import pandas as pd
    pd.DataFrame(all_composition).to_csv(os.path.join(OUT_DIR, "cluster_composition_results.csv"), index=False)
    pd.DataFrame(all_token_geom).to_csv(os.path.join(OUT_DIR, "geometric_compression_results.csv"), index=False)
    pd.DataFrame(all_region).to_csv(os.path.join(OUT_DIR, "token_region_results.csv"), index=False)
    pd.DataFrame(all_distortion).to_csv(os.path.join(OUT_DIR, "distortion_curve_results.csv"), index=False)
    pd.DataFrame(all_retrieval).to_csv(os.path.join(OUT_DIR, "retrieval_fidelity_results.csv"), index=False)
    pd.DataFrame(all_importance).to_csv(os.path.join(OUT_DIR, "predictive_results.csv"), index=False)
    pd.DataFrame(all_allocation).to_csv(os.path.join(OUT_DIR, "capacity_allocation_results.csv"), index=False)

    memory_rows = [{"cluster": r["cluster"], "r_total_conditioned": r["r_total_conditioned"],
                     "bytes_conditioned": r["bytes_conditioned"], "bytes_agnostic": r["bytes_agnostic"],
                     "bytes_ratio_agn_over_cond": (r["bytes_agnostic"] / r["bytes_conditioned"]) if r["bytes_conditioned"] else None}
                    for r in all_retrieval]
    pd.DataFrame(memory_rows).to_csv(os.path.join(OUT_DIR, "memory_results.csv"), index=False)

    def paired_bootstrap(key_a, key_b):
        diffs = [r[key_a] - r[key_b] for r in all_retrieval
                 if r.get(key_a) is not None and r.get(key_b) is not None]
        return cluster_level_bootstrap_ci(diffs)

    comparisons = {
        "tv_all_member_cond_minus_agn": paired_bootstrap("tv_all_member_conditioned", "tv_all_member_agnostic"),
        "tv_topm_cond_minus_agn": paired_bootstrap("tv_topm_conditioned", "tv_topm_agnostic"),
        "candidate_recall_cond_minus_agn": paired_bootstrap("candidate_recall_topm_conditioned", "candidate_recall_topm_agnostic"),
        "distortion_cond_minus_agn": paired_bootstrap("cond_mean_distortion", "agn_distortion"),
    }
    outcome_code, outcome_text = classify_outcome(comparisons)
    print(f"Outcome: {outcome_code} -- {outcome_text}")

    qc_cond_vals = [r["conditioned"]["mean_e_delta_model"] for r in all_query_change if r["conditioned"]]
    qc_agn_vals = [r["agnostic"]["mean_e_delta_model"] for r in all_query_change if r["agnostic"]]
    qc_static_vals = [r["conditioned"]["mean_e_delta_static_baseline"] for r in all_query_change if r["conditioned"]]
    qc_cond_beats_static = [r["conditioned"]["beats_static_baseline_frac"] for r in all_query_change if r["conditioned"]]
    qc_agn_beats_static = [r["agnostic"]["beats_static_baseline_frac"] for r in all_query_change if r["agnostic"]]

    freq_imp_corrs = [r["freq_importance_correlation"] for r in all_retrieval if r.get("freq_importance_correlation") is not None]

    summary = {
        "n_ds": int(len(ds_values)), "n_ct": int(len(ct_values)), "n_val": int(len(val_values)),
        "n_clusters": N_CLUSTERS, "seed": SEED, "smoke": SMOKE, "tau": TAU, "primary_m": PRIMARY_M,
        "n_eligible_clusters": len(eligible), "sample_clusters": sample_clusters,
        "comparisons": comparisons,
        "outcome_code": outcome_code, "outcome_text": outcome_text,
        "query_change": {
            "mean_e_delta_conditioned": float(np.mean(qc_cond_vals)) if qc_cond_vals else None,
            "mean_e_delta_agnostic": float(np.mean(qc_agn_vals)) if qc_agn_vals else None,
            "mean_e_delta_static_baseline": float(np.mean(qc_static_vals)) if qc_static_vals else None,
            "conditioned_beats_static_frac_mean": float(np.mean(qc_cond_beats_static)) if qc_cond_beats_static else None,
            "agnostic_beats_static_frac_mean": float(np.mean(qc_agn_beats_static)) if qc_agn_beats_static else None,
        },
        "mean_freq_importance_correlation": float(np.mean(freq_imp_corrs)) if freq_imp_corrs else None,
        "elapsed_seconds": time.perf_counter() - t_start,
    }
    save_results(os.path.join(OUT_DIR, "dime_v2_summary.json"), {"synthetic_checks": synthetic_results, **summary})

    write_markdown_deliverables(all_composition, all_token_geom, all_region, synthetic_results,
                                  all_retrieval, all_allocation, comparisons, summary)
    make_plots(all_distortion, all_retrieval, all_allocation, all_token_geom, all_query_change)

    print(f"DIME-V2 job done in {time.perf_counter() - t_start:.1f}s")


def write_markdown_deliverables(all_composition, all_token_geom, all_region, synthetic_results,
                                  all_retrieval, all_allocation, comparisons, summary):
    import pandas as pd

    with open(os.path.join(OUT_DIR, "synthetic_results.md"), "w") as f:
        f.write("# Synthetic Correctness Checks (S0-S5 + exact-reduction)\n\n")
        f.write("All checks run and assert-pass BEFORE any real cache is touched.\n\n")
        for r in synthetic_results:
            extras = ", ".join(f"{k}={v}" for k, v in r.items() if k not in ("name", "passed"))
            f.write(f"- **{r['name']}**: passed={r['passed']}, {extras}\n")

    with open(os.path.join(OUT_DIR, "mathematical_analysis.md"), "w") as f:
        f.write("# Mathematical Analysis (Phase A)\n\n")
        f.write("## Centering identity (S0)\n\n")
        f.write("The Euclidean distance between a query and an observation equals the distance between "
                 "their centroid-relative residuals exactly, by translation invariance -- verified "
                 "numerically to 1e-9 tolerance, not merely asserted.\n\n")
        f.write("## Exact-reduction identity\n\n")
        f.write("Rule (i), the all-member weighted-sum reconstruction, reduces EXACTLY to R1's "
                 "exact_evidence when every observation is its own weight-1 prototype (component count "
                 "equals population per token) -- verified to under 1e-8 max absolute error on a "
                 "synthetic two-token cluster.\n\n")
        f.write("## Two distinct, explicitly documented approximation rules (Section 8, point D3)\n\n")
        f.write("Rule (i), ALL-MEMBER weighted sum, is used for the all-member predictive target and the "
                 "geometric-compression comparison: exact at full component count, approximate otherwise.\n\n")
        f.write("Rule (ii), TOP-m TRUNCATED: prototypes are pooled across tokens for the label-conditioned "
                 "case, or drawn from the single shared codebook for the label-agnostic case; the m "
                 "closest prototypes by raw distance (not weight-adjusted) form the truncated candidate "
                 "set, and each selected prototype's softmax contribution is scaled by its own "
                 "multiplicity, or by its per-token histogram fraction in the agnostic case. This is one "
                 "reasonable, explicitly stated rule, never silently substituted for rule (i) -- the two "
                 "are evaluated and reported separately throughout this study.\n\n")
        f.write("## Compressibility metric\n\n")
        f.write("r_min(eps) is the smallest component count r in the grid " + str(COMPONENT_GRID) +
                 " whose mean squared within-token distortion is at most eps times the r=1 (single-"
                 "prototype) distortion, for eps in " + str(EPS_GRID) + ". Falls back to the largest grid "
                 "value if no eps threshold is reached.\n\n")
        f.write("## Phase B proxies\n\n")
        f.write("Support/dominance region shape in 768 dimensions is not claimed to be fully recoverable "
                 "from finite held-out samples. Two bounded, explicit proxies are reported instead: "
                 "direct support/dominance counts over sampled held-out queries, and a k=1-vs-k=2 "
                 "within-token k-means inertia-reduction-ratio multimodality proxy.\n\n")

    comp_df = pd.DataFrame(all_composition)
    tok_df = pd.DataFrame(all_token_geom)
    with open(os.path.join(OUT_DIR, "cluster_geometry_audit.md"), "w") as f:
        f.write("# Cluster Geometry Audit (Phase A1/A2)\n\n")
        f.write(f"Sampled {len(comp_df)} clusters, rho_j-stratified (low/mid/high average observations "
                 "per token).\n\n")
        if len(comp_df):
            f.write("## Composition (Phase A1)\n\n")
            f.write("```\n" + comp_df.describe().to_string() + "\n```\n\n")
        if len(tok_df):
            f.write("## Within-token geometry vs. size-matched random baseline (Phase A2)\n\n")
            valid_ratio = tok_df["compactness_ratio"].dropna()
            frac_more_compact = float((valid_ratio < 1.0).mean()) if len(valid_ratio) else None
            f.write("Fraction of analyzed tokens whose own residuals are more compact than an equal-size "
                     f"random same-cluster baseline: {frac_more_compact}\n\n")
            f.write("```\n" + tok_df[["mean_dist_to_centroid", "effective_dim_90pct", "compactness_ratio",
                              "multimodality_reduction_ratio"]].describe().to_string() + "\n```\n\n")

    reg_df = pd.DataFrame(all_region)
    with open(os.path.join(OUT_DIR, "token_region_analysis.md"), "w") as f:
        f.write("# Token Region Analysis (Phase B, measurable proxies)\n\n")
        f.write("Full support/dominance region topology in 768 dimensions is not claimed to be "
                 "recoverable from finite held-out samples, per the spec's own disclaimer. Reported here: "
                 "direct counts over the held-out test split, plus the multimodality proxy reported "
                 "jointly with Phase A2 in cluster_geometry_audit.md.\n\n")
        if len(reg_df):
            f.write("```\n" + reg_df.describe().to_string() + "\n```\n\n")

    with open(os.path.join(OUT_DIR, "research_questions.md"), "w") as f:
        f.write("# Research Questions\n\n")
        tv_all = comparisons["tv_all_member_cond_minus_agn"]
        tv_topm = comparisons["tv_topm_cond_minus_agn"]
        cand = comparisons["candidate_recall_cond_minus_agn"]
        dist = comparisons["distortion_cond_minus_agn"]
        f.write("Q1. Is token-conditioned residual geometry more compact than a label-agnostic baseline "
                 f"at matched storage? Mean distortion difference (conditioned minus agnostic) = "
                 f"{dist['mean']}, 95% CI [{dist['ci_95_low']}, {dist['ci_95_high']}] over "
                 f"{dist['n_clusters']} clusters.\n\n")
        f.write("Q2. Does that geometric compactness translate into better all-member predictive "
                 f"fidelity? Mean TV difference = {tv_all['mean']}, 95% CI [{tv_all['ci_95_low']}, "
                 f"{tv_all['ci_95_high']}].\n\n")
        f.write("Q3. Does it translate into better top-m truncated predictive fidelity (the "
                 f"candidate-selection-relevant target)? Mean TV difference = {tv_topm['mean']}, 95% CI "
                 f"[{tv_topm['ci_95_low']}, {tv_topm['ci_95_high']}].\n\n")
        f.write("Q4. Are candidate-set fidelity and candidate-weight fidelity genuinely distinct "
                 "quantities (per S5)? Yes, by construction, verified synthetically -- reported "
                 f"separately throughout. Candidate-recall difference = {cand['mean']}, 95% CI "
                 f"[{cand['ci_95_low']}, {cand['ci_95_high']}].\n\n")
        f.write("Q5. Is predictive importance of individual components predictable from naive frequency "
                 "alone? Mean correlation between log weight and leave-one-out TV impact across sampled "
                 f"clusters = {summary['mean_freq_importance_correlation']}.\n\n")
        f.write("Q6. Does any representation beat the static-histogram zero-change baseline on "
                 "query-change preservation? Conditioned beats the static baseline on "
                 f"{summary['query_change']['conditioned_beats_static_frac_mean']} of query pairs on "
                 f"average; agnostic on {summary['query_change']['agnostic_beats_static_frac_mean']} -- "
                 "see final_report.md for discussion of this project's recurring finding.\n\n")
        f.write("Q7-Q10 (capacity allocation, rare-token preservation, adaptive allocation, overall "
                 f"verdict). See capacity_allocation_results.csv and final_report.md's outcome "
                 f"classification ({summary['outcome_code']}).\n\n")

    with open(os.path.join(OUT_DIR, "failure_cases.md"), "w") as f:
        f.write("# Failure Cases\n\n")
        ret_df = pd.DataFrame(all_retrieval)
        if len(ret_df):
            worse_all = ret_df[ret_df["tv_all_member_conditioned"] > ret_df["tv_all_member_agnostic"]]
            f.write("## Clusters where label-conditioned did worse than label-agnostic (all-member TV)\n\n")
            f.write(f"{len(worse_all)}/{len(ret_df)} clusters.\n\n")
            if len(worse_all):
                f.write("```\n" + worse_all[["cluster", "tv_all_member_conditioned", "tv_all_member_agnostic"]].to_string(index=False) + "\n```\n\n")
            worse_dist = ret_df[ret_df["cond_mean_distortion"] > ret_df["agn_distortion"]]
            f.write("## Clusters where token-conditioned geometry was less compact than label-agnostic\n\n")
            f.write(f"{len(worse_dist)}/{len(ret_df)} clusters.\n\n")
        if len(tok_df):
            not_compact = tok_df[tok_df["compactness_ratio"] >= 1.0]
            f.write("## Tokens whose own residuals were not more compact than a random same-size baseline\n\n")
            f.write(f"{len(not_compact)}/{len(tok_df)} analyzed tokens (compactness_ratio >= 1.0 means the "
                     "token-conditioned group is no tighter than an equal-size random group of cluster "
                     "members -- a direct counterexample to the compressibility hypothesis where it "
                     "occurs).\n\n")

    with open(os.path.join(OUT_DIR, "final_report.md"), "w") as f:
        f.write("# DIME-V2 Token-Conditioned Geometric Compressibility Study: Final Report\n\n")
        f.write(f"SMOKE={summary['smoke']}, n_clusters_total={summary['n_clusters']}, "
                 f"n_eligible={summary['n_eligible_clusters']}, n_sampled={len(summary['sample_clusters'])}.\n\n")
        f.write("## Primary comparisons (cluster-level bootstrap, 95% CI)\n\n")
        for k, v in comparisons.items():
            f.write(f"- {k}: mean={v['mean']}, CI=[{v['ci_95_low']}, {v['ci_95_high']}], "
                     f"n_clusters={v['n_clusters']}\n")
        f.write(f"\n## Outcome classification: {summary['outcome_code']}\n\n{summary['outcome_text']}\n\n")
        f.write("## Query-change preservation\n\n")
        qc = summary["query_change"]
        f.write(f"mean E_delta (conditioned) = {qc['mean_e_delta_conditioned']}; "
                 f"mean E_delta (agnostic) = {qc['mean_e_delta_agnostic']}; "
                 f"mean E_delta (static zero-change baseline) = {qc['mean_e_delta_static_baseline']}; "
                 f"conditioned beats the static baseline on {qc['conditioned_beats_static_frac_mean']} of "
                 f"pairs; agnostic beats it on {qc['agnostic_beats_static_frac_mean']} of pairs.\n\n")
        f.write("## Limitations\n\n")
        f.write("This is a diagnostic study, not a new compressed retrieval algorithm. The "
                 "controller_train/val query splits reused here have been used repeatedly across every "
                 "prior CRISP and Y-series experiment in this project, so comparisons remain exploratory, "
                 "not independently confirmatory. The cluster sample is rho_j-stratified and bounded, not "
                 "the full cluster population. Phase B's region analysis uses explicit measurable "
                 "proxies, not full topological recovery. Deliverables beyond the primary named set "
                 "(cluster_composition_results.csv, token_region_results.csv, "
                 "distortion_curve_results.csv) are included for full traceability, not to hit a "
                 "specific file count.\n\n")


def make_plots(all_distortion, all_retrieval, all_allocation, all_token_geom, all_query_change):
    import pandas as pd
    dist_df = pd.DataFrame(all_distortion)
    ret_df = pd.DataFrame(all_retrieval)
    alloc_df = pd.DataFrame(all_allocation)
    tok_df = pd.DataFrame(all_token_geom)

    if len(dist_df):
        fig, ax = plt.subplots(figsize=(6, 4))
        rs = [r for r in COMPONENT_GRID if f"distortion_r{r}" in dist_df.columns]
        for r in rs:
            col = f"distortion_r{r}"
            ax.scatter([r] * len(dist_df), dist_df[col], alpha=0.3, s=10)
        means = [dist_df[f"distortion_r{r}"].mean() for r in rs]
        ax.plot(rs, means, "k-o", label="mean")
        ax.set_xlabel("components r"); ax.set_ylabel("mean squared distortion"); ax.set_yscale("log")
        ax.set_title("Within-token geometric distortion vs. component count")
        ax.legend(); fig.tight_layout()
        fig.savefig(os.path.join(PLOT_DIR, "distortion_curves.png")); plt.close(fig)

    if len(ret_df):
        fig, ax = plt.subplots(figsize=(6, 4))
        labels = ["all-member", "top-m"]
        cond_means = [ret_df["tv_all_member_conditioned"].mean(), ret_df["tv_topm_conditioned"].mean()]
        agn_means = [ret_df["tv_all_member_agnostic"].mean(), ret_df["tv_topm_agnostic"].mean()]
        x = np.arange(len(labels)); w = 0.35
        ax.bar(x - w / 2, cond_means, w, label="label-conditioned")
        ax.bar(x + w / 2, agn_means, w, label="label-agnostic")
        ax.set_xticks(x); ax.set_xticklabels(labels); ax.set_ylabel("mean TV (lower is better)")
        ax.set_title("Predictive fidelity: label-conditioned vs. label-agnostic")
        ax.legend(); fig.tight_layout()
        fig.savefig(os.path.join(PLOT_DIR, "tv_comparison.png")); plt.close(fig)

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar(["label-conditioned", "label-agnostic"],
               [ret_df["candidate_recall_topm_conditioned"].mean(), ret_df["candidate_recall_topm_agnostic"].mean()])
        ax.set_ylabel("mean candidate-set Jaccard recall")
        ax.set_title("Top-m candidate-set fidelity")
        fig.tight_layout(); fig.savefig(os.path.join(PLOT_DIR, "candidate_recall.png")); plt.close(fig)

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar(["label-conditioned", "label-agnostic"],
               [ret_df["bytes_conditioned"].mean(), ret_df["bytes_agnostic"].mean()])
        ax.set_ylabel("mean bytes at matched component count")
        ax.set_title("Storage cost (label-agnostic pays extra for token histograms)")
        fig.tight_layout(); fig.savefig(os.path.join(PLOT_DIR, "storage_bytes.png")); plt.close(fig)

    if len(alloc_df):
        fig, ax = plt.subplots(figsize=(6, 4))
        means = alloc_df.groupby("rule")["tv_all_member"].mean()
        ax.bar(means.index, means.values)
        ax.set_ylabel("mean all-member TV at fixed total budget")
        ax.set_title("Capacity allocation rule comparison")
        fig.tight_layout(); fig.savefig(os.path.join(PLOT_DIR, "capacity_allocation.png")); plt.close(fig)

    if len(tok_df):
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(tok_df["compactness_ratio"].dropna(), bins=20)
        ax.axvline(1.0, color="red", linestyle="--", label="equal to random baseline")
        ax.set_xlabel("compactness ratio (token-group / random-group mean distance)")
        ax.set_title("Token-conditioned compactness vs. random same-size baseline")
        ax.legend(); fig.tight_layout()
        fig.savefig(os.path.join(PLOT_DIR, "compactness_ratio_hist.png")); plt.close(fig)

    if all_query_change:
        cond_vals = [r["conditioned"]["mean_e_delta_model"] for r in all_query_change if r["conditioned"]]
        agn_vals = [r["agnostic"]["mean_e_delta_model"] for r in all_query_change if r["agnostic"]]
        static_vals = [r["conditioned"]["mean_e_delta_static_baseline"] for r in all_query_change if r["conditioned"]]
        if cond_vals:
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.bar(["conditioned", "agnostic", "static baseline"],
                   [np.mean(cond_vals), np.mean(agn_vals) if agn_vals else 0, np.mean(static_vals)])
            ax.set_ylabel("mean E_delta (lower is better)")
            ax.set_title("Query-change preservation vs. zero-change static baseline")
            fig.tight_layout(); fig.savefig(os.path.join(PLOT_DIR, "query_change.png")); plt.close(fig)


if __name__ == "__main__":
    main()
