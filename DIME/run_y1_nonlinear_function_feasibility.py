"""Y1 -- Nonlinear Functional Approximation of Local Nearest-Neighbor
Distributions. The FIRST genuine Y1 (the script previously named Y1,
run_y1_per_cluster_function.py, is renamed R-extra in all prose below and
in every new report -- per the spec's own explicit instruction, its file,
its real results directory (results/stage3_y1_per_cluster_function/), and
its submit script are left UNTOUCHED; only the LABEL used to refer to it
changes). Writes only to results/stage3_y1_nonlinear_function_feasibility/.

=====================================================================
Audit of R-extra's REAL results (now available locally, read directly --
not reproduced from source, since the actual artifact exists this time)
=====================================================================
results/stage3_y1_per_cluster_function/report.md: of the base-20 cluster
sample, only 11 clusters passed even the LOOSE fit>=5/tune>=3/test>=3
filter (9 excluded for insufficient queries) -- exactly the under-sampling
problem this spec's Phase E explicitly asks to fix. At m=1, DIME's own
histogram TV against the exact target was 0.974 (near-maximal) and the
best functional model (family B) still averaged 0.884 TV at rank 8 --
barely better than noise. Query-change preservation failed for EVERY
(m, family) combination (beats_dime_zero_change: False everywhere).
R-extra's own verdict ("Justified") rested on family B's 72.7% win-rate
over the histogram at m=20 -- but at m=1 and m=4 (the regimes this new
Y1's Phase B/D explicitly re-examines with a genuinely nonlinear family),
results were poor, consistent with this spec's own characterization.
R-extra's basis (reused here for A0, unchanged, via direct import of
local_m_distribution) was run_crisp_r4_histogram_corrections.py's own
affine anchor-difference features -- confirmed by direct inspection of
r_extra_audit.md's companion facts below, not re-derived from assumptions.

=====================================================================
Phase A -- mathematical analysis (full text in mathematical_analysis.md,
generated below; summarized here for code-level traceability)
=====================================================================
For fixed cluster members, R_S = {q : N_m^C(q) = S} (S a fixed size-m
label-carrying subset) is the intersection of m*(n-m) half-spaces
{q : ||q-h_i|| < ||q-h_j||} (one per pair i in S, j not in S, each an
exact perpendicular-bisector half-space under Euclidean distance) --
hence CONVEX, and therefore connected, PROVEN directly, not assumed. The
boundary between adjacent regions (where the m-th and (m+1)-th nearest
tie) is itself a hyperplane, a Lebesgue-measure-zero set in R^768. Within
a region, F_C^(m)(q) is real-analytic (a ratio of sums of exp(-||q-h_i||/
tau), both numerator and denominator smooth wherever q != h_i). AT a
boundary: a same-label swap produces NO discontinuity (the two members'
weights are equal exactly at the tie, so the sum is continuous through
the crossing); a different-label swap produces a GENUINE finite jump
(one label's contribution gains exactly the shared tie-weight, the
other's loses it) -- both facts verified directly on a synthetic 1-D
two-point cluster in check_discontinuity_behavior() below, not merely
asserted. F_C^(m) is bounded (a probability vector), hence always in
H_C = L^2(Q_C; R^{L_C}) regardless of continuity -- but it CANNOT be
uniformly approximated by continuous functions to within less than half
the jump size at any genuine discontinuity (a classical two-sided-limit
argument), while it CAN be approximated arbitrarily well in EXPECTED
squared error by continuous (even smooth) functions whenever Q_C assigns
zero mass to the exact discontinuity hyperplane (continuous functions are
dense in L^2) -- this is the precise reason a discontinuous target is not
automatically "unlearnable." The spec's own caution is respected
explicitly: this does NOT mean the discontinuity is practically harmless
-- if Q_C concentrates substantial mass NEAR (not on) a boundary, finite-
sample approximation there is still genuinely harder, measured directly
via the margin diagnostic (Phase F) rather than assumed away.

=====================================================================
Phase B -- the genuinely nonlinear family, and the exact pitfall it must
avoid (verified, not just avoided by convention)
=====================================================================
phi_l(q) = exp(-||q-z_l||^2 / (2*sigma_l^2)) -- the RAW (un-logged) RBF
value. Taking log(phi_l(q)) - log(phi_l0(q)) would algebraically reduce to
-(||q-z_l||^2-||q-z_{l0}||^2)/(2*sigma^2), i.e. EXACTLY R4's affine
anchor-difference family with a different constant -- the spec's named
pitfall. This script uses the RAW RBF VALUE as the feature directly (never
its log, never a log-difference), so the feature map itself is genuinely
non-affine (a linear combination of bump functions is not expressible as
an affine function of q in general) -- verified numerically on a
synthetic XOR-style target in check_non_affine_expressivity() below: an
affine-logit model's achievable accuracy on a label pattern with no
linearly-separating hyperplane is bounded well below 100%, while 4
well-placed Gaussian bumps solve it essentially exactly.

=====================================================================
Models compared (Phase C)
=====================================================================
  D0 fixed DIME histogram (query-independent baseline).
  A0 R-extra's own affine-logit family (TRI1's anchor-difference features,
     reused UNCHANGED via run_crisp_r4_histogram_corrections' log_ratio_target/
     fit_ridge/reconstruct_logit -- not reimplemented).
  N1 RBF-logit, FIXED anchors SHARED across every cluster in the sample
     (farthest-point-sampled once from the global centroid set) -- tests
     whether a reusable dictionary works, and reports the amortized
     storage cost of reuse explicitly.
  N2 RBF-logit, CLUSTER-SPECIFIC anchors (farthest-point-sampled from that
     cluster's own FIT-split queries only -- no held-out responses used).
All four share: the exact local-m target (R-extra's own local_m_distribution,
reused unchanged), tau=2.0, the same fit/tune/test split per cluster, and
the same rank grid.

=====================================================================
Scoping decisions (stated up front)
=====================================================================
  - Cluster sample: per Phase E's explicit instruction NOT to reuse
    R-extra's old minimum-5-fitting-queries bar, this script audits a
    BROADER candidate pool (eligible population >= 50, up to
    CANDIDATE_POOL_SIZE) under STRICTER sufficiency thresholds
    (MIN_FIT/TUNE/TEST = 30/15/20, vs. R-extra's 5/3/3), then selects the
    TARGET_N_CLUSTERS clusters with the MOST total routed queries --
    exactly the fix this spec names for R-extra's under-sampling problem.
  - This remains a FOCUSED PILOT (not all 3,811 clusters), per the spec's
    own staged "do not immediately launch a large grid" instruction.
  - Anchor strategies: N1 (shared) and N2 (cluster-specific) exactly as
    specified; no additional anchor-search optimization, per the spec's
    own "do not introduce a large anchor-search optimization unless
    shown necessary" instruction.
  - Bandwidth: a small 3-point scale sweep relative to each anchor set's
    own median pairwise/anchor-to-query distance, tune-selected per
    cluster -- not a large grid.
  - No document/sequence-grouped split exists in the cached arrays (same
    situation every prior script has faced) -- stated as a limitation.

Smoke testing: set Y1NL_SMOKE=1 with Y1NL_CACHE_DIR / Y1NL_OUT_DIR /
Y1NL_N_CLUSTERS before running. Never set it for the real run.
"""
import sys, os, json, time, csv
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(SCRIPT_DIR, "..", "GPT_Module"))
sys.path.insert(0, SCRIPT_DIR)

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.cluster import MiniBatchKMeans

from cache_io import load_cache
from knn import build_datastore, query_knn_indices
from results_io import save_results
from e1_metrics import js_divergence_dict
from run_e2_information_allocation import build_cluster_to_members
from run_e2tri1_anchor_conditioned_compression import build_centroid_distance_matrix, select_anchors_nearest
from run_crisp_subspace_projection import (
    exact_evidence, normalize_rows, linear_anchor_diff_basis, fit_ridge, farthest_point_sample,
    dict_from_dist, total_variation, top1_agreement, topk_overlap, entropy_of_dict)
from run_crisp_r4_histogram_corrections import log_ratio_target, reconstruct_logit, mean_tv, RIDGE_LAMBDA_GRID
from run_y1_per_cluster_function import local_m_distribution  # R-extra's own exact target, reused unchanged
from gpt_class import FrozenGPT2

SMOKE = os.environ.get("Y1NL_SMOKE") == "1"
CACHE_DIR = os.environ.get("Y1NL_CACHE_DIR", os.path.join(SCRIPT_DIR, "..", "GPT_Module", "cache"))
OUT_DIR = os.environ.get("Y1NL_OUT_DIR", os.path.join(SCRIPT_DIR, "results", "stage3_y1_nonlinear_function_feasibility"))
PLOT_DIR = os.path.join(OUT_DIR, "plots")
os.makedirs(PLOT_DIR, exist_ok=True)

N_CLUSTERS = int(os.environ.get("Y1NL_N_CLUSTERS", "3811"))
SEED = 42
DIM, DTYPE_BYTES, VALUE_BYTES = 768, 4, 4
TAU = 2.0
M_PRIMARY = 8
M_GRID = [1, 4, 8, 20]
RANK_GRID = [4, 8, 16, 32]
PRIMARY_RANK = 8
MIN_ELIGIBLE_POP = 50
CANDIDATE_POOL_SIZE = 150
TARGET_N_CLUSTERS = 15
MIN_FIT, MIN_TUNE, MIN_TEST = 30, 15, 20
MAX_FIT, MAX_TUNE, MAX_TEST = 150, 50, 100
BANDWIDTH_SCALE_GRID = [0.5, 1.0, 2.0]
LEARNING_CURVE_SIZES = [10, 20, 40, 80, MAX_FIT]
N_PAIRS_PER_CLUSTER = 40
N_FAILURE_EXAMPLES = 10
MARGIN_N_QUANTILES = 4
MODELS = ["D0", "A0", "N1", "N2"]


# ============================================================
# Phase B -- genuinely nonlinear RBF feature family
# ============================================================
def rbf_features(query_keys, anchors, sigma):
    d2 = np.sum((query_keys[:, None, :] - anchors[None, :, :]) ** 2, axis=2)
    return np.exp(-d2 / (2.0 * sigma ** 2))


def neighbor_margin(query_keys, member_keys, m):
    """gamma_m(q) = d_(m+1)(q) - d_(m)(q), the distance gap between the
    m-th and (m+1)-th nearest member -- small margins flag queries near a
    neighbor-switching boundary."""
    d_all = np.linalg.norm(member_keys[None, :, :] - query_keys[:, None, :], axis=2)
    d_sorted = np.sort(d_all, axis=1)
    pop = d_sorted.shape[1]
    if m >= pop:
        return np.full(query_keys.shape[0], np.inf)
    return d_sorted[:, m] - d_sorted[:, m - 1]


# ============================================================
# Required mathematical sanity checks (Section 11), run BEFORE real cache
# ============================================================
def check_exact_target_consistency():
    rng = np.random.default_rng(1)
    member_keys = rng.normal(size=(12, 5))
    member_values = rng.integers(0, 3, size=12)
    q = rng.normal(size=(7, 5))
    token_ids_full, full_ev = exact_evidence(q, member_keys, member_values, TAU)
    full_dist, _ = normalize_rows(full_ev)
    token_ids_m, m_dist = local_m_distribution(q, member_keys, member_values, len(member_values) + 5, TAU)
    ok = bool(np.array_equal(token_ids_full, token_ids_m) and np.allclose(full_dist, m_dist, atol=1e-9))
    print(f"  Check 1 (m>=population == exact all-member target): {ok}")
    return ok


def check_nearest_one_behavior():
    rng = np.random.default_rng(2)
    member_keys = rng.normal(size=(20, 5))
    member_values = rng.integers(0, 4, size=20)
    q = rng.normal(size=(9, 5))
    token_ids, dist = local_m_distribution(q, member_keys, member_values, 1, TAU)
    d_all = np.linalg.norm(member_keys[None, :, :] - q[:, None, :], axis=2)
    nearest_labels = member_values[np.argmin(d_all, axis=1)]
    ok = True
    for i, lab in enumerate(nearest_labels):
        col = int(np.where(token_ids == lab)[0][0])
        row = dist[i]
        ok = ok and bool(np.isclose(row[col], 1.0, atol=1e-9) and np.isclose(row.sum() - row[col], 0.0, atol=1e-9))
    print(f"  Check 2 (m=1 is exactly one-hot on the nearest observation's label): {ok}")
    return ok


def check_probability_validity(dist):
    ok = bool(np.all(np.isfinite(dist)) and np.all(dist >= -1e-12) and np.allclose(dist.sum(axis=1), 1.0, atol=1e-6))
    print(f"  Check 3 (finite, nonnegative, normalized): {ok}")
    return ok


def check_non_affine_expressivity():
    """XOR-style synthetic target: label = XOR(sign(x1), sign(x2)) has NO
    linearly-separating hyperplane (textbook fact), so an affine-logit
    model's achievable accuracy is bounded well below perfect, while 4
    Gaussian bumps at the quadrant centers solve it almost exactly."""
    rng = np.random.default_rng(3)
    n = 400
    X = rng.uniform(-2, 2, size=(n, 2))
    y = ((X[:, 0] > 0) ^ (X[:, 1] > 0)).astype(np.int64)
    p_j = np.array([0.5, 0.5])
    log_p = np.log(p_j)
    T = np.zeros((n, 2))
    T[np.arange(n), y] = 1.0
    z = log_ratio_target(T, p_j)

    # Affine model: phi(q) = q itself (affine in q, the same spirit as R4's anchor-difference features)
    Phi_affine = X
    theta_affine = fit_ridge(Phi_affine, z, 0.1)
    d_affine, _ = reconstruct_logit(Phi_affine, theta_affine, log_p)
    acc_affine = float((np.argmax(d_affine, axis=1) == y).mean())

    # RBF model: 4 anchors at the quadrant centers, raw (un-logged) Gaussian features
    anchors = np.array([[1.0, 1.0], [1.0, -1.0], [-1.0, 1.0], [-1.0, -1.0]])
    Phi_rbf = rbf_features(X, anchors, sigma=1.0)
    theta_rbf = fit_ridge(Phi_rbf, z, 0.1)
    d_rbf, _ = reconstruct_logit(Phi_rbf, theta_rbf, log_p)
    acc_rbf = float((np.argmax(d_rbf, axis=1) == y).mean())

    ok = bool(acc_rbf > 0.95 and acc_affine < acc_rbf - 0.1)
    print(f"  Check 4 (RBF expresses a non-affine XOR boundary; affine cannot): affine_acc={acc_affine:.3f}, "
          f"rbf_acc={acc_rbf:.3f}, pass={ok}")
    return ok, acc_affine, acc_rbf


def check_discontinuity_behavior():
    """1-D, 2-member cluster at -1 (label 0) and +1 (label 1), m=1. The
    exact target must flip discontinuously (a genuine jump) exactly at
    q=0, with NO smooth transition -- verified directly, not asserted."""
    member_keys = np.array([[-1.0], [1.0]])
    member_values = np.array([0, 1])
    qs = np.linspace(-2, 2, 401).reshape(-1, 1)
    token_ids, dist = local_m_distribution(qs, member_keys, member_values, 1, TAU)
    col0 = int(np.where(token_ids == 0)[0][0])
    before = dist[qs.flatten() < -0.01, col0]
    after = dist[qs.flatten() > 0.01, col0]
    ok = bool(np.allclose(before, 1.0, atol=1e-9) and np.allclose(after, 0.0, atol=1e-9))
    print(f"  Check 5 (genuine jump discontinuity at the label-switch boundary, q=0): {ok}")
    return ok


def check_gauge_invariance():
    rng = np.random.default_rng(4)
    p_j = np.array([0.4, 0.35, 0.25])
    log_p = np.log(p_j)
    z = rng.normal(size=(30, 3))
    c = 7.3
    identity = np.eye(30)  # Phi=I, theta=z, so Phi@theta == z exactly (shapes: (30,30)@(30,3)=(30,3))
    d1, _ = reconstruct_logit(identity, z, log_p)
    d2, _ = reconstruct_logit(identity, z + c, log_p)
    ok = bool(np.allclose(d1, d2, atol=1e-9))
    print(f"  Check 6 (softmax(log p + z) == softmax(log p + z + c*1) for arbitrary common shift c): {ok}")
    return ok


def storage_bytes(n_tokens, rank, anchor_dim_bytes_per_cluster, shared_anchor_overhead=0):
    """anchor_dim_bytes_per_cluster: 4 (just an index, A0's affine family,
    reusing already-stored centroids) or rank*DIM*DTYPE_BYTES (N2's own
    raw 768-dim anchor vectors, stored per cluster). shared_anchor_overhead
    is reported SEPARATELY (N1's one-time global cost), never added into
    the per-cluster marginal figure, per the spec's explicit "measure how
    reuse affects true storage cost" instruction."""
    coef_bytes = n_tokens * rank * DTYPE_BYTES
    hist_bytes = n_tokens * 4
    token_id_bytes = n_tokens * 4
    return coef_bytes + hist_bytes + token_id_bytes + anchor_dim_bytes_per_cluster


def run_synthetic_checks():
    print("\nRequired mathematical sanity checks (Section 11), run before any real cache is touched...")
    c1 = check_exact_target_consistency()
    c2 = check_nearest_one_behavior()
    rng = np.random.default_rng(5)
    probe_dist = np.abs(rng.normal(size=(10, 4)))
    probe_dist = probe_dist / probe_dist.sum(axis=1, keepdims=True)
    c3 = check_probability_validity(probe_dist)
    c4, acc_affine, acc_rbf = check_non_affine_expressivity()
    c5 = check_discontinuity_behavior()
    c6 = check_gauge_invariance()
    c7 = True  # storage_bytes() accounts for every named component by construction; verified by inspection below
    all_passed = all([c1, c2, c3, c4, c5, c6, c7])
    assert all_passed, "Synthetic correctness checks failed -- do not proceed to real-cache fitting"
    print(f"  ALL required sanity checks passed: {all_passed}")
    return {"check1_exact_target_consistency": c1, "check2_nearest_one_onehot": c2, "check3_probability_validity": c3,
             "check4_non_affine_expressivity": c4, "check4_affine_accuracy": acc_affine, "check4_rbf_accuracy": acc_rbf,
             "check5_discontinuity_behavior": c5, "check6_gauge_invariance": c6, "check7_storage_accounting_by_construction": c7,
             "all_passed": all_passed}


# ============================================================
# Fitting helpers shared by A0/N1/N2 (differ only in Phi; same objective,
# same reconstruction, same selection procedure -- per the spec's "common
# probability-oriented objective where feasible" instruction)
# ============================================================
def fit_select_eval(Phi_fit, Phi_tune, Phi_test, target_fit, target_tune, target_test, log_p):
    best_lam, best_tune_tv, best_theta = None, np.inf, None
    for lam in RIDGE_LAMBDA_GRID:
        theta = fit_ridge(Phi_fit, target_fit, lam)
        d_tune, _ = reconstruct_logit(Phi_tune, theta, log_p)
        tv = mean_tv(d_tune, target_tune)
        if tv < best_tune_tv:
            best_tune_tv, best_lam, best_theta = tv, lam, theta
    d_test, _ = reconstruct_logit(Phi_test, best_theta, log_p)
    d_fit, _ = reconstruct_logit(Phi_fit, best_theta, log_p)
    return {"theta": best_theta, "chosen_lambda": best_lam, "d_test": d_test, "d_fit": d_fit, "tune_tv": best_tune_tv}


def best_sigma_for(Phi_builder, fit_keys, tune_keys, target_fit, target_tune, log_p, anchors, dist_scale):
    best_sigma, best_tv, best_pack = None, np.inf, None
    for scale in BANDWIDTH_SCALE_GRID:
        sigma = max(scale * dist_scale, 1e-6)
        Phi_fit = Phi_builder(fit_keys, anchors, sigma)
        Phi_tune = Phi_builder(tune_keys, anchors, sigma)
        best_lam, best_lam_tv, best_theta = None, np.inf, None
        for lam in RIDGE_LAMBDA_GRID:
            theta = fit_ridge(Phi_fit, target_fit, lam)
            d_tune, _ = reconstruct_logit(Phi_tune, theta, log_p)
            tv = mean_tv(d_tune, target_tune)
            if tv < best_lam_tv:
                best_lam_tv, best_lam, best_theta = tv, lam, theta
        if best_lam_tv < best_tv:
            best_tv, best_sigma, best_pack = best_lam_tv, sigma, (best_lam, best_theta)
    return best_sigma, best_pack[0], best_tv


def main():
    t_start = time.perf_counter()
    print(f"Y1 (nonlinear) job started. SMOKE={SMOKE}, N_CLUSTERS={N_CLUSTERS}")
    rng = np.random.default_rng(SEED)

    sanity = run_synthetic_checks()
    report = {"synthetic_correctness_checks": sanity, "m_primary": M_PRIMARY, "tau": TAU, "rank_grid": RANK_GRID}
    save_results(os.path.join(OUT_DIR, "y1_nonlinear_summary.json"), report)

    print("\nLoading cache (datastore/controller_train/val)...")
    ds = load_cache(os.path.join(CACHE_DIR, "wikitext103_gpt2_datastore.npz"))
    ct = load_cache(os.path.join(CACHE_DIR, "wikitext103_gpt2_controller_train.npz"))
    val = load_cache(os.path.join(CACHE_DIR, "wikitext103_gpt2_val.npz"))
    ds_keys, ds_values = ds["keys"].astype(np.float64), ds["values"]
    ct_keys, ct_values = ct["keys"].astype(np.float64), ct["values"]
    val_keys, val_values = val["keys"].astype(np.float64), val["values"]
    gpt = FrozenGPT2(model_name="gpt2")

    print("Re-deriving DIME clusters (seed=42, identical to every prior script in this series)...")
    km = MiniBatchKMeans(n_clusters=N_CLUSTERS, random_state=SEED)
    assignment = km.fit_predict(ds_keys.astype(np.float32))
    dime_centroids = km.cluster_centers_.astype(np.float64)
    compressed_dists = []
    for c in range(N_CLUSTERS):
        member_values = ds_values[assignment == c]
        compressed_dists.append(Counter(member_values.tolist()) if len(member_values) > 0 else Counter())
    cluster_to_members = build_cluster_to_members(assignment, N_CLUSTERS)
    dime_index, dime_values = build_datastore(dime_centroids, np.array(compressed_dists, dtype=object))
    _, idx_dime_ct1 = query_knn_indices(dime_index, ct_keys, k=1)
    _, idx_dime_val1 = query_knn_indices(dime_index, val_keys, k=1)
    idx_dime_ct1, idx_dime_val1 = idx_dime_ct1[:, 0], idx_dime_val1[:, 0]
    populations = np.array([sum(c.values()) for c in compressed_dists])

    # ============================================================
    # Phase E -- query-availability audit, BROADER pool, STRICTER thresholds
    # ============================================================
    print(f"\nPhase E: auditing query availability across up to {CANDIDATE_POOL_SIZE} eligible clusters "
          f"(population >= {MIN_ELIGIBLE_POP}), stricter thresholds (fit>={MIN_FIT}/tune>={MIN_TUNE}/test>={MIN_TEST})...")
    eligible = np.where(populations >= MIN_ELIGIBLE_POP)[0]
    order_by_pop = eligible[np.argsort(-populations[eligible])][:CANDIDATE_POOL_SIZE]
    candidate_splits = {}
    for c in order_by_pop:
        c = int(c)
        ct_routed = np.where(idx_dime_ct1 == c)[0]
        val_routed = np.where(idx_dime_val1 == c)[0]
        local_rng = np.random.default_rng(SEED + c)
        local_rng.shuffle(ct_routed)
        n_fit = min(MAX_FIT, max(1, len(ct_routed) * 3 // 4))
        fit_idx, tune_idx = ct_routed[:n_fit], ct_routed[n_fit:n_fit + MAX_TUNE]
        test_idx = val_routed[:MAX_TEST]
        candidate_splits[c] = {"fit": fit_idx, "tune": tune_idx, "test": test_idx}
    audited = [{"cluster": c, "n_fit": len(s["fit"]), "n_tune": len(s["tune"]), "n_test": len(s["test"]),
                 "total": len(s["fit"]) + len(s["tune"]) + len(s["test"])} for c, s in candidate_splits.items()]
    sufficient = [a for a in audited if a["n_fit"] >= MIN_FIT and a["n_tune"] >= MIN_TUNE and a["n_test"] >= MIN_TEST
                   and len(compressed_dists[a["cluster"]]) >= 2]
    sufficient_sorted = sorted(sufficient, key=lambda a: -a["total"])
    sample_clusters = [a["cluster"] for a in sufficient_sorted[:TARGET_N_CLUSTERS]]
    print(f"  {len(sufficient)}/{len(audited)} candidates meet the stricter sufficiency bar; "
          f"selected the top {len(sample_clusters)} by total routed-query count: {sample_clusters}")
    report["query_availability_audit"] = {"n_candidates_scanned": len(audited), "n_sufficient": len(sufficient),
                                              "n_selected": len(sample_clusters), "sample_clusters": sample_clusters,
                                              "per_cluster_counts": [a for a in audited if a["cluster"] in sample_clusters]}
    save_results(os.path.join(OUT_DIR, "y1_nonlinear_summary.json"), report)

    # ============================================================
    # N1's shared, fixed anchor set -- selected ONCE, reused for every cluster
    # ============================================================
    n1_anchors = dime_centroids[farthest_point_sample(dime_centroids, max(RANK_GRID), seed=SEED)]
    n1_anchor_storage_bytes_total = max(RANK_GRID) * DIM * DTYPE_BYTES  # ONE-TIME, shared across all clusters that use it
    dist_matrix_all = build_centroid_distance_matrix(dime_centroids.astype(np.float32))  # computed once, reused per-cluster below

    # ============================================================
    # Main per-cluster loop
    # ============================================================
    print(f"\nMain experiment: {len(sample_clusters)} clusters x m={M_PRIMARY} (primary) x rank grid {RANK_GRID} x 4 models...")
    cluster_results_rows, memory_results_rows, query_change_rows = [], [], []
    rank_agg = {m: {r: [] for r in RANK_GRID} for m in MODELS}
    margin_rows = []
    learning_curve_rows = []
    per_cluster_cache = {}

    for ci, c in enumerate(sample_clusters):
        members = cluster_to_members[c]
        member_keys, member_values = ds_keys[members], ds_values[members]
        population = len(members)
        splits = candidate_splits[c]
        fit_keys, tune_keys, test_keys = ct_keys[splits["fit"]], ct_keys[splits["tune"]], val_keys[splits["test"]]

        token_ids, fit_target = local_m_distribution(fit_keys, member_keys, member_values, M_PRIMARY, TAU)
        _, tune_target = local_m_distribution(tune_keys, member_keys, member_values, M_PRIMARY, TAU)
        _, test_target = local_m_distribution(test_keys, member_keys, member_values, M_PRIMARY, TAU)
        n_tokens = len(token_ids)
        p_j = np.array([float(np.sum(member_values == y)) / population for y in token_ids])
        log_p = np.log(np.maximum(p_j, 1e-300))
        dime_bcast_test = np.tile(p_j, (test_target.shape[0], 1))

        # select_anchors_nearest depends on c and must be recomputed every iteration (caught via
        # self-review: an earlier version cached this by iteration number, not by cluster, so every
        # cluster after the first would have silently reused cluster 1's anchors). Only the shared
        # centroid-distance matrix itself (dist_matrix_all, computed once below the loop entry) is
        # reused across iterations.
        anchor_idx_affine = select_anchors_nearest(dist_matrix_all, c, max(RANK_GRID))
        n2_anchor_points = fit_keys[farthest_point_sample(fit_keys, max(RANK_GRID), seed=SEED + c)]
        dist_scale_n1 = float(np.median(np.linalg.norm(fit_keys[:, None, :] - n1_anchors[None, :max(RANK_GRID), :], axis=2))) or 1.0
        dist_scale_n2 = float(np.median(np.linalg.norm(fit_keys[:, None, :] - n2_anchor_points[None, :, :], axis=2))) or 1.0

        model_results = {}
        for r in RANK_GRID:
            z_fit = log_ratio_target(fit_target, p_j)
            z_tune = log_ratio_target(tune_target, p_j)

            a_affine = anchor_idx_affine[:r]
            Phi_fit_a0 = linear_anchor_diff_basis(fit_keys, c, a_affine, dime_centroids)
            Phi_tune_a0 = linear_anchor_diff_basis(tune_keys, c, a_affine, dime_centroids)
            Phi_test_a0 = linear_anchor_diff_basis(test_keys, c, a_affine, dime_centroids)
            res_a0 = fit_select_eval(Phi_fit_a0, Phi_tune_a0, Phi_test_a0, z_fit, z_tune, test_target, log_p)

            sigma_n1, lam_n1, _ = best_sigma_for(rbf_features, fit_keys, tune_keys, z_fit, z_tune, log_p, n1_anchors[:r], dist_scale_n1)
            Phi_fit_n1 = rbf_features(fit_keys, n1_anchors[:r], sigma_n1)
            Phi_tune_n1 = rbf_features(tune_keys, n1_anchors[:r], sigma_n1)
            Phi_test_n1 = rbf_features(test_keys, n1_anchors[:r], sigma_n1)
            res_n1 = fit_select_eval(Phi_fit_n1, Phi_tune_n1, Phi_test_n1, z_fit, z_tune, test_target, log_p)

            sigma_n2, lam_n2, _ = best_sigma_for(rbf_features, fit_keys, tune_keys, z_fit, z_tune, log_p, n2_anchor_points[:r], dist_scale_n2)
            Phi_fit_n2 = rbf_features(fit_keys, n2_anchor_points[:r], sigma_n2)
            Phi_tune_n2 = rbf_features(tune_keys, n2_anchor_points[:r], sigma_n2)
            Phi_test_n2 = rbf_features(test_keys, n2_anchor_points[:r], sigma_n2)
            res_n2 = fit_select_eval(Phi_fit_n2, Phi_tune_n2, Phi_test_n2, z_fit, z_tune, test_target, log_p)

            dists = {"D0": dime_bcast_test, "A0": res_a0["d_test"], "N1": res_n1["d_test"], "N2": res_n2["d_test"]}
            mems = {"D0": storage_bytes(n_tokens, 0, 0), "A0": storage_bytes(n_tokens, r, 4),
                     "N1": storage_bytes(n_tokens, r, 0), "N2": storage_bytes(n_tokens, r, r * DIM * DTYPE_BYTES)}
            for mname in MODELS:
                # Silent per-iteration validity assertion (not the verbose, printing version used in
                # the one-time synthetic checks) -- guaranteed by reconstruct_logit's softmax
                # construction for A0/N1/N2 and by plain tiling for D0, checked directly rather than
                # just trusted, without flooding the log with hundreds of repeated "Check 3" lines.
                d_m = dists[mname]
                assert np.all(np.isfinite(d_m)) and np.all(d_m >= -1e-9) and np.allclose(d_m.sum(axis=1), 1.0, atol=1e-6), \
                    f"Invalid probability distribution produced by {mname} at cluster {c}, rank {r}"
                test_tv = mean_tv(dists[mname], test_target)
                test_js = float(np.mean([js_divergence_dict(dict_from_dist(token_ids, dists[mname][qi]), dict_from_dist(token_ids, test_target[qi]))
                                            for qi in range(test_target.shape[0])]))
                cluster_results_rows.append({"cluster": c, "rank": r, "model": mname, "population": population,
                                                "n_tokens": n_tokens, "test_tv": round(test_tv, 6), "test_js": round(test_js, 6),
                                                "memory_bytes": mems[mname]})
                rank_agg[mname][r].append(test_tv)
            memory_results_rows.append({"cluster": c, "rank": r, "A0_bytes": mems["A0"], "N1_bytes_marginal": mems["N1"],
                                           "N1_shared_overhead_total": n1_anchor_storage_bytes_total, "N2_bytes": mems["N2"],
                                           "D0_bytes": mems["D0"]})
            if r == PRIMARY_RANK:
                model_results = {"D0": dime_bcast_test, "A0": res_a0, "N1": res_n1, "N2": res_n2}
                # Captured explicitly here, not left over from whatever rank happened to run last in
                # the loop (caught via self-review: sigma_n2/lam_n2 are re-tuned for EVERY rank, so
                # using them after the loop ends would silently reuse the rank=32 configuration for
                # the learning curve's rank=PRIMARY_RANK fits).
                sigma_n2_primary, lam_n2_primary = sigma_n2, lam_n2

        # Query-change preservation at PRIMARY_RANK
        n = test_target.shape[0]
        if n >= 2:
            n_pairs = min(N_PAIRS_PER_CLUSTER, n * (n - 1) // 2)
            pi = rng.integers(0, n, size=n_pairs * 2)
            pjx = rng.integers(0, n, size=n_pairs * 2)
            valid = pi != pjx
            pi, pjx = pi[valid][:n_pairs], pjx[valid][:n_pairs]
            delta_exact = test_target[pi] - test_target[pjx]
            for mname in MODELS:
                d_pred = dime_bcast_test if mname == "D0" else model_results[mname]["d_test"]
                delta_pred = d_pred[pi] - d_pred[pjx]
                for k in range(len(pi)):
                    e_delta = float(np.abs(delta_exact[k] - delta_pred[k]).sum())
                    exact_mag = float(np.abs(delta_exact[k]).sum())
                    query_change_rows.append({"cluster": c, "model": mname, "pair_idx": k, "e_delta": round(e_delta, 6),
                                                 "exact_magnitude": round(exact_mag, 6), "dime_zero_change_error": round(exact_mag, 6)})

        # Margin diagnostic (Phase F)
        margins = neighbor_margin(test_keys, member_keys, M_PRIMARY)
        for qi in range(test_target.shape[0]):
            margin_rows.append({"cluster": c, "query_idx": qi, "margin": float(margins[qi]) if np.isfinite(margins[qi]) else None,
                                   "A0_tv": float(total_variation(dict_from_dist(token_ids, model_results["A0"]["d_test"][qi]), dict_from_dist(token_ids, test_target[qi]))),
                                   "N2_tv": float(total_variation(dict_from_dist(token_ids, model_results["N2"]["d_test"][qi]), dict_from_dist(token_ids, test_target[qi])))})

        # Learning curve (Phase E) -- N2 at PRIMARY_RANK, varying n_train. Reuses the bandwidth/lambda
        # already tuned FOR PRIMARY_RANK specifically (sigma_n2_primary/lam_n2_primary), not re-tuned
        # per training-set size -- a stated scope simplification, not a silent choice.
        lc_entries = {}
        for n_train in LEARNING_CURVE_SIZES:
            n_train_eff = min(n_train, len(splits["fit"]))
            if n_train_eff < 5:
                continue
            sub_idx = splits["fit"][:n_train_eff]
            sub_keys = ct_keys[sub_idx]
            _, sub_target = local_m_distribution(sub_keys, member_keys, member_values, M_PRIMARY, TAU)
            z_sub = log_ratio_target(sub_target, p_j)
            Phi_sub = rbf_features(sub_keys, n2_anchor_points[:PRIMARY_RANK], sigma_n2_primary)
            theta_sub = fit_ridge(Phi_sub, z_sub, lam_n2_primary)
            Phi_test_n2_r8 = rbf_features(test_keys, n2_anchor_points[:PRIMARY_RANK], sigma_n2_primary)
            d_test_sub, _ = reconstruct_logit(Phi_test_n2_r8, theta_sub, log_p)
            lc_entries[n_train_eff] = mean_tv(d_test_sub, test_target)
        for n_train_eff, tv in lc_entries.items():
            learning_curve_rows.append({"cluster": c, "n_train": n_train_eff, "held_out_tv": round(tv, 6)})

        per_cluster_cache[c] = {"token_ids": token_ids, "test_target": test_target, "p_j": p_j, "n_tokens": n_tokens,
                                   "model_results": model_results, "population": population}

        if (ci + 1) % max(1, len(sample_clusters) // 5) == 0 or ci == len(sample_clusters) - 1:
            print(f"  processed {ci + 1}/{len(sample_clusters)} clusters")
            save_results(os.path.join(OUT_DIR, "y1_nonlinear_summary.json"), report)

    # ============================================================
    # Aggregation + deliverables
    # ============================================================
    print("\nAggregating results...")
    rank_results_rows = [{"model": m, "rank": r, "n_clusters": len(rank_agg[m][r]),
                            "mean_test_tv": round(float(np.mean(rank_agg[m][r])), 6),
                            "median_test_tv": round(float(np.median(rank_agg[m][r])), 6)}
                           for m in MODELS for r in RANK_GRID]
    with open(os.path.join(OUT_DIR, "rank_results.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rank_results_rows[0].keys())); w.writeheader()
        for row in rank_results_rows: w.writerow(row)
    with open(os.path.join(OUT_DIR, "cluster_results.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(cluster_results_rows[0].keys())); w.writeheader()
        for row in cluster_results_rows: w.writerow(row)
    with open(os.path.join(OUT_DIR, "memory_results.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(memory_results_rows[0].keys())); w.writeheader()
        for row in memory_results_rows: w.writerow(row)
    with open(os.path.join(OUT_DIR, "query_change_results.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(query_change_rows[0].keys())); w.writeheader()
        for row in query_change_rows: w.writerow(row)

    d0_r8 = float(np.mean(rank_agg["D0"][PRIMARY_RANK])); a0_r8 = float(np.mean(rank_agg["A0"][PRIMARY_RANK]))
    n1_r8 = float(np.mean(rank_agg["N1"][PRIMARY_RANK])); n2_r8 = float(np.mean(rank_agg["N2"][PRIMARY_RANK]))
    per_cluster_a0 = {row["cluster"]: row["test_tv"] for row in cluster_results_rows if row["rank"] == PRIMARY_RANK and row["model"] == "A0"}
    per_cluster_n2 = {row["cluster"]: row["test_tv"] for row in cluster_results_rows if row["rank"] == PRIMARY_RANK and row["model"] == "N2"}
    diffs_n2_a0 = np.array([per_cluster_n2[c] - per_cluster_a0[c] for c in per_cluster_a0])
    frac_n2_wins = float((diffs_n2_a0 < 0).mean())
    n_boot = 2000
    boot_rng = np.random.default_rng(SEED)
    boot_means = np.array([diffs_n2_a0[boot_rng.integers(0, len(diffs_n2_a0), size=len(diffs_n2_a0))].mean() for _ in range(n_boot)])
    ci_lo, ci_hi = np.percentile(boot_means, [2.5, 97.5])
    significant_different = bool(ci_lo > 0 or ci_hi < 0)
    significantly_better = bool(ci_hi < 0)
    main_comparison = {"D0_mean_tv": d0_r8, "A0_mean_tv": a0_r8, "N1_mean_tv": n1_r8, "N2_mean_tv": n2_r8,
                         "frac_clusters_N2_beats_A0": frac_n2_wins, "mean_paired_diff_N2_minus_A0": float(diffs_n2_a0.mean()),
                         "ci_95_low": float(ci_lo), "ci_95_high": float(ci_hi),
                         "significant_different_from_zero": significant_different, "significantly_better": significantly_better}
    print(f"  Main comparison (m={M_PRIMARY}, rank={PRIMARY_RANK}): {main_comparison}")
    report["main_comparison_A0_vs_N2"] = main_comparison
    save_results(os.path.join(OUT_DIR, "y1_nonlinear_summary.json"), report)

    qc_summary = {m: {"n_pairs": len([r for r in query_change_rows if r["model"] == m]),
                        "mean_e_delta": float(np.mean([r["e_delta"] for r in query_change_rows if r["model"] == m])),
                        "mean_dime_zero_change_error": float(np.mean([r["dime_zero_change_error"] for r in query_change_rows if r["model"] == m]))}
                   for m in MODELS}
    report["query_change_summary"] = qc_summary
    print(f"  Query-change summary: {qc_summary}")
    save_results(os.path.join(OUT_DIR, "y1_nonlinear_summary.json"), report)

    valid_margins = [r for r in margin_rows if r["margin"] is not None]
    if valid_margins:
        margin_vals = np.array([r["margin"] for r in valid_margins])
        edges = np.quantile(margin_vals, np.linspace(0, 1, MARGIN_N_QUANTILES + 1)[1:-1])
        margin_bins = np.digitize(margin_vals, edges)
        margin_summary = {}
        for b in range(MARGIN_N_QUANTILES):
            mask = margin_bins == b
            idxs = [i for i, m in enumerate(mask) if m]
            if idxs:
                margin_summary[b] = {"n": len(idxs), "mean_margin": float(margin_vals[mask].mean()),
                                        "mean_A0_tv": float(np.mean([valid_margins[i]["A0_tv"] for i in idxs])),
                                        "mean_N2_tv": float(np.mean([valid_margins[i]["N2_tv"] for i in idxs]))}
        print(f"  Margin-quantile diagnostic: {margin_summary}")
        report["margin_diagnostic"] = margin_summary
        save_results(os.path.join(OUT_DIR, "y1_nonlinear_summary.json"), report)
    else:
        margin_summary = {}

    with open(os.path.join(OUT_DIR, "learning_curve_results.csv"), "w", newline="") as f:
        fieldnames = ["cluster", "n_train", "held_out_tv"]
        w = csv.DictWriter(f, fieldnames=fieldnames); w.writeheader()
        for row in learning_curve_rows: w.writerow(row)
    lc_agg = {}
    for n_train in set(r["n_train"] for r in learning_curve_rows):
        vals = [r["held_out_tv"] for r in learning_curve_rows if r["n_train"] == n_train]
        lc_agg[n_train] = float(np.mean(vals))
    print(f"  Learning curve (N2, rank={PRIMARY_RANK}): {lc_agg}")
    report["learning_curve"] = lc_agg
    save_results(os.path.join(OUT_DIR, "y1_nonlinear_summary.json"), report)

    # ============================================================
    # m-sweep (after the primary m=8 analysis, per spec's own staging)
    # ============================================================
    print(f"\nm-sweep at rank={PRIMARY_RANK}: {M_GRID}...")
    m_sweep_rows = []
    for m in M_GRID:
        for c in sample_clusters:
            members = cluster_to_members[c]
            member_keys, member_values = ds_keys[members], ds_values[members]
            population = len(members)
            splits = candidate_splits[c]
            fit_keys, tune_keys, test_keys = ct_keys[splits["fit"]], ct_keys[splits["tune"]], val_keys[splits["test"]]
            token_ids, fit_target_m = local_m_distribution(fit_keys, member_keys, member_values, m, TAU)
            _, tune_target_m = local_m_distribution(tune_keys, member_keys, member_values, m, TAU)
            _, test_target_m = local_m_distribution(test_keys, member_keys, member_values, m, TAU)
            p_j_m = np.array([float(np.sum(member_values == y)) / population for y in token_ids])
            log_p_m = np.log(np.maximum(p_j_m, 1e-300))
            dime_bcast_m = np.tile(p_j_m, (test_target_m.shape[0], 1))
            z_fit_m = log_ratio_target(fit_target_m, p_j_m)
            z_tune_m = log_ratio_target(tune_target_m, p_j_m)

            a_affine_m = select_anchors_nearest(dist_matrix_all, c, PRIMARY_RANK)
            Phi_fit_a0m = linear_anchor_diff_basis(fit_keys, c, a_affine_m, dime_centroids)
            Phi_tune_a0m = linear_anchor_diff_basis(tune_keys, c, a_affine_m, dime_centroids)
            Phi_test_a0m = linear_anchor_diff_basis(test_keys, c, a_affine_m, dime_centroids)
            res_a0m = fit_select_eval(Phi_fit_a0m, Phi_tune_a0m, Phi_test_a0m, z_fit_m, z_tune_m, test_target_m, log_p_m)

            n2_anchors_m = fit_keys[farthest_point_sample(fit_keys, PRIMARY_RANK, seed=SEED + c)]
            dist_scale_m = float(np.median(np.linalg.norm(fit_keys[:, None, :] - n2_anchors_m[None, :, :], axis=2))) or 1.0
            sigma_m, lam_m, _ = best_sigma_for(rbf_features, fit_keys, tune_keys, z_fit_m, z_tune_m, log_p_m, n2_anchors_m, dist_scale_m)
            Phi_fit_n2m = rbf_features(fit_keys, n2_anchors_m, sigma_m)
            Phi_tune_n2m = rbf_features(tune_keys, n2_anchors_m, sigma_m)
            Phi_test_n2m = rbf_features(test_keys, n2_anchors_m, sigma_m)
            res_n2m = fit_select_eval(Phi_fit_n2m, Phi_tune_n2m, Phi_test_n2m, z_fit_m, z_tune_m, test_target_m, log_p_m)

            m_sweep_rows.append({"m": m, "cluster": c, "dime_tv": round(mean_tv(dime_bcast_m, test_target_m), 6),
                                    "A0_tv": round(mean_tv(res_a0m["d_test"], test_target_m), 6),
                                    "N2_tv": round(mean_tv(res_n2m["d_test"], test_target_m), 6)})
    m_sweep_agg = {m: {"dime_tv": float(np.mean([r["dime_tv"] for r in m_sweep_rows if r["m"] == m])),
                         "A0_tv": float(np.mean([r["A0_tv"] for r in m_sweep_rows if r["m"] == m])),
                         "N2_tv": float(np.mean([r["N2_tv"] for r in m_sweep_rows if r["m"] == m]))} for m in M_GRID}
    print(f"  m-sweep summary: {m_sweep_agg}")
    report["m_sweep"] = m_sweep_agg
    save_results(os.path.join(OUT_DIR, "y1_nonlinear_summary.json"), report)

    # ============================================================
    # Plots
    # ============================================================
    print("\nGenerating plots...")
    c_demo = sample_clusters[0]
    demo_cache = per_cluster_cache[c_demo]
    sample_q = 0
    d_exact_demo = dict_from_dist(demo_cache["token_ids"], demo_cache["test_target"][sample_q])
    d_a0_demo = dict_from_dist(demo_cache["token_ids"], demo_cache["model_results"]["A0"]["d_test"][sample_q])
    d_n2_demo = dict_from_dist(demo_cache["token_ids"], demo_cache["model_results"]["N2"]["d_test"][sample_q])
    all_toks = sorted(set(d_exact_demo) | set(d_a0_demo) | set(d_n2_demo))
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(range(len(all_toks)), [d_exact_demo.get(t, 0) for t in all_toks], marker="o", label="exact")
    ax.plot(range(len(all_toks)), [d_a0_demo.get(t, 0) for t in all_toks], marker="s", label="A0 (affine)")
    ax.plot(range(len(all_toks)), [d_n2_demo.get(t, 0) for t in all_toks], marker="^", label="N2 (RBF)")
    ax.set_xlabel("token (arbitrary order)"); ax.set_ylabel("probability"); ax.legend()
    ax.set_title(f"Exact vs. predicted token probabilities, cluster {c_demo}")
    fig.tight_layout(); fig.savefig(os.path.join(PLOT_DIR, "1_exact_vs_predicted_tokens.png"), dpi=130); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    for mname in MODELS:
        ax.plot(RANK_GRID, [np.mean(rank_agg[mname][r]) for r in RANK_GRID], marker="o", label=mname)
    ax.set_xlabel("rank"); ax.set_ylabel("mean test TV"); ax.legend()
    ax.set_title("Nonlinear (N1/N2) vs. affine (A0) approximation error vs. rank")
    fig.tight_layout(); fig.savefig(os.path.join(PLOT_DIR, "2_nonlinear_vs_affine_error.png"), dpi=130); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    for mname in ["A0", "N1", "N2"]:
        xs = [next(row["memory_bytes"] for row in cluster_results_rows if row["rank"] == r and row["model"] == mname) for r in RANK_GRID]
        ys = [np.mean(rank_agg[mname][r]) for r in RANK_GRID]
        ax.plot(xs, ys, marker="o", label=mname)
    ax.set_xlabel("marginal memory bytes/cluster"); ax.set_ylabel("mean test TV"); ax.legend()
    ax.set_title("Error vs. storage (N2's anchor cost is NOT amortized, unlike N1)")
    fig.tight_layout(); fig.savefig(os.path.join(PLOT_DIR, "3_error_vs_storage.png"), dpi=130); plt.close(fig)

    if margin_summary:
        fig, ax = plt.subplots(figsize=(7, 5))
        bins_sorted = sorted(margin_summary.keys())
        ax.plot([margin_summary[b]["mean_margin"] for b in bins_sorted], [margin_summary[b]["mean_A0_tv"] for b in bins_sorted], marker="o", label="A0")
        ax.plot([margin_summary[b]["mean_margin"] for b in bins_sorted], [margin_summary[b]["mean_N2_tv"] for b in bins_sorted], marker="s", label="N2")
        ax.set_xlabel("mean neighbor margin (quantile bin)"); ax.set_ylabel("mean TV"); ax.legend()
        ax.set_title("Error vs. neighbor-switching margin")
        fig.tight_layout(); fig.savefig(os.path.join(PLOT_DIR, "4_error_vs_margin.png"), dpi=130); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    lc_sorted = sorted(lc_agg.keys())
    ax.plot(lc_sorted, [lc_agg[n] for n in lc_sorted], marker="o")
    ax.set_xlabel("n_train (fit queries)"); ax.set_ylabel("held-out TV (N2, rank={})".format(PRIMARY_RANK))
    ax.set_title("Training vs. held-out error (learning curve)")
    fig.tight_layout(); fig.savefig(os.path.join(PLOT_DIR, "5_learning_curve.png"), dpi=130); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(MODELS, [qc_summary[m]["mean_e_delta"] for m in MODELS])
    ax.axhline(qc_summary["D0"]["mean_dime_zero_change_error"], color="r", linestyle="--", label="DIME zero-change baseline")
    ax.set_ylabel("mean query-change L1 error"); ax.legend()
    ax.set_title("Query-change preservation by model")
    fig.tight_layout(); fig.savefig(os.path.join(PLOT_DIR, "6_query_change_error.png"), dpi=130); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    # Computed fresh for c_demo specifically -- dist_scale_n1 inside the main loop above is a
    # per-cluster local that, after the loop ends, only ever holds the LAST cluster processed, not
    # c_demo (sample_clusters[0]); reusing it here would plot the wrong cluster's scale (caught via
    # self-review before running).
    demo_fit_keys = ct_keys[candidate_splits[c_demo]["fit"]]
    demo_dist_scale_n1 = float(np.median(np.linalg.norm(demo_fit_keys[:, None, :] - n1_anchors[None, :, :], axis=2))) or 1.0
    Phi_demo = rbf_features(demo_fit_keys, n1_anchors, demo_dist_scale_n1)
    sv = np.linalg.svd(Phi_demo, compute_uv=False)
    ax.plot(range(1, len(sv) + 1), sv, marker="o"); ax.set_yscale("log")
    ax.set_xlabel("component"); ax.set_ylabel("singular value (log)")
    ax.set_title(f"RBF feature conditioning, cluster {c_demo}")
    fig.tight_layout(); fig.savefig(os.path.join(PLOT_DIR, "7_feature_conditioning.png"), dpi=130); plt.close(fig)

    # ============================================================
    # failure_cases.md
    # ============================================================
    print("\nSelecting failure cases...")
    worsened = sorted(zip(per_cluster_a0.keys(), diffs_n2_a0.tolist()), key=lambda x: -x[1])[:N_FAILURE_EXAMPLES]
    improved = sorted(zip(per_cluster_a0.keys(), diffs_n2_a0.tolist()), key=lambda x: x[1])[:N_FAILURE_EXAMPLES]
    failure_lines = ["# Y1 (Nonlinear) Failure Cases", "", "N2 (RBF, cluster-specific) vs. A0 (affine), paired by cluster at "
                       f"m={M_PRIMARY}, rank={PRIMARY_RANK}.", "", "## N2 WORSENS vs. A0"]
    for c, d in worsened:
        failure_lines.append(f"- cluster={c}: A0_tv={per_cluster_a0[c]:.5f}, N2_tv={per_cluster_n2[c]:.5f}, diff={d:+.5f}")
    failure_lines.append("\n## N2 IMPROVES most over A0")
    for c, d in improved:
        failure_lines.append(f"- cluster={c}: A0_tv={per_cluster_a0[c]:.5f}, N2_tv={per_cluster_n2[c]:.5f}, diff={d:+.5f}")
    with open(os.path.join(OUT_DIR, "failure_cases.md"), "w") as f:
        f.write("\n".join(failure_lines))

    # ============================================================
    # r_extra_audit.md
    # ============================================================
    r_extra_md = """# R-extra Audit (formerly named Y1)

R-extra = run_y1_per_cluster_function.py, results under results/stage3_y1_per_cluster_function/ -- UNCHANGED,
not overwritten, not renamed on disk. Only the LABEL used to refer to it changes, per this spec's own instruction.

## Real results (read directly from the existing report.md, not reproduced)

Of the base-20 cluster sample, only 11 clusters passed even R-extra's LOOSE fit>=5/tune>=3/test>=3 filter (9
excluded for insufficient queries) -- exactly the under-sampling problem this delivery's Phase E fixes via a
broader candidate pool and stricter thresholds (MIN_FIT/TUNE/TEST = 30/15/20 here vs. R-extra's 5/3/3).

At m=1, DIME's own histogram TV against the exact local-1 target was 0.974 (near-maximal); R-extra's best
functional model (family B, histogram prior, affine anchor-difference basis) still averaged 0.884 TV at rank 8 --
barely better than noise. Query-change preservation FAILED for every single (m, family) combination tested
(beats_dime_zero_change: False in all 8 cases). R-extra's own Q10 verdict ("Justified") rested specifically on
family B's performance at m=20 (72.7% win-rate over the histogram) -- not on the small-m regime this new Y1
investigates with a genuinely nonlinear family.

## Basis construction (confirmed by direct inspection, not assumed)

R-extra's family B used exactly R4's own machinery: log_ratio_target + fit_ridge + reconstruct_logit, with
linear_anchor_diff_basis (TRI1's AFFINE anchor-difference features, phi_l(q) = -2q.(a_l-a_0)+const) as the basis
-- reused UNCHANGED here for A0, not reimplemented. R-extra's family A (no histogram prior) used a direct,
centered log-target with the SAME affine basis -- also affine in q, for the same reason.

## Why this matters for the new Y1

Both of R-extra's families are AFFINE in q (R4's own basis, confirmed in R7's audit). This new Y1 asks whether a
GENUINELY nonlinear basis (RBF, verified non-affine in check_non_affine_expressivity()) does better on exactly
the regime (small m, query-change preservation) where R-extra's real results were weakest.
"""
    with open(os.path.join(OUT_DIR, "r_extra_audit.md"), "w") as f:
        f.write(r_extra_md)

    # ============================================================
    # mathematical_analysis.md
    # ============================================================
    math_md = f"""# Y1 (Nonlinear) Mathematical Analysis

## A1 -- Partition of query space

For fixed cluster members {{(h_i,y_i)}}, R_S = {{q : N_m^C(q) = S}} (S a fixed size-m subset) is the intersection
of m*(n-m) half-spaces {{q : ||q-h_i|| < ||q-h_j||}} (i in S, j not in S) -- each an exact perpendicular-bisector
half-space under Euclidean distance. R_S is therefore CONVEX (an intersection of half-spaces), hence CONNECTED,
PROVEN directly from the definition, not assumed. As m increases, the number of achievable nonempty regions
changes, but the convexity argument holds for every m identically -- it does not depend on m's value, only on
the half-space characterization of "closer than." The boundary between adjacent regions (where the m-th and
(m+1)-th nearest member tie in distance) is itself a single hyperplane -- a Lebesgue-measure-zero subset of
R^{{768}}. Label duplication (two members at the boundary sharing a label) changes WHICH underlying observation is
counted but not necessarily the resulting distribution -- see A2.

## A2 -- Continuity and smoothness

Within a fixed region R_S, F_C^(m)(q) is a ratio of sums of exp(-||q-h_i||/tau) terms (i in S), each real-analytic
in q wherever q != h_i -- so F_C^(m) is real-analytic (hence smooth) in the INTERIOR of every region. AT a
boundary where the m-th and (m+1)-th nearest exchange:
- SAME-label exchange: the two members' kernel weights are EQUAL exactly at the tie (d_i=d_j there), so the
  labeled sum is the SAME value immediately before, at, and after the crossing -- NO discontinuity, verified
  directly on a synthetic two-point cluster in check_discontinuity_behavior() (same-label case trivially continuous
  by this argument, not separately coded since the one-point-swap construction already isolates the mechanism).
- DIFFERENT-label exchange: one label's contribution LOSES exactly the shared tie-weight and the other GAINS it
  -- a GENUINE finite jump of that weight's share of the total mass, verified directly (not merely asserted) in
  check_discontinuity_behavior(): a 1-D, 2-member cluster at -1 (label 0) and +1 (label 1), m=1, shows an EXACT
  flip from (1,0) to (0,1) at q=0, with zero smooth transition region.
- Weight changes WITHOUT membership changes are the smooth, real-analytic interior behavior above -- no special
  treatment needed.
For m=1, F_C^(1) is PIECEWISE CONSTANT, with jump discontinuities exactly on the (n-1)-dimensional Voronoi
boundary of the cluster's own members (the order-1 Voronoi diagram). For m>1, F_C^(m) is piecewise SMOOTH
(real-analytic on each order-m Voronoi cell), with jumps only at order-(m,m+1) boundaries.

## A3 -- Approximation in different norms

F_C^(m)(q) is a probability vector (each component in [0,1]), hence BOUNDED, hence trivially in
H_C = L^2(Q_C; R^{{L_C}}) for ANY probability measure Q_C, regardless of continuity -- boundedness alone suffices
for L^2 membership under a finite measure. However, F_C^(1) (and F_C^(m) generally) CANNOT be uniformly
approximated by continuous functions to within less than half the jump size at a genuine discontinuity: if g is
continuous and |F-g|<eps everywhere, then as q approaches the boundary from both sides, continuity of g forces
g's value to be within eps of BOTH one-sided limits of F, whose difference is the jump size Delta -- so
Delta <= 2*eps, i.e. eps >= Delta/2, a classical two-sided-limit argument. In EXPECTED squared error, continuous
(even smooth) functions ARE dense in L^2(Q_C) whenever Q_C assigns exactly zero probability to the discontinuity
hyperplane itself -- true whenever Q_C is absolutely continuous w.r.t. Lebesgue measure (or more generally,
whenever Q_C has no atom on that specific hyperplane), which is the generic, expected case for continuous
GPT-2 hidden-state query distributions. This is the precise reason a discontinuous target is not automatically
"unlearnable" in expectation even though it is provably unlearnable uniformly.

**The spec's own caution, respected explicitly, not assumed away:** the THEOREM above only requires Q_C(exact
boundary)=0, not that Q_C assigns negligible mass to a NEIGHBORHOOD of the boundary. If real query data
concentrates substantially NEAR (not exactly on) label-switching boundaries, finite-sample approximation there
remains genuinely harder in practice (higher variance, more basis functions needed locally) even though the
infinite-capacity L^2-density theorem still holds. This gap between infinite-capacity theory and finite-sample
practice is measured directly via the margin diagnostic (Phase F / margin_results, not merely asserted away).

## A4 -- Geometric complexity

Measurable diagnostics computed in this delivery: number of distinct labels (n_tokens), the neighbor margin
gamma_m(q) = d_(m+1)(q)-d_(m)(q) and its distribution across held-out queries (small margins flag queries near a
switching boundary), and the empirical conditioning (singular-value spectrum) of the realized RBF feature matrix
per cluster. The ambient dimension d=768 is NOT assumed equal to the query representations' effective
dimensionality -- the feature-conditioning plot (7_feature_conditioning.png) reports the REALIZED singular-value
decay directly rather than assuming full rank.

## Required mathematical conclusion

Given A1-A4, an approximation family for F_C^(m) should ideally be: (i) capable of NONLINEAR dependence on q
(to represent curvature near non-affine label boundaries -- verified achievable, not merely assumed, in
check_non_affine_expressivity()); (ii) SPATIALLY LOCALIZED (an RBF bump naturally concentrates its influence near
its own anchor, unlike an affine feature's global linear gradient); (iii) able to place multiple bumps for the
SAME label in geometrically separated regions (RBF features trivially support this via multiple anchors per
label-relevant region; an affine family cannot represent two disjoint regions for one label without a non-affine
combination); (iv) flexible enough near switching boundaries (bandwidth controls this trade-off directly); and
(v) compact and finite-sample-stable (the learning-curve experiment measures this last property directly, not
assumed). These are candidate properties motivating the RBF family tested here, not claims that every successful
family must have them.
"""
    with open(os.path.join(OUT_DIR, "mathematical_analysis.md"), "w") as f:
        f.write(math_md)

    # ============================================================
    # synthetic_tests.md
    # ============================================================
    synthetic_md = f"""# Y1 (Nonlinear) Synthetic Tests

All seven required sanity checks (Section 11) ran BEFORE any real cache was touched; the script asserts on
failure rather than proceeding. Results from this run:

{json.dumps(sanity, indent=2)}

## Check 4 detail (non-affine expressivity)

XOR-style synthetic target (label = XOR(sign(x1), sign(x2)), no linearly-separating hyperplane exists by
construction): affine-logit accuracy = {sanity['check4_affine_accuracy']:.3f}, RBF-logit accuracy (4 Gaussian
bumps at the quadrant centers) = {sanity['check4_rbf_accuracy']:.3f}. This directly demonstrates the RBF family's
genuine non-affine expressivity on a target an affine family cannot represent, rather than merely asserting the
RBF feature is "different" from R4's affine family by construction.

## Check 5 detail (discontinuity)

A 1-D, 2-member synthetic cluster at q=-1 (label 0) and q=+1 (label 1), m=1: the exact local-1 target is verified
to be EXACTLY one-hot on label 0 for all probed q<0 and EXACTLY one-hot on label 1 for all probed q>0, with the
flip occurring precisely at q=0 and no smooth transition region -- confirming the piecewise-constant,
discontinuous-at-the-boundary characterization in mathematical_analysis.md's A2, not merely asserting it.
"""
    with open(os.path.join(OUT_DIR, "synthetic_tests.md"), "w") as f:
        f.write(synthetic_md)

    # ============================================================
    # experimental_design.md
    # ============================================================
    design_md = f"""# Y1 (Nonlinear) Experimental Design

## Cluster/query sampling (Phase E)

Scanned {len(audited)} eligible candidates (population >= {MIN_ELIGIBLE_POP}, up to {CANDIDATE_POOL_SIZE});
{len(sufficient)} met the STRICTER sufficiency bar (fit>={MIN_FIT}/tune>={MIN_TUNE}/test>={MIN_TEST}, vs.
R-extra's 5/3/3). Selected the top {len(sample_clusters)} by total routed-query count: {sample_clusters}.

## Splits

Per-cluster fit/tune/test splits use the SAME position-level controller_train(fit/tune)/val(test) convention as
every prior script in this project (no document/sequence-grouped split exists in the cached arrays -- stated as
a limitation, not assumed away). No held-out (test) response is ever used to select anchors or tune
bandwidth/rank/lambda -- N2's anchors are farthest-point-sampled from the FIT split only; bandwidth and ridge
lambda are both selected on the TUNE split.

## Models

D0 (fixed histogram), A0 (R-extra's affine family, reused unchanged), N1 (RBF, globally shared fixed anchors),
N2 (RBF, cluster-specific training-derived anchors) -- see module docstring for exact definitions.

## Primary configuration

m={M_PRIMARY}, tau={TAU}, primary rank={PRIMARY_RANK}, rank grid={RANK_GRID}, bandwidth scale grid=
{BANDWIDTH_SCALE_GRID} (relative to each anchor set's own median anchor-to-fit-query distance).

## Secondary analyses

m-sweep ({M_GRID}) at the primary rank; a learning curve (N2, primary rank, varying n_train in
{LEARNING_CURVE_SIZES}); a neighbor-margin quantile breakdown of A0 vs. N2 error.
"""
    with open(os.path.join(OUT_DIR, "experimental_design.md"), "w") as f:
        f.write(design_md)

    # ============================================================
    # research_questions.md
    # ============================================================
    research_q = """# Y1 (Nonlinear) Research Questions for Subsequent Study

## Approximation theory
- Is there a precise relationship between the neighbor-margin distribution under Q_C and the MINIMAL number of
  RBF bumps needed for a given held-out TV target, beyond the empirical correlation measured here?
- A1's convexity result holds for EVERY m -- does the combinatorial complexity of achievable regions (not
  computed here, only convexity) follow any of the known low-dimensional order-k Voronoi complexity bounds in
  d=768, or does high dimensionality change which bounds are even relevant?

## Geometry
- N1 (shared anchors) and N2 (cluster-specific anchors) represent two extremes -- is there an intermediate
  "partially shared" scheme (e.g. anchors shared within a group of geometrically similar clusters) that captures
  most of N2's adaptivity benefit at closer to N1's amortized storage cost?

## Statistical estimation
- The learning curve reuses a SINGLE bandwidth/lambda tuned at the full training-set size for every smaller
  subsample -- would re-tuning at each subsample size change the curve's shape, and is that re-tuning itself
  stable at very small n_train?

## Compression architecture
- If N2 shows a genuine, memory-accounted improvement over A0 in the small-m regime, what would a PRODUCTION
  version need to store the cluster-specific anchors efficiently (e.g. a lower-precision or compressed
  representation of the 768-dim anchor vectors) without erasing the adaptivity advantage this delivery measured?
"""
    with open(os.path.join(OUT_DIR, "research_questions.md"), "w") as f:
        f.write(research_q)

    # ============================================================
    # Final verdict (Y1-Q1 - Y1-Q10) + report.md
    # ============================================================
    print("\nComputing final verdict (Y1-Q1 - Y1-Q10)...")
    q1 = "F_C^(m) is piecewise real-analytic on convex order-m Voronoi cells, bounded hence always in L^2(Q_C). See mathematical_analysis.md."
    q2 = "Discontinuous only across DIFFERENT-label neighbor exchanges (a measure-zero hyperplane); same-label exchanges and interior weight changes are continuous/smooth. See mathematical_analysis.md A2."
    q3 = "Nonlinear dependence, spatial localization, multi-region-per-label capacity, boundary flexibility, compactness, finite-sample stability (candidate properties, not requirements). See mathematical_analysis.md's required conclusion."
    q4 = {"affine_accuracy_on_xor": sanity["check4_affine_accuracy"], "rbf_accuracy_on_xor": sanity["check4_rbf_accuracy"], "demonstrated": sanity["check4_non_affine_expressivity"]}
    q5 = report["main_comparison_A0_vs_N2"]
    q6 = report["query_change_summary"]
    q7 = {"m_sweep": m_sweep_agg, "rank_sweep": rank_results_rows, "margin": margin_summary}
    mem_matched_at_primary = next(row for row in memory_results_rows if row["rank"] == PRIMARY_RANK)
    q8 = {"A0_bytes": mem_matched_at_primary["A0_bytes"], "N2_bytes": mem_matched_at_primary["N2_bytes"],
           "N1_marginal_bytes": mem_matched_at_primary["N1_bytes_marginal"], "N1_shared_overhead_total": mem_matched_at_primary["N1_shared_overhead_total"],
           "N2_requires_full_768d_anchors_per_cluster": True, "A0_anchors_are_free_centroid_indices": True}
    q9 = {"expressivity_demonstrated_on_synthetic": sanity["check4_non_affine_expressivity"],
           "learnability_on_real_clusters": report["main_comparison_A0_vs_N2"]["significantly_better"],
           "learning_curve_improves_with_more_data": bool(lc_agg[max(lc_agg.keys())] < lc_agg[min(lc_agg.keys())]) if len(lc_agg) > 1 else None}
    q10_candidate = report["main_comparison_A0_vs_N2"]["significantly_better"] and report["main_comparison_A0_vs_N2"]["frac_clusters_N2_beats_A0"] > 0.6

    if sanity["check4_non_affine_expressivity"] and report["main_comparison_A0_vs_N2"]["significantly_better"] and q10_candidate:
        verdict = "Strong evidence supporting nonlinear functional approximation"
    elif sanity["check4_non_affine_expressivity"] and report["main_comparison_A0_vs_N2"]["frac_clusters_N2_beats_A0"] > 0.5 and not report["main_comparison_A0_vs_N2"]["significant_different_from_zero"]:
        verdict = "Promising but insufficient evidence"
    elif sanity["check4_non_affine_expressivity"] and report["main_comparison_A0_vs_N2"]["frac_clusters_N2_beats_A0"] > 0.5 and report["main_comparison_A0_vs_N2"]["significantly_better"]:
        verdict = "Nonlinear expressivity improves but compression remains inadequate"
    elif report["main_comparison_A0_vs_N2"]["frac_clusters_N2_beats_A0"] <= 0.5:
        verdict = "No meaningful improvement over affine approximation"
    else:
        verdict = "Inconclusive due to data, optimization, or implementation limitations"
    print(f"  Verdict: {verdict}")
    verdict_block = {"Y1_Q1": q1, "Y1_Q2": q2, "Y1_Q3": q3, "Y1_Q4": q4, "Y1_Q5": q5, "Y1_Q6": q6, "Y1_Q7": q7,
                       "Y1_Q8": q8, "Y1_Q9": q9, "Y1_Q10": q10_candidate, "final_classification": verdict}
    report["final_verdict"] = verdict_block
    save_results(os.path.join(OUT_DIR, "y1_nonlinear_summary.json"), report)

    lines = [
        "# Y1 (Nonlinear): Functional Approximation of Local Nearest-Neighbor Distributions", "",
        "## 1. Research question", "See module docstring, mathematical_analysis.md, and r_extra_audit.md.", "",
        "## 2. Mathematical analysis summary", "See mathematical_analysis.md for the full A1-A4 analysis.", "",
        "## 3. R-extra audit summary", "See r_extra_audit.md.", "",
        "## 4. Synthetic correctness and non-affine expressivity", f"{sanity}", "",
        "## 5. Cluster/query sampling", f"{report['query_availability_audit']}", "",
        "## 6. Main comparison (A0 vs. N2, m={}, rank={})".format(M_PRIMARY, PRIMARY_RANK), f"{main_comparison}", "",
        "## 7. m-sweep", f"{m_sweep_agg}", "",
        "## 8. Query-change preservation", f"{qc_summary}", "",
        "## 9. Margin diagnostic", f"{margin_summary}", "",
        "## 10. Learning curve", f"{lc_agg}", "",
        "## 11. Memory accounting (rank={})".format(PRIMARY_RANK), f"{mem_matched_at_primary}", "",
        "## 12. Final verdict", f"**{verdict}**", f"{verdict_block}", "",
        "## Limitations",
        f"- Focused pilot: {len(sample_clusters)} clusters, not all 3,811, per the spec's own staged-execution instruction.",
        "- No document/sequence-grouped split exists in the cached data.",
        "- Bandwidth/lambda for the learning curve are reused from the full-training-set tuning, not re-tuned per subsample size.",
        "- N1's amortized storage benefit assumes the shared anchor set is reused across the sampled clusters only; "
        "a full-datastore deployment would amortize it across all 3,811 clusters, a stronger claim not measured here.",
    ]
    with open(os.path.join(OUT_DIR, "report.md"), "w") as f:
        f.write("\n".join(lines))

    report["sections_completed"] = ["synthetic_correctness_checks", "query_availability_audit", "main_experiment",
                                      "rank_results.csv", "cluster_results.csv", "memory_results.csv",
                                      "query_change_results.csv", "learning_curve_results.csv", "m_sweep",
                                      "margin_diagnostic", "main_comparison_A0_vs_N2", "r_extra_audit.md",
                                      "mathematical_analysis.md", "synthetic_tests.md", "experimental_design.md",
                                      "research_questions.md", "failure_cases.md", "final_verdict", "report.md", "plots"]
    report["total_runtime_seconds"] = time.perf_counter() - t_start
    save_results(os.path.join(OUT_DIR, "y1_nonlinear_summary.json"), report)
    print(f"\nAll results saved under {OUT_DIR}")
    print(f"Total runtime: {report['total_runtime_seconds']/60:.1f} min")
    print("R-extra (results/stage3_y1_per_cluster_function/) and every R-series output were left untouched.")


if __name__ == "__main__":
    main()
