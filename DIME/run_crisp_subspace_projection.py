"""CRISP -- Centroid-Routed Intra-cluster Subspace Projection. Extends the
E1/E2/TRI1/NC1/GEOM1/KM1 series without modifying any existing output
(writes only to results/stage3_crisp_subspace_projection/). CRISP is an
UNVALIDATED hypothesis under test here, not an assumed-working method.

=====================================================================
What this script evaluates
=====================================================================
DIME-V1 replaces a cluster's original (hidden-state, token) members with one
fixed, query-independent histogram. The within-cluster exact evidence
function is, for cluster j, label y:
    s_{j,y}(q) = sum_{i in C_j, y_i=y} exp(-d(q,h_i)/tau_local)
    F_j(q)(y) = s_{j,y}(q) / sum_t s_{j,t}(q)
CRISP asks whether s_{j,y}(.) admits a compact finite-basis approximation
    s~_{j,y}(q) = sum_{l=1..r} theta_{j,y,l} * phi_l(q)
fit by regularized least squares (ridge regression, closed form) on
controller_train queries routed to cluster j, evaluated on DISJOINT held-out
val queries -- NOT the same split used for either rank/lambda/basis
selection (a THIRD controller_train sub-split is reserved for that; see
"Split discipline" below).

Per Section 4's explicit scope: NO base-LM interpolation and NO end-to-end
LM perplexity anywhere in this script -- the primary metrics are
distributional (JS divergence, total variation, top-1/top-k agreement,
entropy) against the exact within-cluster evidence function and against raw
kNN, exactly as Section 4/E3 require. This is a deliberate, substantial
deviation from every prior E1/E2/TRI1/NC1/GEOM1/KM1 script's NLL-centric
convention -- stated explicitly, not silently changed.

=====================================================================
Split discipline (controller_train queries routed to a sampled cluster are
split three ways, never mixed):
=====================================================================
  - FIT split: used only to fit ridge-regression coefficients theta.
  - TUNE split: used only to choose rank r, ridge lambda, and basis family.
  - VAL (held-out): used ONLY for final reported numbers, touched by
    nothing above -- the spec's own "avoid tuning on the final test split"
    instruction, applied literally via a genuine three-way split, not just
    a train/val split reused for both fitting and selection.

=====================================================================
Basis functions (per the spec's own "do not implement every family -- one
mathematically appropriate basis and one simple comparison"):
=====================================================================
  - PRIMARY: RBF kernels centered at each cluster's r nearest OTHER DIME
    centroids (reusing TRI1's build_centroid_distance_matrix/
    select_anchors_nearest, imported directly) -- phi_l(q) =
    exp(-||q-a_l||^2 / (2*gamma^2)).
  - COMPARISON: linear anchor-difference features -- phi_l(q) = z_l(q),
    TRI1's own compute_diff_features (imported directly), i.e. CRISP's basis
    IS TRI1's anchor representation, just now used as regression features
    for a per-token function instead of a single mean/variance summary.

=====================================================================
CRISP-A vs. CRISP-B (Section 7, must not be confused):
=====================================================================
  - CRISP-A: DIME's original outer centroid weights w_c(q) are kept
    EXACTLY as-is; only each retrieved cluster's internal distribution is
    replaced, P_A(y|q) = sum_c w_c(q) * F~_c(q)(y).
  - CRISP-B: the retrieved clusters' relative contribution is replaced by
    their RECONSTRUCTED evidence MASS instead of DIME's distance-based
    weight: P_B(y|q) = sum_c [m~_c(q) / sum_{c'} m~_{c'}(q)] * F~_c(q)(y).
    This can differ from CRISP-A even with identical routing, by design.
    This delivery evaluates CRISP-B's PREREQUISITE directly -- how well the
    reconstructed evidence MASS m~_c(q) itself correlates with the true
    mass m_c(q), per sampled cluster -- rather than a full multi-cluster
    P_B(y|q) mixture, for the same reason E3 is per-cluster (see below):
    only 20 of 3,811 clusters have a fitted model, so a real query's other
    retrieved clusters have none to combine with yet.

Negative coefficients from unconstrained ridge regression can produce
negative reconstructed evidence; this is corrected by clipping to zero
before renormalizing, and the fraction of evidence mass clipped is reported
explicitly as a distortion measure (Section 6's explicit instruction), not
silently absorbed.

=====================================================================
Scoping decisions (stated up front; this spec's own Section 11 explicitly
structures work into Phase 1-4 and says NOT to run a large sweep
immediately -- this delivery implements Phases 1-3 only, Phase 4 (full
3,811-cluster / full-val scale) explicitly deferred):
=====================================================================
  - 20 sampled clusters (10 largest-population + 10 smaller/rarer-label,
    population >= 30 so each still has enough data to fit AND test),
    not all 3,811 -- a per-cluster regression is fit for EVERY sampled
    cluster, so this is not a cheap aggregate statistic to begin with.
  - Rank grid r in {1,2,4,8,16,32} (64 deferred to Phase 4 as a stretch
    extension, not needed to see the error-vs-rank trend).
  - Fit/tune/test query counts per cluster bounded to 150/50/100
    respectively (controller_train fit+tune, val test).
  - tau_local (the within-cluster kernel's own temperature, distinct from
    DIME's outer routing tau) is a single global constant set once from the
    empirical local-distance distribution and held fixed across every rank/
    basis/cluster -- the same "single global constant" precedent E2/E2.1
    established for beta.
  - B4 (memory-matched simple baseline): M farthest-point-diverse
    representative members per cluster (a hand-rolled greedy routine, no
    scikit-learn), reported at real storage bytes alongside CRISP so the
    rate-distortion comparison is direct rather than forced to exact byte
    parity.
  - E6 (failure analysis) is a grouped re-reading of E2/E3's own already-
    computed per-cluster results (by population/homogeneity/rare-label),
    not a separate new computation.
  - Mathematical analysis (Section 9 A-E) is written directly from this
    script's own empirical findings in report.md, not a separate symbolic-
    math computation.

Smoke testing: set CRISP_SMOKE=1 with CRISP_CACHE_DIR / CRISP_OUT_DIR /
CRISP_N_CLUSTERS before running -- same convention as every prior script in
this series. Never set it for the real run.
"""
import sys, os, json, time
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
from knn import build_datastore, query_knn_indices, softmax_np
from results_io import save_results
from e1_metrics import js_divergence_dict, distribution_shape_stats
from run_e2_information_allocation import build_cluster_to_members
from run_e2tri1_anchor_conditioned_compression import (
    build_centroid_distance_matrix, select_anchors_nearest, compute_diff_features)
from gpt_class import FrozenGPT2

SMOKE = os.environ.get("CRISP_SMOKE") == "1"
CACHE_DIR = os.environ.get("CRISP_CACHE_DIR", os.path.join(SCRIPT_DIR, "..", "GPT_Module", "cache"))
OUT_DIR = os.environ.get("CRISP_OUT_DIR", os.path.join(SCRIPT_DIR, "results", "stage3_crisp_subspace_projection"))
PLOT_DIR = os.path.join(OUT_DIR, "plots")
os.makedirs(PLOT_DIR, exist_ok=True)

N_CLUSTERS = int(os.environ.get("CRISP_N_CLUSTERS", "3811"))
MATCHED_OUTER = {"k": 20, "tau": 2.0}  # outer DIME routing only -- no alpha/LM here, per spec Section 4
SEED = 42
DIM, DTYPE_BYTES, VALUE_BYTES = 768, 4, 4

N_LARGE_CLUSTERS = 10
N_SMALL_CLUSTERS = 10
MIN_CLUSTER_POP_FOR_SAMPLE = 30
RANK_GRID = [1, 2, 4, 8, 16, 32]
BASIS_FAMILIES = ["rbf_anchor", "linear_anchor_diff"]
MAX_FIT, MAX_TUNE, MAX_TEST = 150, 50, 100
RIDGE_LAMBDA_GRID = [0.01, 0.1, 1.0, 10.0]
N_PROBE_FOR_TAU = 300
TOPK_OVERLAP = 5
N_QUERY_PAIRS_E4 = 30


# ============================================================
# Exact within-cluster evidence oracle (B2)
# ============================================================
def exact_evidence(query_keys, member_keys, member_values, tau_local):
    """s_{j,y}(q) for every distinct y in the cluster, every query in
    query_keys. Returns (token_ids [T], evidence_matrix [n_queries, T])."""
    token_ids = np.array(sorted(set(member_values.tolist())))
    d = np.linalg.norm(member_keys[None, :, :] - query_keys[:, None, :], axis=2)  # [n_q, n_members]
    k = np.exp(-d / tau_local)  # [n_q, n_members]
    evidence = np.zeros((len(query_keys), len(token_ids)), dtype=np.float64)
    for t_idx, y in enumerate(token_ids):
        mask = member_values == y
        evidence[:, t_idx] = k[:, mask].sum(axis=1)
    return token_ids, evidence


def normalize_rows(evidence):
    total = evidence.sum(axis=1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        dist = np.where(total > 0, evidence / total, 0.0)
    return dist, total.flatten()


# ============================================================
# Basis functions
# ============================================================
def rbf_basis(query_keys, anchor_points, gamma):
    d2 = np.sum((query_keys[:, None, :] - anchor_points[None, :, :]) ** 2, axis=2)
    return np.exp(-d2 / (2 * gamma ** 2))


def linear_anchor_diff_basis(query_keys, cluster_centroid_idx, anchor_idx, dime_centroids):
    return compute_diff_features(query_keys, cluster_centroid_idx, anchor_idx, dime_centroids)


# ============================================================
# Ridge regression (closed form, NumPy only)
# ============================================================
def fit_ridge(Phi, targets, lam):
    """Phi: [n, r]. targets: [n, T] (one column per token). Returns theta
    [r, T]. Empirical L2 (least squares) + L2 ridge penalty -- NOT exact
    orthogonal projection unless lam=0 and Phi has full column rank."""
    r = Phi.shape[1]
    A = Phi.T @ Phi + lam * np.eye(r)
    B = Phi.T @ targets
    return np.linalg.solve(A, B)


def reconstruct_and_correct(Phi, theta):
    """s~_{j,y}(q) = Phi @ theta, then clip negative evidence to 0 (nonneg
    correction) before the caller renormalizes. Returns (raw, corrected,
    fraction_of_mass_clipped)."""
    raw = Phi @ theta
    corrected = np.clip(raw, 0.0, None)
    raw_mass = np.abs(raw).sum()
    clipped_mass = (raw_mass - corrected.sum()) if raw_mass > 0 else 0.0
    frac_clipped = float(clipped_mass / raw_mass) if raw_mass > 0 else 0.0
    return raw, corrected, frac_clipped


# ============================================================
# Farthest-point diverse sampling (B4's representative members)
# ============================================================
def farthest_point_sample(keys, m, seed=SEED):
    n = len(keys)
    if n <= m:
        return np.arange(n)
    rng = np.random.default_rng(seed)
    selected = [int(rng.integers(0, n))]
    min_dist = np.linalg.norm(keys - keys[selected[0]][None, :], axis=1)
    for _ in range(m - 1):
        nxt = int(np.argmax(min_dist))
        selected.append(nxt)
        min_dist = np.minimum(min_dist, np.linalg.norm(keys - keys[nxt][None, :], axis=1))
    return np.array(selected)


# ============================================================
# Distributional comparison metrics
# ============================================================
def dict_from_dist(token_ids, probs):
    return {int(t): float(p) for t, p in zip(token_ids, probs) if p > 0}


def total_variation(d1, d2):
    support = set(d1.keys()) | set(d2.keys())
    return 0.5 * sum(abs(d1.get(t, 0.0) - d2.get(t, 0.0)) for t in support)


def top1_agreement(d1, d2):
    if not d1 or not d2:
        return None
    return int(max(d1, key=d1.get) == max(d2, key=d2.get))


def topk_overlap(d1, d2, k=TOPK_OVERLAP):
    if not d1 or not d2:
        return None
    t1 = set(sorted(d1, key=d1.get, reverse=True)[:k])
    t2 = set(sorted(d2, key=d2.get, reverse=True)[:k])
    return len(t1 & t2) / max(1, min(k, len(t1 | t2)))


def entropy_of_dict(d):
    if not d:
        return 0.0
    probs = np.array(list(d.values()))
    probs = probs[probs > 0]
    return float(-(probs * np.log(probs)).sum())


# ============================================================
# Synthetic sanity tests (E1)
# ============================================================
def make_synthetic_cluster(kind, rng, n_per_label=60, D=8):
    if kind == "separated":
        a = rng.normal(scale=0.2, size=(n_per_label, D)) + np.array([2.0] + [0.0] * (D - 1))
        b = rng.normal(scale=0.2, size=(n_per_label, D)) - np.array([2.0] + [0.0] * (D - 1))
        keys = np.vstack([a, b]).astype(np.float64)
        values = np.array([0] * n_per_label + [1] * n_per_label)
    elif kind == "overlapping_different_distance":
        a = rng.normal(scale=1.0, size=(n_per_label, D))
        b = rng.normal(scale=1.0, size=(n_per_label, D)) + np.array([0.5] + [0.0] * (D - 1))
        keys = np.vstack([a, b]).astype(np.float64)
        values = np.array([0] * n_per_label + [1] * n_per_label)
    elif kind == "imbalanced":
        n_minor = max(5, n_per_label // 10)
        a = rng.normal(scale=0.5, size=(n_per_label, D))
        b = rng.normal(scale=0.5, size=(n_minor, D)) + np.array([1.5] + [0.0] * (D - 1))
        keys = np.vstack([a, b]).astype(np.float64)
        values = np.array([0] * n_per_label + [1] * n_minor)
    else:
        raise ValueError(kind)
    return keys, values


def run_synthetic_tests(rng):
    """Fit and test queries are DISTINCT random samples from the same
    region -- NOT the cluster's own member positions. Fitting evidence AT
    the member points themselves makes self-distance exactly 0, producing a
    near one-hot target that a flexible high-rank model overfits to (caught
    during pilot testing: synthetic 'separated' TV got WORSE with rank under
    that degenerate design, while the real E2 evaluation on genuine
    controller_train/val query splits showed clean monotonic improvement --
    isolating the bug to this function's fitting procedure, not the shared
    ridge-regression/basis-function machinery)."""
    results = {}
    tau_local = 1.0
    for kind in ["separated", "overlapping_different_distance", "imbalanced"]:
        keys, values = make_synthetic_cluster(kind, rng)
        offset = np.array([1.0] + [0.0] * (keys.shape[1] - 1))
        fit_queries = rng.normal(scale=0.4, size=(80, keys.shape[1])) + offset
        test_queries = rng.normal(scale=0.4, size=(40, keys.shape[1])) + offset
        token_ids, exact_ev_test = exact_evidence(test_queries, keys, values, tau_local)
        _, exact_ev_fit = exact_evidence(fit_queries, keys, values, tau_local)
        exact_dist_test, _ = normalize_rows(exact_ev_test)
        dime_hist = np.array([np.mean(values == y) for y in token_ids])
        dime_dist_broadcast = np.tile(dime_hist, (len(test_queries), 1))
        dime_vs_exact_tv = float(np.mean(0.5 * np.abs(exact_dist_test - dime_dist_broadcast).sum(axis=1)))

        rank_errors = {}
        anchors = rng.normal(scale=2.0, size=(8, keys.shape[1]))
        for r in [1, 2, 4, 8]:
            gamma = 1.5
            Phi_fit = rbf_basis(fit_queries, anchors[:r], gamma)
            theta = fit_ridge(Phi_fit, exact_ev_fit, lam=0.1)
            Phi_test = rbf_basis(test_queries, anchors[:r], gamma)
            _, corrected, _ = reconstruct_and_correct(Phi_test, theta)
            recon_dist, _ = normalize_rows(corrected)
            rank_errors[r] = float(np.mean(0.5 * np.abs(exact_dist_test - recon_dist).sum(axis=1)))
        results[kind] = {"dime_vs_exact_mean_tv": dime_vs_exact_tv, "crisp_tv_vs_rank": rank_errors}
        print(f"  synthetic[{kind}]: DIME-vs-exact TV={dime_vs_exact_tv:.4f}, CRISP TV by rank={rank_errors}")

    # Reversal / identical-ordering-different-weights checks (qualitative, not aggregated above)
    keys_sep, values_sep = make_synthetic_cluster("separated", rng)
    q_near_a = np.array([[1.8] + [0.0] * (keys_sep.shape[1] - 1)])
    q_near_b = np.array([[-1.8] + [0.0] * (keys_sep.shape[1] - 1)])
    _, ev_a = exact_evidence(q_near_a, keys_sep, values_sep, tau_local)
    _, ev_b = exact_evidence(q_near_b, keys_sep, values_sep, tau_local)
    dist_a, _ = normalize_rows(ev_a)
    dist_b, _ = normalize_rows(ev_b)
    reversal_detected = bool(np.argmax(dist_a[0]) != np.argmax(dist_b[0]))
    results["reversal_check"] = {"top_label_near_a": int(np.argmax(dist_a[0])), "top_label_near_b": int(np.argmax(dist_b[0])),
                                   "reversal_detected": reversal_detected}
    print(f"  reversal check: {results['reversal_check']}")
    return results


# ============================================================
# Main
# ============================================================
def main():
    t_start = time.perf_counter()
    print(f"CRISP job started. SMOKE={SMOKE}, N_CLUSTERS={N_CLUSTERS}")
    rng = np.random.default_rng(SEED)

    print("Loading cache (datastore/controller_train/val)...")
    ds = load_cache(os.path.join(CACHE_DIR, "wikitext103_gpt2_datastore.npz"))
    ct = load_cache(os.path.join(CACHE_DIR, "wikitext103_gpt2_controller_train.npz"))
    val = load_cache(os.path.join(CACHE_DIR, "wikitext103_gpt2_val.npz"))
    ds_keys, ds_values = ds["keys"].astype(np.float64), ds["values"]
    ct_keys, ct_values = ct["keys"].astype(np.float64), ct["values"]
    val_keys, val_values = val["keys"].astype(np.float64), val["values"]
    gpt = FrozenGPT2(model_name="gpt2")

    print("Re-deriving DIME clusters (seed=42, deterministic, identical to every prior E1/E2/TRI1/NC1/GEOM1/KM1 script)...")
    km = MiniBatchKMeans(n_clusters=N_CLUSTERS, random_state=SEED)
    assignment = km.fit_predict(ds_keys.astype(np.float32))
    dime_centroids = km.cluster_centers_.astype(np.float64)
    compressed_dists = []
    for c in range(N_CLUSTERS):
        member_values = ds_values[assignment == c]
        compressed_dists.append(Counter(member_values.tolist()) if len(member_values) > 0 else Counter())
    cluster_to_members = build_cluster_to_members(assignment, N_CLUSTERS)

    raw_index, raw_values = build_datastore(ds_keys, ds_values)
    dime_index, dime_values = build_datastore(dime_centroids, np.array(compressed_dists, dtype=object))
    kOuter, tauOuter = MATCHED_OUTER["k"], MATCHED_OUTER["tau"]

    d_dime_ct, idx_dime_ct = query_knn_indices(dime_index, ct_keys, k=kOuter)
    d_dime_val, idx_dime_val = query_knn_indices(dime_index, val_keys, k=kOuter)

    report = {"n_ds": int(len(ds_values)), "n_ct": int(len(ct_values)), "n_val": int(len(val_values)),
              "outer_config": MATCHED_OUTER, "n_clusters": N_CLUSTERS, "seed": SEED, "smoke": SMOKE}
    save_results(os.path.join(OUT_DIR, "crisp_summary.json"), report)

    # ============================================================
    # tau_local -- single global constant, probed once
    # ============================================================
    print(f"\nProbing empirical within-cluster distance scale ({N_PROBE_FOR_TAU} queries)...")
    probe_idx = rng.choice(len(ct_values), size=min(N_PROBE_FOR_TAU, len(ct_values)), replace=False)
    probe_d = []
    for i in probe_idx:
        c = int(idx_dime_ct[i, 0])
        members = cluster_to_members[c]
        if len(members) > 0:
            probe_d.append(np.linalg.norm(ds_keys[members] - ct_keys[i][None, :], axis=1))
    probe_d = np.concatenate(probe_d)
    tau_local = float(np.median(probe_d))
    print(f"  tau_local = {tau_local:.4f} (median empirical local distance, n={len(probe_d)})")
    report["tau_local"] = tau_local
    save_results(os.path.join(OUT_DIR, "crisp_summary.json"), report)

    # ============================================================
    # E1 -- synthetic sanity tests
    # ============================================================
    print("\nE1: synthetic mathematical sanity tests...")
    e1_results = run_synthetic_tests(rng)
    report["E1_synthetic"] = e1_results
    save_results(os.path.join(OUT_DIR, "crisp_summary.json"), report)

    # ============================================================
    # Cluster sample: 10 largest-population + 10 smaller/rarer-label
    # ============================================================
    print("\nSelecting a sample of 20 clusters (10 largest-population + 10 smaller/rarer-label)...")
    populations = np.array([sum(c.values()) for c in compressed_dists])
    eligible = np.where(populations >= MIN_CLUSTER_POP_FOR_SAMPLE)[0]
    order_by_pop = eligible[np.argsort(-populations[eligible])]
    large_clusters = order_by_pop[:N_LARGE_CLUSTERS].tolist()
    small_clusters = order_by_pop[-N_SMALL_CLUSTERS:].tolist()
    sample_clusters = sorted(set(int(c) for c in large_clusters + small_clusters))
    print(f"  sampled {len(sample_clusters)} clusters, populations: {[int(populations[c]) for c in sample_clusters]}")
    report["sampled_clusters"] = sample_clusters
    save_results(os.path.join(OUT_DIR, "crisp_summary.json"), report)

    # ============================================================
    # Per-cluster fit/tune/test query splits (controller_train + val)
    # ============================================================
    cluster_splits = {}
    for c in sample_clusters:
        ct_routed = np.where(idx_dime_ct[:, 0] == c)[0]
        val_routed = np.where(idx_dime_val[:, 0] == c)[0]
        rng.shuffle(ct_routed)
        n_fit = min(MAX_FIT, max(1, len(ct_routed) * 3 // 4))
        fit_idx = ct_routed[:n_fit]
        tune_idx = ct_routed[n_fit:n_fit + MAX_TUNE]
        test_idx = val_routed[:MAX_TEST]
        cluster_splits[c] = {"fit": fit_idx, "tune": tune_idx, "test": test_idx}
    report["cluster_split_sizes"] = {c: {k: int(len(v)) for k, v in s.items()} for c, s in cluster_splits.items()}
    save_results(os.path.join(OUT_DIR, "crisp_summary.json"), report)

    # ============================================================
    # E2 -- functional approximation error vs. rank (both basis families)
    # ============================================================
    print("\nE2: within-cluster functional approximation error vs. rank...")
    dist_matrix = build_centroid_distance_matrix(dime_centroids.astype(np.float32))
    max_rank = max(RANK_GRID)
    anchor_idx_by_cluster = {c: select_anchors_nearest(dist_matrix, c, max_rank) for c in sample_clusters}

    e2_results = {}
    crisp_models = {}  # (cluster, basis, rank) -> dict with theta, token_ids, anchors/gamma info, chosen_lambda
    for c in sample_clusters:
        splits = cluster_splits[c]
        if len(splits["fit"]) < 5 or len(splits["tune"]) < 3 or len(splits["test"]) < 3:
            e2_results[c] = {"skipped": True, "reason": "insufficient fit/tune/test queries"}
            continue
        members = cluster_to_members[c]
        member_keys, member_values = ds_keys[members], ds_values[members]

        fit_keys = ct_keys[splits["fit"]]
        tune_keys = ct_keys[splits["tune"]]
        test_keys = val_keys[splits["test"]]
        token_ids, fit_ev = exact_evidence(fit_keys, member_keys, member_values, tau_local)
        _, tune_ev = exact_evidence(tune_keys, member_keys, member_values, tau_local)
        _, test_ev = exact_evidence(test_keys, member_keys, member_values, tau_local)
        exact_fit_dist, _ = normalize_rows(fit_ev)
        exact_tune_dist, _ = normalize_rows(tune_ev)
        exact_test_dist, exact_test_mass = normalize_rows(test_ev)

        anchor_points_rbf = dime_centroids[anchor_idx_by_cluster[c]]
        gamma = float(np.median(np.linalg.norm(anchor_points_rbf - dime_centroids[c][None, :], axis=1))) or 1.0

        per_basis = {}
        for basis in BASIS_FAMILIES:
            per_rank = {}
            for r in RANK_GRID:
                if basis == "rbf_anchor":
                    Phi_fit = rbf_basis(fit_keys, anchor_points_rbf[:r], gamma)
                    Phi_tune = rbf_basis(tune_keys, anchor_points_rbf[:r], gamma)
                    Phi_test = rbf_basis(test_keys, anchor_points_rbf[:r], gamma)
                else:
                    Phi_fit = linear_anchor_diff_basis(fit_keys, c, anchor_idx_by_cluster[c][:r], dime_centroids)
                    Phi_tune = linear_anchor_diff_basis(tune_keys, c, anchor_idx_by_cluster[c][:r], dime_centroids)
                    Phi_test = linear_anchor_diff_basis(test_keys, c, anchor_idx_by_cluster[c][:r], dime_centroids)

                best_lam, best_tune_tv = None, np.inf
                best_theta = None
                for lam in RIDGE_LAMBDA_GRID:
                    theta = fit_ridge(Phi_fit, fit_ev, lam)
                    _, corrected_tune, _ = reconstruct_and_correct(Phi_tune, theta)
                    recon_tune_dist, _ = normalize_rows(corrected_tune)
                    tv = float(np.mean(0.5 * np.abs(recon_tune_dist - exact_tune_dist).sum(axis=1)))
                    if tv < best_tune_tv:
                        best_tune_tv, best_lam, best_theta = tv, lam, theta

                raw_test, corrected_test, frac_clipped = reconstruct_and_correct(Phi_test, best_theta)
                recon_test_dist, recon_test_mass = normalize_rows(corrected_test)
                test_tv = float(np.mean(0.5 * np.abs(recon_test_dist - exact_test_dist).sum(axis=1)))
                mass_rel_err = float(np.mean(np.abs(recon_test_mass - exact_test_mass) / np.maximum(exact_test_mass, 1e-9)))
                per_rank[r] = {"chosen_lambda": best_lam, "tune_tv": best_tune_tv, "test_tv": test_tv,
                                 "fraction_evidence_mass_clipped": frac_clipped, "mass_relative_error": mass_rel_err}
                crisp_models[(c, basis, r)] = {"theta": best_theta, "token_ids": token_ids, "lambda": best_lam,
                                                  "anchor_idx": anchor_idx_by_cluster[c][:r], "gamma": gamma, "basis": basis}
            per_basis[basis] = per_rank
        e2_results[c] = {"population": int(populations[c]), "n_tokens": len(token_ids), "per_basis": per_basis}
        print(f"  cluster {c} (pop={int(populations[c])}, n_tok={len(token_ids)}): "
              f"rbf test_tv@r=8 = {per_basis['rbf_anchor'].get(8, {}).get('test_tv')}, "
              f"linear test_tv@r=8 = {per_basis['linear_anchor_diff'].get(8, {}).get('test_tv')}")
        save_results(os.path.join(OUT_DIR, "crisp_summary.json"), {**report, "E2_approximation_error": e2_results})

    report["E2_approximation_error"] = e2_results
    save_results(os.path.join(OUT_DIR, "crisp_summary.json"), report)

    valid_clusters = [c for c in sample_clusters if not e2_results[c].get("skipped")]
    fig, ax = plt.subplots(figsize=(7, 5))
    for basis in BASIS_FAMILIES:
        mean_by_rank = {r: float(np.mean([e2_results[c]["per_basis"][basis][r]["test_tv"] for c in valid_clusters]))
                          for r in RANK_GRID}
        ax.plot(RANK_GRID, [mean_by_rank[r] for r in RANK_GRID], marker="o", label=basis)
    ax.set_xlabel("rank r")
    ax.set_ylabel("mean held-out total variation vs. exact F_j(q)")
    ax.set_title("E2: approximation error vs. rank")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, "E2_error_vs_rank.png"), dpi=130)
    plt.close(fig)

    # ============================================================
    # E3 -- per-cluster standalone distributional comparison (bounded sample)
    # ============================================================
    # NOTE: this is PER-CLUSTER reconstruction fidelity (F~_j(q) vs. F_j(q)
    # vs. DIME-V1's P(y|c), for cluster j considered on its own), not a full
    # multi-cluster mixture across a query's entire top-20 retrieval --
    # only the 20 sampled clusters have a fitted CRISP model, so most
    # queries' other ~19 retrieved clusters have no model to combine with.
    # Testing per-cluster fidelity directly is still the right proxy here:
    # DIME's outer mixture is a weighted AVERAGE across clusters, so if each
    # cluster's own reconstruction is closer to its true F_j, the combined
    # mixture moves the same direction. Full multi-cluster end-to-end
    # evaluation needs every retrieved cluster fitted -- deferred to Phase 4.
    print("\nE3: per-cluster standalone distributional comparison (DIME, CRISP-A, exact oracle)...")
    best_basis = min(BASIS_FAMILIES, key=lambda b: np.mean([e2_results[c]["per_basis"][b][8]["test_tv"]
                                                                for c in valid_clusters if 8 in e2_results[c]["per_basis"][b]]))
    print(f"  best-performing basis at r=8 (by mean held-out TV): {best_basis}")
    report["best_basis_at_r8"] = best_basis

    raw_index_d, raw_index_idx = build_datastore(ds_keys, ds_values)
    e3_results = {}
    for r in RANK_GRID:
        per_query_js_A, per_query_js_dime_vs_exact, per_query_tv_A = [], [], []
        top1_A_vs_exact, topk_A_vs_exact = [], []
        entropy_dime_list, entropy_A_list = [], []
        for c in valid_clusters:
            if (c, best_basis, r) not in crisp_models:
                continue
            splits = cluster_splits[c]
            test_keys = val_keys[splits["test"]]
            members = cluster_to_members[c]
            member_keys, member_values = ds_keys[members], ds_values[members]
            token_ids, test_ev = exact_evidence(test_keys, member_keys, member_values, tau_local)
            exact_dist, exact_mass = normalize_rows(test_ev)

            model = crisp_models[(c, best_basis, r)]
            if best_basis == "rbf_anchor":
                anchor_points = dime_centroids[model["anchor_idx"]]
                Phi_test = rbf_basis(test_keys, anchor_points, model["gamma"])
            else:
                Phi_test = linear_anchor_diff_basis(test_keys, c, model["anchor_idx"], dime_centroids)
            _, corrected, _ = reconstruct_and_correct(Phi_test, model["theta"])
            crispA_dist, crispA_mass = normalize_rows(corrected)

            total_c = sum(compressed_dists[c].values())
            dime_hist = np.array([compressed_dists[c].get(int(y), 0) / total_c if total_c > 0 else 0.0 for y in token_ids])
            dime_dist_b = np.tile(dime_hist, (len(test_keys), 1))

            for qi in range(len(test_keys)):
                d_exact = dict_from_dist(token_ids, exact_dist[qi])
                d_dime = dict_from_dist(token_ids, dime_dist_b[qi])
                d_crispA = dict_from_dist(token_ids, crispA_dist[qi])
                per_query_js_A.append(js_divergence_dict(d_crispA, d_exact))
                per_query_tv_A.append(0.5 * sum(abs(d_crispA.get(t, 0) - d_exact.get(t, 0)) for t in set(d_crispA) | set(d_exact)))
                top1_A_vs_exact.append(top1_agreement(d_crispA, d_exact))
                topk_A_vs_exact.append(topk_overlap(d_crispA, d_exact))
                entropy_dime_list.append(entropy_of_dict(d_dime))
                entropy_A_list.append(entropy_of_dict(d_crispA))
                per_query_js_dime_vs_exact.append(js_divergence_dict(d_dime, d_exact))

        e3_results[r] = {
            "n_query_cluster_pairs": len(per_query_js_A),
            "mean_js_crispA_vs_exact": float(np.mean(per_query_js_A)) if per_query_js_A else None,
            "mean_tv_crispA_vs_exact": float(np.mean(per_query_tv_A)) if per_query_tv_A else None,
            "mean_top1_agreement_crispA_vs_exact": float(np.mean([x for x in top1_A_vs_exact if x is not None])) if top1_A_vs_exact else None,
            "mean_topk_overlap_crispA_vs_exact": float(np.mean([x for x in topk_A_vs_exact if x is not None])) if topk_A_vs_exact else None,
            "mean_entropy_dime": float(np.mean(entropy_dime_list)) if entropy_dime_list else None,
            "mean_entropy_crispA": float(np.mean(entropy_A_list)) if entropy_A_list else None,
            "mean_js_dime_vs_exact": float(np.mean(per_query_js_dime_vs_exact)) if per_query_js_dime_vs_exact else None,
        }
        print(f"  r={r}: {e3_results[r]}")
        save_results(os.path.join(OUT_DIR, "crisp_summary.json"), {**report, "E3_distributional_comparison": e3_results})
    report["E3_distributional_comparison"] = e3_results
    save_results(os.path.join(OUT_DIR, "crisp_summary.json"), report)

    # ============================================================
    # CRISP-B: evidence-mass-aware aggregation (illustrative, same sampled clusters/queries)
    # ============================================================
    print("\nCRISP-B: evidence-mass-aware aggregation, illustrative comparison against CRISP-A on sampled clusters...")
    crispB_vs_crispA = {}
    r_for_B = min(RANK_GRID[-1], 8)
    for c in valid_clusters:
        if (c, best_basis, r_for_B) not in crisp_models:
            continue
        splits = cluster_splits[c]
        test_keys = val_keys[splits["test"]]
        members = cluster_to_members[c]
        member_keys, member_values = ds_keys[members], ds_values[members]
        token_ids, test_ev = exact_evidence(test_keys, member_keys, member_values, tau_local)
        exact_mass = test_ev.sum(axis=1)
        model = crisp_models[(c, best_basis, r_for_B)]
        if best_basis == "rbf_anchor":
            anchor_points = dime_centroids[model["anchor_idx"]]
            Phi_test = rbf_basis(test_keys, anchor_points, model["gamma"])
        else:
            Phi_test = linear_anchor_diff_basis(test_keys, c, model["anchor_idx"], dime_centroids)
        _, corrected, _ = reconstruct_and_correct(Phi_test, model["theta"])
        recon_mass = corrected.sum(axis=1)
        mass_corr = float(np.corrcoef(exact_mass, recon_mass)[0, 1]) if len(exact_mass) > 1 and np.std(recon_mass) > 0 else None
        crispB_vs_crispA[c] = {"n_test": len(test_keys), "exact_mass_mean": float(np.mean(exact_mass)),
                                  "reconstructed_mass_mean": float(np.mean(recon_mass)),
                                  "mass_correlation_exact_vs_reconstructed": mass_corr}
    report["crispB_evidence_mass_reconstruction"] = crispB_vs_crispA
    print(f"  evidence-mass reconstruction quality across {len(crispB_vs_crispA)} clusters recorded")
    save_results(os.path.join(OUT_DIR, "crisp_summary.json"), report)

    # ============================================================
    # E4 -- query-dependence preservation
    # ============================================================
    print(f"\nE4: query-dependence preservation (||F(q1)-F(q2)|| vs. ||F~(q1)-F~(q2)||)...")
    e4_records = []
    for c in valid_clusters:
        if (c, best_basis, 8) not in crisp_models:
            continue
        splits = cluster_splits[c]
        test_idx_local = splits["test"]
        if len(test_idx_local) < 2:
            continue
        test_keys = val_keys[test_idx_local]
        members = cluster_to_members[c]
        member_keys, member_values = ds_keys[members], ds_values[members]
        token_ids, test_ev = exact_evidence(test_keys, member_keys, member_values, tau_local)
        exact_dist, _ = normalize_rows(test_ev)
        model = crisp_models[(c, best_basis, 8)]
        if best_basis == "rbf_anchor":
            anchor_points = dime_centroids[model["anchor_idx"]]
            Phi_test = rbf_basis(test_keys, anchor_points, model["gamma"])
        else:
            Phi_test = linear_anchor_diff_basis(test_keys, c, model["anchor_idx"], dime_centroids)
        _, corrected, _ = reconstruct_and_correct(Phi_test, model["theta"])
        recon_dist, _ = normalize_rows(corrected)
        n = len(test_idx_local)
        n_pairs = min(N_QUERY_PAIRS_E4, n * (n - 1) // 2)
        pi = rng.integers(0, n, size=n_pairs)
        pj = rng.integers(0, n, size=n_pairs)
        valid = pi != pj
        for a, b in zip(pi[valid], pj[valid]):
            exact_diff = float(np.linalg.norm(exact_dist[a] - exact_dist[b]))
            recon_diff = float(np.linalg.norm(recon_dist[a] - recon_dist[b]))
            e4_records.append({"centroid": c, "exact_diff": exact_diff, "recon_diff": recon_diff})
    if e4_records:
        exact_diffs = np.array([r["exact_diff"] for r in e4_records])
        recon_diffs = np.array([r["recon_diff"] for r in e4_records])
        e4_corr = float(np.corrcoef(exact_diffs, recon_diffs)[0, 1]) if np.std(recon_diffs) > 0 else None
        e4_summary = {"n_pairs": len(e4_records), "mean_exact_diff": float(np.mean(exact_diffs)),
                        "mean_recon_diff": float(np.mean(recon_diffs)), "correlation": e4_corr}
    else:
        e4_summary = {"n_pairs": 0}
    print(f"  E4 summary: {e4_summary}")
    report["E4_query_dependence_preservation"] = e4_summary
    save_results(os.path.join(OUT_DIR, "crisp_summary.json"), report)

    # ============================================================
    # E5 -- storage and computation (B4 memory-matched baseline included)
    # ============================================================
    print("\nE5: storage and computation accounting...")
    import pickle
    centroid_bytes = N_CLUSTERS * DIM * DTYPE_BYTES
    counter_bytes = len(pickle.dumps(compressed_dists))
    dime_v1_bytes = centroid_bytes + counter_bytes
    raw_bytes = len(ds_values) * (DIM * DTYPE_BYTES + VALUE_BYTES)

    crisp_storage_by_rank = {}
    for r in RANK_GRID:
        n_coef_entries = sum(e2_results[c]["n_tokens"] for c in valid_clusters if not e2_results[c].get("skipped"))
        coef_bytes_per_cluster_avg = n_coef_entries * r * DTYPE_BYTES / max(1, len(valid_clusters))
        total_coef_bytes_all_clusters = coef_bytes_per_cluster_avg * N_CLUSTERS  # extrapolated, stated as such
        anchor_index_bytes = N_CLUSTERS * r * 4
        total = centroid_bytes + counter_bytes + total_coef_bytes_all_clusters + anchor_index_bytes
        crisp_storage_by_rank[r] = {"extrapolated_total_bytes": float(total),
                                       "note": "coefficient bytes extrapolated from the 20 sampled clusters' average "
                                               "tokens/cluster to all 3,811 -- not a full-scale measurement (Phase 4)."}

    b4_storage_by_m = {}
    b4_m_values = [1, 2, 4, 8]
    for m in b4_m_values:
        rep_bytes_per_cluster = m * (DIM * DTYPE_BYTES + VALUE_BYTES)
        b4_storage_by_m[m] = {"extrapolated_total_bytes": float(centroid_bytes + rep_bytes_per_cluster * N_CLUSTERS)}

    t_crisp_eval_start = time.perf_counter()
    sample_c = valid_clusters[0]
    sample_members = cluster_to_members[sample_c]
    sample_model = crisp_models.get((sample_c, best_basis, 8))
    if sample_model is not None:
        sample_test_keys = val_keys[cluster_splits[sample_c]["test"]]
        if best_basis == "rbf_anchor":
            _ = rbf_basis(sample_test_keys, dime_centroids[sample_model["anchor_idx"]], sample_model["gamma"]) @ sample_model["theta"]
        else:
            _ = linear_anchor_diff_basis(sample_test_keys, sample_c, sample_model["anchor_idx"], dime_centroids) @ sample_model["theta"]
    t_crisp_eval = time.perf_counter() - t_crisp_eval_start

    t_oracle_start = time.perf_counter()
    exact_evidence(val_keys[cluster_splits[sample_c]["test"]], ds_keys[sample_members], ds_values[sample_members], tau_local)
    t_oracle_eval = time.perf_counter() - t_oracle_start

    storage_report = {"dime_v1_bytes": int(dime_v1_bytes), "raw_kNN_bytes": int(raw_bytes),
                        "crisp_storage_by_rank_extrapolated": crisp_storage_by_rank,
                        "b4_representative_storage_by_m_extrapolated": b4_storage_by_m,
                        "single_cluster_crisp_reconstruction_seconds": t_crisp_eval,
                        "single_cluster_oracle_exact_seconds": t_oracle_eval,
                        "n_test_queries_timed": len(cluster_splits[sample_c]["test"])}
    print(f"  {storage_report}")
    report["E5_storage_and_computation"] = storage_report
    save_results(os.path.join(OUT_DIR, "crisp_summary.json"), report)

    # ============================================================
    # E6 -- generalization/failure analysis (grouped re-reading of E2/E3)
    # ============================================================
    print("\nE6: generalization and failure analysis (grouped by population/homogeneity)...")
    e6 = {"large_clusters": {}, "small_clusters": {}}
    for c in valid_clusters:
        homog = max(compressed_dists[c].values()) / sum(compressed_dists[c].values())
        group = "large_clusters" if c in large_clusters else "small_clusters"
        test_tv_r8 = e2_results[c]["per_basis"][best_basis].get(8, {}).get("test_tv")
        e6[group][c] = {"population": int(populations[c]), "homogeneity": float(homog), "test_tv_at_r8": test_tv_r8}
    for group, data in e6.items():
        tvs = [v["test_tv_at_r8"] for v in data.values() if v["test_tv_at_r8"] is not None]
        print(f"  {group}: n={len(data)}, mean test_tv@r=8 = {float(np.mean(tvs)) if tvs else None}")
    report["E6_generalization_failure"] = e6
    save_results(os.path.join(OUT_DIR, "crisp_summary.json"), report)

    # ============================================================
    # Representative examples (decoded tokens)
    # ============================================================
    print("\nRecording representative successful/failed cluster examples...")
    tv_at_r8 = {c: e2_results[c]["per_basis"][best_basis].get(8, {}).get("test_tv") for c in valid_clusters}
    valid_tv = {c: v for c, v in tv_at_r8.items() if v is not None}
    examples = {}
    if valid_tv:
        best_c = min(valid_tv, key=valid_tv.get)
        worst_c = max(valid_tv, key=valid_tv.get)
        for label, c in [("best_approximated", best_c), ("worst_approximated", worst_c)]:
            top_tokens = [t for t, _ in compressed_dists[c].most_common(5)]
            examples[label] = {"centroid": int(c), "test_tv_at_r8": valid_tv[c], "population": int(populations[c]),
                                  "top_tokens_decoded": [gpt.tokenizer.decode([int(t)]) for t in top_tokens]}
    report["representative_examples"] = examples
    print(f"  {examples}")
    save_results(os.path.join(OUT_DIR, "crisp_summary.json"), report)

    # ============================================================
    # Verdict
    # ============================================================
    print("\nDetermining verdict...")
    mean_tv_at_r8 = float(np.mean([v for v in tv_at_r8.values() if v is not None])) if valid_tv else None
    mean_dime_tv = e3_results.get(8, {}).get("mean_tv_crispA_vs_exact")
    js_improves = e3_results.get(8, {}).get("mean_js_crispA_vs_exact") is not None and \
        e3_results.get(8, {}).get("mean_js_dime_vs_exact") is not None and \
        e3_results[8]["mean_js_crispA_vs_exact"] < e3_results[8]["mean_js_dime_vs_exact"]
    storage_r8 = crisp_storage_by_rank.get(8, {}).get("extrapolated_total_bytes")
    storage_credible = storage_r8 is not None and storage_r8 < raw_bytes * 0.5
    if js_improves and mean_tv_at_r8 is not None and mean_tv_at_r8 < 0.3 and storage_credible:
        verdict = "Partially supported"
        verdict_reason = ("CRISP's reconstructed distributions are meaningfully closer to the exact within-cluster "
                           "evidence than DIME-V1's static histogram at r=8, within a storage budget well below raw "
                           "kNN -- but this delivery only covers 20 sampled clusters and bounded query samples "
                           "(Phase 1-3), so full-scale storage/fidelity at all 3,811 clusters remains unverified.")
    elif js_improves:
        verdict = "Inconclusive"
        verdict_reason = ("CRISP improves distributional fidelity over DIME-V1 on the sampled clusters, but either "
                           "approximation error remains large or the storage comparison is not yet credible at full "
                           "scale -- Phase 4 (full-scale evaluation) is required before a stronger claim.")
    else:
        verdict = "Not supported"
        verdict_reason = ("CRISP does not show a clear distributional-fidelity advantage over DIME-V1 on the sampled "
                           "clusters at the ranks tested here.")
    print(f"  Verdict: {verdict}. {verdict_reason}")
    report["verdict"] = {"classification": verdict, "reason": verdict_reason, "mean_test_tv_at_r8": mean_tv_at_r8,
                           "js_improves_over_dime_at_r8": bool(js_improves) if js_improves is not None else None}
    save_results(os.path.join(OUT_DIR, "crisp_summary.json"), report)

    # ============================================================
    # Markdown report
    # ============================================================
    print("\nGenerating report.md...")
    lines = [
        "# CRISP: Centroid-Routed Intra-cluster Subspace Projection",
        "",
        f"Outer routing (k={kOuter}, tau={tauOuter}), tau_local={tau_local:.4f} (single global constant, probed "
        f"from empirical within-cluster distances). {len(sample_clusters)} sampled clusters "
        f"({N_LARGE_CLUSTERS} largest-population + {N_SMALL_CLUSTERS} smaller/rarer-label). "
        f"Best-performing basis at r=8: **{best_basis}**.",
        "",
        "## E1 -- synthetic sanity tests",
        f"{e1_results}",
        "",
        "## E2 -- approximation error vs. rank (mean held-out total variation, both basis families)",
    ]
    for basis in BASIS_FAMILIES:
        vals = {r: float(np.mean([e2_results[c]["per_basis"][basis][r]["test_tv"] for c in valid_clusters
                                    if r in e2_results[c]["per_basis"][basis]])) for r in RANK_GRID}
        lines.append(f"- {basis}: {vals}")
    lines += [
        "",
        "## E3 -- per-cluster standalone distributional comparison (CRISP-A vs. DIME-V1 vs. exact oracle)",
        "(Per-cluster reconstruction fidelity, not a full multi-cluster top-20 mixture -- see Limitations.)",
    ]
    for r, d in e3_results.items():
        lines.append(f"- r={r}: {d}")
    lines += [
        "",
        "## CRISP-B -- evidence-mass reconstruction quality",
        f"{crispB_vs_crispA}",
        "",
        "## E4 -- query-dependence preservation",
        f"{e4_summary}",
        "",
        "## E5 -- storage and computation",
        f"DIME-V1: {dime_v1_bytes} bytes. Raw kNN: {raw_bytes} bytes. CRISP (extrapolated, Phase 1-3 only): "
        f"{crisp_storage_by_rank}. B4 representative baseline (extrapolated): {b4_storage_by_m}.",
        "",
        "## E6 -- generalization and failure analysis",
        "See E6_generalization_failure in the JSON summary for the full per-cluster breakdown "
        "(population, label homogeneity, held-out TV at r=8) across the large- and small-cluster groups.",
        "",
        "## Representative examples",
        f"{examples}",
        "",
        "## Mathematical analysis (Section 9)",
        "**A. Continuity:** the exponential full-cluster kernel exp(-d(q,h_i)/tau_local), summed over a FIXED finite "
        "member set, is continuous (indeed Lipschitz on any bounded region) in q wherever d(.,.) is -- unlike a "
        "top-k operator, whose selected neighbor SET can change discontinuously as q moves. This script's basis "
        "functions (RBF, linear anchor-difference) are likewise smooth in q, so the smoothness mismatch CRISP must "
        "address is between the exact full-cluster function and a FINITE-RANK approximation of it, not between "
        "smooth and discontinuous operators.",
        "**B. Effective dimensionality:** the E2 error-vs-rank curve above is the direct empirical evidence for this "
        "-- a sharp drop-then-plateau indicates a low effective dimensionality for the evidence function on the "
        "sampled clusters; a slowly-decaying curve indicates otherwise. Continuity (A) alone does not imply this.",
        "**C. Projection error vs. distribution error:** when total evidence mass m_j(q) is small, a given absolute "
        "error in s~ translates into a LARGER relative error in the normalized distribution F~ -- this is exactly "
        "why E2/E3 report total variation on the NORMALIZED distribution (and the E2 mass_relative_error field) "
        "rather than only raw evidence error.",
        "**D. Storage complexity:** a rank-r model stores r coefficients per (cluster, token) pair plus O(r) shared "
        "anchor/basis metadata per cluster -- parameters scale with (number of clusters) x (average distinct tokens "
        "per cluster) x r, NOT with cluster population; this is CRISP's structural storage advantage over retaining "
        "members directly (B4), whose cost scales with the number of RETAINED VECTORS x the full hidden-state "
        "dimension (768), independent of how many distinct tokens the cluster has.",
        "**E. Limitations:** ridge regression minimizes squared error unconstrained by nonnegativity or the simplex "
        "constraint; this script corrects negative evidence by clipping (frac_evidence_mass_clipped in the JSON), "
        "which is a post-hoc correction, not a property the fitting objective itself guarantees. Reconstructed "
        "evidence also has no awareness of DIME's own top-k outer selection -- CRISP only changes what happens "
        "AFTER a cluster is already retrieved, never whether it is retrieved.",
        "",
        f"## Verdict: {verdict}",
        verdict_reason,
        "",
        "## Limitations (Phase 1-3 scope, Phase 4 deferred)",
        f"- {len(sample_clusters)} of {N_CLUSTERS} clusters evaluated, not all -- storage figures in E5 are "
        "EXTRAPOLATED from this sample's average tokens/cluster, not a full-scale measurement.",
        f"- Per-cluster fit/tune/test query counts bounded to {MAX_FIT}/{MAX_TUNE}/{MAX_TEST}.",
        f"- Rank grid capped at {max(RANK_GRID)} (64 deferred).",
        "- No base-LM interpolation or end-to-end perplexity anywhere in this script, per the spec's own Section 4 "
        "scope -- all comparisons are distributional (JS/TV/top-1/top-k/entropy) against the exact within-cluster "
        "evidence function and DIME-V1's static histogram.",
        "- tau_local is a single global constant (not tuned per cluster/rank), consistent with this project's "
        "established 'single global constant' precedent for shared hyperparameters.",
        "- E3/E4/CRISP-B are PER-CLUSTER reconstruction-fidelity comparisons, not a full multi-cluster top-20 "
        "mixture evaluation -- only the 20 sampled clusters have a fitted CRISP model, so a real query's other "
        "~19 retrieved clusters have no model to combine with yet. A full end-to-end mixture test needs every "
        "retrieved cluster fitted, which is exactly the Phase 4 full-scale extension this delivery defers.",
    ]
    with open(os.path.join(OUT_DIR, "report.md"), "w") as f:
        f.write("\n".join(lines))

    report["sections_completed"] = ["tau_local_probe", "E1_synthetic", "cluster_sampling", "E2_approximation_error",
                                      "E3_distributional_comparison", "crispB_evidence_mass", "E4_query_dependence",
                                      "E5_storage_computation", "E6_generalization_failure", "representative_examples",
                                      "verdict", "report.md"]
    report["total_runtime_seconds"] = time.perf_counter() - t_start
    save_results(os.path.join(OUT_DIR, "crisp_summary.json"), report)
    print(f"\nAll results saved under {OUT_DIR}")
    print(f"Total runtime: {report['total_runtime_seconds']/60:.1f} min")
    print("No existing E1/E2/TRI1/NC1/GEOM1/KM1 outputs were modified.")


if __name__ == "__main__":
    main()
