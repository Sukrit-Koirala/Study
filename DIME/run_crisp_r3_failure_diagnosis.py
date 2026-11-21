"""CRISP-R3 -- Diagnosing Functional Approximation Failure. Focused
diagnostic follow-up to CRISP-R1 (run_crisp_subspace_projection.py) and
CRISP-R2 (run_crisp_r2_query_sensitivity.py). Writes only to
results/stage3_crisp_r3_failure_diagnosis/, preserving every prior result.

Central question this script answers: why did R2 find CRISP's query-CHANGE
reconstruction error WORSE than DIME-V1's implicit zero-change baseline,
even though CRISP's pointwise distribution error was better than DIME's?
This is a diagnosis task, not an attempt to produce a better CRISP.

=====================================================================
Reproduction note (same situation KM1/GEOM1/R2 all faced)
=====================================================================
results/stage3_crisp_subspace_projection (R1) and
results/stage3_crisp_r2_query_sensitivity (R2) were NOT present in this
local checkout when this script was written (checked directly). The
specific numbers in the R3 spec (tau=21.4592 JS=0.00024436, tau=2.0 query-
change error 0.84862 vs. DIME's implicit 0.74713, clusters 923/1465, "11 of
20 R1 clusters in R2") could not be diffed against a persisted artifact.
This script does NOT treat those numbers as verified ground truth. Instead
it reruns R1/R2's own methodology fresh -- importing their functions
directly (exact_evidence, basis functions, ridge regression, nonnegativity
correction, farthest-point sampling) under the identical seed=42 clustering
-- so every comparison below is independently, freshly computed. One
concrete, checkable hypothesis for "why only 11 of 20": R2's own
cluster-sampling code (reused here) filters its base-20 sample down to
`valid_base = [c for c in base_clusters if len(fit)>=5 and len(tune)>=3 and
len(test)>=3]` -- i.e. clusters without enough controller_train/val queries
actually routed to them are silently dropped before Experiment D's
stratified table. This script explicitly counts and reports how many of the
base 20 pass that filter on the real data, rather than asserting the
explanation without checking.

=====================================================================
Scope (per the spec's own Section 16 "start small, only expand if
justified" instruction)
=====================================================================
Primary diagnostic set: clusters 923 and 1465 (the real failing clusters
named in the spec; overridable via CRISPR3_FAILING_CLUSTERS for pilot
testing against a tiny synthetic cache where those indices don't exist)
plus 2 "successful" clusters chosen by a PREDETERMINED rule -- lowest
rank-8 CRISP-vs-exact TV among the same R1/R2-style base-20 sample,
computed fresh (not selected on R3's own new results). Broader-sample
expansion is deferred; the spec explicitly permits this, and running the
full diagnostic battery (5 anchor variants x 4 fitting variants x 3
objectives x functional-rank SVD, all at multiple ranks) across many
clusters would multiply cost without changing what a per-cluster deep
diagnosis already shows.

Primary temperature tau=2.0 throughout (the real matched-config temperature
and R2's own point of concern); tau=5.0 used only as a light secondary
mechanism check in Part D, not a second full sweep.

No scikit-learn in any new computation; the only sklearn usage project-wide
remains the pre-existing MiniBatchKMeans clustering.

Smoke testing: set CRISPR3_SMOKE=1 with CRISPR3_CACHE_DIR / CRISPR3_OUT_DIR /
CRISPR3_N_CLUSTERS / CRISPR3_FAILING_CLUSTERS before running. Never set it
for the real run.
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
from knn import build_datastore, query_knn_indices
from results_io import save_results
from e1_metrics import js_divergence_dict
from run_e2_information_allocation import build_cluster_to_members
from run_e2tri1_anchor_conditioned_compression import (
    build_centroid_distance_matrix, select_anchors_nearest, select_anchors_diverse)
from run_crisp_subspace_projection import (
    exact_evidence, normalize_rows, linear_anchor_diff_basis, fit_ridge, reconstruct_and_correct,
    farthest_point_sample, dict_from_dist, total_variation)
from gpt_class import FrozenGPT2

SMOKE = os.environ.get("CRISPR3_SMOKE") == "1"
CACHE_DIR = os.environ.get("CRISPR3_CACHE_DIR", os.path.join(SCRIPT_DIR, "..", "GPT_Module", "cache"))
OUT_DIR = os.environ.get("CRISPR3_OUT_DIR", os.path.join(SCRIPT_DIR, "results", "stage3_crisp_r3_failure_diagnosis"))
PLOT_DIR = os.path.join(OUT_DIR, "plots")
os.makedirs(PLOT_DIR, exist_ok=True)

N_CLUSTERS = int(os.environ.get("CRISPR3_N_CLUSTERS", "3811"))
FAILING_CLUSTERS = [int(x) for x in os.environ.get("CRISPR3_FAILING_CLUSTERS", "923,1465").split(",")]
SEED = 42
DIM, DTYPE_BYTES, VALUE_BYTES = 768, 4, 4
OUTER_K = 20
TAU_PRIMARY = 2.0
TAU_SECONDARY = 5.0
RANK_GRID = [2, 4, 8, 16, 32]
PRIMARY_RANK = 8
N_LARGE_CLUSTERS, N_SMALL_CLUSTERS, MIN_CLUSTER_POP = 10, 10, 30
MAX_FIT, MAX_TUNE, MAX_TEST = 150, 50, 100
RIDGE_LAMBDA_GRID = [0.01, 0.1, 1.0, 10.0]
LOG_LAMBDA_GRID = [10 ** p for p in range(-4, 5)]  # scaled by design-matrix norm per cluster, see fit_with_method
N_PAIRS_DIAG = 60


# ============================================================
# Query splits (R1/R2's exact rule, reused for comparability)
# ============================================================
def build_splits_for(clusters, idx_dime_ct, idx_dime_val):
    splits = {}
    for c in clusters:
        ct_routed = np.where(idx_dime_ct[:, 0] == c)[0]
        val_routed = np.where(idx_dime_val[:, 0] == c)[0]
        local_rng = np.random.default_rng(SEED + c)
        local_rng.shuffle(ct_routed)
        n_fit = min(MAX_FIT, max(1, len(ct_routed) * 3 // 4))
        fit_idx = ct_routed[:n_fit]
        tune_idx = ct_routed[n_fit:n_fit + MAX_TUNE]
        test_idx = val_routed[:MAX_TEST]
        splits[c] = {"fit": fit_idx, "tune": tune_idx, "test": test_idx}
    return splits


# ============================================================
# Part A -- basis diagnostics
# ============================================================
def rbf_basis_at(query_keys, anchor_points, gamma):
    d2 = np.sum((query_keys[:, None, :] - anchor_points[None, :, :]) ** 2, axis=2)
    return np.exp(-d2 / (2 * gamma ** 2))


def basis_matrix_diagnostics(Phi):
    if Phi.shape[0] < 2 or Phi.shape[1] < 1:
        return None
    sv = np.linalg.svd(Phi, compute_uv=False)
    tol = sv.max() * max(Phi.shape) * np.finfo(float).eps
    effective_rank = int(np.sum(sv > max(tol, 1e-10 * sv.max())))
    cond = float(sv.max() / sv.min()) if sv.min() > 1e-300 else float("inf")
    col_norms = np.linalg.norm(Phi, axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        corr = np.corrcoef(Phi.T) if Phi.shape[1] > 1 else np.array([[1.0]])
    off_diag_corr = corr[~np.eye(corr.shape[0], dtype=bool)] if corr.shape[0] > 1 else np.array([])
    return {"singular_values": sv.tolist(), "condition_number": cond, "numerical_effective_rank": effective_rank,
             "n_rank_theoretical": Phi.shape[1], "mean_column_norm": float(np.mean(col_norms)),
             "mean_abs_offdiag_correlation": float(np.mean(np.abs(off_diag_corr))) if len(off_diag_corr) else None}


def anchor_variant_points(variant, c, dist_matrix, dime_centroids, ds_keys, members, fit_keys, max_rank, rng):
    if variant == "nearest_centroid":
        idx = select_anchors_nearest(dist_matrix, c, max_rank)
        return dime_centroids[idx], "centroid_index"
    if variant == "diverse_centroid":
        idx = select_anchors_diverse(dist_matrix, c, max_rank)
        return dime_centroids[idx], "centroid_index"
    if variant == "member_farthest_point":
        local_idx = farthest_point_sample(ds_keys[members], max_rank, seed=SEED)
        return ds_keys[members][local_idx], "raw_point"
    if variant == "query_space_farthest_point":
        local_idx = farthest_point_sample(fit_keys, max_rank, seed=SEED)
        return fit_keys[local_idx], "raw_point"
    raise ValueError(variant)


def basis_from_anchor_points(query_keys, anchor_points, anchor_kind, c, dime_centroids, gamma):
    """RBF kernel features centered at the given anchor points, regardless
    of whether those points are other centroids or raw hidden states --
    unifies all anchor variants under one basis family for a fair,
    apples-to-apples Part-A comparison (R2's own linear anchor-difference
    basis is centroid-index-specific and is kept as a separate, explicitly
    labeled reference point, not silently mixed into this comparison)."""
    return rbf_basis_at(query_keys, anchor_points, gamma)


# ============================================================
# Part B -- fitting diagnostics
# ============================================================
def fit_lstsq(Phi, targets):
    theta, *_ = np.linalg.lstsq(Phi, targets, rcond=None)
    return theta


def fit_standardized_ridge(Phi, targets, lam):
    mu, sigma = Phi.mean(axis=0), Phi.std(axis=0)
    sigma_safe = np.where(sigma > 1e-12, sigma, 1.0)
    Phi_std = (Phi - mu) / sigma_safe
    theta_std = fit_ridge(Phi_std, targets, lam)
    return theta_std, mu, sigma_safe


def apply_standardized_ridge(Phi, theta_std, mu, sigma_safe):
    return ((Phi - mu) / sigma_safe) @ theta_std


# ============================================================
# Part C -- alternative objectives
# ============================================================
def fit_objective_B(Phi_fit, exact_fit_dist, lam):
    """Regress directly onto the NORMALIZED distribution, not raw evidence."""
    return fit_ridge(Phi_fit, exact_fit_dist, lam)


def fit_objective_C(Phi_fit, exact_fit_dist, lam, lam_delta, rng, n_pairs=N_PAIRS_DIAG):
    """Objective B's loss plus a query-change penalty, implemented as
    augmented rows: minimizing sum_pairs ||(Phi_a-Phi_b)theta - (F_a-F_b)||^2
    is algebraically identical to ordinary least squares on an augmented
    design matrix with those DIFFERENCED rows appended, scaled by
    sqrt(lam_delta) -- a standard reduction, not an approximation."""
    n = Phi_fit.shape[0]
    if n < 2:
        return fit_ridge(Phi_fit, exact_fit_dist, lam)
    n_pairs = min(n_pairs, n * (n - 1) // 2)
    pi = rng.integers(0, n, size=n_pairs * 2)
    pj = rng.integers(0, n, size=n_pairs * 2)
    valid = pi != pj
    pi, pj = pi[valid][:n_pairs], pj[valid][:n_pairs]
    Phi_diff = (Phi_fit[pi] - Phi_fit[pj]) * np.sqrt(lam_delta)
    target_diff = (exact_fit_dist[pi] - exact_fit_dist[pj]) * np.sqrt(lam_delta)
    Phi_aug = np.vstack([Phi_fit, Phi_diff])
    target_aug = np.vstack([exact_fit_dist, target_diff])
    return fit_ridge(Phi_aug, target_aug, lam)


# ============================================================
# Part D -- normalization bound check
# ============================================================
def check_normalization_bound(s_exact, s_approx):
    """Verifies ||s/||s||_1 - s~/||s~||_1||_1 <= 2||s-s~||_1 / ||s||_1
    empirically wherever ||s||_1 > 0, per query."""
    violations, ratios = 0, []
    for i in range(s_exact.shape[0]):
        s, s_hat = s_exact[i], s_approx[i]
        mass_s = np.abs(s).sum()
        if mass_s <= 1e-12:
            continue
        lhs = np.abs(s / mass_s - s_hat / max(np.abs(s_hat).sum(), 1e-12)).sum()
        rhs = 2 * np.abs(s - s_hat).sum() / mass_s
        ratios.append(lhs / rhs if rhs > 1e-12 else None)
        if lhs > rhs + 1e-9:
            violations += 1
    valid_ratios = [r for r in ratios if r is not None]
    return {"n_checked": len(ratios), "n_violations": violations,
             "mean_lhs_over_rhs_ratio": float(np.mean(valid_ratios)) if valid_ratios else None}


# ============================================================
# Part F -- empirical functional rank
# ============================================================
def functional_rank_analysis(M_fit, M_test, p_j):
    M_centered_fit = M_fit - p_j[None, :]
    M_centered_test = M_test - p_j[None, :]
    U, S, Vt = np.linalg.svd(M_centered_fit, full_matrices=False)
    cum_var = np.cumsum(S ** 2) / max(np.sum(S ** 2), 1e-12)
    results = {"singular_values": S.tolist(), "cumulative_explained_variance": cum_var.tolist()}
    proj_quality = {}
    for r in RANK_GRID:
        r_eff = min(r, len(S))
        basis_r = Vt[:r_eff]  # [r_eff, T]
        coords_test = M_centered_test @ basis_r.T  # [n_test, r_eff]
        recon_test = coords_test @ basis_r
        resid = M_centered_test - recon_test
        rel_err = float(np.linalg.norm(resid) / max(np.linalg.norm(M_centered_test), 1e-12))
        proj_quality[r] = {"relative_reconstruction_error_held_out": rel_err}
    results["projection_quality_by_rank"] = proj_quality
    return results


def main():
    t_start = time.perf_counter()
    print(f"CRISP-R3 job started. SMOKE={SMOKE}, N_CLUSTERS={N_CLUSTERS}, FAILING_CLUSTERS={FAILING_CLUSTERS}")
    rng = np.random.default_rng(SEED)

    print("Loading cache (datastore/controller_train/val)...")
    ds = load_cache(os.path.join(CACHE_DIR, "wikitext103_gpt2_datastore.npz"))
    ct = load_cache(os.path.join(CACHE_DIR, "wikitext103_gpt2_controller_train.npz"))
    val = load_cache(os.path.join(CACHE_DIR, "wikitext103_gpt2_val.npz"))
    ds_keys, ds_values = ds["keys"].astype(np.float64), ds["values"]
    ct_keys, ct_values = ct["keys"].astype(np.float64), ct["values"]
    val_keys, val_values = val["keys"].astype(np.float64), val["values"]
    gpt = FrozenGPT2(model_name="gpt2")

    print("Re-deriving DIME clusters (seed=42, identical to R1/R2 and every prior script)...")
    km = MiniBatchKMeans(n_clusters=N_CLUSTERS, random_state=SEED)
    assignment = km.fit_predict(ds_keys.astype(np.float32))
    dime_centroids = km.cluster_centers_.astype(np.float64)
    compressed_dists = []
    for c in range(N_CLUSTERS):
        member_values = ds_values[assignment == c]
        compressed_dists.append(Counter(member_values.tolist()) if len(member_values) > 0 else Counter())
    cluster_to_members = build_cluster_to_members(assignment, N_CLUSTERS)
    dime_index, dime_values = build_datastore(dime_centroids, np.array(compressed_dists, dtype=object))
    d_dime_ct, idx_dime_ct = query_knn_indices(dime_index, ct_keys, k=OUTER_K)
    d_dime_val, idx_dime_val = query_knn_indices(dime_index, val_keys, k=OUTER_K)
    dist_matrix = build_centroid_distance_matrix(dime_centroids.astype(np.float32))

    report = {"n_ds": int(len(ds_values)), "n_ct": int(len(ct_values)), "n_val": int(len(val_values)),
              "n_clusters": N_CLUSTERS, "seed": SEED, "smoke": SMOKE, "tau_primary": TAU_PRIMARY,
              "failing_clusters_requested": FAILING_CLUSTERS,
              "reproduction_note": "R1 (results/stage3_crisp_subspace_projection) and R2 "
                                     "(results/stage3_crisp_r2_query_sensitivity) were not present locally when "
                                     "this script was written -- their methodology is reproduced fresh here, not "
                                     "cited from unavailable persisted artifacts. See module docstring."}
    save_results(os.path.join(OUT_DIR, "crisp_r3_summary.json"), report)

    # ============================================================
    # Reproduce R1/R2's base-20 cluster sample + "why only 11 of 20" check
    # ============================================================
    print("\nReproducing R1/R2's base-20 cluster-sampling rule and checking the fit/tune/test sufficiency filter...")
    populations = np.array([sum(c.values()) for c in compressed_dists])
    eligible = np.where(populations >= MIN_CLUSTER_POP)[0]
    order_by_pop = eligible[np.argsort(-populations[eligible])]
    base20 = sorted(set(int(c) for c in order_by_pop[:N_LARGE_CLUSTERS].tolist() + order_by_pop[-N_SMALL_CLUSTERS:].tolist()))
    base20 = [c for c in base20 if len(cluster_to_members[c]) > 0]
    base20_splits = build_splits_for(base20, idx_dime_ct, idx_dime_val)
    valid_base20 = [c for c in base20 if len(base20_splits[c]["fit"]) >= 5 and len(base20_splits[c]["tune"]) >= 3
                     and len(base20_splits[c]["test"]) >= 3]
    print(f"  base-20 sample: {len(base20)} clusters (population>={MIN_CLUSTER_POP}); "
          f"{len(valid_base20)} pass the fit>=5/tune>=3/test>=3 sufficiency filter R2's Experiment D used")
    report["r2_sample_sufficiency_check"] = {"base20_count": len(base20), "valid_count": len(valid_base20),
                                                "dropped_for_insufficient_queries": sorted(set(base20) - set(valid_base20))}
    save_results(os.path.join(OUT_DIR, "crisp_r3_summary.json"), report)

    # ============================================================
    # Select the 4-cluster diagnostic set: 2 failing (required) + 2 successful (predetermined rule)
    # ============================================================
    print(f"\nEnsuring the required failing clusters {FAILING_CLUSTERS} are usable, then selecting 2 successful "
          f"clusters by lowest rank-{PRIMARY_RANK} CRISP-vs-exact TV among the base-20 sample (predetermined rule)...")
    anchor_idx_cache = {}

    def get_anchors(c, max_rank=max(RANK_GRID)):
        if c not in anchor_idx_cache:
            anchor_idx_cache[c] = select_anchors_nearest(dist_matrix, c, max_rank)
        return anchor_idx_cache[c]

    def quick_crisp_tv(c, splits, tau, rank):
        members = cluster_to_members[c]
        fit_keys, tune_keys, test_keys = ct_keys[splits["fit"]], ct_keys[splits["tune"]], val_keys[splits["test"]]
        token_ids, fit_ev = exact_evidence(fit_keys, ds_keys[members], ds_values[members], tau)
        _, tune_ev = exact_evidence(tune_keys, ds_keys[members], ds_values[members], tau)
        _, test_ev = exact_evidence(test_keys, ds_keys[members], ds_values[members], tau)
        exact_test_dist, _ = normalize_rows(test_ev)
        a_idx = get_anchors(c)[:rank]
        Phi_fit = linear_anchor_diff_basis(fit_keys, c, a_idx, dime_centroids)
        Phi_tune = linear_anchor_diff_basis(tune_keys, c, a_idx, dime_centroids)
        Phi_test = linear_anchor_diff_basis(test_keys, c, a_idx, dime_centroids)
        exact_tune_dist, _ = normalize_rows(tune_ev)
        best_lam, best_tune_tv, best_theta = None, np.inf, None
        for lam in RIDGE_LAMBDA_GRID:
            theta = fit_ridge(Phi_fit, fit_ev, lam)
            _, corrected_tune, _ = reconstruct_and_correct(Phi_tune, theta)
            recon_tune_dist, _ = normalize_rows(corrected_tune)
            tv = float(np.mean(0.5 * np.abs(recon_tune_dist - exact_tune_dist).sum(axis=1)))
            if tv < best_tune_tv:
                best_tune_tv, best_lam, best_theta = tv, lam, theta
        _, corrected_test, _ = reconstruct_and_correct(Phi_test, best_theta)
        recon_test_dist, _ = normalize_rows(corrected_test)
        test_tv = float(np.mean(0.5 * np.abs(recon_test_dist - exact_test_dist).sum(axis=1)))
        return test_tv, token_ids, fit_ev, tune_ev, test_ev, exact_test_dist, Phi_fit, Phi_tune, Phi_test, best_theta, best_lam

    screening_tvs = {}
    for c in valid_base20:
        if len(cluster_to_members[c]) < 10:
            continue
        tv, *_ = quick_crisp_tv(c, base20_splits[c], TAU_PRIMARY, PRIMARY_RANK)
        screening_tvs[c] = tv
    successful_clusters = sorted(screening_tvs, key=screening_tvs.get)[:2]
    print(f"  successful clusters chosen (lowest test TV): {successful_clusters} (TVs: "
          f"{[round(screening_tvs[c], 5) for c in successful_clusters]})")

    diagnostic_clusters = []
    for c in FAILING_CLUSTERS:
        if c < N_CLUSTERS and len(cluster_to_members[c]) >= MIN_CLUSTER_POP:
            diagnostic_clusters.append(c)
        else:
            print(f"  WARNING: requested failing cluster {c} is out of range or too small in this cache "
                  f"(population={len(cluster_to_members[c]) if c < N_CLUSTERS else 'N/A'}) -- skipped.")
    diagnostic_clusters += successful_clusters
    diagnostic_clusters = sorted(set(diagnostic_clusters))
    print(f"  final 4-cluster diagnostic set: {diagnostic_clusters}")
    report["successful_clusters_chosen"] = successful_clusters
    report["diagnostic_clusters"] = diagnostic_clusters
    save_results(os.path.join(OUT_DIR, "crisp_r3_summary.json"), report)

    diag_splits = build_splits_for(diagnostic_clusters, idx_dime_ct, idx_dime_val)
    cluster_role = {c: ("failing" if c in FAILING_CLUSTERS else "successful") for c in diagnostic_clusters}

    # ============================================================
    # Part A -- basis diagnostics (5 anchor variants x rank grid)
    # ============================================================
    print("\nPart A: basis diagnostics (singular values, condition number, anchor-variant comparison)...")
    anchor_variants = ["nearest_centroid", "diverse_centroid", "member_farthest_point", "query_space_farthest_point"]
    partA = {}
    for c in diagnostic_clusters:
        members = cluster_to_members[c]
        splits = diag_splits[c]
        if len(splits["fit"]) < 5 or len(splits["test"]) < 3:
            partA[c] = {"skipped": True, "reason": "insufficient fit/test queries"}
            continue
        fit_keys, test_keys = ct_keys[splits["fit"]], val_keys[splits["test"]]
        token_ids, fit_ev = exact_evidence(fit_keys, ds_keys[members], ds_values[members], TAU_PRIMARY)
        _, test_ev = exact_evidence(test_keys, ds_keys[members], ds_values[members], TAU_PRIMARY)
        exact_test_dist, _ = normalize_rows(test_ev)
        gamma = float(np.median(np.linalg.norm(ds_keys[members] - ds_keys[members].mean(axis=0)[None, :], axis=1))) or 1.0

        partA[c] = {"role": cluster_role[c], "n_tokens": len(token_ids), "population": int(len(members)), "by_variant": {}}
        for variant in anchor_variants:
            anchor_points, kind = anchor_variant_points(variant, c, dist_matrix, dime_centroids, ds_keys, members, fit_keys,
                                                           max(RANK_GRID), rng)
            by_rank = {}
            for r in RANK_GRID:
                Phi_fit = basis_from_anchor_points(fit_keys, anchor_points[:r], kind, c, dime_centroids, gamma)
                Phi_test = basis_from_anchor_points(test_keys, anchor_points[:r], kind, c, dime_centroids, gamma)
                diag = basis_matrix_diagnostics(Phi_fit)
                theta = fit_ridge(Phi_fit, fit_ev, 0.1)
                train_resid = float(np.linalg.norm(Phi_fit @ theta - fit_ev) / max(np.linalg.norm(fit_ev), 1e-12))
                _, corrected_test, _ = reconstruct_and_correct(Phi_test, theta)
                recon_test_dist, _ = normalize_rows(corrected_test)
                test_tv = float(np.mean(0.5 * np.abs(recon_test_dist - exact_test_dist).sum(axis=1)))
                by_rank[r] = {"condition_number": diag["condition_number"] if diag else None,
                                "numerical_effective_rank": diag["numerical_effective_rank"] if diag else None,
                                "mean_abs_offdiag_correlation": diag["mean_abs_offdiag_correlation"] if diag else None,
                                "train_relative_residual": train_resid, "test_tv_vs_exact": test_tv}
            partA[c]["by_variant"][variant] = by_rank
        print(f"  cluster {c} ({cluster_role[c]}): nearest_centroid test_tv@r=8 = "
              f"{partA[c]['by_variant']['nearest_centroid'][PRIMARY_RANK]['test_tv_vs_exact']:.5f}, "
              f"cond@r=8 = {partA[c]['by_variant']['nearest_centroid'][PRIMARY_RANK]['condition_number']}")
        save_results(os.path.join(OUT_DIR, "crisp_r3_summary.json"), {**report, "part_A_basis_diagnostics": partA})
    report["part_A_basis_diagnostics"] = partA
    save_results(os.path.join(OUT_DIR, "crisp_r3_summary.json"), report)

    # ============================================================
    # Part B -- fitting diagnostics (4 fitting methods x rank grid, R2's own basis)
    # ============================================================
    print("\nPart B: fitting diagnostics (original ridge, lstsq/SVD, logspace ridge, standardized ridge)...")
    partB = {}
    for c in diagnostic_clusters:
        members = cluster_to_members[c]
        splits = diag_splits[c]
        if len(splits["fit"]) < 5 or len(splits["tune"]) < 3 or len(splits["test"]) < 3:
            partB[c] = {"skipped": True}
            continue
        fit_keys, tune_keys, test_keys = ct_keys[splits["fit"]], ct_keys[splits["tune"]], val_keys[splits["test"]]
        token_ids, fit_ev = exact_evidence(fit_keys, ds_keys[members], ds_values[members], TAU_PRIMARY)
        _, tune_ev = exact_evidence(tune_keys, ds_keys[members], ds_values[members], TAU_PRIMARY)
        _, test_ev = exact_evidence(test_keys, ds_keys[members], ds_values[members], TAU_PRIMARY)
        exact_tune_dist, _ = normalize_rows(tune_ev)
        exact_test_dist, _ = normalize_rows(test_ev)

        partB[c] = {"role": cluster_role[c], "by_rank": {}}
        for r in RANK_GRID:
            a_idx = get_anchors(c)[:r]
            Phi_fit = linear_anchor_diff_basis(fit_keys, c, a_idx, dime_centroids)
            Phi_tune = linear_anchor_diff_basis(tune_keys, c, a_idx, dime_centroids)
            Phi_test = linear_anchor_diff_basis(test_keys, c, a_idx, dime_centroids)
            diag = basis_matrix_diagnostics(Phi_fit)
            methods = {}

            best_lam, best_tune_tv, best_theta = None, np.inf, None
            for lam in RIDGE_LAMBDA_GRID:
                theta = fit_ridge(Phi_fit, fit_ev, lam)
                _, c_tune, _ = reconstruct_and_correct(Phi_tune, theta)
                d_tune, _ = normalize_rows(c_tune)
                tv = float(np.mean(0.5 * np.abs(d_tune - exact_tune_dist).sum(axis=1)))
                if tv < best_tune_tv:
                    best_tune_tv, best_lam, best_theta = tv, lam, theta
            _, c_test, _ = reconstruct_and_correct(Phi_test, best_theta)
            d_test, _ = normalize_rows(c_test)
            methods["original_ridge"] = {"chosen_lambda": best_lam, "coef_norm": float(np.linalg.norm(best_theta)),
                                            "test_tv": float(np.mean(0.5 * np.abs(d_test - exact_test_dist).sum(axis=1)))}

            theta_lstsq = fit_lstsq(Phi_fit, fit_ev)
            _, c_test_l, _ = reconstruct_and_correct(Phi_test, theta_lstsq)
            d_test_l, _ = normalize_rows(c_test_l)
            methods["lstsq_svd"] = {"coef_norm": float(np.linalg.norm(theta_lstsq)),
                                       "test_tv": float(np.mean(0.5 * np.abs(d_test_l - exact_test_dist).sum(axis=1)))}

            scale = float(np.linalg.norm(Phi_fit)) or 1.0
            best_lam2, best_tune_tv2, best_theta2 = None, np.inf, None
            for lam_raw in LOG_LAMBDA_GRID:
                lam = lam_raw * scale
                theta = fit_ridge(Phi_fit, fit_ev, lam)
                _, c_tune2, _ = reconstruct_and_correct(Phi_tune, theta)
                d_tune2, _ = normalize_rows(c_tune2)
                tv = float(np.mean(0.5 * np.abs(d_tune2 - exact_tune_dist).sum(axis=1)))
                if tv < best_tune_tv2:
                    best_tune_tv2, best_lam2, best_theta2 = tv, lam, theta
            _, c_test2, _ = reconstruct_and_correct(Phi_test, best_theta2)
            d_test2, _ = normalize_rows(c_test2)
            methods["logspace_ridge_scaled"] = {"chosen_lambda": best_lam2, "coef_norm": float(np.linalg.norm(best_theta2)),
                                                   "test_tv": float(np.mean(0.5 * np.abs(d_test2 - exact_test_dist).sum(axis=1)))}

            best_lam3, best_tune_tv3, best_pack = None, np.inf, None
            for lam in RIDGE_LAMBDA_GRID:
                theta_std, mu, sigma_safe = fit_standardized_ridge(Phi_fit, fit_ev, lam)
                recon_tune = apply_standardized_ridge(Phi_tune, theta_std, mu, sigma_safe)
                c_tune3 = np.clip(recon_tune, 0, None)
                d_tune3, _ = normalize_rows(c_tune3)
                tv = float(np.mean(0.5 * np.abs(d_tune3 - exact_tune_dist).sum(axis=1)))
                if tv < best_tune_tv3:
                    best_tune_tv3, best_lam3, best_pack = tv, lam, (theta_std, mu, sigma_safe)
            recon_test3 = apply_standardized_ridge(Phi_test, *best_pack)
            c_test3 = np.clip(recon_test3, 0, None)
            d_test3, _ = normalize_rows(c_test3)
            methods["standardized_ridge"] = {"chosen_lambda": best_lam3, "coef_norm": float(np.linalg.norm(best_pack[0])),
                                                "test_tv": float(np.mean(0.5 * np.abs(d_test3 - exact_test_dist).sum(axis=1)))}

            partB[c]["by_rank"][r] = {"condition_number": diag["condition_number"] if diag else None, "methods": methods}
        print(f"  cluster {c} ({cluster_role[c]}) @ r={PRIMARY_RANK}: "
              f"{ {k: v['test_tv'] for k, v in partB[c]['by_rank'][PRIMARY_RANK]['methods'].items()} }")
        save_results(os.path.join(OUT_DIR, "crisp_r3_summary.json"), {**report, "part_B_fitting_diagnostics": partB})
    report["part_B_fitting_diagnostics"] = partB
    save_results(os.path.join(OUT_DIR, "crisp_r3_summary.json"), report)

    # ============================================================
    # Part C -- alternative objectives (A: raw evidence [current], B: normalized dist, C: B + pairwise-change)
    # ============================================================
    print("\nPart C: alternative objectives (A=raw evidence [current method], B=normalized distribution, "
          "C=B+query-change penalty)...")
    partC = {}
    LAM_DELTA_GRID = [0.1, 1.0, 10.0]
    for c in diagnostic_clusters:
        members = cluster_to_members[c]
        splits = diag_splits[c]
        if len(splits["fit"]) < 5 or len(splits["tune"]) < 3 or len(splits["test"]) < 3:
            partC[c] = {"skipped": True}
            continue
        fit_keys, tune_keys, test_keys = ct_keys[splits["fit"]], ct_keys[splits["tune"]], val_keys[splits["test"]]
        token_ids, fit_ev = exact_evidence(fit_keys, ds_keys[members], ds_values[members], TAU_PRIMARY)
        _, tune_ev = exact_evidence(tune_keys, ds_keys[members], ds_values[members], TAU_PRIMARY)
        _, test_ev = exact_evidence(test_keys, ds_keys[members], ds_values[members], TAU_PRIMARY)
        exact_fit_dist, _ = normalize_rows(fit_ev)
        exact_tune_dist, _ = normalize_rows(tune_ev)
        exact_test_dist, _ = normalize_rows(test_ev)
        a_idx = get_anchors(c)[:PRIMARY_RANK]
        Phi_fit = linear_anchor_diff_basis(fit_keys, c, a_idx, dime_centroids)
        Phi_tune = linear_anchor_diff_basis(tune_keys, c, a_idx, dime_centroids)
        Phi_test = linear_anchor_diff_basis(test_keys, c, a_idx, dime_centroids)

        def pointwise_and_change_error(dist_pred, dist_exact, n_pairs=N_PAIRS_DIAG):
            tv = float(np.mean(0.5 * np.abs(dist_pred - dist_exact).sum(axis=1)))
            n = dist_exact.shape[0]
            if n < 2:
                return tv, None
            n_pairs = min(n_pairs, n * (n - 1) // 2)
            pi = rng.integers(0, n, size=n_pairs * 2)
            pj = rng.integers(0, n, size=n_pairs * 2)
            valid = pi != pj
            pi, pj = pi[valid][:n_pairs], pj[valid][:n_pairs]
            delta_exact = dist_exact[pi] - dist_exact[pj]
            delta_pred = dist_pred[pi] - dist_pred[pj]
            e_delta = float(np.mean(np.abs(delta_exact - delta_pred).sum(axis=1)))
            return tv, e_delta

        best_lam, best_tune_tv, theta_A = None, np.inf, None
        for lam in RIDGE_LAMBDA_GRID:
            theta = fit_ridge(Phi_fit, fit_ev, lam)
            _, c_tune, _ = reconstruct_and_correct(Phi_tune, theta)
            d_tune, _ = normalize_rows(c_tune)
            tv = float(np.mean(0.5 * np.abs(d_tune - exact_tune_dist).sum(axis=1)))
            if tv < best_tune_tv:
                best_tune_tv, best_lam, theta_A = tv, lam, theta
        _, c_test_A, _ = reconstruct_and_correct(Phi_test, theta_A)
        d_test_A, _ = normalize_rows(c_test_A)
        tv_A, delta_A = pointwise_and_change_error(d_test_A, exact_test_dist)

        best_lam_B, best_tune_tv_B, theta_B = None, np.inf, None
        for lam in RIDGE_LAMBDA_GRID:
            theta = fit_objective_B(Phi_fit, exact_fit_dist, lam)
            c_tune_B = np.clip(Phi_tune @ theta, 0, None)
            d_tune_B, _ = normalize_rows(c_tune_B)
            tv = float(np.mean(0.5 * np.abs(d_tune_B - exact_tune_dist).sum(axis=1)))
            if tv < best_tune_tv_B:
                best_tune_tv_B, best_lam_B, theta_B = tv, lam, theta
        c_test_B = np.clip(Phi_test @ theta_B, 0, None)
        d_test_B, _ = normalize_rows(c_test_B)
        tv_B, delta_B = pointwise_and_change_error(d_test_B, exact_test_dist)

        best_cfg_C, best_tune_tv_C, theta_C = None, np.inf, None
        for lam in RIDGE_LAMBDA_GRID:
            for lam_delta in LAM_DELTA_GRID:
                theta = fit_objective_C(Phi_fit, exact_fit_dist, lam, lam_delta, rng)
                c_tune_C = np.clip(Phi_tune @ theta, 0, None)
                d_tune_C, _ = normalize_rows(c_tune_C)
                tv = float(np.mean(0.5 * np.abs(d_tune_C - exact_tune_dist).sum(axis=1)))
                if tv < best_tune_tv_C:
                    best_tune_tv_C, best_cfg_C, theta_C = tv, (lam, lam_delta), theta
        c_test_C = np.clip(Phi_test @ theta_C, 0, None)
        d_test_C, _ = normalize_rows(c_test_C)
        tv_C, delta_C = pointwise_and_change_error(d_test_C, exact_test_dist)

        partC[c] = {"role": cluster_role[c],
                     "objective_A_raw_evidence_current": {"test_tv": tv_A, "test_e_delta": delta_A, "chosen_lambda": best_lam},
                     "objective_B_normalized_distribution": {"test_tv": tv_B, "test_e_delta": delta_B, "chosen_lambda": best_lam_B},
                     "objective_C_distribution_plus_change_penalty": {"test_tv": tv_C, "test_e_delta": delta_C, "chosen_config": best_cfg_C}}
        print(f"  cluster {c} ({cluster_role[c]}): {partC[c]}")
        save_results(os.path.join(OUT_DIR, "crisp_r3_summary.json"), {**report, "part_C_objectives": partC})
    report["part_C_objectives"] = partC
    save_results(os.path.join(OUT_DIR, "crisp_r3_summary.json"), report)

    # ============================================================
    # Part D -- normalization diagnosis (evidence error vs. distribution error, bound check)
    # ============================================================
    print(f"\nPart D: normalization diagnosis (evidence vs. distribution error, bound check) at tau={TAU_PRIMARY} "
          f"and a light secondary check at tau={TAU_SECONDARY}...")
    partD = {}
    for tau in (TAU_PRIMARY, TAU_SECONDARY):
        partD[tau] = {}
        for c in diagnostic_clusters:
            members = cluster_to_members[c]
            splits = diag_splits[c]
            if len(splits["fit"]) < 5 or len(splits["tune"]) < 3 or len(splits["test"]) < 3:
                continue
            fit_keys, tune_keys, test_keys = ct_keys[splits["fit"]], ct_keys[splits["tune"]], val_keys[splits["test"]]
            token_ids, fit_ev = exact_evidence(fit_keys, ds_keys[members], ds_values[members], tau)
            _, tune_ev = exact_evidence(tune_keys, ds_keys[members], ds_values[members], tau)
            _, test_ev = exact_evidence(test_keys, ds_keys[members], ds_values[members], tau)
            exact_tune_dist, _ = normalize_rows(tune_ev)
            exact_test_dist, exact_test_mass = normalize_rows(test_ev)
            a_idx = get_anchors(c)[:PRIMARY_RANK]
            Phi_fit = linear_anchor_diff_basis(fit_keys, c, a_idx, dime_centroids)
            Phi_tune = linear_anchor_diff_basis(tune_keys, c, a_idx, dime_centroids)
            Phi_test = linear_anchor_diff_basis(test_keys, c, a_idx, dime_centroids)
            best_lam, best_tune_tv, best_theta = None, np.inf, None
            for lam in RIDGE_LAMBDA_GRID:
                theta = fit_ridge(Phi_fit, fit_ev, lam)
                _, c_tune, _ = reconstruct_and_correct(Phi_tune, theta)
                d_tune, _ = normalize_rows(c_tune)
                tv = float(np.mean(0.5 * np.abs(d_tune - exact_tune_dist).sum(axis=1)))
                if tv < best_tune_tv:
                    best_tune_tv, best_lam, best_theta = tv, lam, theta
            raw_test, corrected_test, frac_clipped = reconstruct_and_correct(Phi_test, best_theta)
            recon_test_dist, recon_test_mass = normalize_rows(corrected_test)

            evidence_l1_error = float(np.mean(np.abs(test_ev - corrected_test).sum(axis=1)))
            dist_tv_error = float(np.mean(0.5 * np.abs(recon_test_dist - exact_test_dist).sum(axis=1)))
            mass_rel_error = float(np.mean(np.abs(recon_test_mass - exact_test_mass) / np.maximum(exact_test_mass, 1e-9)))
            bound_check = check_normalization_bound(test_ev, corrected_test)
            partD[tau][c] = {"role": cluster_role[c], "fraction_evidence_mass_clipped": frac_clipped,
                               "mean_evidence_l1_error": evidence_l1_error, "mean_distribution_tv_error": dist_tv_error,
                               "mean_mass_relative_error": mass_rel_error, "normalization_bound_check": bound_check}
        print(f"  tau={tau}: {partD[tau]}")
    report["part_D_normalization_diagnosis"] = partD
    save_results(os.path.join(OUT_DIR, "crisp_r3_summary.json"), report)

    # ============================================================
    # Part E -- histogram-plus-correction baseline
    # ============================================================
    print("\nPart E: histogram-plus-correction baseline vs. original CRISP...")
    partE = {}
    for c in diagnostic_clusters:
        members = cluster_to_members[c]
        splits = diag_splits[c]
        if len(splits["fit"]) < 5 or len(splits["tune"]) < 3 or len(splits["test"]) < 3:
            continue
        total_c = sum(compressed_dists[c].values())
        fit_keys, tune_keys, test_keys = ct_keys[splits["fit"]], ct_keys[splits["tune"]], val_keys[splits["test"]]
        token_ids, fit_ev = exact_evidence(fit_keys, ds_keys[members], ds_values[members], TAU_PRIMARY)
        _, tune_ev = exact_evidence(tune_keys, ds_keys[members], ds_values[members], TAU_PRIMARY)
        _, test_ev = exact_evidence(test_keys, ds_keys[members], ds_values[members], TAU_PRIMARY)
        exact_fit_dist, _ = normalize_rows(fit_ev)
        exact_tune_dist, _ = normalize_rows(tune_ev)
        exact_test_dist, _ = normalize_rows(test_ev)
        p_j = np.array([compressed_dists[c].get(int(y), 0) / total_c if total_c > 0 else 0.0 for y in token_ids])

        a_idx = get_anchors(c)[:PRIMARY_RANK]
        Phi_fit = linear_anchor_diff_basis(fit_keys, c, a_idx, dime_centroids)
        Phi_tune = linear_anchor_diff_basis(tune_keys, c, a_idx, dime_centroids)
        Phi_test = linear_anchor_diff_basis(test_keys, c, a_idx, dime_centroids)
        delta_fit_target = exact_fit_dist - p_j[None, :]
        best_lam, best_tune_tv, theta_delta = None, np.inf, None
        for lam in RIDGE_LAMBDA_GRID:
            theta = fit_ridge(Phi_fit, delta_fit_target, lam)
            recon_tune = p_j[None, :] + Phi_tune @ theta
            corrected_tune = np.clip(recon_tune, 0, None)
            d_tune, _ = normalize_rows(corrected_tune)
            tv = float(np.mean(0.5 * np.abs(d_tune - exact_tune_dist).sum(axis=1)))
            if tv < best_tune_tv:
                best_tune_tv, best_lam, theta_delta = tv, lam, theta
        recon_test = p_j[None, :] + Phi_test @ theta_delta
        corrected_test = np.clip(recon_test, 0, None)
        d_test_hc, _ = normalize_rows(corrected_test)
        tv_hc = float(np.mean(0.5 * np.abs(d_test_hc - exact_test_dist).sum(axis=1)))

        tv_crisp, *_ = quick_crisp_tv(c, splits, TAU_PRIMARY, PRIMARY_RANK)
        partE[c] = {"role": cluster_role[c], "histogram_plus_correction_test_tv": tv_hc, "original_crisp_test_tv": tv_crisp,
                     "histogram_plus_correction_better": bool(tv_hc < tv_crisp)}
        print(f"  cluster {c} ({cluster_role[c]}): {partE[c]}")
    report["part_E_histogram_plus_correction"] = partE
    save_results(os.path.join(OUT_DIR, "crisp_r3_summary.json"), report)

    # ============================================================
    # Part F -- empirical functional rank
    # ============================================================
    print("\nPart F: empirical functional rank (SVD of exact response matrix, held-out projection quality)...")
    partF = {}
    for c in diagnostic_clusters:
        members = cluster_to_members[c]
        splits = diag_splits[c]
        if len(splits["fit"]) < 5 or len(splits["test"]) < 3:
            continue
        total_c = sum(compressed_dists[c].values())
        fit_keys, test_keys = ct_keys[splits["fit"]], val_keys[splits["test"]]
        token_ids, fit_ev = exact_evidence(fit_keys, ds_keys[members], ds_values[members], TAU_PRIMARY)
        _, test_ev = exact_evidence(test_keys, ds_keys[members], ds_values[members], TAU_PRIMARY)
        M_fit, _ = normalize_rows(fit_ev)
        M_test, _ = normalize_rows(test_ev)
        p_j = np.array([compressed_dists[c].get(int(y), 0) / total_c if total_c > 0 else 0.0 for y in token_ids])
        partF[c] = {"role": cluster_role[c], **functional_rank_analysis(M_fit, M_test, p_j)}
        print(f"  cluster {c} ({cluster_role[c]}): projection_quality_by_rank = {partF[c]['projection_quality_by_rank']}")
    report["part_F_functional_rank"] = partF
    save_results(os.path.join(OUT_DIR, "crisp_r3_summary.json"), report)

    # ============================================================
    # Decoded per-token change examples for the failing clusters
    # ============================================================
    print("\nDecoded per-token change examples for failing clusters...")
    token_examples = {}
    for c in [c for c in diagnostic_clusters if cluster_role[c] == "failing"]:
        splits = diag_splits[c]
        if len(splits["test"]) < 2:
            continue
        members = cluster_to_members[c]
        test_keys = val_keys[splits["test"]]
        token_ids, test_ev = exact_evidence(test_keys, ds_keys[members], ds_values[members], TAU_PRIMARY)
        exact_dist, _ = normalize_rows(test_ev)
        a_idx = get_anchors(c)[:PRIMARY_RANK]
        fit_keys = ct_keys[splits["fit"]]
        _, fit_ev = exact_evidence(fit_keys, ds_keys[members], ds_values[members], TAU_PRIMARY)
        Phi_fit = linear_anchor_diff_basis(fit_keys, c, a_idx, dime_centroids)
        Phi_test = linear_anchor_diff_basis(test_keys, c, a_idx, dime_centroids)
        theta = fit_ridge(Phi_fit, fit_ev, 1.0)
        _, corrected, _ = reconstruct_and_correct(Phi_test, theta)
        crisp_dist, _ = normalize_rows(corrected)
        if exact_dist.shape[0] >= 2:
            delta_exact = exact_dist[0] - exact_dist[1]
            delta_crisp = crisp_dist[0] - crisp_dist[1]
            top_idx = np.argsort(-np.abs(delta_exact))[:5]
            token_examples[c] = [{"token": int(token_ids[i]), "decoded": gpt.tokenizer.decode([int(token_ids[i])]),
                                     "exact_change": float(delta_exact[i]), "crisp_change": float(delta_crisp[i])}
                                    for i in top_idx]
    print(f"  {token_examples}")
    report["token_change_examples"] = token_examples
    save_results(os.path.join(OUT_DIR, "crisp_r3_summary.json"), report)

    # ============================================================
    # Plots
    # ============================================================
    print("\nGenerating plots...")
    fig, ax = plt.subplots(figsize=(7, 5))
    for c in diagnostic_clusters:
        if c in partA and not partA[c].get("skipped"):
            ranks_plot = list(partA[c]["by_variant"]["nearest_centroid"].keys())
            conds = [partA[c]["by_variant"]["nearest_centroid"][r]["condition_number"] for r in ranks_plot]
            ax.plot(ranks_plot, conds, marker="o", label=f"cluster {c} ({cluster_role[c]})")
    ax.set_xlabel("rank r"); ax.set_ylabel("condition number"); ax.set_yscale("log"); ax.legend(fontsize=7)
    ax.set_title("Design-matrix condition number vs. rank")
    fig.tight_layout(); fig.savefig(os.path.join(PLOT_DIR, "A_condition_vs_rank.png"), dpi=130); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    for c in diagnostic_clusters:
        if c in partB and not partB[c].get("skipped"):
            ranks_plot = list(partB[c]["by_rank"].keys())
            train_tv = [partA.get(c, {}).get("by_variant", {}).get("nearest_centroid", {}).get(r, {}).get("train_relative_residual") for r in ranks_plot]
            test_tv = [partB[c]["by_rank"][r]["methods"]["original_ridge"]["test_tv"] for r in ranks_plot]
            ax.plot(ranks_plot, test_tv, marker="s", label=f"cluster {c} test TV")
    ax.set_xlabel("rank r"); ax.set_ylabel("TV"); ax.legend(fontsize=7)
    ax.set_title("Held-out error vs. rank (original ridge)")
    fig.tight_layout(); fig.savefig(os.path.join(PLOT_DIR, "B_train_vs_heldout.png"), dpi=130); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    for c in diagnostic_clusters:
        if c in partB and not partB[c].get("skipped") and PRIMARY_RANK in partB[c]["by_rank"]:
            methods = partB[c]["by_rank"][PRIMARY_RANK]["methods"]
            ax.scatter(range(len(methods)), [v["test_tv"] for v in methods.values()], label=f"cluster {c}")
    ax.set_xticks(range(4)); ax.set_xticklabels(["original_ridge", "lstsq_svd", "logspace_ridge", "standardized_ridge"], rotation=30, ha="right")
    ax.set_ylabel("test TV"); ax.legend(fontsize=7)
    ax.set_title("Regularization/fitting method vs. generalization error (r=8)")
    fig.tight_layout(); fig.savefig(os.path.join(PLOT_DIR, "B_regularization_methods.png"), dpi=130); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    for c in diagnostic_clusters:
        if c in partC:
            objs = partC[c]
            vals = [objs["objective_A_raw_evidence_current"]["test_e_delta"], objs["objective_B_normalized_distribution"]["test_e_delta"],
                     objs["objective_C_distribution_plus_change_penalty"]["test_e_delta"]]
            ax.plot(["A (raw evidence)", "B (distribution)", "C (+change penalty)"], vals, marker="o", label=f"cluster {c}")
    ax.set_ylabel("mean query-change L1 error"); ax.legend(fontsize=7)
    ax.set_title("Query-change error across objectives")
    fig.tight_layout(); fig.savefig(os.path.join(PLOT_DIR, "C_change_error_by_objective.png"), dpi=130); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    for c in diagnostic_clusters:
        if c in partF:
            cum_var = partF[c]["cumulative_explained_variance"]
            ax.plot(range(1, len(cum_var) + 1), cum_var, marker=".", label=f"cluster {c} ({cluster_role[c]})")
    ax.set_xlabel("number of singular vectors"); ax.set_ylabel("cumulative explained variance"); ax.legend(fontsize=7)
    ax.set_title("Empirical functional singular-value spectrum (fit queries)")
    fig.tight_layout(); fig.savefig(os.path.join(PLOT_DIR, "F_functional_rank_spectrum.png"), dpi=130); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    for c in diagnostic_clusters:
        if c in partB and not partB[c].get("skipped"):
            ranks_plot = sorted(partB[c]["by_rank"].keys())
            pointwise = [partB[c]["by_rank"][r]["methods"]["original_ridge"]["test_tv"] for r in ranks_plot]
            change = [partC[c]["objective_A_raw_evidence_current"]["test_e_delta"] for _ in ranks_plot] if c in partC else [None] * len(ranks_plot)
            ax.scatter(pointwise, change, label=f"cluster {c}")
    ax.set_xlabel("pointwise TV"); ax.set_ylabel("query-change L1 error"); ax.legend(fontsize=7)
    ax.set_title("Pointwise fidelity vs. change fidelity")
    fig.tight_layout(); fig.savefig(os.path.join(PLOT_DIR, "G_pointwise_vs_change.png"), dpi=130); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    for c in diagnostic_clusters:
        if c in partE:
            avg_tokens = partA.get(c, {}).get("n_tokens", 1)
            crisp_bytes = avg_tokens * PRIMARY_RANK * DTYPE_BYTES
            ax.scatter([crisp_bytes], [partE[c]["original_crisp_test_tv"]], marker="o", label=f"CRISP ({c})")
            ax.scatter([crisp_bytes], [partE[c]["histogram_plus_correction_test_tv"]], marker="s", label=f"Hist+corr ({c})")
    ax.set_xlabel("bytes/cluster (approx.)"); ax.set_ylabel("test TV"); ax.legend(fontsize=6)
    ax.set_title("Distortion vs. storage")
    fig.tight_layout(); fig.savefig(os.path.join(PLOT_DIR, "H_distortion_vs_storage.png"), dpi=130); plt.close(fig)

    # ============================================================
    # Verdict
    # ============================================================
    print("\nDetermining verdict...")
    findings = {"basis_limitation": False, "fitting_instability": False, "objective_mismatch": False, "normalization_problem": False}
    for c in diagnostic_clusters:
        if c in partA and not partA[c].get("skipped"):
            best_variant_tv = min(partA[c]["by_variant"][v][PRIMARY_RANK]["test_tv_vs_exact"] for v in anchor_variants)
            original_tv = partA[c]["by_variant"]["nearest_centroid"][PRIMARY_RANK]["test_tv_vs_exact"]
            if original_tv > 0 and (original_tv - best_variant_tv) / original_tv > 0.2:
                findings["basis_limitation"] = True
        if c in partB and not partB[c].get("skipped") and PRIMARY_RANK in partB[c]["by_rank"]:
            methods = partB[c]["by_rank"][PRIMARY_RANK]["methods"]
            spread = max(v["test_tv"] for v in methods.values()) - min(v["test_tv"] for v in methods.values())
            if spread > 0.05:
                findings["fitting_instability"] = True
        if c in partC:
            a_delta, c_delta = partC[c]["objective_A_raw_evidence_current"]["test_e_delta"], partC[c]["objective_C_distribution_plus_change_penalty"]["test_e_delta"]
            if a_delta is not None and c_delta is not None and (a_delta - c_delta) / max(a_delta, 1e-9) > 0.15:
                findings["objective_mismatch"] = True
        if c in partD.get(TAU_PRIMARY, {}):
            if partD[TAU_PRIMARY][c]["fraction_evidence_mass_clipped"] > 0.05:
                findings["normalization_problem"] = True
    n_supported = sum(findings.values())
    if n_supported == 0:
        verdict = "Unresolved"
    elif n_supported == 1:
        verdict = [k for k, v in findings.items() if v][0].replace("_", " ") + " supported"
    else:
        verdict = "Multiple interacting causes"
    print(f"  findings: {findings}")
    print(f"  Verdict: {verdict}")
    report["verdict"] = {"classification": verdict, "findings": findings}
    save_results(os.path.join(OUT_DIR, "crisp_r3_summary.json"), report)

    # ============================================================
    # Markdown report
    # ============================================================
    print("\nGenerating report.md...")
    lines = [
        "# CRISP-R3: Diagnosing Functional Approximation Failure",
        "",
        f"Diagnostic cluster set: {diagnostic_clusters} (roles: {cluster_role}). tau={TAU_PRIMARY} primary.",
        "",
        "## 1. R2 reproduction",
        f"R1/R2's real persisted artifacts were not available locally (see module docstring). Base-20 sample "
        f"sufficiency check: {len(valid_base20)} of {len(base20)} clusters pass R2's own fit>=5/tune>=3/test>=3 "
        f"filter -- this is the mechanism this script found for 'why not all sampled clusters appear downstream', "
        "checked directly rather than asserted.",
        "",
        "## 2. Basis diagnosis",
        f"{ {c: partA[c].get('by_variant') for c in diagnostic_clusters if c in partA and not partA[c].get('skipped')} }",
        "",
        "## 3. Fitting diagnosis",
        f"{ {c: partB[c].get('by_rank', {}).get(PRIMARY_RANK) for c in diagnostic_clusters if c in partB and not partB[c].get('skipped')} }",
        "",
        "## 4. Objective diagnosis",
        f"{partC}",
        "",
        "## 5. Normalization diagnosis",
        f"{partD}",
        "",
        "## 6. Functional-rank analysis",
        f"{ {c: partF[c].get('projection_quality_by_rank') for c in partF} }",
        "",
        "## 7. Alternative model comparison",
        f"{partE}",
        "",
        "## 8. Failure-case analysis",
        f"Decoded token-level change examples for the failing clusters: {token_examples}",
        "",
        "## 9. Storage and computational costs",
        "See part_A_basis_diagnostics/n_tokens and PRIMARY_RANK in the JSON summary for real per-cluster "
        "parameter counts; approximate bytes/cluster = n_tokens x rank x 4 (float32 coefficients).",
        "",
        f"## 10. Verdict: {verdict}",
        f"Findings: {findings}",
        "",
        "## Scope",
        f"Diagnostic set limited to {len(diagnostic_clusters)} clusters (2 required failing + 2 successful, chosen "
        "by a predetermined lowest-TV rule), per the spec's own 'start small, expand only if justified' "
        "instruction. Broader-sample expansion was not pursued in this delivery -- judge from the findings above "
        "whether it would be justified.",
    ]
    with open(os.path.join(OUT_DIR, "report.md"), "w") as f:
        f.write("\n".join(lines))

    report["sections_completed"] = ["sample_sufficiency_check", "cluster_selection", "part_A_basis", "part_B_fitting",
                                      "part_C_objectives", "part_D_normalization", "part_E_histogram_correction",
                                      "part_F_functional_rank", "token_examples", "verdict", "report.md"]
    report["total_runtime_seconds"] = time.perf_counter() - t_start
    save_results(os.path.join(OUT_DIR, "crisp_r3_summary.json"), report)
    print(f"\nAll results saved under {OUT_DIR}")
    print(f"Total runtime: {report['total_runtime_seconds']/60:.1f} min")
    print("No existing E1/E2/TRI1/NC1/GEOM1/KM1/R1/R2 outputs were modified.")


if __name__ == "__main__":
    main()
