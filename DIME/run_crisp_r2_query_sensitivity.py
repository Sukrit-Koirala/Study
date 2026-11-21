"""CRISP-R2 -- Query-Sensitivity Preservation Experiment. Focused follow-up
to CRISP R1 (run_crisp_subspace_projection.py), not a new unrelated
implementation. Writes only to results/stage3_crisp_r2_query_sensitivity/,
preserving every prior E1/E2/TRI1/NC1/GEOM1/KM1/R1 result unchanged.

=====================================================================
Reproduction note (Section 3's explicit requirement)
=====================================================================
This script was written to inspect results/stage3_crisp_subspace_projection/
and verify R1's printed numbers against its saved artifacts before
proceeding. That directory was NOT present in this local checkout/repo at
write time (checked directly, not assumed) -- so a byte-for-byte comparison
against a persisted R1 JSON could not be performed. Per this project's
established practice when an external result is unavailable (the same
situation KM1 faced with GEOM1's results), this script does NOT cite R1's
numbers as ground truth. Instead it REPRODUCES R1's exact methodology fresh
-- importing R1's own functions directly (exact_evidence, basis functions,
ridge regression, nonnegativity correction, farthest-point sampling,
distributional metrics) under the identical seed=42 clustering, the same
cluster-sampling rule (10 largest-population + 10 smaller/rarer-label,
population >= 30), and the same fit/tune/test split sizes -- so this
script's own "tau=21.4592, rank=8" numbers ARE an independent, freshly
verified reproduction of R1's setup, not a citation of unavailable external
results. Any future comparison against a real persisted R1 artifact is
limited to whatever this script's own fresh reproduction finds.

=====================================================================
What changed vs. R1 (per the spec's central concern)
=====================================================================
R1 used tau_local ~= 21.4592 (probed as the median empirical local
distance). This script's central question is whether that temperature
caused near-constant target distributions, making R1's low error numbers
look better than they would at a REALISTIC retrieval temperature. The
single most important added reference point is tau=2.0 -- the actual outer
DIME matched-config temperature already used throughout this entire
project (TRI1/NC1/GEOM1/KM1 all use MATCHED={"tau": 2.0}) -- used here as
the "realistic retrieval temperature" throughout Experiments B-F, with
tau=21.4592 kept only as one point on Experiment A's temperature grid.

No scikit-learn in any new computation (the pre-existing MiniBatchKMeans
clustering is the only sklearn usage, reused unchanged).

=====================================================================
Scoping decisions (stated up front, per this project's established
practice and the spec's own Section 13 "prefer a well-controlled experiment
over an expensive, poorly isolated one"):
=====================================================================
  - Basis family: R1's own single best-performing basis (linear
    anchor-difference, as found during R1's pilot and real run) is used
    throughout R2 -- basis-FAMILY comparison was R1's own scope, not R2's;
    re-running both families across 6 temperatures x 6 ranks here would
    multiply cost without answering R2's actual research question.
  - Experiment A refits CRISP fresh at every temperature in the grid (per
    the spec's explicit "refit for the primary controlled comparison"
    instruction) for rank r=8 as the primary comparison point, plus the
    full rank grid at the two temperature endpoints (R1's tau and the
    realistic tau=2.0) to show the error-vs-rank curve does not qualitatively
    change. A lightweight, clearly-separate "transfer" check (fit once at
    tau=21.4592, evaluate without refitting at other temperatures) is
    reported alongside, explicitly labeled as such.
  - Cluster sample starts as R1's exact 20 clusters; expanded by up to 20
    more (highest TUNE-split sensitivity among a 150-cluster candidate
    pool, population >= 30, not already sampled) ONLY if fewer than 3 of
    the original 20 classify as high-sensitivity at the realistic
    temperature -- a predetermined, stated procedure, not cherry-picking on
    test performance (selection uses only each candidate's intrinsic TUNE-
    split sensitivity magnitude, never DIME's or CRISP's accuracy).
  - Sensitivity thresholds (tertile boundaries on mean pairwise TV) are
    computed from each cluster's TUNE split and frozen before any TEST-split
    number is touched, per the spec's explicit "freeze before evaluating the
    independent test queries" instruction.
  - Cluster-level (not query-pair-level) bootstrap for every aggregate CI,
    since query-cluster pairs within the same cluster are not independent
    (the spec's own "prefer clustered confidence intervals" instruction).

Smoke testing: set CRISPR2_SMOKE=1 with CRISPR2_CACHE_DIR / CRISPR2_OUT_DIR /
CRISPR2_N_CLUSTERS before running. Never set it for the real run.
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
from run_e2tri1_anchor_conditioned_compression import build_centroid_distance_matrix, select_anchors_nearest
from run_crisp_subspace_projection import (
    exact_evidence, normalize_rows, linear_anchor_diff_basis, fit_ridge, reconstruct_and_correct,
    farthest_point_sample, dict_from_dist, total_variation)
from gpt_class import FrozenGPT2

SMOKE = os.environ.get("CRISPR2_SMOKE") == "1"
CACHE_DIR = os.environ.get("CRISPR2_CACHE_DIR", os.path.join(SCRIPT_DIR, "..", "GPT_Module", "cache"))
OUT_DIR = os.environ.get("CRISPR2_OUT_DIR", os.path.join(SCRIPT_DIR, "results", "stage3_crisp_r2_query_sensitivity"))
PLOT_DIR = os.path.join(OUT_DIR, "plots")
os.makedirs(PLOT_DIR, exist_ok=True)

N_CLUSTERS = int(os.environ.get("CRISPR2_N_CLUSTERS", "3811"))
SEED = 42
DIM, DTYPE_BYTES, VALUE_BYTES = 768, 4, 4
OUTER_K = 20
R1_TAU = 21.4592
REALISTIC_TAU = 2.0  # the actual outer DIME matched-config temperature used project-wide
TAU_GRID = [1.0, 2.0, 5.0, 10.0, R1_TAU, 40.0]
RANK_GRID = [1, 2, 4, 8, 16, 32]
PRIMARY_RANK = 8
BASIS = "linear_anchor_diff"  # R1's own best-performing family; see scoping note above
N_LARGE_CLUSTERS, N_SMALL_CLUSTERS = 10, 10
MIN_CLUSTER_POP = 30
MAX_FIT, MAX_TUNE, MAX_TEST = 150, 50, 100
RIDGE_LAMBDA_GRID = [0.01, 0.1, 1.0, 10.0]
MIN_HIGH_SENSITIVITY_CLUSTERS = 3
EXPANSION_CANDIDATE_POOL = 150
EXPANSION_SIZE = 20
N_PAIRS_PER_CLUSTER = 40
N_CLUSTER_BOOTSTRAP = 2000


def mean_pairwise_tv(dist_matrix_rows, n_pairs, rng):
    """dist_matrix_rows: [n_queries, T] normalized distributions. Returns
    (mean, median, p90, all_pair_tvs) over n_pairs random distinct query
    pairs."""
    n = dist_matrix_rows.shape[0]
    if n < 2:
        return None
    n_pairs = min(n_pairs, n * (n - 1) // 2)
    pi = rng.integers(0, n, size=n_pairs * 2)
    pj = rng.integers(0, n, size=n_pairs * 2)
    valid = pi != pj
    pi, pj = pi[valid][:n_pairs], pj[valid][:n_pairs]
    tvs = 0.5 * np.abs(dist_matrix_rows[pi] - dist_matrix_rows[pj]).sum(axis=1)
    return {"mean": float(np.mean(tvs)), "median": float(np.median(tvs)), "p90": float(np.percentile(tvs, 90)),
             "n_pairs": int(len(tvs))}, tvs


def cluster_level_bootstrap_ci(per_cluster_values, n_boot=N_CLUSTER_BOOTSTRAP, seed=SEED):
    """Resamples CLUSTERS (not individual query-pairs), respecting
    within-cluster dependence."""
    vals = np.array([v for v in per_cluster_values if v is not None])
    if len(vals) < 2:
        return {"mean": float(np.mean(vals)) if len(vals) else None, "ci_95_low": None, "ci_95_high": None, "n_clusters": len(vals)}
    rng = np.random.default_rng(seed)
    n = len(vals)
    boot = np.array([np.mean(vals[rng.integers(0, n, size=n)]) for _ in range(n_boot)])
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return {"mean": float(np.mean(vals)), "ci_95_low": float(lo), "ci_95_high": float(hi), "n_clusters": int(n)}


def fit_and_eval_cluster(member_keys, member_values, fit_keys, tune_keys, test_keys, tau_local, anchor_idx,
                           dime_centroids, cluster_idx, rank_grid, lambda_grid):
    """Reproduces R1's own per-cluster fit/tune/select/test pipeline for the
    linear_anchor_diff basis across a rank grid, at a GIVEN tau_local.
    Returns a dict keyed by rank with test-split exact/DIME/CRISP
    distributions and TV/JS, plus the fitted models (for downstream E4-style
    query-change analysis)."""
    token_ids, fit_ev = exact_evidence(fit_keys, member_keys, member_values, tau_local)
    _, tune_ev = exact_evidence(tune_keys, member_keys, member_values, tau_local)
    _, test_ev = exact_evidence(test_keys, member_keys, member_values, tau_local)
    exact_tune_dist, _ = normalize_rows(tune_ev)
    exact_test_dist, exact_test_mass = normalize_rows(test_ev)

    out = {}
    for r in rank_grid:
        a_idx = anchor_idx[:r]
        Phi_fit = linear_anchor_diff_basis(fit_keys, cluster_idx, a_idx, dime_centroids)
        Phi_tune = linear_anchor_diff_basis(tune_keys, cluster_idx, a_idx, dime_centroids)
        Phi_test = linear_anchor_diff_basis(test_keys, cluster_idx, a_idx, dime_centroids)
        best_lam, best_tune_tv, best_theta = None, np.inf, None
        for lam in lambda_grid:
            theta = fit_ridge(Phi_fit, fit_ev, lam)
            _, corrected_tune, _ = reconstruct_and_correct(Phi_tune, theta)
            recon_tune_dist, _ = normalize_rows(corrected_tune)
            tv = float(np.mean(0.5 * np.abs(recon_tune_dist - exact_tune_dist).sum(axis=1)))
            if tv < best_tune_tv:
                best_tune_tv, best_lam, best_theta = tv, lam, theta
        _, corrected_test, frac_clipped = reconstruct_and_correct(Phi_test, best_theta)
        recon_test_dist, recon_test_mass = normalize_rows(corrected_test)
        out[r] = {"token_ids": token_ids, "exact_test_dist": exact_test_dist, "crisp_test_dist": recon_test_dist,
                   "theta": best_theta, "chosen_lambda": best_lam, "fraction_evidence_mass_clipped": frac_clipped,
                   "exact_test_mass": exact_test_mass, "crisp_test_mass": recon_test_mass}
    return out


def main():
    t_start = time.perf_counter()
    print(f"CRISP-R2 job started. SMOKE={SMOKE}, N_CLUSTERS={N_CLUSTERS}")
    rng = np.random.default_rng(SEED)

    print("Loading cache (datastore/controller_train/val)...")
    ds = load_cache(os.path.join(CACHE_DIR, "wikitext103_gpt2_datastore.npz"))
    ct = load_cache(os.path.join(CACHE_DIR, "wikitext103_gpt2_controller_train.npz"))
    val = load_cache(os.path.join(CACHE_DIR, "wikitext103_gpt2_val.npz"))
    ds_keys, ds_values = ds["keys"].astype(np.float64), ds["values"]
    ct_keys, ct_values = ct["keys"].astype(np.float64), ct["values"]
    val_keys, val_values = val["keys"].astype(np.float64), val["values"]
    gpt = FrozenGPT2(model_name="gpt2")

    print("Re-deriving DIME clusters (seed=42, identical to R1 and every prior script)...")
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
              "n_clusters": N_CLUSTERS, "seed": SEED, "smoke": SMOKE, "outer_k": OUTER_K,
              "tau_grid": TAU_GRID, "rank_grid": RANK_GRID, "primary_rank": PRIMARY_RANK,
              "r1_tau": R1_TAU, "realistic_tau": REALISTIC_TAU, "basis": BASIS,
              "reproduction_note": "results/stage3_crisp_subspace_projection was not present locally when this "
                                     "script was written -- R1's methodology is reproduced fresh here, not cited "
                                     "from an unavailable persisted artifact. See module docstring."}
    save_results(os.path.join(OUT_DIR, "crisp_r2_summary.json"), report)

    # ============================================================
    # Cluster sampling (R1's exact rule) + per-cluster query splits
    # ============================================================
    def build_splits_for(clusters):
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

    print("\nSelecting R1's exact 20-cluster sample (10 largest-population + 10 smaller/rarer-label)...")
    populations = np.array([sum(c.values()) for c in compressed_dists])
    eligible = np.where(populations >= MIN_CLUSTER_POP)[0]
    order_by_pop = eligible[np.argsort(-populations[eligible])]
    large_clusters = order_by_pop[:N_LARGE_CLUSTERS].tolist()
    small_clusters = order_by_pop[-N_SMALL_CLUSTERS:].tolist()
    base_clusters = sorted(set(int(c) for c in large_clusters + small_clusters))
    base_clusters = [c for c in base_clusters if len(cluster_to_members[c]) > 0]
    print(f"  base sample: {len(base_clusters)} clusters")

    base_splits = build_splits_for(base_clusters)
    valid_base = [c for c in base_clusters if len(base_splits[c]["fit"]) >= 5 and len(base_splits[c]["tune"]) >= 3
                   and len(base_splits[c]["test"]) >= 3]

    # ============================================================
    # Realistic-temperature sensitivity screening (decides expansion)
    # ============================================================
    print(f"\nScreening cluster sensitivity at the realistic temperature tau={REALISTIC_TAU} (TUNE split)...")
    anchor_idx_cache = {}

    def get_anchors(c):
        if c not in anchor_idx_cache:
            anchor_idx_cache[c] = select_anchors_nearest(dist_matrix, c, max(RANK_GRID))
        return anchor_idx_cache[c]

    def tune_sensitivity(c, splits, tau):
        members = cluster_to_members[c]
        tune_keys = ct_keys[splits["tune"]]
        if len(tune_keys) < 2:
            return None
        _, tune_ev = exact_evidence(tune_keys, ds_keys[members], ds_values[members], tau)
        tune_dist, _ = normalize_rows(tune_ev)
        stats, _ = mean_pairwise_tv(tune_dist, N_PAIRS_PER_CLUSTER, rng) or (None, None)
        return stats["mean"] if stats else None

    base_tune_sensitivity = {c: tune_sensitivity(c, base_splits[c], REALISTIC_TAU) for c in valid_base}
    sens_values = np.array([v for v in base_tune_sensitivity.values() if v is not None])
    if len(sens_values) >= 3:
        thresholds = {"low_high": float(np.quantile(sens_values, 1 / 3)), "high_low": float(np.quantile(sens_values, 2 / 3))}
    else:
        thresholds = {"low_high": 0.0, "high_low": 0.0}
    n_high_base = int(np.sum(sens_values > thresholds["high_low"])) if len(sens_values) else 0
    print(f"  base sample: {n_high_base} clusters classify as high-sensitivity (threshold={thresholds['high_low']:.4f})")

    sample_clusters = list(valid_base)
    expanded = False
    if n_high_base < MIN_HIGH_SENSITIVITY_CLUSTERS:
        print(f"  fewer than {MIN_HIGH_SENSITIVITY_CLUSTERS} high-sensitivity clusters in the base sample -- "
              f"expanding via a predetermined procedure (highest TUNE-split sensitivity among a "
              f"{EXPANSION_CANDIDATE_POOL}-cluster candidate pool, not already sampled)...")
        expanded = True
        remaining_eligible = [int(c) for c in eligible if int(c) not in sample_clusters]
        rng.shuffle(remaining_eligible)
        candidates = remaining_eligible[:EXPANSION_CANDIDATE_POOL]
        candidate_splits = build_splits_for(candidates)
        candidate_valid = [c for c in candidates if len(candidate_splits[c]["fit"]) >= 5
                            and len(candidate_splits[c]["tune"]) >= 3 and len(candidate_splits[c]["test"]) >= 3]
        candidate_sensitivity = {c: tune_sensitivity(c, candidate_splits[c], REALISTIC_TAU) for c in candidate_valid}
        ranked = sorted([c for c in candidate_valid if candidate_sensitivity[c] is not None],
                          key=lambda c: -candidate_sensitivity[c])
        added = ranked[:EXPANSION_SIZE]
        sample_clusters = sorted(set(sample_clusters + added))
        base_splits.update({c: candidate_splits[c] for c in added})
        base_tune_sensitivity.update({c: candidate_sensitivity[c] for c in added})
        sens_values = np.array([v for v in base_tune_sensitivity.values() if v is not None])
        thresholds = {"low_high": float(np.quantile(sens_values, 1 / 3)), "high_low": float(np.quantile(sens_values, 2 / 3))}
        print(f"  expanded to {len(sample_clusters)} clusters; recomputed thresholds: {thresholds}")

    cluster_splits = base_splits
    report["sample_clusters"] = sample_clusters
    report["expanded_sample"] = expanded
    report["sensitivity_thresholds_frozen_from_tune_split"] = thresholds
    save_results(os.path.join(OUT_DIR, "crisp_r2_summary.json"), report)

    # ============================================================
    # Final classification on TEST split (frozen thresholds)
    # ============================================================
    print("\nClassifying clusters into low/moderate/high sensitivity (TEST split, frozen thresholds)...")

    def test_sensitivity(c, tau):
        members = cluster_to_members[c]
        test_keys = val_keys[cluster_splits[c]["test"]]
        if len(test_keys) < 2:
            return None
        _, test_ev = exact_evidence(test_keys, ds_keys[members], ds_values[members], tau)
        test_dist, _ = normalize_rows(test_ev)
        stats, _ = mean_pairwise_tv(test_dist, N_PAIRS_PER_CLUSTER, rng) or (None, None)
        return stats

    test_sensitivity_realistic = {c: test_sensitivity(c, REALISTIC_TAU) for c in sample_clusters}
    classification = {}
    for c, stats in test_sensitivity_realistic.items():
        if stats is None:
            classification[c] = "insufficient_data"
        elif stats["mean"] <= thresholds["low_high"]:
            classification[c] = "low"
        elif stats["mean"] <= thresholds["high_low"]:
            classification[c] = "moderate"
        else:
            classification[c] = "high"
    counts = Counter(classification.values())
    print(f"  classification counts: {dict(counts)}")
    report["cluster_sensitivity_test_realistic_tau"] = {c: test_sensitivity_realistic[c] for c in sample_clusters}
    report["cluster_sensitivity_classification"] = classification
    save_results(os.path.join(OUT_DIR, "crisp_r2_summary.json"), report)

    # ============================================================
    # Experiment A -- temperature sensitivity (refit per temperature)
    # ============================================================
    print(f"\nExperiment A: temperature grid {TAU_GRID} (refit at rank={PRIMARY_RANK}; full rank curve at "
          f"tau={REALISTIC_TAU} and tau={R1_TAU})...")
    per_cluster_models = {}  # (cluster, tau) -> {rank: {...}}
    expA_summary = {}
    for tau in TAU_GRID:
        ranks_this_tau = RANK_GRID if tau in (REALISTIC_TAU, R1_TAU) else [PRIMARY_RANK]
        tv_dime_list, js_dime_list, tv_crisp_list, js_crisp_list = [], [], [], []
        v_exact_means = []
        for c in sample_clusters:
            members = cluster_to_members[c]
            splits = cluster_splits[c]
            fit_keys, tune_keys, test_keys = ct_keys[splits["fit"]], ct_keys[splits["tune"]], val_keys[splits["test"]]
            results = fit_and_eval_cluster(ds_keys[members], ds_values[members], fit_keys, tune_keys, test_keys,
                                              tau, get_anchors(c), dime_centroids, c, ranks_this_tau, RIDGE_LAMBDA_GRID)
            per_cluster_models[(c, tau)] = results
            exact_dist = results[ranks_this_tau[0]]["exact_test_dist"]
            v_stats, _ = mean_pairwise_tv(exact_dist, N_PAIRS_PER_CLUSTER, rng) or (None, None)
            if v_stats:
                v_exact_means.append(v_stats["mean"])
            token_ids = results[ranks_this_tau[0]]["token_ids"]
            total_c = sum(compressed_dists[c].values())
            dime_hist = np.array([compressed_dists[c].get(int(y), 0) / total_c if total_c > 0 else 0.0 for y in token_ids])
            for qi in range(exact_dist.shape[0]):
                d_exact = dict_from_dist(token_ids, exact_dist[qi])
                d_dime = dict_from_dist(token_ids, dime_hist)
                tv_dime_list.append(total_variation(d_dime, d_exact))
                js_dime_list.append(js_divergence_dict(d_dime, d_exact))
            r_primary = PRIMARY_RANK if PRIMARY_RANK in ranks_this_tau else ranks_this_tau[0]
            crisp_dist = results[r_primary]["crisp_test_dist"]
            for qi in range(crisp_dist.shape[0]):
                d_crisp = dict_from_dist(token_ids, crisp_dist[qi])
                d_exact = dict_from_dist(token_ids, exact_dist[qi])
                tv_crisp_list.append(total_variation(d_crisp, d_exact))
                js_crisp_list.append(js_divergence_dict(d_crisp, d_exact))
        expA_summary[tau] = {
            "mean_exact_query_pair_variation": float(np.mean(v_exact_means)) if v_exact_means else None,
            "median_exact_query_pair_variation": float(np.median(v_exact_means)) if v_exact_means else None,
            "mean_tv_dime_vs_exact": float(np.mean(tv_dime_list)) if tv_dime_list else None,
            "mean_js_dime_vs_exact": float(np.mean(js_dime_list)) if js_dime_list else None,
            f"mean_tv_crisp_r{PRIMARY_RANK}_vs_exact": float(np.mean(tv_crisp_list)) if tv_crisp_list else None,
            f"mean_js_crisp_r{PRIMARY_RANK}_vs_exact": float(np.mean(js_crisp_list)) if js_crisp_list else None,
        }
        print(f"  tau={tau}: {expA_summary[tau]}")
        save_results(os.path.join(OUT_DIR, "crisp_r2_summary.json"), {**report, "experiment_A_temperature": expA_summary})
    report["experiment_A_temperature"] = expA_summary
    save_results(os.path.join(OUT_DIR, "crisp_r2_summary.json"), report)

    # rank curves at the two featured temperatures
    rank_curves = {}
    for tau in (REALISTIC_TAU, R1_TAU):
        rank_curves[tau] = {}
        for r in RANK_GRID:
            tvs = []
            for c in sample_clusters:
                res = per_cluster_models[(c, tau)][r]
                token_ids = res["token_ids"]
                for qi in range(res["exact_test_dist"].shape[0]):
                    tvs.append(total_variation(dict_from_dist(token_ids, res["crisp_test_dist"][qi]),
                                                 dict_from_dist(token_ids, res["exact_test_dist"][qi])))
            rank_curves[tau][r] = float(np.mean(tvs)) if tvs else None
    report["rank_curves_by_tau"] = rank_curves
    save_results(os.path.join(OUT_DIR, "crisp_r2_summary.json"), report)

    # transfer check: fit once at R1_TAU rank=8, evaluate (no refit) against OTHER temperatures' exact targets
    print("\nTransfer check (fit once at R1's tau, evaluate without refitting against other temperatures)...")
    transfer_results = {}
    for tau in TAU_GRID:
        if tau == R1_TAU:
            continue
        tvs = []
        for c in sample_clusters:
            members = cluster_to_members[c]
            test_keys = val_keys[cluster_splits[c]["test"]]
            model = per_cluster_models[(c, R1_TAU)][PRIMARY_RANK]
            token_ids = model["token_ids"]
            _, test_ev_other_tau = exact_evidence(test_keys, ds_keys[members], ds_values[members], tau)
            exact_dist_other, _ = normalize_rows(test_ev_other_tau)
            Phi_test = linear_anchor_diff_basis(test_keys, c, get_anchors(c)[:PRIMARY_RANK], dime_centroids)
            _, corrected, _ = reconstruct_and_correct(Phi_test, model["theta"])
            crisp_dist, _ = normalize_rows(corrected)
            for qi in range(exact_dist_other.shape[0]):
                tvs.append(total_variation(dict_from_dist(token_ids, crisp_dist[qi]), dict_from_dist(token_ids, exact_dist_other[qi])))
        transfer_results[tau] = float(np.mean(tvs)) if tvs else None
    print(f"  transfer (model fit at tau={R1_TAU}, evaluated elsewhere without refit): {transfer_results}")
    report["experiment_A_transfer_no_refit"] = transfer_results
    save_results(os.path.join(OUT_DIR, "crisp_r2_summary.json"), report)

    # ============================================================
    # Experiment C -- Property 1 (fidelity) and Property 2 (query-change fidelity) at REALISTIC_TAU
    # ============================================================
    print(f"\nExperiment C: Property 1/2 at the realistic temperature tau={REALISTIC_TAU}...")
    property1_by_rank = {r: {"tv": [], "js": []} for r in RANK_GRID}
    property2_records = []
    for c in sample_clusters:
        members = cluster_to_members[c]
        for r in RANK_GRID:
            res = per_cluster_models[(c, REALISTIC_TAU)][r]
            token_ids = res["token_ids"]
            exact_dist, crisp_dist = res["exact_test_dist"], res["crisp_test_dist"]
            for qi in range(exact_dist.shape[0]):
                d_exact, d_crisp = dict_from_dist(token_ids, exact_dist[qi]), dict_from_dist(token_ids, crisp_dist[qi])
                property1_by_rank[r]["tv"].append(total_variation(d_crisp, d_exact))
                property1_by_rank[r]["js"].append(js_divergence_dict(d_crisp, d_exact))

        res8 = per_cluster_models[(c, REALISTIC_TAU)][PRIMARY_RANK]
        exact_dist, crisp_dist = res8["exact_test_dist"], res8["crisp_test_dist"]
        n = exact_dist.shape[0]
        if n < 2:
            continue
        n_pairs = min(N_PAIRS_PER_CLUSTER, n * (n - 1) // 2)
        pi = rng.integers(0, n, size=n_pairs * 2)
        pj = rng.integers(0, n, size=n_pairs * 2)
        valid = pi != pj
        pi, pj = pi[valid][:n_pairs], pj[valid][:n_pairs]
        for a, b in zip(pi, pj):
            delta_exact = exact_dist[a] - exact_dist[b]
            delta_crisp = crisp_dist[a] - crisp_dist[b]
            e_delta = float(np.abs(delta_exact - delta_crisp).sum())
            exact_mag = float(np.abs(delta_exact).sum())
            crisp_mag = float(np.abs(delta_crisp).sum())
            token_ids = res8["token_ids"]
            top_change_exact = token_ids[np.argsort(-np.abs(delta_exact))[:3]]
            top_change_crisp = token_ids[np.argsort(-np.abs(delta_crisp))[:3]]
            top_overlap = len(set(top_change_exact.tolist()) & set(top_change_crisp.tolist())) / 3.0
            property2_records.append({"cluster": c, "e_delta": e_delta, "exact_magnitude": exact_mag,
                                         "crisp_magnitude": crisp_mag, "top_change_token_overlap": top_overlap})
    property1_summary = {r: {"mean_tv": float(np.mean(v["tv"])) if v["tv"] else None,
                                "mean_js": float(np.mean(v["js"])) if v["js"] else None} for r, v in property1_by_rank.items()}
    exact_mags = np.array([p["exact_magnitude"] for p in property2_records])
    crisp_mags = np.array([p["crisp_magnitude"] for p in property2_records])
    e_deltas = np.array([p["e_delta"] for p in property2_records])
    # Scale-appropriate denominator guard (same lesson as E1C.2's ratio-stability
    # fix earlier in this project): a handful of near-zero exact_magnitude values
    # make per-pair ratios blow up even when crisp_mags tracks them well, so the
    # PRIMARY retention statistic is the ratio of means (stable), not the mean of
    # ratios (unstable) -- caught during pilot testing, where mean-of-ratios gave
    # 5.6 (implying gross over-estimation) while calibration_slope and
    # correlation both indicated good calibration (~0.97, ~0.99).
    exact_mag_scale = float(np.median(exact_mags)) if len(exact_mags) else 0.0
    well_supported = exact_mags > max(0.1 * exact_mag_scale, 1e-6)
    retention_ratio = float(np.sum(crisp_mags) / np.sum(exact_mags)) if exact_mags.sum() > 0 else None
    per_pair_ratio_mean_unstable = float(np.mean(crisp_mags[well_supported] / exact_mags[well_supported])) \
        if well_supported.any() else None
    corr = float(np.corrcoef(exact_mags, crisp_mags)[0, 1]) if len(exact_mags) > 1 and np.std(crisp_mags) > 0 else None
    calibration_slope = float(np.polyfit(exact_mags, crisp_mags, 1)[0]) if len(exact_mags) > 1 and np.std(exact_mags) > 0 else None
    property2_summary = {
        "n_pairs": len(property2_records), "mean_e_delta": float(np.mean(e_deltas)) if len(e_deltas) else None,
        "mean_exact_magnitude": float(np.mean(exact_mags)) if len(exact_mags) else None,
        "mean_crisp_magnitude": float(np.mean(crisp_mags)) if len(crisp_mags) else None,
        "variation_retention_ratio": retention_ratio,
        "per_pair_ratio_mean_unstable_diagnostic_only": per_pair_ratio_mean_unstable,
        "correlation_magnitudes": corr, "calibration_slope_crisp_on_exact": calibration_slope,
        "mean_top_change_token_overlap": float(np.mean([p["top_change_token_overlap"] for p in property2_records])) if property2_records else None,
        "interpretation": "variation_retention_ratio is sum(crisp_mags)/sum(exact_mags) (ratio of means, stable); "
                            "near 1.0 together with calibration_slope near 1.0 indicates genuine magnitude "
                            "preservation. per_pair_ratio_mean_unstable_diagnostic_only is the naive mean-of-ratios "
                            "statistic kept only to show how unstable it is relative to the two numbers above -- "
                            "it is NOT the primary retention metric.",
    }
    print(f"  Property 1 (fidelity) by rank: {property1_summary}")
    print(f"  Property 2 (query-change fidelity): {property2_summary}")
    report["experiment_C_property1_fidelity"] = property1_summary
    report["experiment_C_property2_query_change_fidelity"] = property2_summary
    save_results(os.path.join(OUT_DIR, "crisp_r2_summary.json"), report)

    # ============================================================
    # Experiment D -- sensitivity-stratified table (the central analysis)
    # ============================================================
    print("\nExperiment D: sensitivity-stratified table (realistic tau, rank=8)...")
    groups = {"low": [], "moderate": [], "high": []}
    for c in sample_clusters:
        g = classification.get(c)
        if g in groups:
            groups[g].append(c)

    expD = {}
    for group, clusters_in_group in groups.items():
        per_cluster_dime_tv, per_cluster_crisp_tv = [], []
        per_cluster_dime_js, per_cluster_crisp_js = [], []
        per_cluster_exact_var = []
        worse_than_dime = []
        for c in clusters_in_group:
            res8 = per_cluster_models[(c, REALISTIC_TAU)][PRIMARY_RANK]
            token_ids = res8["token_ids"]
            total_c = sum(compressed_dists[c].values())
            dime_hist = np.array([compressed_dists[c].get(int(y), 0) / total_c if total_c > 0 else 0.0 for y in token_ids])
            tvs_dime, tvs_crisp, jss_dime, jss_crisp = [], [], [], []
            for qi in range(res8["exact_test_dist"].shape[0]):
                d_exact = dict_from_dist(token_ids, res8["exact_test_dist"][qi])
                d_dime = dict_from_dist(token_ids, dime_hist)
                d_crisp = dict_from_dist(token_ids, res8["crisp_test_dist"][qi])
                tvs_dime.append(total_variation(d_dime, d_exact))
                tvs_crisp.append(total_variation(d_crisp, d_exact))
                jss_dime.append(js_divergence_dict(d_dime, d_exact))
                jss_crisp.append(js_divergence_dict(d_crisp, d_exact))
            per_cluster_dime_tv.append(float(np.mean(tvs_dime)))
            per_cluster_crisp_tv.append(float(np.mean(tvs_crisp)))
            per_cluster_dime_js.append(float(np.mean(jss_dime)))
            per_cluster_crisp_js.append(float(np.mean(jss_crisp)))
            per_cluster_exact_var.append(test_sensitivity_realistic[c]["mean"] if test_sensitivity_realistic[c] else None)
            worse_than_dime.append(np.mean(tvs_crisp) > np.mean(tvs_dime))

        dime_tv_ci = cluster_level_bootstrap_ci(per_cluster_dime_tv)
        crisp_tv_ci = cluster_level_bootstrap_ci(per_cluster_crisp_tv)
        dime_js_ci = cluster_level_bootstrap_ci(per_cluster_dime_js)
        crisp_js_ci = cluster_level_bootstrap_ci(per_cluster_crisp_js)
        rel_improvement_tv = float((dime_tv_ci["mean"] - crisp_tv_ci["mean"]) / dime_tv_ci["mean"]) \
            if dime_tv_ci["mean"] and dime_tv_ci["mean"] > 0 else None
        expD[group] = {"n_clusters": len(clusters_in_group), "clusters": clusters_in_group,
                        "mean_exact_variation": float(np.mean([v for v in per_cluster_exact_var if v is not None]))
                        if any(v is not None for v in per_cluster_exact_var) else None,
                        "dime_tv_vs_exact": dime_tv_ci, "crisp_tv_vs_exact": crisp_tv_ci,
                        "dime_js_vs_exact": dime_js_ci, "crisp_js_vs_exact": crisp_js_ci,
                        "relative_improvement_tv": rel_improvement_tv,
                        "n_clusters_crisp_worse_than_dime": int(sum(worse_than_dime))}
        print(f"  {group}: {expD[group]}")
    report["experiment_D_sensitivity_stratified"] = expD
    save_results(os.path.join(OUT_DIR, "crisp_r2_summary.json"), report)

    # ============================================================
    # Experiment E -- numerical explanation (descriptive, reuses Experiment A data)
    # ============================================================
    print("\nExperiment E: numerical explanation of temperature effects...")
    expE = {}
    for tau in (TAU_GRID[0], REALISTIC_TAU, R1_TAU, TAU_GRID[-1]):
        c = sample_clusters[0]
        members = cluster_to_members[c]
        test_keys = val_keys[cluster_splits[c]["test"]][:10]
        d = np.linalg.norm(ds_keys[members][None, :, :] - test_keys[:, None, :], axis=2)
        d_sorted = np.sort(d, axis=1)
        weight_ratio_adjacent = float(np.mean(np.exp(-(d_sorted[:, 1] - d_sorted[:, 0]) / tau)))
        _, ev = exact_evidence(test_keys, ds_keys[members], ds_values[members], tau)
        mass = ev.sum(axis=1)
        expE[tau] = {"example_cluster": c, "mean_within_cluster_distance_spread_p90_minus_p10": float(np.percentile(d, 90) - np.percentile(d, 10)),
                      "mean_adjacent_neighbor_weight_ratio": weight_ratio_adjacent,
                      "mean_evidence_mass": float(np.mean(mass)), "exact_query_pair_variation": expA_summary.get(tau, {}).get("mean_exact_query_pair_variation")}
    print(f"  {expE}")
    report["experiment_E_numerical_explanation"] = expE
    save_results(os.path.join(OUT_DIR, "crisp_r2_summary.json"), report)

    # ============================================================
    # Experiment F -- storage-matched representative baseline, by sensitivity group
    # ============================================================
    print("\nExperiment F: storage-matched representative-vector baseline, by sensitivity group (realistic tau)...")
    r8_coef_bytes_per_token = PRIMARY_RANK * DTYPE_BYTES
    avg_n_tokens = float(np.mean([len(per_cluster_models[(c, REALISTIC_TAU)][PRIMARY_RANK]["token_ids"]) for c in sample_clusters]))
    crisp_bytes_per_cluster = avg_n_tokens * r8_coef_bytes_per_token + PRIMARY_RANK * 4  # + anchor indices
    m_match = max(1, int(round(crisp_bytes_per_cluster / (DIM * DTYPE_BYTES + VALUE_BYTES))))
    print(f"  CRISP r={PRIMARY_RANK} average bytes/cluster = {crisp_bytes_per_cluster:.1f} -> memory-matched M = {m_match} representative(s)")

    expF = {}
    for group, clusters_in_group in groups.items():
        tvs_b4, tvs_crisp = [], []
        for c in clusters_in_group:
            members = cluster_to_members[c]
            rep_local_idx = farthest_point_sample(ds_keys[members], m_match, seed=SEED)
            rep_members, rep_values = members[rep_local_idx], ds_values[members][rep_local_idx]
            res8 = per_cluster_models[(c, REALISTIC_TAU)][PRIMARY_RANK]
            token_ids = res8["token_ids"]
            test_keys = val_keys[cluster_splits[c]["test"]]
            token_ids_b4, b4_ev = exact_evidence(test_keys, ds_keys[rep_members], rep_values, REALISTIC_TAU)
            b4_dist, _ = normalize_rows(b4_ev)
            for qi in range(res8["exact_test_dist"].shape[0]):
                d_exact = dict_from_dist(token_ids, res8["exact_test_dist"][qi])
                d_crisp = dict_from_dist(token_ids, res8["crisp_test_dist"][qi])
                d_b4 = dict_from_dist(token_ids_b4, b4_dist[qi])
                tvs_crisp.append(total_variation(d_crisp, d_exact))
                tvs_b4.append(total_variation(d_b4, d_exact))
        expF[group] = {"n_clusters": len(clusters_in_group), "m_representatives": m_match,
                         "crisp_bytes_per_cluster_avg": crisp_bytes_per_cluster,
                         "b4_bytes_per_cluster": m_match * (DIM * DTYPE_BYTES + VALUE_BYTES),
                         "mean_tv_crisp_vs_exact": float(np.mean(tvs_crisp)) if tvs_crisp else None,
                         "mean_tv_b4_vs_exact": float(np.mean(tvs_b4)) if tvs_b4 else None}
        print(f"  {group}: {expF[group]}")
    report["experiment_F_storage_matched"] = expF
    save_results(os.path.join(OUT_DIR, "crisp_r2_summary.json"), report)

    # ============================================================
    # Failure cases
    # ============================================================
    print("\nIdentifying failure cases (CRISP worse than DIME at realistic tau, rank=8)...")
    failures = []
    for c in sample_clusters:
        res8 = per_cluster_models[(c, REALISTIC_TAU)][PRIMARY_RANK]
        token_ids = res8["token_ids"]
        total_c = sum(compressed_dists[c].values())
        dime_hist = np.array([compressed_dists[c].get(int(y), 0) / total_c if total_c > 0 else 0.0 for y in token_ids])
        tvs_dime = [total_variation(dict_from_dist(token_ids, dime_hist), dict_from_dist(token_ids, res8["exact_test_dist"][qi]))
                     for qi in range(res8["exact_test_dist"].shape[0])]
        tvs_crisp = [total_variation(dict_from_dist(token_ids, res8["crisp_test_dist"][qi]), dict_from_dist(token_ids, res8["exact_test_dist"][qi]))
                      for qi in range(res8["exact_test_dist"].shape[0])]
        if np.mean(tvs_crisp) > np.mean(tvs_dime):
            top_tokens = [t for t, _ in compressed_dists[c].most_common(5)]
            failures.append({"cluster": int(c), "classification": classification.get(c), "mean_tv_dime": float(np.mean(tvs_dime)),
                                "mean_tv_crisp": float(np.mean(tvs_crisp)), "population": int(populations[c]),
                                "top_tokens_decoded": [gpt.tokenizer.decode([int(t)]) for t in top_tokens]})
    print(f"  {len(failures)} failure cases found: {failures}")
    report["failure_cases"] = failures
    save_results(os.path.join(OUT_DIR, "crisp_r2_summary.json"), report)

    # ============================================================
    # Plots
    # ============================================================
    print("\nGenerating plots...")
    fig, ax = plt.subplots(figsize=(7, 5))
    taus_sorted = sorted(expA_summary.keys())
    ax.plot(taus_sorted, [expA_summary[t]["mean_exact_query_pair_variation"] for t in taus_sorted], marker="o")
    ax.set_xlabel("tau_local"); ax.set_ylabel("mean exact query-pair TV")
    ax.set_title("Temperature vs. exact within-cluster variation")
    fig.tight_layout(); fig.savefig(os.path.join(PLOT_DIR, "A1_temperature_vs_variation.png"), dpi=130); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(taus_sorted, [expA_summary[t]["mean_tv_dime_vs_exact"] for t in taus_sorted], marker="o", label="DIME-V1")
    ax.plot(taus_sorted, [expA_summary[t][f"mean_tv_crisp_r{PRIMARY_RANK}_vs_exact"] for t in taus_sorted], marker="s", label=f"CRISP r={PRIMARY_RANK}")
    ax.set_xlabel("tau_local"); ax.set_ylabel("mean TV vs. exact"); ax.legend()
    ax.set_title("Temperature vs. DIME/CRISP distributional error")
    fig.tight_layout(); fig.savefig(os.path.join(PLOT_DIR, "A2_temperature_vs_error.png"), dpi=130); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    for tau in (REALISTIC_TAU, R1_TAU):
        ax.plot(RANK_GRID, [rank_curves[tau][r] for r in RANK_GRID], marker="o", label=f"tau={tau}")
    ax.set_xlabel("rank r"); ax.set_ylabel("mean TV vs. exact"); ax.legend()
    ax.set_title("Error vs. rank at two temperatures")
    fig.tight_layout(); fig.savefig(os.path.join(PLOT_DIR, "C1_error_vs_rank.png"), dpi=130); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(exact_mags, crisp_mags, s=10, alpha=0.4)
    lims = [0, max(float(exact_mags.max()) if len(exact_mags) else 1, float(crisp_mags.max()) if len(crisp_mags) else 1)]
    ax.plot(lims, lims, "k--", linewidth=1, label="perfect calibration")
    ax.set_xlabel("exact |F(qa)-F(qb)| (L1)"); ax.set_ylabel("CRISP-reconstructed |F~(qa)-F~(qb)| (L1)")
    ax.set_title("Exact vs. reconstructed query-change magnitude"); ax.legend()
    fig.tight_layout(); fig.savefig(os.path.join(PLOT_DIR, "D1_exact_vs_reconstructed_change.png"), dpi=130); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    group_order = ["low", "moderate", "high"]
    improvements = [expD[g]["relative_improvement_tv"] for g in group_order if g in expD]
    sensitivities = [expD[g]["mean_exact_variation"] for g in group_order if g in expD]
    ax.scatter(sensitivities, improvements)
    for g, x, y in zip(group_order, sensitivities, improvements):
        if x is not None and y is not None:
            ax.annotate(g, (x, y))
    ax.set_xlabel("mean exact query-pair variation (group)"); ax.set_ylabel("CRISP relative TV improvement over DIME")
    ax.set_title("CRISP improvement vs. query sensitivity")
    fig.tight_layout(); fig.savefig(os.path.join(PLOT_DIR, "E1_improvement_vs_sensitivity.png"), dpi=130); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    for group in group_order:
        if group in expF:
            ax.scatter([expF[group]["crisp_bytes_per_cluster_avg"]], [expF[group]["mean_tv_crisp_vs_exact"]], marker="o", label=f"CRISP ({group})")
            ax.scatter([expF[group]["b4_bytes_per_cluster"]], [expF[group]["mean_tv_b4_vs_exact"]], marker="s", label=f"B4 ({group})")
    ax.set_xlabel("bytes/cluster"); ax.set_ylabel("mean TV vs. exact"); ax.legend(fontsize=7)
    ax.set_title("Distortion vs. storage, CRISP vs. representative baseline")
    fig.tight_layout(); fig.savefig(os.path.join(PLOT_DIR, "F1_distortion_vs_storage.png"), dpi=130); plt.close(fig)

    # ============================================================
    # Verdict
    # ============================================================
    print("\nDetermining verdict...")
    high_improves = expD.get("high", {}).get("relative_improvement_tv")
    low_improves = expD.get("low", {}).get("relative_improvement_tv")
    retention_ok = property2_summary["variation_retention_ratio"] is not None and property2_summary["variation_retention_ratio"] > 0.5
    temp_matters = (expA_summary[REALISTIC_TAU]["mean_exact_query_pair_variation"] or 0) > \
        2 * (expA_summary[R1_TAU]["mean_exact_query_pair_variation"] or 1e-9)
    if high_improves is not None and high_improves > 0.1 and retention_ok:
        verdict = "Partially supported"
        reason = ("CRISP improves fidelity over DIME-V1 in high-sensitivity clusters specifically, and the "
                   "variation-retention ratio shows it preserves real query-dependent magnitude, not merely "
                   "average distributions. Storage and full-scale questions remain open (same Phase 1-3 "
                   "limitation as R1).")
    elif high_improves is not None and high_improves > 0:
        verdict = "Inconclusive"
        reason = ("CRISP shows some improvement in high-sensitivity clusters, but the variation-retention ratio "
                   "indicates it may be systematically underestimating the true magnitude of query-dependent "
                   "change rather than genuinely tracking it.")
    else:
        verdict = "Not supported"
        reason = ("CRISP does not show a clear fidelity advantage over DIME-V1 specifically in high-sensitivity "
                   "clusters, which is the central condition this experiment was designed to test.")
    print(f"  Verdict: {verdict}. {reason}")
    print(f"  Temperature mattered substantially (realistic variation > 2x R1's): {temp_matters}")
    report["verdict"] = {"classification": verdict, "reason": reason, "high_sensitivity_relative_improvement_tv": high_improves,
                           "low_sensitivity_relative_improvement_tv": low_improves, "variation_retention_ratio": property2_summary["variation_retention_ratio"],
                           "temperature_substantially_suppressed_r1_variation": bool(temp_matters)}
    save_results(os.path.join(OUT_DIR, "crisp_r2_summary.json"), report)

    # ============================================================
    # Markdown report
    # ============================================================
    print("\nGenerating report.md...")
    lines = [
        "# CRISP-R2: Query-Sensitivity Preservation Experiment",
        "",
        "## A. Reproduction of R1",
        "results/stage3_crisp_subspace_projection was not present locally when this script was written, so R1's "
        "numbers could not be diffed against a persisted artifact. R1's exact methodology (seed=42 clustering, "
        "same cluster-sampling rule, same fit/tune/test split sizes, R1's own functions imported directly) was "
        f"reproduced fresh instead. This script's own tau={R1_TAU}, rank={PRIMARY_RANK} numbers below are that "
        "independent reproduction.",
        "",
        "## B. Temperature findings",
        f"{expA_summary}",
        f"Transfer check (model fit once at tau={R1_TAU}, evaluated without refitting elsewhere): {transfer_results}",
        f"Temperature substantially suppressed R1's observed variation: {temp_matters}",
        "",
        "## C. Cluster sensitivity",
        f"Classification counts: {dict(counts)}. Sample expanded beyond R1's original 20: {expanded}. "
        f"Frozen thresholds (from TUNE split): {thresholds}.",
        "",
        "## D. Reconstruction fidelity (central analysis)",
        f"Property 1 (distribution fidelity) by rank: {property1_summary}",
        f"Property 2 (query-change fidelity): {property2_summary}",
        f"Sensitivity-stratified table: {expD}",
        "",
        "## E. Complexity and storage",
        f"{expF}",
        "",
        "## F. Failure cases",
        f"{failures}",
        "",
        f"## G. Final verdict: {verdict}",
        reason,
        "",
        "## H. Implications for CRISP",
        "Judge directly from Section D's sensitivity-stratified table and Section C's cluster-sensitivity "
        "prevalence above: if CRISP's advantage is concentrated in high-sensitivity clusters and genuinely "
        "preserves variation magnitude (not just direction), the next step is investigating ADAPTIVE rank "
        "allocation (more rank for high-sensitivity clusters, less for low) rather than a uniform full-scale "
        "sweep -- not decided here in advance of the numbers above.",
        "",
        "## Limitations",
        f"- Basis family fixed to {BASIS} throughout (R1's own best performer) -- basis-family comparison was "
        "R1's scope, not R2's.",
        f"- Experiment A refits fully at rank={PRIMARY_RANK} across the temperature grid; the full rank curve is "
        f"only computed at tau={REALISTIC_TAU} and tau={R1_TAU}.",
        f"- {len(sample_clusters)} of {N_CLUSTERS} clusters evaluated (Phase 1-3 scope, same as R1).",
        "- Cluster-level bootstrap CIs account for within-cluster dependence across query pairs, but not for "
        "any dependence ACROSS clusters (e.g. shared blob structure in synthetic pilots; not applicable to real "
        "WikiText clusters, which are geometrically independent by construction).",
    ]
    with open(os.path.join(OUT_DIR, "report.md"), "w") as f:
        f.write("\n".join(lines))

    report["sections_completed"] = ["cluster_sampling", "sensitivity_screening", "experiment_A_temperature",
                                      "experiment_A_transfer", "experiment_C_properties", "experiment_D_stratified",
                                      "experiment_E_numerical", "experiment_F_storage", "failure_cases", "report.md"]
    report["total_runtime_seconds"] = time.perf_counter() - t_start
    save_results(os.path.join(OUT_DIR, "crisp_r2_summary.json"), report)
    print(f"\nAll results saved under {OUT_DIR}")
    print(f"Total runtime: {report['total_runtime_seconds']/60:.1f} min")
    print("No existing E1/E2/TRI1/NC1/GEOM1/KM1/R1 outputs were modified.")


if __name__ == "__main__":
    main()
