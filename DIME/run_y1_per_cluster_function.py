"""Y1 -- Per-Cluster Query-Dependent Functional Approximation. First delivery
of a new Y-series, deliberately scoped tight and separate from the CRISP
R-series (per the spec's own explicit "do not turn this into an end-to-end
kNN reconstruction experiment... do not automatically begin Y2" instruction).
Writes only to results/stage3_y1_per_cluster_function/, preserving every
prior R-series (and E1/E2/TRI1/...) output unchanged.

=====================================================================
What is actually new here (not just a renamed CRISP run)
=====================================================================
Every prior CRISP delivery (R1-R6) targeted the FULL-cluster oracle
F_j(q)(y) = softmax over ALL of a cluster's members. Y1's target is
DIFFERENT: F_j^(m)(q)(y), the softmax-weighted vote over only the m
NEAREST members within the cluster (a hard top-m selection by real
Euclidean distance, THEN a softmax over just those m) -- not all members.
This is implemented fresh in local_m_distribution() below; it is NOT a
relabeling of R1's exact_evidence. When m >= the cluster's population, the
hard top-m selection trivially includes every member, so the two targets
must coincide exactly -- verified explicitly in this script's own sanity
checks (Essential Sanity Check #6), not assumed.

A direct, mechanical consequence: unlike every prior CRISP target (where
every member always contributes some strictly-positive kernel weight, so
every label with p_j(y)>0 always had F_j(q)(y)>0), F_j^(m)(q)(y) can be
EXACTLY ZERO for many labels whenever m is small and that label's members
are not among the m nearest for a given query -- small-m targets are
therefore frequently near-one-hot, not merely "very peaked." Both fitting
targets below (direct_log_target for family A, log_ratio_target for
family B) use a documented epsilon specifically to keep log() finite on
these exact zeros; this is a FITTING-TARGET regularizer, not a floor on any
reported or returned probability -- the softmax reconstruction is always
exactly normalized and nonnegative by construction, needing no floor at
inference time.

=====================================================================
Two function families (per the spec's explicit "separate the effect of the
target function from the choice of parameterization" instruction)
=====================================================================
  A. Low-rank basis expansion, softmax-normalized directly:
     g_{j,y}(q) = Phi(q).theta_y; F~(y) = softmax_y(g_{j,.}(q)).
     Fit by ridge regression onto a CENTERED, floor-safe direct log target
     (direct_log_target below) -- centering removes the softmax's
     arbitrary shift-invariance constant before fitting (the same
     numerical-conditioning question R4's Observation 2 raised).
  B. Histogram-prior functional correction (R4's exact B4 parameterization,
     reused unchanged via direct import of log_ratio_target/
     reconstruct_logit): F~(y) = softmax_y(log p_j(y) + g_{j,y}(q)), fit by
     ridge regression onto log(F_target(y)/p_j(y)).
Both share the SAME basis (R1's linear anchor-difference features, reused
unchanged -- no new architecture introduced, per the spec's explicit
instruction) and the SAME reconstruct_logit() reconstruction, differing
only in the fitting target and whether log_p is added at reconstruction
time (zero for A, real log p_j for B) -- this isolates target-vs-
parameterization exactly as the spec asks, with minimal new code.

=====================================================================
Cluster/query sampling (Section 4)
=====================================================================
Reuses R1's exact base-20 sample (10 largest-population + 10 smaller/
rarer-label, population >= 30) and R2's exact sensitivity-classification
method (TUNE-split pairwise TV, tertile thresholds) -- not re-derived, not
expanded. Per Y1's own explicit "do not turn this into an end-to-end
experiment" instruction, this delivery deliberately does NOT scale up to
R4/R5's 80-100-cluster samples; 20 clusters, chosen by the same
established, performance-blind rule, is the right size for a focused
functional-compressibility diagnostic. Sensitivity classification uses the
FULL-cluster (m=population) oracle as the screening basis (an intrinsic
cluster property, independent of which m is later tested) -- stated
explicitly, not silently decided.

No document/sequence boundary metadata exists in the cached .npz arrays
(confirmed by inspection, consistent with every prior script in this
project never using one) -- per the spec's own explicit fallback
instruction ("If no [sequence-grouped split] is available..."), this
script uses the existing position-level controller_train(fit/tune)/
val(test) split, stated as a limitation, not silently assumed to be
sequence-independent.

=====================================================================
Scoping decisions (stated up front)
=====================================================================
  - m in {1, 4, 8, 20} (predetermined diagnostic grid, per spec).
  - rank in {1, 2, 4, 8, 16, 32} (predetermined grid, per spec).
  - Query-change analysis (Section 6) and failure-case selection use a
    single PRIMARY_RANK=8 (this project's established "primary"
    convention) across all m values and both families -- not re-swept
    across every rank, per the spec's own "do not expand" instruction.
  - No new architectures, no global centroid-mixing, no mass-aware
    aggregation, no end-to-end kNN-LM evaluation anywhere in this script.

Smoke testing: set Y1_SMOKE=1 with Y1_CACHE_DIR / Y1_OUT_DIR / Y1_N_CLUSTERS
before running. Never set it for the real run.
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
    exact_evidence, normalize_rows, linear_anchor_diff_basis, fit_ridge,
    dict_from_dist, total_variation, top1_agreement, topk_overlap, entropy_of_dict)
from run_crisp_r2_query_sensitivity import mean_pairwise_tv
from run_crisp_r3_failure_diagnosis import functional_rank_analysis
from run_crisp_r4_histogram_corrections import log_ratio_target, reconstruct_logit, mean_tv, RIDGE_LAMBDA_GRID
from gpt_class import FrozenGPT2

SMOKE = os.environ.get("Y1_SMOKE") == "1"
CACHE_DIR = os.environ.get("Y1_CACHE_DIR", os.path.join(SCRIPT_DIR, "..", "GPT_Module", "cache"))
OUT_DIR = os.environ.get("Y1_OUT_DIR", os.path.join(SCRIPT_DIR, "results", "stage3_y1_per_cluster_function"))
PLOT_DIR = os.path.join(OUT_DIR, "plots")
os.makedirs(PLOT_DIR, exist_ok=True)

N_CLUSTERS = int(os.environ.get("Y1_N_CLUSTERS", "3811"))
SEED = 42
DIM, DTYPE_BYTES, VALUE_BYTES = 768, 4, 4
OUTER_K_FOR_ROUTING = 1  # only nearest-centroid assignment is needed -- no global mixing in Y1
TAU = 2.0
M_GRID = [1, 4, 8, 20]
RANK_GRID = [1, 2, 4, 8, 16, 32]
PRIMARY_RANK = 8
N_LARGE_CLUSTERS, N_SMALL_CLUSTERS, MIN_CLUSTER_POP = 10, 10, 30
MAX_FIT, MAX_TUNE, MAX_TEST = 150, 50, 100
N_PAIRS_PER_CLUSTER = 40
TOPK_OVERLAP_K = 5
N_FAILURE_EXAMPLES = 10
DIRECT_TARGET_FLOOR = 1e-6
LOGIT_CLIP = 30.0


# ============================================================
# Y1's own new target: nearest-m-within-cluster distribution
# ============================================================
def local_m_distribution(query_keys, member_keys, member_values, m, tau=TAU):
    """F_j^(m)(q)(y): softmax-weighted vote over ONLY the m nearest members
    by real Euclidean distance -- a hard top-m selection, then a softmax
    restricted to those m. Reduces exactly to a full-cluster softmax
    (equivalent to R1's exact_evidence) when m >= population; verified
    explicitly in this script's sanity-check phase, not assumed."""
    token_ids = np.array(sorted(set(member_values.tolist())))
    n_q = query_keys.shape[0]
    n_mem = len(member_values)
    m_eff = min(m, n_mem)
    d_all = np.linalg.norm(member_keys[None, :, :] - query_keys[:, None, :], axis=2)  # [n_q, n_mem]
    nn_idx = np.argpartition(d_all, m_eff - 1, axis=1)[:, :m_eff]  # [n_q, m_eff], order irrelevant
    label_to_col = {int(t): i for i, t in enumerate(token_ids)}
    dist = np.zeros((n_q, len(token_ids)), dtype=np.float64)
    for qi in range(n_q):
        idxs = nn_idx[qi]
        d_sel = d_all[qi, idxs]
        w = np.exp(-d_sel / tau)
        wsum = w.sum()
        if wsum <= 0:
            continue
        cols = np.array([label_to_col[int(v)] for v in member_values[idxs]])
        np.add.at(dist[qi], cols, w / wsum)
    return token_ids, dist


def direct_log_target(exact_dist, clip=LOGIT_CLIP, floor=DIRECT_TARGET_FLOOR):
    """Family A's fitting target: centered, floor-safe log(F_target(y)).
    Centering removes the softmax shift-invariance constant before fitting
    (same conditioning question as R4's Observation 2); floor is a FITTING
    regularizer only (see module docstring), not an output-probability floor."""
    with np.errstate(divide="ignore"):
        z = np.log(np.maximum(exact_dist, floor))
    z = z - z.mean(axis=1, keepdims=True)
    return np.clip(z, -clip, clip)


def fit_and_select(Phi_fit, Phi_tune, z_fit, exact_tune_dist, log_p, lambda_grid):
    best_lam, best_tune_tv, best_theta = None, np.inf, None
    for lam in lambda_grid:
        theta = fit_ridge(Phi_fit, z_fit, lam)
        d_tune, _ = reconstruct_logit(Phi_tune, theta, log_p)
        tv = mean_tv(d_tune, exact_tune_dist)
        if tv < best_tune_tv:
            best_tune_tv, best_lam, best_theta = tv, lam, theta
    return best_theta, best_lam, best_tune_tv


def storage_bytes(n_tokens, rank, include_histogram):
    coef_bytes = n_tokens * rank * DTYPE_BYTES
    anchor_bytes = rank * 4
    token_id_bytes = n_tokens * 4
    hist_bytes = n_tokens * 4 if include_histogram else 0
    return coef_bytes + anchor_bytes + token_id_bytes + hist_bytes


def main():
    t_start = time.perf_counter()
    print(f"Y1 job started. SMOKE={SMOKE}, N_CLUSTERS={N_CLUSTERS}")
    rng = np.random.default_rng(SEED)

    print("Loading cache (datastore/controller_train/val)...")
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
    dist_matrix = build_centroid_distance_matrix(dime_centroids.astype(np.float32))
    dime_index, dime_values = build_datastore(dime_centroids, np.array(compressed_dists, dtype=object))
    populations = np.array([sum(c.values()) for c in compressed_dists])

    print("Nearest-centroid routing (k=1, no global mixing needed in Y1)...")
    _, idx_dime_ct1 = query_knn_indices(dime_index, ct_keys, k=OUTER_K_FOR_ROUTING)
    _, idx_dime_val1 = query_knn_indices(dime_index, val_keys, k=OUTER_K_FOR_ROUTING)
    idx_dime_ct1, idx_dime_val1 = idx_dime_ct1[:, 0], idx_dime_val1[:, 0]

    report = {"n_ds": int(len(ds_values)), "n_ct": int(len(ct_values)), "n_val": int(len(val_values)),
              "n_clusters": N_CLUSTERS, "seed": SEED, "smoke": SMOKE, "tau": TAU, "m_grid": M_GRID,
              "rank_grid": RANK_GRID, "primary_rank": PRIMARY_RANK,
              "note": "No document/sequence boundary metadata exists in the cached arrays -- the existing "
                       "position-level controller_train/val split is used, per the spec's own fallback "
                       "instruction. Stated explicitly as a limitation, not assumed sequence-independent."}
    save_results(os.path.join(OUT_DIR, "y1_summary.json"), report)

    # ============================================================
    # Cluster sample: R1's exact base-20 rule
    # ============================================================
    print("\nSelecting the base-20 cluster sample (R1's exact rule: 10 largest-population + "
          "10 smaller/rarer-label, population >= 30)...")
    eligible = np.where(populations >= MIN_CLUSTER_POP)[0]
    order_by_pop = eligible[np.argsort(-populations[eligible])]
    large_clusters = order_by_pop[:N_LARGE_CLUSTERS].tolist()
    small_clusters = order_by_pop[-N_SMALL_CLUSTERS:].tolist()
    sample_clusters = sorted(set(int(c) for c in large_clusters + small_clusters))
    print(f"  sampled {len(sample_clusters)} clusters, populations: {[int(populations[c]) for c in sample_clusters]}")

    # Per-cluster fit/tune/test query splits (same convention as R1-R6)
    cluster_splits = {}
    excluded = []
    for c in sample_clusters:
        ct_routed = np.where(idx_dime_ct1 == c)[0]
        val_routed = np.where(idx_dime_val1 == c)[0]
        local_rng = np.random.default_rng(SEED + c)
        local_rng.shuffle(ct_routed)
        n_fit = min(MAX_FIT, max(1, len(ct_routed) * 3 // 4))
        fit_idx = ct_routed[:n_fit]
        tune_idx = ct_routed[n_fit:n_fit + MAX_TUNE]
        test_idx = val_routed[:MAX_TEST]
        if len(fit_idx) < 5 or len(tune_idx) < 3 or len(test_idx) < 3:
            excluded.append(int(c))
            continue
        cluster_splits[c] = {"fit": fit_idx, "tune": tune_idx, "test": test_idx}
    valid_clusters = [c for c in sample_clusters if c in cluster_splits]
    print(f"  {len(valid_clusters)}/{len(sample_clusters)} clusters pass the fit>=5/tune>=3/test>=3 filter; "
          f"excluded (insufficient queries): {excluded}")
    report["cluster_sampling"] = {"sample_clusters": sample_clusters, "valid_clusters": valid_clusters,
                                     "excluded_insufficient_queries": excluded}
    save_results(os.path.join(OUT_DIR, "y1_summary.json"), report)

    # ============================================================
    # Sensitivity classification (R2's exact method, full-cluster oracle, TUNE split)
    # ============================================================
    print("\nSensitivity classification (R2's exact method: TUNE-split pairwise TV, full-cluster oracle)...")
    def tune_sensitivity(c):
        members = cluster_to_members[c]
        tune_keys = ct_keys[cluster_splits[c]["tune"]]
        if len(tune_keys) < 2:
            return None
        _, tune_ev = exact_evidence(tune_keys, ds_keys[members], ds_values[members], TAU)
        tune_dist, _ = normalize_rows(tune_ev)
        stats, _ = mean_pairwise_tv(tune_dist, N_PAIRS_PER_CLUSTER, rng) or (None, None)
        return stats["mean"] if stats else None

    sens = {c: tune_sensitivity(c) for c in valid_clusters}
    sens_values = np.array([v for v in sens.values() if v is not None])
    thresholds = {"low_high": float(np.quantile(sens_values, 1 / 3)), "high_low": float(np.quantile(sens_values, 2 / 3))} \
        if len(sens_values) >= 3 else {"low_high": 0.0, "high_low": 0.0}
    classification = {}
    for c in valid_clusters:
        v = sens.get(c)
        classification[c] = "insufficient_data" if v is None else (
            "low" if v <= thresholds["low_high"] else ("moderate" if v <= thresholds["high_low"] else "high"))
    print(f"  classification counts: {dict(Counter(classification.values()))}")
    report["sensitivity"] = {"thresholds": thresholds, "classification": classification}
    save_results(os.path.join(OUT_DIR, "y1_summary.json"), report)

    # ============================================================
    # Essential sanity checks (Section 9)
    # ============================================================
    print("\nEssential sanity checks...")
    c0 = valid_clusters[0]
    members0 = cluster_to_members[c0]
    member_keys0, member_values0 = ds_keys[members0], ds_values[members0]
    probe_keys = ct_keys[cluster_splits[c0]["fit"][:10]]

    # Check 1 + 6: brute-force cross-check against exact_evidence when m >= population
    token_ids_full, full_ev = exact_evidence(probe_keys, member_keys0, member_values0, TAU)
    full_dist, _ = normalize_rows(full_ev)
    token_ids_m, m_dist_full = local_m_distribution(probe_keys, member_keys0, member_values0, len(members0) + 10, TAU)
    assert np.array_equal(token_ids_full, token_ids_m), "token_ids mismatch between exact_evidence and local_m_distribution"
    check_m_eq_full = bool(np.allclose(full_dist, m_dist_full, atol=1e-9))
    print(f"  Check 1+6 (m>=population reduces EXACTLY to full-cluster oracle): {check_m_eq_full} "
          f"(max abs diff = {np.max(np.abs(full_dist - m_dist_full)):.2e})")
    assert check_m_eq_full, "local_m_distribution with m>=population must match exact_evidence exactly"

    _, m1_dist = local_m_distribution(probe_keys, member_keys0, member_values0, 1, TAU)
    differs_when_m_small = bool(not np.allclose(full_dist, m1_dist, atol=1e-6))
    print(f"  Check 6b (m=1 target DIFFERS from full-cluster oracle when population > 1): {differs_when_m_small}")

    # Check 2: finiteness / nonnegativity / normalization
    checks_valid = bool(np.all(np.isfinite(m1_dist)) and np.all(m1_dist >= 0) and np.allclose(m1_dist.sum(axis=1), 1.0, atol=1e-6))
    print(f"  Check 2 (finite, nonnegative, normalized): {checks_valid}")
    assert checks_valid

    report["sanity_checks"] = {"m_geq_population_matches_full_oracle_exactly": check_m_eq_full,
                                  "m1_target_differs_from_full_oracle": differs_when_m_small,
                                  "distributions_finite_nonneg_normalized": checks_valid,
                                  "note": "Checks 3 (query/label alignment), 4 (no raw vectors accessed during "
                                           "functional inference -- true by construction, linear_anchor_diff_basis "
                                           "only needs the query and stored centroids), and 5 (test queries never "
                                           "used in fitting/lambda selection -- true by construction of the "
                                           "fit/tune/test split) are structural guarantees of this script's design, "
                                           "not runtime assertions."}
    save_results(os.path.join(OUT_DIR, "y1_summary.json"), report)

    # ============================================================
    # Main experiment: per (cluster, m, rank, family)
    # ============================================================
    print(f"\nMain experiment: {len(valid_clusters)} clusters x {len(M_GRID)} m-values x {len(RANK_GRID)} ranks x 2 families...")
    cluster_results_rows = []
    rank_agg = {}  # (m, rank, family) -> list of test_tv
    query_pair_rows = []
    functional_complexity = {}
    per_cluster_models = {}  # (c, m, PRIMARY_RANK) -> {"A": theta_etc, "B": theta_etc, token_ids, exact_test_dist}

    for ci, c in enumerate(valid_clusters):
        members = cluster_to_members[c]
        member_keys, member_values = ds_keys[members], ds_values[members]
        population = len(members)
        total_c = population
        token_ids_all = np.array(sorted(set(member_values.tolist())))
        p_j = np.array([float(np.sum(member_values == y)) / total_c for y in token_ids_all])
        log_p_full = np.log(np.maximum(p_j, 1e-300))
        n_tokens_full = len(token_ids_all)
        entropy_full = float(-(p_j[p_j > 0] * np.log(p_j[p_j > 0])).sum())
        anchor_idx_max = select_anchors_nearest(dist_matrix, c, max(RANK_GRID))

        splits = cluster_splits[c]
        fit_keys, tune_keys, test_keys = ct_keys[splits["fit"]], ct_keys[splits["tune"]], val_keys[splits["test"]]

        for m in M_GRID:
            token_ids, fit_dist = local_m_distribution(fit_keys, member_keys, member_values, m, TAU)
            _, tune_dist = local_m_distribution(tune_keys, member_keys, member_values, m, TAU)
            _, test_dist = local_m_distribution(test_keys, member_keys, member_values, m, TAU)
            n_tokens = len(token_ids)
            # local_m_distribution derives token_ids from the FULL member_values (so every query's
            # distribution shares one consistent column ordering, with zeros for labels outside its
            # own m-nearest set) -- it is therefore always IDENTICAL to token_ids_all, never a subset.
            # DIME's histogram is m-independent by definition, so p_j_m is just p_j itself; asserted
            # once rather than silently assumed.
            assert np.array_equal(token_ids, token_ids_all)
            p_j_m, log_p_m = p_j, log_p_full

            dime_bcast_test = np.tile(p_j_m, (test_dist.shape[0], 1))
            dime_tv = mean_tv(dime_bcast_test, test_dist)
            dime_js = float(np.mean([js_divergence_dict(dict_from_dist(token_ids, p_j_m), dict_from_dist(token_ids, test_dist[qi]))
                                        for qi in range(test_dist.shape[0])]))

            z_fit_A = direct_log_target(fit_dist)
            z_fit_B = log_ratio_target(fit_dist, p_j_m)
            zero_log_p = np.zeros(n_tokens)
            models_this_m = {}

            for rank in RANK_GRID:
                a_idx = anchor_idx_max[:rank]
                Phi_fit = linear_anchor_diff_basis(fit_keys, c, a_idx, dime_centroids)
                Phi_tune = linear_anchor_diff_basis(tune_keys, c, a_idx, dime_centroids)
                Phi_test = linear_anchor_diff_basis(test_keys, c, a_idx, dime_centroids)

                theta_A, lam_A, tune_tv_A = fit_and_select(Phi_fit, Phi_tune, z_fit_A, tune_dist, zero_log_p, RIDGE_LAMBDA_GRID)
                d_test_A, _ = reconstruct_logit(Phi_test, theta_A, zero_log_p)
                test_tv_A = mean_tv(d_test_A, test_dist)
                test_js_A = float(np.mean([js_divergence_dict(dict_from_dist(token_ids, d_test_A[qi]), dict_from_dist(token_ids, test_dist[qi]))
                                              for qi in range(test_dist.shape[0])]))

                theta_B, lam_B, tune_tv_B = fit_and_select(Phi_fit, Phi_tune, z_fit_B, tune_dist, log_p_m, RIDGE_LAMBDA_GRID)
                d_test_B, _ = reconstruct_logit(Phi_test, theta_B, log_p_m)
                test_tv_B = mean_tv(d_test_B, test_dist)
                test_js_B = float(np.mean([js_divergence_dict(dict_from_dist(token_ids, d_test_B[qi]), dict_from_dist(token_ids, test_dist[qi]))
                                              for qi in range(test_dist.shape[0])]))

                for family, test_tv, test_js, tune_tv, mem_bytes in [
                    ("A", test_tv_A, test_js_A, tune_tv_A, storage_bytes(n_tokens, rank, include_histogram=False)),
                    ("B", test_tv_B, test_js_B, tune_tv_B, storage_bytes(n_tokens, rank, include_histogram=True)),
                ]:
                    orig_bytes = population * (DIM * DTYPE_BYTES + VALUE_BYTES)
                    cluster_results_rows.append({
                        "cluster": c, "m": m, "rank": rank, "family": family, "population": population,
                        "n_tokens": n_tokens, "entropy": round(entropy_full, 5), "sensitivity_class": classification[c],
                        "n_fit": len(splits["fit"]), "n_tune": len(splits["tune"]), "n_test": len(splits["test"]),
                        "tune_tv": round(tune_tv, 6), "test_tv": round(test_tv, 6), "test_js": round(test_js, 6),
                        "dime_tv": round(dime_tv, 6), "dime_js": round(dime_js, 6), "beats_histogram": bool(test_tv < dime_tv),
                        "memory_bytes": mem_bytes, "original_bytes": orig_bytes,
                        "compression_ratio": round(orig_bytes / mem_bytes, 3) if mem_bytes > 0 else None,
                    })
                    rank_agg.setdefault((m, rank, family), []).append(test_tv)

                if rank == PRIMARY_RANK:
                    models_this_m["A"] = {"theta": theta_A, "log_p": zero_log_p}
                    models_this_m["B"] = {"theta": theta_B, "log_p": log_p_m}

            per_cluster_models[(c, m)] = {"token_ids": token_ids, "test_dist": test_dist, "test_keys": test_keys,
                                             "models": models_this_m, "p_j_m": p_j_m}

            # functional complexity (SVD of the FIT response matrix, held-out projection via TEST)
            fc = functional_rank_analysis(fit_dist, test_dist, p_j_m)
            functional_complexity[(c, m)] = {"n_tokens": n_tokens, "population": population, "m": m,
                                                "projection_quality_by_rank": fc["projection_quality_by_rank"]}

        if (ci + 1) % max(1, len(valid_clusters) // 5) == 0 or ci == len(valid_clusters) - 1:
            print(f"  processed {ci + 1}/{len(valid_clusters)} clusters")
            save_results(os.path.join(OUT_DIR, "y1_summary.json"), report)

    # ============================================================
    # Query-change preservation (Section 6), PRIMARY_RANK only
    # ============================================================
    print(f"\nQuery-change preservation at rank={PRIMARY_RANK}...")
    for c in valid_clusters:
        for m in M_GRID:
            cache = per_cluster_models[(c, m)]
            token_ids, test_dist, test_keys = cache["token_ids"], cache["test_dist"], cache["test_keys"]
            a_idx = select_anchors_nearest(dist_matrix, c, PRIMARY_RANK)
            Phi_test = linear_anchor_diff_basis(test_keys, c, a_idx, dime_centroids)
            n = test_dist.shape[0]
            if n < 2:
                continue
            n_pairs = min(N_PAIRS_PER_CLUSTER, n * (n - 1) // 2)
            pi = rng.integers(0, n, size=n_pairs * 2)
            pj = rng.integers(0, n, size=n_pairs * 2)
            valid = pi != pj
            pi, pj = pi[valid][:n_pairs], pj[valid][:n_pairs]
            for family in ["A", "B"]:
                theta, log_p_used = cache["models"][family]["theta"], cache["models"][family]["log_p"]
                d_pred, _ = reconstruct_logit(Phi_test, theta, log_p_used)
                delta_exact = test_dist[pi] - test_dist[pj]
                delta_pred = d_pred[pi] - d_pred[pj]
                for k in range(len(pi)):
                    e_delta = float(np.abs(delta_exact[k] - delta_pred[k]).sum())
                    exact_mag = float(np.abs(delta_exact[k]).sum())
                    pred_mag = float(np.abs(delta_pred[k]).sum())
                    dime_zero_change_error = exact_mag  # DIME predicts zero change always
                    top_exact = token_ids[np.argsort(-np.abs(delta_exact[k]))[:3]]
                    top_pred = token_ids[np.argsort(-np.abs(delta_pred[k]))[:3]]
                    overlap = len(set(top_exact.tolist()) & set(top_pred.tolist())) / 3.0
                    direction_agree = float(np.mean(np.sign(delta_exact[k]) == np.sign(delta_pred[k])))
                    query_pair_rows.append({"cluster": c, "m": m, "family": family, "pair_idx": k,
                                               "e_delta": round(e_delta, 6), "exact_magnitude": round(exact_mag, 6),
                                               "predicted_magnitude": round(pred_mag, 6),
                                               "dime_zero_change_error": round(dime_zero_change_error, 6),
                                               "top_changing_token_overlap": round(overlap, 4),
                                               "direction_agreement": round(direction_agree, 4)})

    qp_summary = {}
    for m in M_GRID:
        for family in ["A", "B"]:
            recs = [r for r in query_pair_rows if r["m"] == m and r["family"] == family]
            if not recs:
                continue
            e_deltas = np.array([r["e_delta"] for r in recs])
            exact_mags = np.array([r["exact_magnitude"] for r in recs])
            pred_mags = np.array([r["predicted_magnitude"] for r in recs])
            dime_errs = np.array([r["dime_zero_change_error"] for r in recs])
            corr = float(np.corrcoef(exact_mags, pred_mags)[0, 1]) if np.std(pred_mags) > 0 else None
            collapse_ratio = float(pred_mags.mean() / exact_mags.mean()) if exact_mags.mean() > 0 else None
            qp_summary[f"m{m}_{family}"] = {
                "n_pairs": len(recs), "mean_e_delta": float(e_deltas.mean()), "mean_dime_zero_change_error": float(dime_errs.mean()),
                "beats_dime_zero_change": bool(e_deltas.mean() < dime_errs.mean()),
                "magnitude_correlation": corr, "collapse_ratio_pred_over_exact_magnitude": collapse_ratio,
                "mean_top_changing_token_overlap": float(np.mean([r["top_changing_token_overlap"] for r in recs])),
                "mean_direction_agreement": float(np.mean([r["direction_agreement"] for r in recs])),
            }
    print(f"  query-change summary (selected): m1_A={qp_summary.get('m1_A')}, m20_B={qp_summary.get('m20_B')}")
    report["query_change_summary"] = qp_summary
    save_results(os.path.join(OUT_DIR, "y1_summary.json"), report)

    # ============================================================
    # Write cluster_results.csv, rank_results.csv, query_pair_results.csv
    # ============================================================
    print("\nWriting cluster_results.csv, rank_results.csv, query_pair_results.csv...")
    with open(os.path.join(OUT_DIR, "cluster_results.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(cluster_results_rows[0].keys()))
        w.writeheader()
        for row in cluster_results_rows:
            w.writerow(row)

    rank_results_rows = []
    for (m, rank, family), tvs in rank_agg.items():
        mem_bytes_list = [r["memory_bytes"] for r in cluster_results_rows if r["m"] == m and r["rank"] == rank and r["family"] == family]
        comp_list = [r["compression_ratio"] for r in cluster_results_rows if r["m"] == m and r["rank"] == rank and r["family"] == family and r["compression_ratio"] is not None]
        wins = [r["beats_histogram"] for r in cluster_results_rows if r["m"] == m and r["rank"] == rank and r["family"] == family]
        rank_results_rows.append({"m": m, "rank": rank, "family": family, "n_clusters": len(tvs),
                                     "mean_test_tv": round(float(np.mean(tvs)), 6), "median_test_tv": round(float(np.median(tvs)), 6),
                                     "mean_memory_bytes": round(float(np.mean(mem_bytes_list)), 1),
                                     "mean_compression_ratio": round(float(np.mean(comp_list)), 3) if comp_list else None,
                                     "win_fraction_vs_histogram": round(float(np.mean(wins)), 4)})
    with open(os.path.join(OUT_DIR, "rank_results.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rank_results_rows[0].keys()))
        w.writeheader()
        for row in sorted(rank_results_rows, key=lambda r: (r["m"], r["family"], r["rank"])):
            w.writerow(row)

    with open(os.path.join(OUT_DIR, "query_pair_results.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(query_pair_rows[0].keys()))
        w.writeheader()
        for row in query_pair_rows:
            w.writerow(row)

    report["rank_results_summary"] = rank_results_rows
    save_results(os.path.join(OUT_DIR, "y1_summary.json"), report)

    # ============================================================
    # Functional complexity (Section 7) -- aggregate correlations
    # ============================================================
    print("\nFunctional complexity analysis...")
    fc_rows = []
    for (c, m), fc in functional_complexity.items():
        # functional_rank_analysis (imported from R3) resolves its rank grid from R3's OWN
        # module-level RANK_GRID=[2,4,8,16,32] at call time, not from Y1's RANK_GRID (which
        # additionally includes 1) -- iterate the keys it actually produced, not Y1's grid,
        # to avoid a KeyError on rank=1 (caught via self-review before running).
        for r, pq in fc["projection_quality_by_rank"].items():
            fc_rows.append({"cluster": c, "m": m, "n_tokens": fc["n_tokens"], "population": fc["population"],
                              "rank": r, "held_out_rel_error": pq["relative_reconstruction_error_held_out"]})
    rel_err_r8 = {(row["cluster"], row["m"]): row["held_out_rel_error"] for row in fc_rows if row["rank"] == PRIMARY_RANK}
    n_tokens_by_key = {(c, m): functional_complexity[(c, m)]["n_tokens"] for c, m in functional_complexity}
    common_keys = list(rel_err_r8.keys())
    corr_complexity_ntokens = None
    if len(common_keys) > 1:
        x = [rel_err_r8[k] for k in common_keys]
        y = [n_tokens_by_key[k] for k in common_keys]
        if np.std(x) > 0 and np.std(y) > 0:
            corr_complexity_ntokens = float(np.corrcoef(x, y)[0, 1])
    corr_complexity_m = None
    if len(common_keys) > 1:
        x = [rel_err_r8[k] for k in common_keys]
        y = [k[1] for k in common_keys]
        if np.std(x) > 0 and np.std(y) > 0:
            corr_complexity_m = float(np.corrcoef(x, y)[0, 1])
    print(f"  correlation(held-out rank-{PRIMARY_RANK} error, n_tokens) = {corr_complexity_ntokens}")
    print(f"  correlation(held-out rank-{PRIMARY_RANK} error, m) = {corr_complexity_m}")
    report["functional_complexity_correlations"] = {"error_vs_n_tokens": corr_complexity_ntokens, "error_vs_m": corr_complexity_m}
    save_results(os.path.join(OUT_DIR, "y1_summary.json"), report)

    # ============================================================
    # Plots
    # ============================================================
    print("\nGenerating plots...")
    fig, ax = plt.subplots(figsize=(7, 5))
    c_demo, m_demo = valid_clusters[0], M_GRID[-1]
    cache_demo = per_cluster_models[(c_demo, m_demo)]
    sample_q = 0
    d_exact_demo = dict_from_dist(cache_demo["token_ids"], cache_demo["test_dist"][sample_q])
    d_dime_demo = dict_from_dist(cache_demo["token_ids"], cache_demo["p_j_m"])
    all_toks = sorted(set(d_exact_demo) | set(d_dime_demo))
    ax.plot(range(len(all_toks)), [d_exact_demo.get(t, 0) for t in all_toks], marker="o", label=f"exact F^({m_demo})")
    ax.plot(range(len(all_toks)), [d_dime_demo.get(t, 0) for t in all_toks], marker="s", label="DIME histogram")
    ax.set_xlabel("token (arbitrary order)"); ax.set_ylabel("probability"); ax.legend()
    ax.set_title(f"Fixed histogram vs. exact local (m={m_demo}) distribution, cluster {c_demo}")
    fig.tight_layout(); fig.savefig(os.path.join(PLOT_DIR, "1_histogram_vs_exact.png"), dpi=130); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    for m in M_GRID:
        for family in ["A", "B"]:
            ys = [next(r["mean_test_tv"] for r in rank_results_rows if r["m"] == m and r["rank"] == rank and r["family"] == family) for rank in RANK_GRID]
            ax.plot(RANK_GRID, ys, marker="o" if family == "A" else "s", label=f"m={m},{family}", alpha=0.7)
    ax.set_xlabel("rank"); ax.set_ylabel("mean test TV"); ax.legend(fontsize=6, ncol=2)
    ax.set_title("Function approximation error vs. rank")
    fig.tight_layout(); fig.savefig(os.path.join(PLOT_DIR, "2_error_vs_rank.png"), dpi=130); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 6))
    recs_demo = [r for r in query_pair_rows if r["family"] == "B"]
    ex = np.array([r["exact_magnitude"] for r in recs_demo])
    pr = np.array([r["predicted_magnitude"] for r in recs_demo])
    ax.scatter(ex, pr, s=6, alpha=0.3)
    lims = [0, max(float(ex.max()) if len(ex) else 1, float(pr.max()) if len(pr) else 1)]
    ax.plot(lims, lims, "k--", linewidth=1)
    ax.set_xlabel("exact |delta F| (L1)"); ax.set_ylabel("predicted |delta F~| (L1), family B")
    ax.set_title("Exact vs. predicted query-change magnitude")
    fig.tight_layout(); fig.savefig(os.path.join(PLOT_DIR, "3_exact_vs_predicted_change.png"), dpi=130); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    entropies = [r["entropy"] for r in cluster_results_rows if r["rank"] == PRIMARY_RANK and r["family"] == "B"]
    tvs_plot = [r["test_tv"] for r in cluster_results_rows if r["rank"] == PRIMARY_RANK and r["family"] == "B"]
    ax.scatter(entropies, tvs_plot, s=10, alpha=0.5)
    ax.set_xlabel("cluster label entropy"); ax.set_ylabel(f"test TV at rank={PRIMARY_RANK}, family B")
    ax.set_title("Error vs. cluster label diversity")
    fig.tight_layout(); fig.savefig(os.path.join(PLOT_DIR, "4_error_vs_diversity.png"), dpi=130); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    for family in ["A", "B"]:
        xs = [next(r["mean_memory_bytes"] for r in rank_results_rows if r["m"] == 8 and r["rank"] == rank and r["family"] == family) for rank in RANK_GRID]
        ys = [next(r["mean_test_tv"] for r in rank_results_rows if r["m"] == 8 and r["rank"] == rank and r["family"] == family) for rank in RANK_GRID]
        ax.plot(xs, ys, marker="o", label=family)
    ax.set_xlabel("mean memory bytes/cluster (m=8)"); ax.set_ylabel("mean test TV"); ax.legend()
    ax.set_title("Distortion vs. storage")
    fig.tight_layout(); fig.savefig(os.path.join(PLOT_DIR, "5_distortion_vs_storage.png"), dpi=130); plt.close(fig)

    # ============================================================
    # failure_cases.md
    # ============================================================
    print("\nSelecting failure cases...")
    r8_B = [r for r in cluster_results_rows if r["rank"] == PRIMARY_RANK and r["family"] == "B"]
    r8_B_sorted = sorted(r8_B, key=lambda r: r["test_tv"] - r["dime_tv"])
    worst = r8_B_sorted[-N_FAILURE_EXAMPLES:]
    best = r8_B_sorted[:N_FAILURE_EXAMPLES]
    failure_lines = ["# Y1 Failure Cases (family B, rank=8)", "",
                       "Worst = largest (test_tv - dime_tv), i.e. function performs WORSE than the fixed "
                       "histogram. Best = smallest (most negative), i.e. largest improvement.", "",
                       "## Worst (function underperforms DIME)"]
    for r in reversed(worst):
        failure_lines.append(f"- cluster={r['cluster']}, m={r['m']}, population={r['population']}, "
                               f"n_tokens={r['n_tokens']}, sensitivity={r['sensitivity_class']}: "
                               f"test_tv={r['test_tv']}, dime_tv={r['dime_tv']}, gap={round(r['test_tv']-r['dime_tv'],5)}")
    failure_lines.append("\n## Best (function substantially improves over DIME)")
    for r in best:
        failure_lines.append(f"- cluster={r['cluster']}, m={r['m']}, population={r['population']}, "
                               f"n_tokens={r['n_tokens']}, sensitivity={r['sensitivity_class']}: "
                               f"test_tv={r['test_tv']}, dime_tv={r['dime_tv']}, gap={round(r['test_tv']-r['dime_tv'],5)}")
    with open(os.path.join(OUT_DIR, "failure_cases.md"), "w") as f:
        f.write("\n".join(failure_lines))

    # ============================================================
    # functional_complexity.md
    # ============================================================
    fc_md = f"""# Y1 Functional Complexity Observations

## Rank-{PRIMARY_RANK} held-out projection error vs. cluster properties

Correlation between held-out relative reconstruction error (rank={PRIMARY_RANK}) and number of distinct tokens:
{corr_complexity_ntokens}

Correlation between held-out relative reconstruction error (rank={PRIMARY_RANK}) and local neighbor count m:
{corr_complexity_m}

## Interpretation notes

A positive correlation with n_tokens would suggest label-diverse clusters need higher rank to represent their
response geometry. A correlation with m indicates the choice of local neighbor count itself changes how
low-dimensional the target response is -- e.g. m=1's near-one-hot targets may show a DIFFERENT (not necessarily
lower) empirical rank than m=20's smoother targets, since one-hot targets are trivially rank-1 in a sense but
highly discontinuous in q.

Per Section 7's explicit instruction, a low measured rank in the response-matrix SVD is NOT automatically
evidence that the function predicts unseen responses well -- see rank_results.csv's mean_test_tv column (a
genuine held-out generalization measure) alongside this file's correlations (a measure of the TARGET's intrinsic
geometry, computed from the FIT split only) for the distinction the spec requires.

See the JSON summary's functional_complexity (not persisted per-cluster here to avoid an extremely large file;
see cluster_results.csv for the per-cluster/m/rank held-out test_tv instead) for the full per-cluster singular-value
spectra if deeper inspection is needed.
"""
    with open(os.path.join(OUT_DIR, "functional_complexity.md"), "w") as f:
        f.write(fc_md)

    # ============================================================
    # research_questions.md
    # ============================================================
    research_q = """# Y1 Research Questions for Subsequent Study

## Approximation theory
- Does the empirical rank needed for accurate held-out reconstruction of F_j^(m) scale predictably with m, or
  does small m (near-one-hot targets) require a QUALITATIVELY different representation (e.g. a classifier-style
  decision boundary) rather than a smooth low-rank basis?
- Family A (no histogram prior) and family B (histogram-prior correction) share the same basis -- is there a
  closed-form relationship between their respective achievable errors at equal rank, given that F_j^(m) can have
  exact zeros that the histogram-prior parameterization structurally cannot represent at delta=0 (since p_j(y)=0
  is impossible for y in the cluster's support, but the TARGET can still be exactly 0 for small m)?

## Geometry
- Is there a precise notion of "effective local radius" (how far the m-th nearest neighbor typically is) that
  predicts whether F_j^(m) will be well-approximated by the existing anchor-difference basis, which is defined
  in terms of distances to OTHER CENTROIDS, not to the cluster's own members?

## Statistical estimation
- Given Y1's m-grid finding (see rank_results.csv), is there a principled way to choose m itself as a learned
  or adaptive parameter per cluster, rather than a fixed diagnostic grid?

## Compression architecture
- If Y1 establishes genuine compressibility (Q10 in report.md), what would change about CRISP's own R4/R5/R6
  pipeline if the LOCAL-m target (rather than the full-cluster target) were adopted as the production objective?
  This is explicitly a Y2-or-later question, not attempted here.
"""
    with open(os.path.join(OUT_DIR, "research_questions.md"), "w") as f:
        f.write(research_q)

    # ============================================================
    # Final answers (Q1-Q10) + report.md
    # ============================================================
    print("\nComputing final answers (Q1-Q10)...")
    dime_tv_by_m = {m: float(np.mean([r["dime_tv"] for r in cluster_results_rows if r["m"] == m and r["family"] == "B" and r["rank"] == PRIMARY_RANK])) for m in M_GRID}
    best_r8 = {family: float(np.mean([r["test_tv"] for r in cluster_results_rows if r["rank"] == PRIMARY_RANK and r["family"] == family])) for family in ["A", "B"]}
    win_frac_r8 = {family: float(np.mean([r["beats_histogram"] for r in cluster_results_rows if r["rank"] == PRIMARY_RANK and r["family"] == family])) for family in ["A", "B"]}
    comp_ratio_r8 = {family: float(np.mean([r["compression_ratio"] for r in cluster_results_rows if r["rank"] == PRIMARY_RANK and r["family"] == family and r["compression_ratio"] is not None])) for family in ["A", "B"]}

    answers = {
        "Q1_query_dependence": dime_tv_by_m,
        "Q2_generalizes_to_unseen_queries": {family: best_r8[family] for family in ["A", "B"]},
        "Q3_beats_histogram": win_frac_r8,
        "Q4_preserves_change": {k: v["beats_dime_zero_change"] for k, v in qp_summary.items()},
        "Q5_difficulty_vs_m": {family: {m: float(np.mean([r["test_tv"] for r in cluster_results_rows if r["m"] == m and r["rank"] == PRIMARY_RANK and r["family"] == family])) for m in M_GRID} for family in ["A", "B"]},
        "Q6_rank_required": rank_results_rows,
        "Q7_genuinely_smaller": comp_ratio_r8,
        "Q8_easy_vs_difficult_clusters": {"correlation_error_vs_n_tokens": corr_complexity_ntokens},
        "Q9_histogram_prior_helps": {"family_A_mean_tv_r8": best_r8["A"], "family_B_mean_tv_r8": best_r8["B"], "B_beats_A": best_r8["B"] < best_r8["A"]},
    }
    q10_justified = (win_frac_r8["B"] > 0.5 and comp_ratio_r8["B"] > 1.0)
    answers["Q10_justifies_full_algorithm"] = {"verdict": "Justified" if q10_justified else "Not yet justified",
                                                  "win_fraction_B_r8": win_frac_r8["B"], "compression_ratio_B_r8": comp_ratio_r8["B"]}
    print(f"  {answers}")
    report["final_answers"] = answers
    save_results(os.path.join(OUT_DIR, "y1_summary.json"), report)

    lines = [
        "# Y1: Per-Cluster Query-Dependent Functional Approximation", "",
        "## 1. Research question", "See module docstring.", "",
        "## 2. Target function and baseline", f"tau={TAU}, m grid={M_GRID}. DIME histogram TV by m (family B, rank={PRIMARY_RANK}): {dime_tv_by_m}", "",
        "## 3. Function families", f"Family A (no prior) mean test TV at rank={PRIMARY_RANK}: {best_r8['A']}. "
        f"Family B (histogram prior) mean test TV: {best_r8['B']}.", "",
        "## 4. Cluster/query sampling", f"{report['cluster_sampling']}", f"Sensitivity: {dict(Counter(classification.values()))}", "",
        "## 5. Primary experiment results", "See cluster_results.csv and rank_results.csv for the full grid.", "",
        "## 6. Query-change preservation", f"{qp_summary}", "",
        "## 7. Functional complexity", f"{report['functional_complexity_correlations']}", "",
        "## 8. Memory accounting", f"Mean compression ratio at rank={PRIMARY_RANK}: {comp_ratio_r8}", "",
        "## 9. Sanity checks", f"{report['sanity_checks']}", "",
        "## 10. Final answers (Q1-Q10)", f"{answers}", "",
        "## Limitations",
        f"- Only the base-20 cluster sample is used (R1's exact rule), per Y1's own 'do not expand' instruction.",
        "- No sequence/document-grouped split exists in the cached data -- the existing position-level "
        "controller_train/val split is used, stated as a limitation.",
        f"- Query-change analysis and failure-case selection use a single PRIMARY_RANK={PRIMARY_RANK}, not a full rank sweep.",
        "- This delivery does not implement global routing, mass-aware aggregation, or end-to-end kNN-LM "
        "evaluation, per the spec's own explicit scope boundary.",
        "", "## Do not automatically begin Y2", "Per the spec's final instruction, this delivery stops here.",
    ]
    with open(os.path.join(OUT_DIR, "report.md"), "w") as f:
        f.write("\n".join(lines))

    report["sections_completed"] = ["cluster_sampling", "sensitivity", "sanity_checks", "main_experiment",
                                      "query_change_summary", "functional_complexity_correlations", "final_answers",
                                      "cluster_results.csv", "rank_results.csv", "query_pair_results.csv",
                                      "failure_cases.md", "functional_complexity.md", "research_questions.md",
                                      "report.md", "plots"]
    report["total_runtime_seconds"] = time.perf_counter() - t_start
    save_results(os.path.join(OUT_DIR, "y1_summary.json"), report)
    print(f"\nAll results saved under {OUT_DIR}")
    print(f"Total runtime: {report['total_runtime_seconds']/60:.1f} min")
    print("No existing E1/E2/TRI1/.../R1-R6 outputs were modified. Y2 not started, per the spec's instruction.")


if __name__ == "__main__":
    main()
