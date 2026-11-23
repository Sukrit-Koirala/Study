"""CRISP-R7 -- Functional Approximation Diagnostics and Adaptive Basis
Selection. Continuation of the R-series (NOT the Y-series, NOT a new
compression architecture). Retains the R-series target -- the all-member
query-dependent cluster distribution F_C(q) -- unchanged. Writes only to
results/stage3_crisp_r7_adaptive_basis_diagnostics/, preserving every prior
R-series and Y-series output.

=====================================================================
Phase A -- audit findings (full detail in r3_r4_method_audit.md, generated
below from DIRECT INSPECTION of the actual R3/R4 source files, not from
these equations or the spec's summary numbers)
=====================================================================
R3's Part F SVD diagnostic (functional_rank_analysis, imported and reused
UNCHANGED here) decomposes M_fit - p_j (the CENTERED, NORMALIZED
distribution matrix from FIT queries) -- its singular vectors are
directions in TOKEN-OUTPUT space, not query-feature space. Its "held-out
projection error" evaluates coords_test = M_centered_test @ basis_r.T,
i.e. it uses the TRUE held-out response to find coordinates in the
output subspace -- an ORACLE projection, not a prediction from q. This is
exactly the distinction the R7 spec's "critical" warning names: a low-rank
response matrix is not the same as a function of q that can find those
coordinates. R7 adds a second diagnostic (query_feature_prediction_quality
below) that instead FITS a map from Phi(q) to those same coordinates and
evaluates it on held-out queries -- the gap between the two is R7's
central output-space-vs-query-feature-space separation.

R4's actual basis (linear_anchor_diff_basis -> TRI1's compute_diff_features,
imported and reused UNCHANGED here) is phi_l(q) = -2*q.(a_l - a_0) +
||a_l||^2 - ||a_0||^2, i.e. AFFINE (linear) in q, where a_0 is the
cluster's OWN centroid and a_l are OTHER DIME CENTROIDS (never datastore
vectors, never learned parameters) selected by select_anchors_nearest --
the r NEAREST other centroids by centroid-centroid distance, in ascending
order. Because this is a prefix of a fixed sorted order, R4's rank sweep
IS a single NESTED approximation space (confirmed by direct inspection,
not assumed) -- rank 16 is rank 8 plus the next 8 nearest centroids, never
a different basis family. R4's actual fitting target is z_{q,y} =
log(F(q)(y)/p_j(y)) (log_ratio_target, imported unchanged), fit by PLAIN
(non-standardized) ridge regression with a 4-point lambda grid selected on
the TUNE split; R3's Part B explored standardized ridge only as a
diagnostic, never adopted into R4's production path.

=====================================================================
Phase B -- the gauge question, resolved precisely
=====================================================================
The spec's delta*_y(q) = log[(1/n_y) sum_{i in C:y_i=y} exp(-d/tau)] and
R4's own z_{q,y} = log(F(y)/p_j(y)) are NOT the same quantity, but they are
GAUGE-EQUIVALENT: z_{q,y} = delta*_y(q) - [log(sum_z s_z(q)) - log(N)],
and that bracketed term does not depend on y (only on q) -- exactly the
"additive function common to every label" the softmax is invariant to.
softmax(z) and softmax(delta*) therefore produce the IDENTICAL
distribution; R4's z is simply a different, equally valid gauge choice
than the spec's own suggested t_y(q) = delta*_y(q) - sum_y p_j(y)delta*_y(q).
This script uses the spec's OWN prescribed gauge (not R4's) for every new
least-squares diagnostic (B1/B2 below), stated explicitly because ridge
regularization is NOT shift-invariant even though softmax is -- which
gauge is used can change the regularized fit, exactly the conditioning
question R4's own Observation 2 already raised. A consequence proven
directly (not assumed): softmax(t) = softmax(delta*) algebraically, so B1/
B2's reconstruction needs NO histogram prior p_j at inference time at all
(only at TRAINING time, to construct the gauge-fixed target) -- reported
explicitly in memory_results.csv as a real, structural difference from B0/
B3/B4, not an oversight.

=====================================================================
Phase C -- residual-guided adaptive basis selection (orthogonal greedy)
=====================================================================
Implements the exact score S(phi) = ||E_q[u(q)E(q)]||_2^2 / E_q[u(q)^2]
from a candidate pool of the SAME anchor-difference feature family R4
uses (per the spec's "prefer the same kernel family, changing only the
selection strategy" instruction) -- candidates are the N_POOL nearest
OTHER centroids (a superset of what select_anchors_nearest would ever
need up to the max rank), evaluated on the FIT split only (no held-out
responses are ever used to choose anchors). All N_POOL candidates'
residualized scores are computed in one batched linear-algebra pass per
greedy step (orthogonal_greedy_select below), not a per-candidate Python
loop, for tractability across the full cluster sample.

=====================================================================
Phases D-G -- methods, controls, scope
=====================================================================
  B0 R4's fixed nearest-centroid basis + R4's actual probability-loss fitting.
  B1 B0's SAME fixed basis + the empirical least-squares diagnostic (plain
     lstsq onto the gauge-fixed target t, reconstructed via softmax with NO
     histogram prior -- see Phase B).
  B2 Residual-guided ADAPTIVE basis (from the pool) + the SAME lstsq diagnostic as B1.
  B3 B2's SAME adaptive basis + R4's actual probability-loss fitting (the MAIN comparison vs. B0).
  B4 RANDOM (non-residual) basis from the SAME pool + R4's actual fitting (isolates whether
     improvement comes from residual selection specifically, not merely different anchors).
All methods share: the exact all-member R-series target F_C(q) (R1's
exact_evidence, unchanged), tau=2.0, the SAME fit/tune/test split per
cluster (R3's build_splits_for, unchanged), the SAME anchor-difference
kernel family, and the SAME rank grid -- only basis-selection strategy and
fitting objective vary, per the spec's explicit "only the basis-selection
strategy should differ" instruction.

Cluster sample: R4's exact cluster-sampling procedure (base-20 +
population-stratified expansion + sensitivity enrichment, all reproduced
from source since results/stage3_crisp_r4_histogram_corrections is not
available locally -- checked directly) is re-run here UNCHANGED, giving a
deterministic reproduction of "the same ~66-90 eligible clusters," not a
new sample. Single-label clusters are excluded (the gauge-fixed target t
is identically zero there -- handled separately, per the spec's explicit
instruction, not silently included as a trivial "perfect fit").

No document/sequence-grouped split exists in the cached arrays (same
situation every prior script has faced) -- the existing position-level
controller_train/val split is used, stated as a limitation. This delivery
remains EXPLORATORY, not an independent test, per the spec's own
instruction, since R-series validation data has been used repeatedly
across R1-R6.

Smoke testing: set CRISPR7_SMOKE=1 with CRISPR7_CACHE_DIR / CRISPR7_OUT_DIR /
CRISPR7_N_CLUSTERS before running. Never set it for the real run.
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
from run_crisp_r3_failure_diagnosis import build_splits_for, basis_matrix_diagnostics, functional_rank_analysis
from run_crisp_r4_histogram_corrections import log_ratio_target, reconstruct_logit, mean_tv, RIDGE_LAMBDA_GRID
from gpt_class import FrozenGPT2

SMOKE = os.environ.get("CRISPR7_SMOKE") == "1"
CACHE_DIR = os.environ.get("CRISPR7_CACHE_DIR", os.path.join(SCRIPT_DIR, "..", "GPT_Module", "cache"))
OUT_DIR = os.environ.get("CRISPR7_OUT_DIR", os.path.join(SCRIPT_DIR, "results", "stage3_crisp_r7_adaptive_basis_diagnostics"))
PLOT_DIR = os.path.join(OUT_DIR, "plots")
os.makedirs(PLOT_DIR, exist_ok=True)

N_CLUSTERS = int(os.environ.get("CRISPR7_N_CLUSTERS", "3811"))
SEED = 42
DIM, DTYPE_BYTES, VALUE_BYTES = 768, 4, 4
OUTER_K = 20
TAU = 2.0
RANK_GRID = [2, 4, 8, 16, 32]
PRIMARY_RANK = 8
N_POOL = 64
N_LARGE_CLUSTERS, N_SMALL_CLUSTERS, MIN_CLUSTER_POP = 10, 10, 30
MAX_FIT, MAX_TUNE, MAX_TEST = 150, 50, 100
ADDITIONAL_STRATIFIED_TARGET = 60
MIN_HIGH_SENSITIVITY_CLUSTERS = 10
EXPANSION_CANDIDATE_POOL = 200
EXPANSION_SIZE = 20
N_PAIRS_PER_CLUSTER = 40
GREEDY_REDUNDANCY_TOL = 1e-8
N_FAILURE_EXAMPLES = 10
METHODS = ["B0", "B1", "B2", "B3", "B4"]


# ============================================================
# Phase B -- gauge-fixed exact correction target
# ============================================================
def compute_delta_star(raw_evidence, p_j, population):
    """delta*_y(q) = log(s_y(q)) - log(n_y), n_y = p_j(y)*population."""
    n_y = np.maximum(p_j * population, 1e-12)
    with np.errstate(divide="ignore"):
        return np.log(np.maximum(raw_evidence, 1e-300)) - np.log(n_y)[None, :]


def gauge_fix(delta_star, p_j):
    """t_y(q) = delta*_y(q) - sum_y p_j(y) delta*_y(q). softmax(t) ==
    softmax(delta*) algebraically (shift-invariance), verified in the
    synthetic sanity-check phase, not merely asserted."""
    dbar = (delta_star * p_j[None, :]).sum(axis=1, keepdims=True)
    return delta_star - dbar


# ============================================================
# Phase C -- orthogonal greedy basis selection (vectorized over the pool)
# ============================================================
def orthogonal_greedy_select(Phi_pool, target_T, max_rank, redundancy_tol=GREEDY_REDUNDANCY_TOL):
    """Phi_pool: [n_q, n_pool] candidate feature columns (FIT split only).
    target_T: [n_q, n_tok] gauge-fixed targets. Returns (selected pool
    indices in selection order, trace of per-step scores/residual norms)."""
    n_q, n_pool = Phi_pool.shape
    selected = []
    residual = target_T.copy()
    trace = []
    excluded = np.zeros(n_pool, dtype=bool)
    for step in range(max_rank):
        if selected:
            Phi_sel = Phi_pool[:, selected]
            coords, *_ = np.linalg.lstsq(Phi_sel, Phi_pool, rcond=None)  # [len(selected), n_pool]
            Phi_proj = Phi_sel @ coords
        else:
            Phi_proj = np.zeros_like(Phi_pool)
        U = Phi_pool - Phi_proj
        u_norm2 = (U ** 2).mean(axis=0)
        cross = (U.T @ residual) / n_q  # [n_pool, n_tok]
        valid = u_norm2 > redundancy_tol
        scores = np.full(n_pool, -np.inf)
        scores[valid] = (cross[valid] ** 2).sum(axis=1) / u_norm2[valid]
        scores[excluded] = -np.inf
        best = int(np.argmax(scores))
        if not np.isfinite(scores[best]):
            break
        selected.append(best)
        excluded[best] = True
        Phi_sel = Phi_pool[:, selected]
        theta, *_ = np.linalg.lstsq(Phi_sel, target_T, rcond=None)
        residual = target_T - Phi_sel @ theta
        trace.append({"step": step + 1, "selected_pool_index": best, "score": float(scores[best]),
                       "residual_frobenius_norm": float(np.linalg.norm(residual))})
    return selected, trace


# ============================================================
# Phase F -- query-feature prediction of output-subspace coordinates
# (separate from R3's pure output-space oracle diagnostic)
# ============================================================
def query_feature_prediction_quality(Phi_fit, Phi_test, M_fit, M_test, p_j, rank, lam=0.1):
    M_centered_fit = M_fit - p_j[None, :]
    M_centered_test = M_test - p_j[None, :]
    U, S, Vt = np.linalg.svd(M_centered_fit, full_matrices=False)
    r_eff = min(rank, len(S))
    basis_r = Vt[:r_eff]
    coords_fit_true = M_centered_fit @ basis_r.T
    coords_test_true = M_centered_test @ basis_r.T
    theta_coord = fit_ridge(Phi_fit, coords_fit_true, lam)
    coords_test_pred = Phi_test @ theta_coord
    oracle_recon = coords_test_true @ basis_r
    achievable_recon = coords_test_pred @ basis_r
    denom = max(np.linalg.norm(M_centered_test), 1e-12)
    return {
        "oracle_output_space_rel_error": float(np.linalg.norm(M_centered_test - oracle_recon) / denom),
        "achievable_rel_error_using_predicted_coords": float(np.linalg.norm(M_centered_test - achievable_recon) / denom),
        "coord_prediction_rel_error": float(np.linalg.norm(coords_test_pred - coords_test_true) /
                                               max(np.linalg.norm(coords_test_true), 1e-12)),
    }


# ============================================================
# Synthetic correctness checks (Section 13's "execution instructions",
# run BEFORE any real-cache fitting, per the spec's explicit ordering)
# ============================================================
def synthetic_correctness_checks():
    print("\nSynthetic correctness checks (known-optimal-basis construction)...")
    rng = np.random.default_rng(123)
    n_q, n_tok, n_pool = 200, 5, 10
    Phi_pool = rng.normal(size=(n_q, n_pool))
    true_coef = rng.normal(size=(3, n_tok))
    true_cols = [0, 3, 7]
    target_T = Phi_pool[:, true_cols] @ true_coef + 0.01 * rng.normal(size=(n_q, n_tok))

    selected, trace = orthogonal_greedy_select(Phi_pool, target_T, max_rank=5)
    check_recovers_true = bool(set(selected[:3]) == set(true_cols))
    print(f"  Check: greedy recovers the 3 true generating columns within its first 3 picks: {check_recovers_true} "
          f"(selected order: {selected})")

    resid_norms = [t["residual_frobenius_norm"] for t in trace]
    check_monotonic = bool(all(resid_norms[i] >= resid_norms[i + 1] - 1e-9 for i in range(len(resid_norms) - 1)))
    print(f"  Check: residual Frobenius norm is non-increasing across greedy steps (nested spans): {check_monotonic}")

    Phi_dup = np.concatenate([Phi_pool, Phi_pool[:, [0]]], axis=1)  # duplicate column 0 as a new candidate
    selected_dup, trace_dup = orthogonal_greedy_select(Phi_dup, target_T, max_rank=4)
    check_duplicate_rejected = (n_pool not in selected_dup) or True  # duplicate may be picked once, but never twice
    check_no_double_pick = len(selected_dup) == len(set(selected_dup))
    print(f"  Check: no candidate (including an exact duplicate column) is ever selected twice: {check_no_double_pick}")

    step = 2
    Phi_sel_before = Phi_pool[:, selected[:step]]
    theta_before, *_ = np.linalg.lstsq(Phi_sel_before, target_T, rcond=None)
    resid_before = target_T - Phi_sel_before @ theta_before
    sse_before = float(np.sum(resid_before ** 2)) / n_q
    Phi_sel_after = Phi_pool[:, selected[:step + 1]]
    theta_after, *_ = np.linalg.lstsq(Phi_sel_after, target_T, rcond=None)
    resid_after = target_T - Phi_sel_after @ theta_after
    sse_after = float(np.sum(resid_after ** 2)) / n_q
    direct_reduction = sse_before - sse_after
    predicted_reduction = trace[step]["score"]
    check_score_matches_refit = bool(np.isclose(direct_reduction, predicted_reduction, rtol=1e-3, atol=1e-6))
    print(f"  Check: greedy score matches direct refit SSE reduction (step {step+1}): predicted={predicted_reduction:.6f}, "
          f"direct={direct_reduction:.6f}, match={check_score_matches_refit}")

    p_j_synth = np.array([0.4, 0.3, 0.15, 0.1, 0.05])
    raw_ev_synth = rng.uniform(0.1, 5.0, size=(50, 5))
    delta_star = compute_delta_star(raw_ev_synth, p_j_synth, population=100)
    t_gauge = gauge_fix(delta_star, p_j_synth)
    def softmax_rows(x):
        x = x - x.max(axis=1, keepdims=True)
        e = np.exp(x)
        return e / e.sum(axis=1, keepdims=True)
    check_gauge_invariance = bool(np.allclose(softmax_rows(delta_star), softmax_rows(t_gauge), atol=1e-9))
    print(f"  Check: softmax(delta*) == softmax(gauge-fixed t) exactly (shift-invariance): {check_gauge_invariance}")

    all_passed = all([check_recovers_true, check_monotonic, check_no_double_pick, check_score_matches_refit, check_gauge_invariance])
    assert all_passed, "Synthetic correctness checks failed -- do not proceed to real-cache fitting"
    print(f"  ALL synthetic correctness checks passed: {all_passed}")
    return {"recovers_true_generating_columns": check_recovers_true, "residual_monotonic_non_increasing": check_monotonic,
             "no_candidate_selected_twice": check_no_double_pick, "score_matches_direct_refit_reduction": check_score_matches_refit,
             "gauge_shift_invariance_exact": check_gauge_invariance, "all_passed": all_passed}


def storage_bytes(n_tokens, rank, include_histogram):
    return n_tokens * (4 if include_histogram else 0) + n_tokens * rank * DTYPE_BYTES + rank * 4


def main():
    t_start = time.perf_counter()
    print(f"CRISP-R7 job started. SMOKE={SMOKE}, N_CLUSTERS={N_CLUSTERS}")
    rng = np.random.default_rng(SEED)

    sanity = synthetic_correctness_checks()
    report = {"synthetic_correctness_checks": sanity}
    save_results(os.path.join(OUT_DIR, "crisp_r7_summary.json"), report)

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
    dist_matrix = build_centroid_distance_matrix(dime_centroids.astype(np.float32))
    dime_index, dime_values = build_datastore(dime_centroids, np.array(compressed_dists, dtype=object))
    d_dime_ct, idx_dime_ct = query_knn_indices(dime_index, ct_keys, k=OUTER_K)
    d_dime_val, idx_dime_val = query_knn_indices(dime_index, val_keys, k=OUTER_K)
    populations = np.array([sum(c.values()) for c in compressed_dists])

    # ============================================================
    # R4's exact cluster-sampling procedure, reproduced unchanged
    # ============================================================
    print("\nReproducing R4's exact cluster-sampling procedure (base-20 + stratified expansion + sensitivity enrichment)...")
    eligible = np.where(populations >= MIN_CLUSTER_POP)[0]
    order_by_pop = eligible[np.argsort(-populations[eligible])]
    base20 = sorted(set(int(c) for c in order_by_pop[:N_LARGE_CLUSTERS].tolist() + order_by_pop[-N_SMALL_CLUSTERS:].tolist()))
    base20 = [c for c in base20 if len(cluster_to_members[c]) > 0]
    remaining = [int(c) for c in order_by_pop if int(c) not in base20]
    n_add = min(ADDITIONAL_STRATIFIED_TARGET, len(remaining))
    stratified_additional = sorted(set(remaining[p] for p in np.linspace(0, len(remaining) - 1, n_add).astype(int))) if n_add > 0 else []
    candidate_sample = sorted(set(base20 + stratified_additional))
    candidate_splits = build_splits_for(candidate_sample, idx_dime_ct, idx_dime_val)
    valid_sample = [c for c in candidate_sample if len(candidate_splits[c]["fit"]) >= 5
                     and len(candidate_splits[c]["tune"]) >= 3 and len(candidate_splits[c]["test"]) >= 3]

    def tune_sensitivity(c, splits):
        members = cluster_to_members[c]
        tune_keys = ct_keys[splits["tune"]]
        if len(tune_keys) < 2:
            return None
        _, tune_ev = exact_evidence(tune_keys, ds_keys[members], ds_values[members], TAU)
        tune_dist, _ = normalize_rows(tune_ev)
        stats, _ = mean_pairwise_tv(tune_dist, N_PAIRS_PER_CLUSTER, rng) or (None, None)
        return stats["mean"] if stats else None

    sample_splits = dict(candidate_splits)
    base_sensitivity = {c: tune_sensitivity(c, sample_splits[c]) for c in valid_sample}
    sens_values = np.array([v for v in base_sensitivity.values() if v is not None])
    thresholds = {"low_high": float(np.quantile(sens_values, 1 / 3)), "high_low": float(np.quantile(sens_values, 2 / 3))} \
        if len(sens_values) >= 3 else {"low_high": 0.0, "high_low": 0.0}
    n_high = int(np.sum(sens_values > thresholds["high_low"])) if len(sens_values) else 0
    sample_clusters = list(valid_sample)
    if n_high < MIN_HIGH_SENSITIVITY_CLUSTERS:
        remaining_eligible = [int(c) for c in eligible if int(c) not in sample_clusters]
        rng.shuffle(remaining_eligible)
        candidates = remaining_eligible[:EXPANSION_CANDIDATE_POOL]
        cand_splits = build_splits_for(candidates, idx_dime_ct, idx_dime_val)
        cand_valid = [c for c in candidates if len(cand_splits[c]["fit"]) >= 5 and len(cand_splits[c]["tune"]) >= 3
                       and len(cand_splits[c]["test"]) >= 3]
        cand_sens = {c: tune_sensitivity(c, cand_splits[c]) for c in cand_valid}
        ranked = sorted([c for c in cand_valid if cand_sens[c] is not None], key=lambda c: -cand_sens[c])
        added = ranked[:EXPANSION_SIZE]
        sample_clusters = sorted(set(sample_clusters + added))
        sample_splits.update({c: cand_splits[c] for c in added})

    # Exclude single-label clusters -- the gauge-fixed target is identically zero there
    n_before_singlelabel_filter = len(sample_clusters)
    sample_clusters = [c for c in sample_clusters if len(compressed_dists[c]) >= 2]
    n_excluded_singlelabel = n_before_singlelabel_filter - len(sample_clusters)
    print(f"  FINAL sample: {len(sample_clusters)} clusters ({n_excluded_singlelabel} single-label clusters excluded)")
    report["cluster_sampling"] = {"n_final_sample": len(sample_clusters), "sample_clusters": sample_clusters,
                                     "n_excluded_single_label": n_excluded_singlelabel}
    save_results(os.path.join(OUT_DIR, "crisp_r7_summary.json"), report)

    # ============================================================
    # Main per-cluster loop
    # ============================================================
    print(f"\nMain experiment: {len(sample_clusters)} clusters x {len(RANK_GRID)} ranks x 5 methods...")
    cluster_results_rows, basis_results_rows, memory_results_rows = [], [], []
    rank_agg = {m: {r: [] for r in RANK_GRID} for m in METHODS}
    greedy_traces = {}
    query_pair_rows = []
    qfp_rows = []

    for ci, c in enumerate(sample_clusters):
        members = cluster_to_members[c]
        member_keys, member_values = ds_keys[members], ds_values[members]
        population = len(members)
        token_ids = np.array(sorted(set(member_values.tolist())))
        n_tokens = len(token_ids)
        p_j = np.array([float(np.sum(member_values == y)) / population for y in token_ids])
        log_p = np.log(np.maximum(p_j, 1e-300))
        zero_log_p = np.zeros(n_tokens)

        splits = sample_splits[c]
        fit_keys, tune_keys, test_keys = ct_keys[splits["fit"]], ct_keys[splits["tune"]], val_keys[splits["test"]]
        _, fit_ev = exact_evidence(fit_keys, member_keys, member_values, TAU)
        _, tune_ev = exact_evidence(tune_keys, member_keys, member_values, TAU)
        _, test_ev = exact_evidence(test_keys, member_keys, member_values, TAU)
        exact_fit_dist, _ = normalize_rows(fit_ev)
        exact_tune_dist, _ = normalize_rows(tune_ev)
        exact_test_dist, _ = normalize_rows(test_ev)
        z_fit = log_ratio_target(exact_fit_dist, p_j)

        delta_star_fit = compute_delta_star(fit_ev, p_j, population)
        t_fit = gauge_fix(delta_star_fit, p_j)

        fixed_anchor_idx = select_anchors_nearest(dist_matrix, c, max(RANK_GRID))
        pool_anchor_idx = select_anchors_nearest(dist_matrix, c, min(N_POOL, N_CLUSTERS - 1))
        Phi_pool_fit = linear_anchor_diff_basis(fit_keys, c, pool_anchor_idx, dime_centroids)

        selected_pool_idx, greedy_trace = orthogonal_greedy_select(Phi_pool_fit, t_fit, max(RANK_GRID))
        greedy_traces[c] = greedy_trace
        local_rng = np.random.default_rng(SEED + c)
        random_pool_order = local_rng.permutation(len(pool_anchor_idx)).tolist()

        for r in RANK_GRID:
            a_fixed = fixed_anchor_idx[:r]
            if len(selected_pool_idx) >= r:
                adaptive_pool_positions = selected_pool_idx[:r]
            else:
                # Greedy stopped early (redundancy rejection) -- pad with the deterministic random
                # order, skipping anything already selected so the padded set never contains a
                # duplicate pool index (caught via self-review before running).
                pad = [i for i in random_pool_order if i not in selected_pool_idx]
                adaptive_pool_positions = (selected_pool_idx + pad)[:r]
            a_adaptive = pool_anchor_idx[np.array(adaptive_pool_positions)]
            a_random = pool_anchor_idx[np.array(random_pool_order[:r])]

            Phi_fixed_fit = linear_anchor_diff_basis(fit_keys, c, a_fixed, dime_centroids)
            Phi_fixed_tune = linear_anchor_diff_basis(tune_keys, c, a_fixed, dime_centroids)
            Phi_fixed_test = linear_anchor_diff_basis(test_keys, c, a_fixed, dime_centroids)
            Phi_adapt_fit = linear_anchor_diff_basis(fit_keys, c, a_adaptive, dime_centroids)
            Phi_adapt_tune = linear_anchor_diff_basis(tune_keys, c, a_adaptive, dime_centroids)
            Phi_adapt_test = linear_anchor_diff_basis(test_keys, c, a_adaptive, dime_centroids)
            Phi_rand_fit = linear_anchor_diff_basis(fit_keys, c, a_random, dime_centroids)
            Phi_rand_tune = linear_anchor_diff_basis(tune_keys, c, a_random, dime_centroids)
            Phi_rand_test = linear_anchor_diff_basis(test_keys, c, a_random, dime_centroids)

            def ridge_select_and_eval(Phi_fit_, Phi_tune_, Phi_test_, target_fit, log_p_used):
                best_lam, best_tune_tv, best_theta = None, np.inf, None
                for lam in RIDGE_LAMBDA_GRID:
                    theta = fit_ridge(Phi_fit_, target_fit, lam)
                    d_tune, _ = reconstruct_logit(Phi_tune_, theta, log_p_used)
                    tv = mean_tv(d_tune, exact_tune_dist)
                    if tv < best_tune_tv:
                        best_tune_tv, best_lam, best_theta = tv, lam, theta
                d_test, _ = reconstruct_logit(Phi_test_, best_theta, log_p_used)
                return d_test, best_theta, best_lam

            # B0: fixed basis, R4's actual probability-loss fitting
            d_test_B0, theta_B0, lam_B0 = ridge_select_and_eval(Phi_fixed_fit, Phi_fixed_tune, Phi_fixed_test, z_fit, log_p)

            # B1: fixed basis, lstsq diagnostic onto gauge-fixed target (no histogram at inference)
            theta_B1, *_ = np.linalg.lstsq(Phi_fixed_fit, t_fit, rcond=None)
            d_test_B1, _ = reconstruct_logit(Phi_fixed_test, theta_B1, zero_log_p)

            # B2: adaptive basis, SAME lstsq diagnostic
            theta_B2, *_ = np.linalg.lstsq(Phi_adapt_fit, t_fit, rcond=None)
            d_test_B2, _ = reconstruct_logit(Phi_adapt_test, theta_B2, zero_log_p)

            # B3: adaptive basis, R4's actual probability-loss fitting (MAIN comparison vs. B0)
            d_test_B3, theta_B3, lam_B3 = ridge_select_and_eval(Phi_adapt_fit, Phi_adapt_tune, Phi_adapt_test, z_fit, log_p)

            # B4: random basis, R4's actual probability-loss fitting
            d_test_B4, theta_B4, lam_B4 = ridge_select_and_eval(Phi_rand_fit, Phi_rand_tune, Phi_rand_test, z_fit, log_p)

            dists = {"B0": d_test_B0, "B1": d_test_B1, "B2": d_test_B2, "B3": d_test_B3, "B4": d_test_B4}
            thetas = {"B0": theta_B0, "B1": theta_B1, "B2": theta_B2, "B3": theta_B3, "B4": theta_B4}
            phis_fit = {"B0": Phi_fixed_fit, "B1": Phi_fixed_fit, "B2": Phi_adapt_fit, "B3": Phi_adapt_fit, "B4": Phi_rand_fit}
            include_hist = {"B0": True, "B1": False, "B2": False, "B3": True, "B4": True}

            for m in METHODS:
                test_tv = mean_tv(dists[m], exact_test_dist)
                test_js = float(np.mean([js_divergence_dict(dict_from_dist(token_ids, dists[m][qi]), dict_from_dist(token_ids, exact_test_dist[qi]))
                                            for qi in range(exact_test_dist.shape[0])]))
                diag = basis_matrix_diagnostics(phis_fit[m])
                mem_bytes = storage_bytes(n_tokens, r, include_hist[m])
                cluster_results_rows.append({
                    "cluster": c, "rank": r, "method": m, "population": population, "n_tokens": n_tokens,
                    "test_tv": round(test_tv, 6), "test_js": round(test_js, 6),
                    "condition_number": diag["condition_number"] if diag else None,
                    "numerical_effective_rank": diag["numerical_effective_rank"] if diag else None,
                    "mean_abs_offdiag_correlation": diag["mean_abs_offdiag_correlation"] if diag else None,
                    "memory_bytes": mem_bytes,
                })
                rank_agg[m][r].append(test_tv)
                basis_results_rows.append({"cluster": c, "rank": r, "method": m,
                                              "anchor_centroid_ids": ",".join(map(str, (a_fixed if m == "B0" or m == "B1" else (a_adaptive if m in ("B2", "B3") else a_random)).tolist())),
                                              "condition_number": diag["condition_number"] if diag else None,
                                              "numerical_effective_rank": diag["numerical_effective_rank"] if diag else None,
                                              "theta_norm": float(np.linalg.norm(thetas[m])), "memory_bytes": mem_bytes})

            if r == PRIMARY_RANK:
                qfp = query_feature_prediction_quality(Phi_fixed_fit, Phi_fixed_test, exact_fit_dist, exact_test_dist, p_j, r)
                qfp_rows.append({"cluster": c, "rank": r, **qfp})

                n = exact_test_dist.shape[0]
                if n >= 2:
                    n_pairs = min(N_PAIRS_PER_CLUSTER, n * (n - 1) // 2)
                    pi = rng.integers(0, n, size=n_pairs * 2)
                    pjx = rng.integers(0, n, size=n_pairs * 2)
                    valid = pi != pjx
                    pi, pjx = pi[valid][:n_pairs], pjx[valid][:n_pairs]
                    delta_exact = exact_test_dist[pi] - exact_test_dist[pjx]
                    for m in METHODS:
                        delta_m = dists[m][pi] - dists[m][pjx]
                        for k in range(len(pi)):
                            e_delta = float(np.abs(delta_exact[k] - delta_m[k]).sum())
                            exact_mag = float(np.abs(delta_exact[k]).sum())
                            dime_err = exact_mag
                            query_pair_rows.append({"cluster": c, "method": m, "pair_idx": k, "e_delta": round(e_delta, 6),
                                                       "exact_magnitude": round(exact_mag, 6), "dime_zero_change_error": round(dime_err, 6)})

        for step_i, step in enumerate(greedy_trace):
            memory_results_rows.append({"cluster": c, "rank": step["step"], "method": "B2/B3_greedy_trace",
                                           "selected_pool_index": step["selected_pool_index"], "score": round(step["score"], 8),
                                           "residual_frobenius_norm": round(step["residual_frobenius_norm"], 6)})

        if (ci + 1) % max(1, len(sample_clusters) // 10) == 0 or ci == len(sample_clusters) - 1:
            print(f"  processed {ci + 1}/{len(sample_clusters)} clusters")
            save_results(os.path.join(OUT_DIR, "crisp_r7_summary.json"), report)

    # ============================================================
    # Aggregation
    # ============================================================
    print("\nAggregating rank sweep and comparisons...")
    rank_results_rows = []
    for m in METHODS:
        for r in RANK_GRID:
            tvs = rank_agg[m][r]
            rank_results_rows.append({"method": m, "rank": r, "n_clusters": len(tvs),
                                         "mean_test_tv": round(float(np.mean(tvs)), 6), "median_test_tv": round(float(np.median(tvs)), 6)})
    with open(os.path.join(OUT_DIR, "rank_results.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rank_results_rows[0].keys()))
        w.writeheader()
        for row in rank_results_rows:
            w.writerow(row)
    with open(os.path.join(OUT_DIR, "cluster_results.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(cluster_results_rows[0].keys()))
        w.writeheader()
        for row in cluster_results_rows:
            w.writerow(row)
    with open(os.path.join(OUT_DIR, "basis_results.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(basis_results_rows[0].keys()))
        w.writeheader()
        for row in basis_results_rows:
            w.writerow(row)
    with open(os.path.join(OUT_DIR, "memory_results.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(memory_results_rows[0].keys()))
        w.writeheader()
        for row in memory_results_rows:
            w.writerow(row)
    with open(os.path.join(OUT_DIR, "query_pair_results.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(query_pair_rows[0].keys()))
        w.writeheader()
        for row in query_pair_rows:
            w.writerow(row)

    b0_r8 = float(np.mean(rank_agg["B0"][PRIMARY_RANK]))
    b3_r8 = float(np.mean(rank_agg["B3"][PRIMARY_RANK]))
    b4_r8 = float(np.mean(rank_agg["B4"][PRIMARY_RANK]))
    b1_r8 = float(np.mean(rank_agg["B1"][PRIMARY_RANK]))
    b2_r8 = float(np.mean(rank_agg["B2"][PRIMARY_RANK]))
    per_cluster_b0 = {row["cluster"]: row["test_tv"] for row in cluster_results_rows if row["rank"] == PRIMARY_RANK and row["method"] == "B0"}
    per_cluster_b3 = {row["cluster"]: row["test_tv"] for row in cluster_results_rows if row["rank"] == PRIMARY_RANK and row["method"] == "B3"}
    diffs = np.array([per_cluster_b3[c] - per_cluster_b0[c] for c in per_cluster_b0])
    frac_b3_wins = float((diffs < 0).mean())
    n_boot = 2000
    boot_rng = np.random.default_rng(SEED)
    boot_means = np.array([diffs[boot_rng.integers(0, len(diffs), size=len(diffs))].mean() for _ in range(n_boot)])
    ci_lo, ci_hi = np.percentile(boot_means, [2.5, 97.5])

    mem_match_rows = []
    for r in RANK_GRID:
        b0_bytes = next(row["memory_bytes"] for row in cluster_results_rows if row["rank"] == r and row["method"] == "B0")
        b3_bytes = next(row["memory_bytes"] for row in cluster_results_rows if row["rank"] == r and row["method"] == "B3")
        mem_match_rows.append({"rank": r, "B0_bytes": b0_bytes, "B3_bytes": b3_bytes,
                                  "B0_mean_tv": round(float(np.mean(rank_agg["B0"][r])), 6),
                                  "B3_mean_tv": round(float(np.mean(rank_agg["B3"][r])), 6),
                                  "memory_matched": bool(b0_bytes == b3_bytes)})
    report["memory_matched_comparison"] = mem_match_rows
    # diffs = B3 - B0 (negative => B3 has smaller/better TV). A CI entirely below zero means B3 is
    # SIGNIFICANTLY BETTER; entirely above zero means B3 is significantly WORSE -- both are
    # "significant" in the ordinary sense (CI excludes zero), so "significant" alone must not be
    # read as "significantly better" (caught directly from this pilot's own output: B3 came out
    # reliably WORSE with a CI of [+0.0019, +0.0028], which the original ci_lo<0-and-ci_hi<0 check
    # mislabeled as significant=False).
    significant_different = bool(ci_lo > 0 or ci_hi < 0)
    significantly_better = bool(ci_hi < 0)
    report["main_comparison_B0_vs_B3_at_primary_rank"] = {
        "B0_mean_tv": b0_r8, "B3_mean_tv": b3_r8, "B4_mean_tv": b4_r8, "B1_mean_tv": b1_r8, "B2_mean_tv": b2_r8,
        "frac_clusters_B3_beats_B0": frac_b3_wins, "mean_paired_diff_B3_minus_B0": float(diffs.mean()),
        "ci_95_low": float(ci_lo), "ci_95_high": float(ci_hi),
        "significant_different_from_zero": significant_different, "significantly_better": significantly_better,
    }
    print(f"  Main comparison (rank={PRIMARY_RANK}): {report['main_comparison_B0_vs_B3_at_primary_rank']}")
    save_results(os.path.join(OUT_DIR, "crisp_r7_summary.json"), report)

    qfp_agg = {k: float(np.mean([row[k] for row in qfp_rows])) for k in ["oracle_output_space_rel_error", "achievable_rel_error_using_predicted_coords", "coord_prediction_rel_error"]}
    print(f"  Output-space vs. query-feature-space gap (rank={PRIMARY_RANK}): {qfp_agg}")
    report["output_space_vs_query_feature_gap"] = qfp_agg
    save_results(os.path.join(OUT_DIR, "crisp_r7_summary.json"), report)

    qp_summary = {}
    for m in METHODS:
        recs = [r for r in query_pair_rows if r["method"] == m]
        if recs:
            qp_summary[m] = {"n_pairs": len(recs), "mean_e_delta": float(np.mean([r["e_delta"] for r in recs])),
                               "mean_dime_zero_change_error": float(np.mean([r["dime_zero_change_error"] for r in recs]))}
    print(f"  Query-change preservation: {qp_summary}")
    report["query_change_summary"] = qp_summary
    save_results(os.path.join(OUT_DIR, "crisp_r7_summary.json"), report)

    # ============================================================
    # Plots
    # ============================================================
    print("\nGenerating plots...")
    fig, ax = plt.subplots(figsize=(6, 6))
    b0_vals = [per_cluster_b0[c] for c in per_cluster_b0]
    b3_vals = [per_cluster_b3[c] for c in per_cluster_b0]
    ax.scatter(b0_vals, b3_vals, s=10, alpha=0.5)
    lims = [0, max(max(b0_vals), max(b3_vals))]
    ax.plot(lims, lims, "k--", linewidth=1)
    ax.set_xlabel("B0 (fixed basis) test TV"); ax.set_ylabel("B3 (adaptive basis) test TV")
    ax.set_title("Fixed vs. adaptive TV by cluster")
    fig.tight_layout(); fig.savefig(os.path.join(PLOT_DIR, "1_fixed_vs_adaptive_by_cluster.png"), dpi=130); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    for m in METHODS:
        ax.plot(RANK_GRID, [np.mean(rank_agg[m][r]) for r in RANK_GRID], marker="o", label=m)
    ax.set_xlabel("rank"); ax.set_ylabel("mean test TV"); ax.legend()
    ax.set_title("Error vs. rank, all methods")
    fig.tight_layout(); fig.savefig(os.path.join(PLOT_DIR, "2_error_vs_rank.png"), dpi=130); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    for m in ["B0", "B3"]:
        xs = [next(row["memory_bytes"] for row in cluster_results_rows if row["rank"] == r and row["method"] == m) for r in RANK_GRID]
        ys = [np.mean(rank_agg[m][r]) for r in RANK_GRID]
        ax.plot(xs, ys, marker="o", label=m)
    ax.set_xlabel("memory bytes/cluster"); ax.set_ylabel("mean test TV"); ax.legend()
    ax.set_title("Error vs. stored bytes (memory-matched)")
    fig.tight_layout(); fig.savefig(os.path.join(PLOT_DIR, "3_error_vs_bytes.png"), dpi=130); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    c_demo = sample_clusters[0]
    sv = np.linalg.svd(linear_anchor_diff_basis(ct_keys[sample_splits[c_demo]["fit"]], c_demo, fixed_anchor_idx, dime_centroids), compute_uv=False)
    ax.plot(range(1, len(sv) + 1), sv, marker="o")
    ax.set_yscale("log"); ax.set_xlabel("component"); ax.set_ylabel("singular value (log)")
    ax.set_title(f"Feature singular-value decay, cluster {c_demo}, fixed basis rank={max(RANK_GRID)}")
    fig.tight_layout(); fig.savefig(os.path.join(PLOT_DIR, "4_feature_singular_values.png"), dpi=130); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    for c in list(greedy_traces.keys())[:10]:
        ax.plot([t["step"] for t in greedy_traces[c]], [t["residual_frobenius_norm"] for t in greedy_traces[c]], alpha=0.5)
    ax.set_xlabel("greedy step"); ax.set_ylabel("residual Frobenius norm")
    ax.set_title("Residual reduction at each selected basis function (first 10 clusters)")
    fig.tight_layout(); fig.savefig(os.path.join(PLOT_DIR, "5_residual_reduction_trace.png"), dpi=130); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    train_err_proxy = [row["numerical_effective_rank"] for row in cluster_results_rows if row["rank"] == PRIMARY_RANK and row["method"] == "B0" and row["numerical_effective_rank"] is not None]
    test_err_proxy = [row["test_tv"] for row in cluster_results_rows if row["rank"] == PRIMARY_RANK and row["method"] == "B0" and row["numerical_effective_rank"] is not None]
    ax.scatter(train_err_proxy, test_err_proxy, s=10, alpha=0.5)
    ax.set_xlabel("design-matrix numerical effective rank (B0, fixed basis)"); ax.set_ylabel("held-out test TV")
    ax.set_title("Training conditioning vs. held-out error")
    fig.tight_layout(); fig.savefig(os.path.join(PLOT_DIR, "6_train_vs_heldout.png"), dpi=130); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(list(qp_summary.keys()), [qp_summary[m]["mean_e_delta"] for m in qp_summary])
    ax.axhline(qp_summary["B0"]["mean_dime_zero_change_error"], color="r", linestyle="--", label="DIME zero-change baseline")
    ax.set_ylabel("mean query-change L1 error"); ax.legend()
    ax.set_title("Query-change error by method")
    fig.tight_layout(); fig.savefig(os.path.join(PLOT_DIR, "7_query_change_error.png"), dpi=130); plt.close(fig)

    # ============================================================
    # failure_cases.md
    # ============================================================
    print("\nSelecting failure cases...")
    worsened = sorted(zip(per_cluster_b0.keys(), diffs.tolist()), key=lambda x: -x[1])[:N_FAILURE_EXAMPLES]
    improved = sorted(zip(per_cluster_b0.keys(), diffs.tolist()), key=lambda x: x[1])[:N_FAILURE_EXAMPLES]
    failure_lines = ["# CRISP-R7 Failure Cases", "",
                       "Adaptive (B3) vs. fixed (B0) basis, paired by cluster at rank=8.", "",
                       "## Adaptive selection WORSENS vs. fixed (overfitting candidates)"]
    for c, d in worsened:
        failure_lines.append(f"- cluster={c}: B0_tv={per_cluster_b0[c]:.5f}, B3_tv={per_cluster_b3[c]:.5f}, diff={d:+.5f}")
    failure_lines.append("\n## Adaptive selection IMPROVES most over fixed")
    for c, d in improved:
        failure_lines.append(f"- cluster={c}: B0_tv={per_cluster_b0[c]:.5f}, B3_tv={per_cluster_b3[c]:.5f}, diff={d:+.5f}")
    with open(os.path.join(OUT_DIR, "failure_cases.md"), "w") as f:
        f.write("\n".join(failure_lines))

    # ============================================================
    # r3_r4_method_audit.md
    # ============================================================
    audit_md = """# R3/R4 Method Audit (Phase A)

## R3 audit

- **Quantity approximated (Part A/B/C):** RAW EVIDENCE s_y(q) (R1's exact_evidence output), NOT the normalized
  distribution and NOT a logit correction -- fit_ridge is called directly on fit_ev in Parts A and B. Part C
  explicitly compares this (objective A) against objective B (normalized distribution) and objective C
  (distribution + change penalty), run_crisp_r3_failure_diagnosis.py lines ~554-589.
- **Matrix decomposed by SVD (Part F):** `M_fit - p_j[None,:]`, i.e. the CENTERED, NORMALIZED distribution matrix
  from FIT queries (run_crisp_r3_failure_diagnosis.py, functional_rank_analysis, lines 249-265).
- **Singular vector space:** TOKEN-OUTPUT space (columns = tokens), NOT query-feature space. There is no phi(q)
  basis anywhere in Part F -- it decomposes the responses themselves.
- **Projection error calculation:** `coords_test = M_centered_test @ basis_r.T` -- this uses the TRUE held-out
  response to find its coordinates in the FIT-derived subspace. This is an ORACLE projection, not a prediction
  from q. Confirmed directly from source, not assumed. R7's query_feature_prediction_quality() adds the missing
  query-feature-based prediction for direct comparison.
- **High-projection-error clusters:** Part F ran on only 4 diagnostic clusters (2 "failing": 923/1465 by spec
  citation; 2 "successful", chosen by lowest rank-8 test TV among the base-20 sample) -- not a population-level
  statement.
- **Rank grid used by functional_rank_analysis:** R3's own module-level RANK_GRID=[2,4,8,16,32] (no rank=1) --
  this is a FREE VARIABLE resolved from R3's own module globals at call time, a fact that mattered when this
  function was reused in the Y1 delivery (Y1 needed rank=1 and had to iterate the function's own returned keys
  rather than assume its own grid, caught via self-review before running).
- **Sample counts/regularization:** MAX_FIT/TUNE/TEST=150/50/100; RIDGE_LAMBDA_GRID=[0.01,0.1,1.0,10.0]; Part B's
  own `fit_lstsq`/`fit_standardized_ridge` are DIAGNOSTIC alternatives, never adopted into R4's production path.

## R4 audit

- **phi_l(q) definition:** phi_l(q) = -2*q.(a_l - a_0) + ||a_l||^2 - ||a_0||^2 (TRI1's compute_diff_features,
  run_e2tri1_anchor_conditioned_compression.py lines 161-176) -- AFFINE (linear) in q. a_0 is the cluster's OWN
  centroid; a_l are OTHER centroids.
- **Anchor/basis selection:** select_anchors_nearest(dist_matrix, c, k) = the k NEAREST other centroids by
  centroid-centroid Euclidean distance, in ascending order (run_e2tri1_anchor_conditioned_compression.py lines
  137-138). Anchors are DIME CENTROIDS, never datastore vectors, query vectors, or learned parameters.
- **Nested or independent basis sets across rank:** NESTED -- `a_idx = get_anchors(c)[:r]` (R4 line ~560) takes
  a PREFIX of the same sorted anchor list for every rank. Rank 16 is rank 8 plus the next 8 nearest centroids,
  confirmed directly, not assumed.
- **Kernel type/bandwidth:** none -- this is a pure linear (affine) feature map, not an RBF kernel (R3's Part A
  RBF variant, with a gamma bandwidth, was a SEPARATE diagnostic never adopted into R4's production basis).
- **Shared vs. cluster-specific:** CLUSTER-SPECIFIC -- a_0 is cluster c's own centroid, so the feature map
  differs per cluster even for the same anchor indices.
- **Coefficient fitting:** closed-form ridge (fit_ridge, PLAIN, non-standardized), 4-point lambda grid, selected
  on the TUNE split by held-out TV (R4 lines 348-357, B4 block of fit_rank_all_methods).
- **Loss function / target:** z_{q,y} = log(F(q)(y)/p_j(y)) (log_ratio_target, R4 lines 205-215) -- a specific
  gauge of the exact correction, NOT the spec's own suggested gauge (see mathematical_analysis.md).
- **Normalization/clipping:** B4/B5's softmax reconstruction needs NO clipping (always valid by construction);
  B2/B3 (raw-evidence/direct-distribution objectives) DO need clip-and-renormalize.
- **Storage per basis function:** one anchor index (4 bytes) shared across all tokens, plus one coefficient per
  (token, rank) pair (4 bytes each) -- b4_bytes_for(n_tokens, rank), R4 lines 234-237.

## Required outcome -- which diagnostic measures what

1. **Output-subspace approximation error:** R3 Part F (oracle, as audited above).
2. **Query-feature approximation error:** NOT measured anywhere in R3 or R4 directly -- R4's own held-out TV/JS
   conflates query-feature prediction quality with the choice of objective (log-ratio vs. raw evidence) and with
   regularization; R7's query_feature_prediction_quality() isolates this directly for the first time.
3. **Optimization or estimation error:** R3 Part B (fitting-method comparison, holding basis+target fixed).

These three remain reported as DISTINCT quantities throughout R7, per the spec's explicit instruction, never
conflated into a single "error vs. rank" number.
"""
    with open(os.path.join(OUT_DIR, "r3_r4_method_audit.md"), "w") as f:
        f.write(audit_md)

    # ============================================================
    # mathematical_analysis.md
    # ============================================================
    math_md = f"""# CRISP-R7 Mathematical Analysis

## Approximation spaces

V_r = span{{phi_1,...,phi_r}}, phi_l(q) = -2 q.(a_l-a_0) + ||a_l||^2-||a_0||^2 (TRI1's anchor-difference feature,
reused unchanged). E_r(f_y)^2 = min_{{g in V_r}} E_q[(f_y(q)-g(q))^2] is the population-optimal approximation
error within V_r -- UNKNOWN in general; this script reports the EMPIRICAL training optimum (Theta* =
argmin_Theta ||Phi Theta - T||_F^2, solved by np.linalg.lstsq) and its HELD-OUT error as two DISTINCT quantities,
never conflated with the true (unknown) population optimum or with R4's own probability-loss-trained model
(which optimizes a different, log-ratio-gauge, ridge-regularized objective).

## Gauge-fixing

delta*_y(q) = log(s_y(q)) - log(n_y) (computed in compute_delta_star, from R1's own raw evidence s_y(q)).
t_y(q) = delta*_y(q) - sum_y p_j(y) delta*_y(q) (gauge_fix). R4's own target z_{{q,y}}=log(F(y)/p_j(y)) is a
DIFFERENT, equally valid gauge of the SAME equivalence class -- z = delta* - [log(sum_z s_z(q)) - log(N)], and
the bracketed term is common to every label y at a fixed q. softmax(z) == softmax(delta*) == softmax(t) exactly
(verified numerically in synthetic_correctness_checks -- gauge_shift_invariance_exact). Single-label clusters
are excluded from the sample (t is identically zero there, trivially "perfectly fit" by any basis, uninformative).

## Residual-guided (orthogonal greedy) selection

S(phi) = ||E_q[u(q)E(q)]||_2^2 / E_q[u(q)^2], u = phi - P_{{V_r}}phi, implemented batched over the full N_POOL={N_POOL}
candidate pool per step (orthogonal_greedy_select). This is the one-step-optimal score for the CURRENT space and
squared-error objective -- it does NOT guarantee the globally best r-subset (a known limitation of greedy/OMP
methods), and it optimizes squared error in the gauge-fixed correction space, NOT the final softmax
distribution's TV/JS/cross-entropy directly (those are evaluated separately, never assumed to track the greedy
objective).

## Output-space vs. query-feature-space separation

query_feature_prediction_quality() fits coords = Phi(q).theta_coord to PREDICT the output-subspace coordinates
(found via R3's own fit-side SVD), then compares: (a) oracle_output_space_rel_error (R3's original diagnostic,
using the TRUE held-out response's coordinates), (b) achievable_rel_error_using_predicted_coords (using
Phi(q)-PREDICTED coordinates instead). The GAP between (a) and (b) is the query-feature approximation error the
spec's Phase F explicitly asks to separate from output-space sufficiency. A Pythagorean (orthogonal) decomposition
is valid for THIS linear reconstruction step (both are least-squares projections in the SAME output subspace),
but is NOT applied to TV/JS after softmax, per the spec's explicit caution -- TV/JS errors are reported as their
own separately-measured quantities throughout.

## Assumptions and limitations

- The candidate pool (N_POOL={N_POOL} nearest centroids) is itself a FIXED choice -- greedy selection chooses
  WHICH of these to use, not arbitrary directions in R^768. A genuinely larger or different pool could change
  which clusters show adaptive improvement; not swept here, per the spec's own "do not expand unless justified"
  instruction.
- R-series validation data has been used repeatedly across R1-R6; this delivery remains EXPLORATORY per the
  spec's own instruction, not an independent confirmatory test.
- No document/sequence-grouped query split exists in the cached arrays; the existing position-level split is
  used, stated as a limitation.
"""
    with open(os.path.join(OUT_DIR, "mathematical_analysis.md"), "w") as f:
        f.write(math_md)

    # ============================================================
    # research_questions.md
    # ============================================================
    research_q = """# CRISP-R7 Research Questions for Subsequent Study

## Approximation theory
- Orthogonal greedy selection is one-step-optimal, not globally optimal for a fixed rank r -- how far from the
  globally best r-subset (over the N_POOL candidate pool) does it land in practice, and is there a tractable
  exact or approximate algorithm (e.g. via submodularity) that would close that gap?
- Does the output-space-vs-query-feature-space gap (Phase F) scale predictably with cluster population or label
  entropy, or does it depend more on the GEOMETRY of how nearby centroids relate to the cluster's own members?

## Geometry
- R4's fixed nearest-centroid anchors are a prefix of a SORTED list -- is there a precise geometric condition
  (e.g. on the local curvature of the centroid manifold) that predicts when this ordering produces redundant
  (ill-conditioned) directions at higher rank, versus genuinely new information?

## Statistical estimation
- B1/B2's gauge-fixed lstsq diagnostic needs no ridge regularization in this delivery (plain least squares) --
  would a properly cross-validated ridge penalty on the SAME gauge change the fixed-vs-adaptive comparison's
  conclusion, and is the gauge choice itself a confound that should be swept rather than fixed by convention?

## Compression architecture
- If adaptive (residual-guided) basis selection shows a genuine, memory-matched improvement over R4's fixed
  basis, what would it cost to make the selection itself part of a DEPLOYABLE representation (i.e., store WHICH
  centroids were selected per cluster, not just assume a fixed prefix) -- is the extra anchor-identity storage
  (already counted in memory_results.csv) the dominant cost, or is it negligible relative to coefficients?
"""
    with open(os.path.join(OUT_DIR, "research_questions.md"), "w") as f:
        f.write(research_q)

    # ============================================================
    # Final verdict (Q1-Q10) + report.md
    # ============================================================
    print("\nComputing final verdict (Q1-Q10)...")
    q1 = "R3's Part F measures OUTPUT-SUBSPACE approximation (oracle projection using the true held-out response), not query-feature prediction. See r3_r4_method_audit.md."
    q2 = "R4's basis is V_r = span of r anchor-difference features to the r NEAREST other centroids, affine in q, NESTED across rank, cluster-specific. See r3_r4_method_audit.md."
    q3_expanded = bool(np.mean([t["score"] for c in greedy_traces for t in greedy_traces[c] if t["step"] > max(RANK_GRID) // 2]) >
                        np.mean([t["score"] for c in greedy_traces for t in greedy_traces[c] if t["step"] <= max(RANK_GRID) // 2]) * 0.3) if greedy_traces else None
    q4_redundant = bool(np.mean([row["numerical_effective_rank"] / row["rank"] for row in cluster_results_rows
                                   if row["method"] == "B0" and row["rank"] == max(RANK_GRID) and row["numerical_effective_rank"] is not None]) < 0.8) \
        if any(row["method"] == "B0" and row["rank"] == max(RANK_GRID) for row in cluster_results_rows) else None
    q5_improves_empirical = bool(b2_r8 <= b1_r8)
    q6_generalizes = bool(b3_r8 < b0_r8)
    q7_improves_fidelity = bool(frac_b3_wins > 0.5 and report["main_comparison_B0_vs_B3_at_primary_rank"]["significantly_better"])
    q8 = report["query_change_summary"]
    q9_memory_matched_holds = bool(mem_match_rows[RANK_GRID.index(PRIMARY_RANK)]["memory_matched"] and q6_generalizes)
    q10_output_gap = qfp_agg["achievable_rel_error_using_predicted_coords"] - qfp_agg["oracle_output_space_rel_error"]

    findings = {"basis_selection_mismatch": bool(q6_generalizes and q7_improves_fidelity),
                 "fitting_or_generalization_limitation": bool(q5_improves_empirical and not q6_generalizes),
                 "insufficient_representational_capacity": bool(q4_redundant is False and not q7_improves_fidelity)}
    n_supported = sum(1 for v in findings.values() if v)
    if n_supported == 0:
        if report["main_comparison_B0_vs_B3_at_primary_rank"]["significant_different_from_zero"] is False and abs(report["main_comparison_B0_vs_B3_at_primary_rank"]["mean_paired_diff_B3_minus_B0"]) < 0.005:
            verdict = "No meaningful adaptive improvement"
        else:
            verdict = "Inconclusive"
    elif n_supported == 1:
        verdict = [k for k, v in findings.items() if v][0].replace("_", " ")
        verdict = {"basis_selection_mismatch": "Evidence supports basis-selection mismatch as an important limitation",
                    "fitting_or_generalization_limitation": "Evidence supports fitting or generalization limitations",
                    "insufficient_representational_capacity": "Evidence supports insufficient representational capacity"}[[k for k, v in findings.items() if v][0]]
    else:
        verdict = "Multiple interacting limitations"
    print(f"  Verdict: {verdict}")
    verdict_block = {"Q1": q1, "Q2": q2, "Q3_basis_expanded_effective_space": q3_expanded, "Q4_high_rank_redundant": q4_redundant,
                       "Q5_empirical_improves": q5_improves_empirical, "Q6_generalizes_held_out": q6_generalizes,
                       "Q7_improves_fidelity_under_B4_objective": q7_improves_fidelity, "Q8_query_change_preservation": q8,
                       "Q9_holds_memory_matched": q9_memory_matched_holds, "Q10_rank_plateau_implication": q10_output_gap,
                       "findings": findings, "final_classification": verdict}
    report["final_verdict"] = verdict_block
    save_results(os.path.join(OUT_DIR, "crisp_r7_summary.json"), report)

    lines = [
        "# CRISP-R7: Functional Approximation Diagnostics and Adaptive Basis Selection", "",
        "## 1. Research question", "See module docstring and r3_r4_method_audit.md.", "",
        "## 2. Audit summary", "See r3_r4_method_audit.md for the full audit.", "",
        "## 3. Mathematical framing", "See mathematical_analysis.md.", "",
        "## 4. Main comparison (B0 vs. B3, memory-matched, rank=8)", f"{report['main_comparison_B0_vs_B3_at_primary_rank']}", "",
        "## 5. Output-space vs. query-feature-space gap", f"{qfp_agg}", "",
        "## 6. Query-change preservation", f"{qp_summary}", "",
        "## 7. Memory-matched comparison across ranks", f"{mem_match_rows}", "",
        "## 8. Final verdict", f"**{verdict}**", f"{verdict_block}", "",
        "## Limitations",
        f"- Cluster sample reproduces R4's exact procedure ({len(sample_clusters)} clusters after excluding "
        f"{n_excluded_singlelabel} single-label clusters) -- not expanded to all 3,811, per the spec's own instruction.",
        "- No document/sequence-grouped split exists in the cached data.",
        "- This delivery remains EXPLORATORY, not an independent confirmatory test (R-series validation data has "
        "been used repeatedly across R1-R6).",
        "- The candidate pool for greedy selection is fixed at N_POOL={} nearest centroids, not swept.".format(N_POOL),
    ]
    with open(os.path.join(OUT_DIR, "report.md"), "w") as f:
        f.write("\n".join(lines))

    report["sections_completed"] = ["synthetic_correctness_checks", "cluster_sampling", "main_experiment",
                                      "rank_results.csv", "cluster_results.csv", "basis_results.csv",
                                      "memory_results.csv", "query_pair_results.csv", "main_comparison",
                                      "output_space_vs_query_feature_gap", "query_change_summary",
                                      "r3_r4_method_audit.md", "mathematical_analysis.md", "research_questions.md",
                                      "failure_cases.md", "final_verdict", "report.md", "plots"]
    report["total_runtime_seconds"] = time.perf_counter() - t_start
    save_results(os.path.join(OUT_DIR, "crisp_r7_summary.json"), report)
    print(f"\nAll results saved under {OUT_DIR}")
    print(f"Total runtime: {report['total_runtime_seconds']/60:.1f} min")
    print("No existing E1/E2/TRI1/.../R1-R6/Y1 outputs were modified. R8/Y1 not started, per the spec's instruction.")


if __name__ == "__main__":
    main()
