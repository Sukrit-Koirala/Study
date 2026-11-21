"""CRISP-R5 -- Global Retrieval Fidelity and Compression Evaluation. Bridges
R1-R4's within-cluster-only diagnostics to the actual end-to-end retrieval
process. Writes only to results/stage3_crisp_r5_global_retrieval/, preserving
every prior E1/E2/TRI1/NC1/GEOM1/KM1/R1/R2/R3/R4 output unchanged.

=====================================================================
Reproduction note
=====================================================================
None of R1/R2/R3/R4's real persisted results are present locally (R1-R3's
results directories exist but are empty -- no summary.json; R4's directory
does not exist at all -- checked directly). The specific R4 numbers cited in
this spec (66 eligible clusters, B4 mean TV=0.279975 at rank 8, etc.) could
NOT be diffed against a persisted artifact. This script does not treat them
as verified ground truth. What IS available locally and real:
  - results/wikitext103_gpt2_tier1.json -- real tier-1 tuned-config results
    (raw kNN best: k=300,tau=1,alpha=0.25; DIME best: k=100,tau=2,alpha=0.1;
    gpt_only_mean_nll=4.360981...). Used only as the secondary/tuned-config
    REFERENCE this spec names -- not re-run in this delivery (see Scoping).
  - results/stage3_e1b3_cluster_vs_aggregation/e1b3_summary.json (real,
    present) -- E1B.3's own "System 3" (DIME centroid selection + fresh
    query-conditioned per-member softmax aggregation) is EXACTLY this spec's
    S3 oracle, same math, same matched config (k=20, tau=2.0, alpha=0.25
    irrelevant here since no LM interpolation). This script's S3/S5/S6
    combination logic reuses that exact pattern (verified inline, see Phase
    2 checks), not a new definition.
  - mixing.py / e1_metrics.py / knn.py -- confirm DIME-V1's real production
    formula: beta_j(q) = softmax(-d_centroid/tau) (knn.softmax_np), and
    P_DIME(y|q) = sum_j beta_j(q) * (count_{j,y}/|C_j|) with NO Dirichlet
    smoothing in the standalone path used throughout E1/E1B/E2 (e1_metrics.
    dime_standalone_prob/dime_full_distribution take no beta-smoothing
    argument; mixing.py's OWN "beta" parameter is a DIFFERENT thing --
    Dirichlet/global-frequency smoothing for the LM-interpolated NLL path,
    not the mixing weight the R5 spec also calls beta_j(q). This script
    follows e1_metrics' unsmoothed convention throughout, matching R5's
    boxed S1 formula literally.
  - knn.py's build_datastore always uses metric="euclidean" via sklearn's
    exact NearestNeighbors (brute-force/tree, never approximate) -- so S0's
    "audit ANN recall" instruction is moot here: there is no ANN index
    anywhere in this codebase, confirmed by reading knn.py directly, not
    assumed. Recall against exact search is trivially 100% by construction.

=====================================================================
R4's B4 model, reused exactly (per the spec's explicit "do not invent a new
CRISP model" instruction)
=====================================================================
F~_j(q)(y) = p_j(y)*exp(delta_{j,y}(q)) / Z, delta_{j,y}(q)=Phi(q).theta_{j,y},
fit by ridge regression onto z_{q,y}=log(F_exact(q)(y)/p_j(y)), basis =
R1's linear anchor-difference features. log_ratio_target, reconstruct_logit,
RIDGE_LAMBDA_GRID are imported directly from run_crisp_r4_histogram_corrections
(not redefined). Rank 8, exactly R4's primary rank.

=====================================================================
Essential issue (spec Section 4): fitting CRISP for ALL 3,811 clusters
=====================================================================
This script fits a rank-8 B4 correction for EVERY cluster with population
>= MIN_FIT_POP and enough controller_train queries TOP-1-routed to it (the
same routing convention R1-R4 use for per-cluster training splits); clusters
that fail this bar fall back to the plain histogram (rank 0, i.e. identical
to DIME-V1 for that cluster). Fitting coverage is reported BOTH by cluster
count and by held-out query-ROUTING FREQUENCY (how often a real evaluation
query's selected top-20 centroids actually land on a fitted vs. fallback
cluster) -- the second number is what actually matters for end-to-end
fidelity, and is computed at FULL val scale (cheap: a lookup against each
selected centroid's fitted/fallback flag, not a per-member computation).

=====================================================================
Retrieval systems implemented: S0-S6 (S7 explicitly NOT implemented --
deferred per the spec's own "only if S5/S6 establish mass matters, and
never let it delay primary results" instruction; see Limitations/Q7)
=====================================================================
  S0 raw global kNN            -- e1_metrics.raw_full_distribution/raw_standalone_prob, reused directly.
  S1 DIME-V1                    -- e1_metrics.dime_full_distribution/dime_standalone_prob, reused directly.
  S2 centroid-routing candidate oracle -- NEW: within U(q) (union of the 20 selected
                                    clusters' REAL members), find the actual k=20 nearest
                                    by real Euclidean distance, apply the SAME raw-kNN
                                    weighting (raw_full_distribution/raw_standalone_prob,
                                    reused, just given the candidate-restricted
                                    distances/values instead of the global top-20).
  S3 exact normalized cluster-function oracle -- E1B.3's "System 3" pattern (same
                                    beta_j(q) as S1, each cluster's inner term replaced
                                    by a freshly computed query-conditioned softmax vote
                                    over ALL of that cluster's real members).
  S4 CRISP-R4 rank 8            -- same beta_j(q) as S1/S3, inner term = the GLOBALLY
                                    fitted B4 model's F~_j(q) (or histogram fallback).
  S5 mass-aware exact oracle     -- same F_j(q) as S3, combined via the REAL evidence
                                    mass m_j(q)=sum_i exp(-d/tau) instead of beta_j(q).
  S6 CRISP + exact masses (diagnostic) -- CRISP's F~_j(q), combined via the REAL mass
                                    m_j(q) (requires raw vectors; not deployable).
  S7 optional compressed mass predictor -- NOT implemented; see Limitations/Q7.

Per-query cost design: for each of the 20 selected clusters, member distances
are computed ONCE (for cluster_query_conditioned's exact F_j/m_j) and reused
for BOTH S2's candidate pool (concatenated across all 20 clusters) and S3/S5
-- avoiding duplicate distance computation that would otherwise make S2 cost
as much again as S3/S5 combined.

=====================================================================
Scoping decisions (stated up front; Phase 1-4 of the spec's own staged plan
only -- Phase 5 and several named "if resources allow" items are explicitly
deferred, not silently dropped):
=====================================================================
  - PRIMARY matched configuration only: raw k=20/tau=2.0, DIME k=20
    centroids/tau=2.0, CRISP rank=8. The secondary TUNED configuration
    (raw k=300/tau=1 and DIME k=100/tau=2, both real numbers confirmed in
    results/wikitext103_gpt2_tier1.json) is NOT re-run here -- stated
    explicitly as deferred, per the spec's own "do not sacrifice the main
    comparison" instruction.
  - CRISP rank sensitivity (4, 16) and alternative centroid counts: NOT run
    in this delivery (Phase 5, deferred).
  - S7 (compressed mass predictor): NOT implemented; S5-vs-S3 and S6-vs-S4
    gaps are reported so a future delivery can judge whether it would be
    justified, per the spec's own conditional instruction.
  - Full-scale (S0, S1) statistics AND the routing-overlap diagnostic
    (frac_neighbor_overlap / frac_mass_overlap, reusing E1B.3's exact
    Experiment-A pattern) are computed over the FULL 77,089-query val set
    -- cheap, since they need only batched kNN queries and assignment
    lookups, no per-member loops (E1B.3 already established this is
    feasible at full scale).
  - The expensive systems (S2, S3, S4, S5, S6, all of which need per-member
    distance computations) run on a BOUNDED, JS-decile-STRATIFIED sample of
    N_EVAL held-out val queries (default 8,000; overridable), reusing
    E1B.3's exact stratification method so rare high-divergence queries are
    not drowned out. This is stated explicitly, not a silent full-vs-sample
    substitution -- the exact query count and selection rule are reported.
  - Experiment G (latency): a LIGHT, in-loop timing capture (cumulative
    wall-clock per system across the main evaluation loop, divided by
    N_EVAL), not a dedicated isolated microbenchmark -- reported with the
    spec's own explicit caution that this is not a deployment-speed claim.
  - Data independence (spec Section 14): val has never been used to FIT any
    CRISP coefficients in R1-R4 or here (every per-cluster ridge fit uses
    controller_train fit/tune splits only) -- but the DECISION to adopt
    rank=8 / the logit-correction representation as "primary" was itself
    informed by R1-R4's own val-set results across this project's
    development history. R5's main comparison is therefore reported as
    CONFIRMATORY BUT NOT FULLY INDEPENDENT, not a pristine held-out test --
    stated explicitly rather than silently assumed pristine.
  - Retrieval-only NLL uses a single documented epsilon floor (1e-6),
    applied IDENTICALLY to every system, with correct-token COVERAGE
    (fraction with nonzero probability) always reported alongside and
    separately -- no base-LM interpolation anywhere in this script.

Smoke testing: set CRISPR5_SMOKE=1 with CRISPR5_CACHE_DIR / CRISPR5_OUT_DIR /
CRISPR5_N_CLUSTERS / CRISPR5_N_EVAL before running. Never set it for the real
run.
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
from knn import build_datastore, query_knn_indices, softmax_np
from results_io import save_results
from e1_metrics import (
    raw_standalone_prob, dime_standalone_prob, raw_full_distribution, dime_full_distribution,
    js_divergence_dict, distribution_shape_stats, retrieval_coverage)
from run_e2_information_allocation import build_cluster_to_members
from run_e2tri1_anchor_conditioned_compression import build_centroid_distance_matrix, select_anchors_nearest
from run_crisp_subspace_projection import (
    exact_evidence, normalize_rows, linear_anchor_diff_basis, fit_ridge,
    dict_from_dist, total_variation, top1_agreement, topk_overlap, entropy_of_dict)
from run_crisp_r4_histogram_corrections import log_ratio_target, reconstruct_logit, mean_tv, RIDGE_LAMBDA_GRID
from gpt_class import FrozenGPT2

SMOKE = os.environ.get("CRISPR5_SMOKE") == "1"
CACHE_DIR = os.environ.get("CRISPR5_CACHE_DIR", os.path.join(SCRIPT_DIR, "..", "GPT_Module", "cache"))
OUT_DIR = os.environ.get("CRISPR5_OUT_DIR", os.path.join(SCRIPT_DIR, "results", "stage3_crisp_r5_global_retrieval"))
PLOT_DIR = os.path.join(OUT_DIR, "plots")
os.makedirs(PLOT_DIR, exist_ok=True)

N_CLUSTERS = int(os.environ.get("CRISPR5_N_CLUSTERS", "3811"))
SEED = 42
DIM, DTYPE_BYTES, VALUE_BYTES = 768, 4, 4
OUTER_K = 20
TAU = 2.0
CRISP_RANK = 8
MIN_FIT_POP = 10
MIN_FIT_QUERIES = 5
MIN_TUNE_QUERIES = 3
MAX_FIT, MAX_TUNE = 150, 50
N_EVAL = int(os.environ.get("CRISPR5_N_EVAL", "8000"))
TOPK_LIST = [5, 10]
EPSILON_FLOOR = 1e-6
N_FAILURE_PER_CATEGORY = 10
SYSTEMS = ["S0", "S1", "S2", "S3", "S4", "S5", "S6"]


# ============================================================
# Shared per-(query, cluster) computation
# ============================================================
def cluster_distances(q_key, member_keys):
    return np.linalg.norm(member_keys - q_key[None, :], axis=1)


def exact_cluster_dist_and_mass(d, member_values, tau=TAU):
    """F_j(q) (dict, sums to 1) and m_j(q) = sum_i exp(-d_i/tau) (raw,
    UN-normalized evidence mass) from already-computed member distances."""
    w = np.exp(-d / tau)
    mass = float(w.sum())
    if mass <= 0:
        return {}, 0.0
    dist = {}
    for v, wi in zip(member_values, w):
        vi = int(v)
        dist[vi] = dist.get(vi, 0.0) + float(wi)
    return {k: v / mass for k, v in dist.items()}, mass


def support_mass_overlap(d_method, d_reference):
    """Fraction of d_method's probability mass that lands on tokens
    d_reference also considers (a clearly defined 'support recovery' metric,
    not an attempt to recover raw kNN information in a formal
    information-theoretic sense)."""
    if not d_reference:
        return None
    return float(sum(d_method.get(t, 0.0) for t in d_reference))


def retrieval_only_nll(p_true, floor=EPSILON_FLOOR):
    return -np.log(np.maximum(p_true, floor))


# ============================================================
# Global CRISP-R4 B4 fitting (all N_CLUSTERS clusters, rank=8, with fallback)
# ============================================================
def build_fit_tune_split(ct_routed, seed):
    local_rng = np.random.default_rng(seed)
    idx = ct_routed.copy()
    local_rng.shuffle(idx)
    n_fit = min(MAX_FIT, max(1, len(idx) * 3 // 4))
    fit_idx = idx[:n_fit]
    tune_idx = idx[n_fit:n_fit + MAX_TUNE]
    return fit_idx, tune_idx


def fit_global_crisp_cluster(c, members, ds_keys, ds_values, ct_keys, idx_dime_ct_top1,
                               dist_matrix, dime_centroids, rank, lambda_grid):
    population = len(members)
    if population == 0:
        # Genuinely empty cluster -- no histogram can be formed at all. This
        # is distinct from "too small to fit a correction" (any population
        # >= 1 can still form a valid, if degenerate, histogram fallback --
        # caught during self-review: MIN_FIT_POP was originally used as the
        # gate for THIS branch, which wrongly gave CRISP zero representation
        # for any cluster below it instead of the required histogram
        # fallback, while the exact oracle (S3/S5) still used its real
        # histogram -- an unfair, bug-driven asymmetry, not a real limitation).
        return {"type": "empty_or_tiny", "population": 0, "n_tokens": 0}
    member_keys, member_values = ds_keys[members], ds_values[members]
    token_ids = np.array(sorted(set(member_values.tolist())))
    p_j = np.array([float(np.sum(member_values == y)) / population for y in token_ids])
    log_p = np.log(np.maximum(p_j, 1e-300))
    base = {"population": population, "n_tokens": len(token_ids), "p_j": p_j, "log_p": log_p, "token_ids": token_ids}

    if population < MIN_FIT_POP:
        return {**base, "type": "fallback_histogram", "n_ct_routed": None}

    ct_routed = np.where(idx_dime_ct_top1 == c)[0]
    if len(ct_routed) < (MIN_FIT_QUERIES + MIN_TUNE_QUERIES):
        return {**base, "type": "fallback_histogram", "n_ct_routed": int(len(ct_routed))}

    fit_idx, tune_idx = build_fit_tune_split(ct_routed, SEED + c)
    if len(fit_idx) < MIN_FIT_QUERIES or len(tune_idx) < MIN_TUNE_QUERIES:
        return {**base, "type": "fallback_histogram", "n_ct_routed": int(len(ct_routed))}

    fit_keys, tune_keys = ct_keys[fit_idx], ct_keys[tune_idx]
    _, fit_ev = exact_evidence(fit_keys, member_keys, member_values, TAU)
    _, tune_ev = exact_evidence(tune_keys, member_keys, member_values, TAU)
    exact_fit_dist, _ = normalize_rows(fit_ev)
    exact_tune_dist, _ = normalize_rows(tune_ev)
    z_fit = log_ratio_target(exact_fit_dist, p_j)
    z_tune = log_ratio_target(exact_tune_dist, p_j)
    anchor_idx = select_anchors_nearest(dist_matrix, c, rank)
    Phi_fit = linear_anchor_diff_basis(fit_keys, c, anchor_idx, dime_centroids)
    Phi_tune = linear_anchor_diff_basis(tune_keys, c, anchor_idx, dime_centroids)

    best_lam, best_tune_tv, best_theta = None, np.inf, None
    for lam in lambda_grid:
        theta = fit_ridge(Phi_fit, z_fit, lam)
        d_tune, _ = reconstruct_logit(Phi_tune, theta, log_p)
        tv = mean_tv(d_tune, exact_tune_dist)
        if tv < best_tune_tv:
            best_tune_tv, best_lam, best_theta = tv, lam, theta
    return {**base, "type": "fitted", "anchor_idx": anchor_idx, "theta": best_theta,
             "chosen_lambda": best_lam, "n_fit": int(len(fit_idx)), "n_tune": int(len(tune_idx))}


def crisp_predict(model, q_key, c, dime_centroids):
    if model["type"] == "empty_or_tiny":
        return {}
    token_ids, p_j = model["token_ids"], model["p_j"]
    if model["type"] == "fallback_histogram":
        return {int(t): float(p) for t, p in zip(token_ids, p_j) if p > 0}
    Phi = linear_anchor_diff_basis(q_key[None, :], c, model["anchor_idx"], dime_centroids)
    d, _ = reconstruct_logit(Phi, model["theta"], model["log_p"])
    return {int(t): float(p) for t, p in zip(token_ids, d[0]) if p > 1e-12}


def main():
    t_start = time.perf_counter()
    print(f"CRISP-R5 job started. SMOKE={SMOKE}, N_CLUSTERS={N_CLUSTERS}, N_EVAL={N_EVAL}")
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
    compressed_dists = np.array(compressed_dists, dtype=object)
    cluster_to_members = build_cluster_to_members(assignment, N_CLUSTERS)
    dist_matrix = build_centroid_distance_matrix(dime_centroids.astype(np.float32))

    raw_index, raw_values = build_datastore(ds_keys, ds_values)
    dime_index, dime_values = build_datastore(dime_centroids, compressed_dists)

    report = {"n_ds": int(len(ds_values)), "n_ct": int(len(ct_values)), "n_val": int(len(val_values)),
              "n_clusters": N_CLUSTERS, "seed": SEED, "smoke": SMOKE, "outer_k": OUTER_K, "tau": TAU,
              "crisp_rank": CRISP_RANK, "n_eval_target": N_EVAL,
              "reproduction_note": "R1-R4's real persisted results were not available locally (checked directly); "
                                     "this script reproduces R4's B4 methodology fresh via direct import, and reuses "
                                     "E1B.3's real, present System-3 pattern for the S3 oracle. See module docstring."}
    save_results(os.path.join(OUT_DIR, "crisp_r5_summary.json"), report)

    # ============================================================
    # Phase 1 -- reproduce baselines at FULL val scale (cheap: batched kNN only)
    # ============================================================
    print(f"\nPhase 1: full-scale ({len(val_values)} queries) raw/DIME baseline reproduction...")
    d_raw_val, idx_raw_val = query_knn_indices(raw_index, val_keys, k=OUTER_K)
    d_dime_val, idx_dime_val = query_knn_indices(dime_index, val_keys, k=OUTER_K)
    d_dime_ct, idx_dime_ct = query_knn_indices(dime_index, ct_keys, k=OUTER_K)
    retrieved_raw_val = raw_values[idx_raw_val]
    retrieved_dime_val = dime_values[idx_dime_val]

    p_raw_full = raw_standalone_prob(d_raw_val, retrieved_raw_val, val_values, tau=TAU)
    p_dime_full = dime_standalone_prob(d_dime_val, retrieved_dime_val, val_values, tau=TAU)
    raw_dists_full = raw_full_distribution(d_raw_val, retrieved_raw_val, tau=TAU)
    dime_dists_full = dime_full_distribution(d_dime_val, retrieved_dime_val, tau=TAU)
    js_full = np.array([js_divergence_dict(raw_dists_full[i], dime_dists_full[i]) for i in range(len(val_values))])

    tier1_path = os.path.join(SCRIPT_DIR, "results", "wikitext103_gpt2_tier1.json")
    tier1_note = None
    if os.path.exists(tier1_path):
        with open(tier1_path) as f:
            tier1 = json.load(f)
        tier1_note = {"gpt_only_mean_nll": tier1.get("gpt_only_mean_nll"),
                       "raw_knn_tuned": tier1.get("raw_knn", {}).get("best_config"),
                       "dime_tuned": tier1.get("dime_minibatch_kmeans", {}).get("best_config")}
        print(f"  tier1.json found (real, secondary/tuned-config reference, NOT re-run here): {tier1_note}")
    else:
        print("  tier1.json not found locally -- tuned-config reference unavailable, noted as a limitation.")
    report["phase1_baseline"] = {
        "matched_raw_mean_p_true": float(p_raw_full.mean()), "matched_dime_mean_p_true": float(p_dime_full.mean()),
        "matched_raw_coverage": retrieval_coverage(p_raw_full), "matched_dime_coverage": retrieval_coverage(p_dime_full),
        "mean_js_raw_vs_dime_full_val": float(js_full.mean()), "tier1_tuned_reference": tier1_note,
    }
    save_results(os.path.join(OUT_DIR, "crisp_r5_summary.json"), report)

    # ============================================================
    # Routing-overlap diagnostic at FULL val scale (E1B.3's exact Experiment-A pattern)
    # ============================================================
    print("\nRouting-overlap diagnostic (full val scale, cheap -- assignment lookups only)...")
    raw_neighbor_clusters_full = assignment[idx_raw_val]
    w_raw_full_val = softmax_np(-d_raw_val / TAU, axis=1)
    frac_neighbor_overlap_full = np.zeros(len(val_values))
    frac_mass_overlap_full = np.zeros(len(val_values))
    for i in range(len(val_values)):
        in_dime = np.isin(raw_neighbor_clusters_full[i], idx_dime_val[i])
        frac_neighbor_overlap_full[i] = in_dime.mean()
        frac_mass_overlap_full[i] = w_raw_full_val[i][in_dime].sum()
    report["routing_overlap_full_scale"] = {
        "mean_frac_neighbor_overlap": float(frac_neighbor_overlap_full.mean()),
        "mean_frac_mass_overlap": float(frac_mass_overlap_full.mean()),
        "frac_queries_with_full_top20_recall": float((frac_neighbor_overlap_full >= 1.0 - 1e-9).mean()),
    }
    print(f"  {report['routing_overlap_full_scale']}")
    save_results(os.path.join(OUT_DIR, "crisp_r5_summary.json"), report)

    # ============================================================
    # Phase 3 -- fit CRISP-R4 B4 globally, all N_CLUSTERS clusters, rank=8
    # ============================================================
    print(f"\nPhase 3: fitting CRISP-R4 B4 (rank={CRISP_RANK}) for all {N_CLUSTERS} clusters...")
    idx_dime_ct_top1 = idx_dime_ct[:, 0]
    models = {}
    n_fitted, n_fallback, n_empty = 0, 0, 0
    t_fit_start = time.perf_counter()
    for c in range(N_CLUSTERS):
        m = fit_global_crisp_cluster(c, cluster_to_members[c], ds_keys, ds_values, ct_keys, idx_dime_ct_top1,
                                        dist_matrix, dime_centroids, CRISP_RANK, RIDGE_LAMBDA_GRID)
        models[c] = m
        if m["type"] == "fitted":
            n_fitted += 1
        elif m["type"] == "fallback_histogram":
            n_fallback += 1
        else:
            n_empty += 1
        if (c + 1) % max(1, N_CLUSTERS // 10) == 0:
            print(f"  fitted {c + 1}/{N_CLUSTERS} clusters (fitted={n_fitted}, fallback={n_fallback}, empty/tiny={n_empty})")
    t_fit = time.perf_counter() - t_fit_start
    print(f"  Global fitting done in {t_fit:.1f}s. fitted={n_fitted} ({100*n_fitted/N_CLUSTERS:.1f}%), "
          f"fallback={n_fallback} ({100*n_fallback/N_CLUSTERS:.1f}%), empty/tiny={n_empty} ({100*n_empty/N_CLUSTERS:.1f}%)")

    # Fitting coverage by held-out query-ROUTING FREQUENCY (full val scale, cheap lookup)
    is_fitted = np.array([models[c]["type"] == "fitted" for c in range(N_CLUSTERS)])
    routed_fitted_frac = float(is_fitted[idx_dime_val].mean())
    routed_fallback_or_empty_frac = 1.0 - routed_fitted_frac
    report["crisp_fitting_coverage"] = {
        "by_cluster_count": {"n_fitted": n_fitted, "n_fallback": n_fallback, "n_empty_or_tiny": n_empty,
                               "frac_fitted": n_fitted / N_CLUSTERS},
        "by_held_out_query_routing_frequency_full_val": {
            "frac_selected_centroid_slots_fitted": routed_fitted_frac,
            "frac_selected_centroid_slots_fallback_or_empty": routed_fallback_or_empty_frac,
            "note": "fraction of (query, selected-centroid) pairs across ALL 77,089 val queries x top-20 "
                     "selections that land on a FITTED cluster -- this, not the raw cluster count, is what "
                     "matters for end-to-end fidelity."},
        "global_fit_wall_clock_seconds": t_fit,
    }
    print(f"  Routing-frequency-weighted coverage: {report['crisp_fitting_coverage']['by_held_out_query_routing_frequency_full_val']}")
    save_results(os.path.join(OUT_DIR, "crisp_r5_summary.json"), report)

    # ============================================================
    # Phase 2 -- small-scale sanity checks of S2/S3 definitions (on real data)
    # ============================================================
    print("\nPhase 2: sanity-checking S2/S3 definitions on a small real-data probe...")
    probe_n = min(50, len(val_values))
    probe_idx = np.arange(probe_n)
    s2_equals_s0_when_full_recall = []
    s3_sums_to_one = []
    w_dime_probe = softmax_np(-d_dime_val[probe_idx] / TAU, axis=1)
    for pi, qi in enumerate(probe_idx):
        q_key = val_keys[qi]
        selected = idx_dime_val[qi]
        if frac_neighbor_overlap_full[qi] >= 1.0 - 1e-9:
            cand_idx = np.concatenate([cluster_to_members[c] for c in selected if len(cluster_to_members[c]) > 0])
            d_cand = np.linalg.norm(ds_keys[cand_idx] - q_key[None, :], axis=1)
            d_s2 = np.sort(d_cand)[:OUTER_K]
            d_s0 = np.sort(d_raw_val[qi])
            s2_equals_s0_when_full_recall.append(bool(np.allclose(d_s2, d_s0, atol=1e-6)))
        s3_total = 0.0
        for j, c in enumerate(selected):
            members_j = cluster_to_members[c]
            if len(members_j) == 0:
                continue
            d_j = cluster_distances(q_key, ds_keys[members_j])
            dist_j, _ = exact_cluster_dist_and_mass(d_j, ds_values[members_j])
            s3_total += float(w_dime_probe[pi, j]) * sum(dist_j.values())
        s3_sums_to_one.append(s3_total)
    frac_s3_valid = float(np.mean(np.isclose(s3_sums_to_one, 1.0, atol=1e-6))) if s3_sums_to_one else None
    print(f"  S3's reconstructed distribution sums to 1 (within 1e-6) on {sum(np.isclose(s3_sums_to_one, 1.0, atol=1e-6)) if s3_sums_to_one else 0}/{len(s3_sums_to_one)} probe queries")
    frac_s2_matches_s0 = float(np.mean(s2_equals_s0_when_full_recall)) if s2_equals_s0_when_full_recall else None
    print(f"  S2's candidate-restricted top-20 distances match S0's global top-20 distances exactly on "
          f"{sum(s2_equals_s0_when_full_recall) if s2_equals_s0_when_full_recall else 0}/"
          f"{len(s2_equals_s0_when_full_recall)} full-recall probe queries (expected: all, per the spec's own "
          f"'verify S2 equals original kNN whenever its candidate set contains every original top-k neighbor' check)")
    report["phase2_sanity_checks"] = {"n_probed": probe_n, "n_full_recall_in_probe": len(s2_equals_s0_when_full_recall),
                                         "frac_s2_matches_s0_on_full_recall_subset": frac_s2_matches_s0,
                                         "frac_s3_sums_to_one": frac_s3_valid}
    save_results(os.path.join(OUT_DIR, "crisp_r5_summary.json"), report)

    # ============================================================
    # Query sample for the expensive systems (E1B.3's exact JS-decile stratification)
    # ============================================================
    print(f"\nSelecting the N_EVAL={N_EVAL} stratified evaluation sample (JS deciles, reusing E1B.3's method)...")
    n_eval_actual = min(N_EVAL, len(val_values))
    js_deciles = np.digitize(js_full, np.quantile(js_full, np.linspace(0, 1, 11))[1:-1])
    per_decile = n_eval_actual // 10
    sample_idx = []
    for dcl in range(10):
        pool = np.where(js_deciles == dcl)[0]
        take = min(per_decile, len(pool))
        sample_idx.append(rng.choice(pool, size=take, replace=False))
    sample_idx = np.concatenate(sample_idx)
    n_eval_actual = len(sample_idx)
    print(f"  evaluation sample: {n_eval_actual} queries ({per_decile} target per JS decile)")
    report["evaluation_sample"] = {"n_eval": n_eval_actual, "selection_rule": "stratified across JS(raw,DIME) "
                                       "deciles at matched config, same method as E1B.3's Experiments B/C"}
    save_results(os.path.join(OUT_DIR, "crisp_r5_summary.json"), report)

    # ============================================================
    # Phase 4 -- main per-query evaluation loop: S2-S6 + all experiments
    # ============================================================
    print(f"\nPhase 4: main evaluation loop over {n_eval_actual} queries (S2-S6 + diagnostics)...")
    per_query_dists = {s: [] for s in SYSTEMS}
    per_query_p_true = {s: [] for s in SYSTEMS}
    routing_rows = []
    error_attr_rows = []
    timing = {s: 0.0 for s in SYSTEMS}
    w_dime_full_val = softmax_np(-d_dime_val / TAU, axis=1)

    for rank_i, qi in enumerate(sample_idx):
        q_key = val_keys[qi]
        true_tok = int(val_values[qi])
        selected = idx_dime_val[qi]
        w_dime_q = w_dime_full_val[qi]

        t0 = time.perf_counter()
        d_s0 = raw_dists_full[qi]
        timing["S0"] += time.perf_counter() - t0
        t0 = time.perf_counter()
        d_s1 = dime_dists_full[qi]
        timing["S1"] += time.perf_counter() - t0

        all_d, all_v = [], []
        s3_accum, s4_accum, s5_accum, s6_accum = {}, {}, {}, {}
        mass_sum = 0.0
        t0 = time.perf_counter()
        for j, c in enumerate(selected):
            members = cluster_to_members[c]
            if len(members) == 0:
                continue
            member_keys, member_values = ds_keys[members], ds_values[members]
            d = cluster_distances(q_key, member_keys)
            all_d.append(d); all_v.append(member_values)
            exact_dist_j, mass_j = exact_cluster_dist_and_mass(d, member_values)
            wj = float(w_dime_q[j])
            for t, p in exact_dist_j.items():
                s3_accum[t] = s3_accum.get(t, 0.0) + wj * p
            crisp_dist_j = crisp_predict(models[c], q_key, c, dime_centroids)
            for t, p in crisp_dist_j.items():
                s4_accum[t] = s4_accum.get(t, 0.0) + wj * p
            if mass_j > 0:
                mass_sum += mass_j
                for t, p in exact_dist_j.items():
                    s5_accum[t] = s5_accum.get(t, 0.0) + mass_j * p
                for t, p in crisp_dist_j.items():
                    s6_accum[t] = s6_accum.get(t, 0.0) + mass_j * p
        timing["S3"] += time.perf_counter() - t0
        d_s3 = s3_accum
        t0 = time.perf_counter()
        d_s4 = s4_accum
        timing["S4"] += time.perf_counter() - t0
        t0 = time.perf_counter()
        d_s5 = {t: p / mass_sum for t, p in s5_accum.items()} if mass_sum > 0 else {}
        timing["S5"] += time.perf_counter() - t0
        t0 = time.perf_counter()
        d_s6 = {t: p / mass_sum for t, p in s6_accum.items()} if mass_sum > 0 else {}
        timing["S6"] += time.perf_counter() - t0

        t0 = time.perf_counter()
        if all_d:
            cand_d = np.concatenate(all_d)
            cand_v = np.concatenate(all_v)
            topk_local = np.argsort(cand_d)[:OUTER_K]
            d_topk, v_topk = cand_d[topk_local], cand_v[topk_local]
            d_s2 = raw_full_distribution(d_topk[None, :], v_topk[None, :], tau=TAU)[0]
            p_s2 = raw_standalone_prob(d_topk[None, :], v_topk[None, :], np.array([true_tok]), tau=TAU)[0]
            candidate_size = int(len(cand_d))
        else:
            d_s2, p_s2, candidate_size = {}, 0.0, 0
        timing["S2"] += time.perf_counter() - t0

        dists = {"S0": d_s0, "S1": d_s1, "S2": d_s2, "S3": d_s3, "S4": d_s4, "S5": d_s5, "S6": d_s6}
        for s, d in dists.items():
            per_query_dists[s].append(d)
            per_query_p_true[s].append(d.get(true_tok, 0.0))

        routing_rows.append({"query_idx": int(qi), "n_selected_centroids": int(len(selected)),
                               "candidate_set_size": candidate_size,
                               "raw_top20_neighbor_recall_in_candidate_set": float(frac_neighbor_overlap_full[qi]),
                               "raw_top20_mass_recall_in_candidate_set": float(frac_mass_overlap_full[qi]),
                               "s0_vs_s2_tv": total_variation(d_s0, d_s2)})

        error_attr_rows.append({
            "query_idx": int(qi),
            "tv_S0_S1": total_variation(d_s0, d_s1), "tv_S0_S2": total_variation(d_s0, d_s2),
            "tv_S0_S3": total_variation(d_s0, d_s3), "tv_S0_S4": total_variation(d_s0, d_s4),
            "tv_S0_S5": total_variation(d_s0, d_s5), "tv_S2_S3": total_variation(d_s2, d_s3),
            "tv_S3_S4": total_variation(d_s3, d_s4), "tv_S3_S5": total_variation(d_s3, d_s5),
            "tv_S5_S6": total_variation(d_s5, d_s6),
        })
        if (rank_i + 1) % max(1, n_eval_actual // 10) == 0:
            print(f"  evaluated {rank_i + 1}/{n_eval_actual} queries")
            save_results(os.path.join(OUT_DIR, "crisp_r5_summary.json"), report)

    report["error_attribution_schema"] = {
        "tv_S0_S1": "DIME's total global error vs. raw kNN.",
        "tv_S0_S2": "Routing error -- how much centroid-candidate-set restriction alone costs.",
        "tv_S0_S3": "Routing + aggregation error with exact local functions (no compression yet).",
        "tv_S0_S4": "CRISP's total global error vs. raw kNN.",
        "tv_S0_S5": "Error of the mass-aware exact-oracle variant vs. raw kNN.",
        "tv_S2_S3": "Aggregation gap -- candidate-restricted raw retrieval vs. exact query-conditioned cluster aggregation.",
        "tv_S3_S4": "Compression approximation gap -- CRISP's local approximation error, isolated from routing/aggregation.",
        "tv_S3_S5": "Effect of centroid-softmax weights vs. real evidence masses, holding exact local functions fixed.",
        "tv_S5_S6": "Effect of replacing exact local functions with CRISP's, holding real evidence masses fixed.",
        "note": "These are PAIRWISE errors, not an additive decomposition -- TV obeys the triangle inequality but "
                 "components can cancel or interact; no claim of tv_S0_S2 + tv_S2_S3 + tv_S3_S4 == tv_S0_S4 is made.",
    }
    save_results(os.path.join(OUT_DIR, "crisp_r5_summary.json"), report)

    # ============================================================
    # Experiment A -- global retrieval fidelity
    # ============================================================
    print("\nExperiment A: global retrieval fidelity (TV/JS/top-k/entropy/support vs. S0)...")
    true_toks_sample = val_values[sample_idx]
    expA = {}
    for s in SYSTEMS:
        d_s0_list = per_query_dists["S0"]
        d_s_list = per_query_dists[s]
        tvs = np.array([total_variation(d_s_list[i], d_s0_list[i]) for i in range(n_eval_actual)])
        jss = np.array([js_divergence_dict(d_s_list[i], d_s0_list[i]) for i in range(n_eval_actual)])
        top1 = np.array([top1_agreement(d_s_list[i], d_s0_list[i]) for i in range(n_eval_actual)], dtype=object)
        top1_valid = np.array([x for x in top1 if x is not None], dtype=float)
        topk_overlaps = {k: np.array([x for x in (topk_overlap(d_s_list[i], d_s0_list[i], k=k) for i in range(n_eval_actual)) if x is not None]) for k in TOPK_LIST}
        ent_diff = np.array([entropy_of_dict(d_s_list[i]) - entropy_of_dict(d_s0_list[i]) for i in range(n_eval_actual)])
        support_rec = np.array([x for x in (support_mass_overlap(d_s_list[i], d_s0_list[i]) for i in range(n_eval_actual)) if x is not None])
        p_true = np.array(per_query_p_true[s])
        expA[s] = {
            "mean_tv": float(tvs.mean()), "median_tv": float(np.median(tvs)),
            "p90_tv": float(np.percentile(tvs, 90)), "p99_tv": float(np.percentile(tvs, 99)),
            "mean_js": float(jss.mean()), "median_js": float(np.median(jss)),
            "top1_agreement_with_S0": float(top1_valid.mean()) if len(top1_valid) else None,
            **{f"top{k}_overlap_with_S0": float(topk_overlaps[k].mean()) if len(topk_overlaps[k]) else None for k in TOPK_LIST},
            "mean_entropy_diff_vs_S0": float(ent_diff.mean()),
            "mean_s0_support_mass_recovered": float(support_rec.mean()) if len(support_rec) else None,
            "mean_p_true": float(p_true.mean()), "median_p_true": float(np.median(p_true)),
            "coverage": retrieval_coverage(p_true),
        }
    print(f"  S1 vs S0: {expA['S1']}")
    print(f"  S4 vs S0: {expA['S4']}")
    report["experiment_A_global_fidelity"] = expA
    save_results(os.path.join(OUT_DIR, "crisp_r5_summary.json"), report)

    # ============================================================
    # Experiment B -- error attribution (aggregate pairwise errors)
    # ============================================================
    print("\nExperiment B: error attribution (aggregate pairwise TV, no additive decomposition claimed)...")
    attr_keys = [k for k in error_attr_rows[0].keys() if k != "query_idx"]
    mean_pairwise_errors = {k: float(np.mean([r[k] for r in error_attr_rows])) for k in attr_keys}
    attr_arrays = {k: np.array([r[k] for r in error_attr_rows]) for k in attr_keys}
    corr_with_final = {}
    for k in ["tv_S0_S2", "tv_S2_S3", "tv_S3_S4"]:
        if np.std(attr_arrays[k]) > 0 and np.std(attr_arrays["tv_S0_S4"]) > 0:
            corr_with_final[k] = float(np.corrcoef(attr_arrays[k], attr_arrays["tv_S0_S4"])[0, 1])
        else:
            corr_with_final[k] = None
    print(f"  mean pairwise errors: {mean_pairwise_errors}")
    print(f"  correlation of each component with final tv_S0_S4: {corr_with_final}")
    expB = {"mean_pairwise_errors": mean_pairwise_errors, "component_correlation_with_final_error": corr_with_final}
    report["experiment_B_error_attribution"] = expB
    save_results(os.path.join(OUT_DIR, "crisp_r5_summary.json"), report)

    # ============================================================
    # Experiment C -- relative improvement R_D (CRISP vs. DIME) + oracle opportunity (S3 vs. DIME)
    # ============================================================
    print("\nExperiment C: relative improvement R_D (CRISP vs. DIME) and oracle opportunity (S3 vs. DIME)...")
    def relative_improvement(metric_name):
        d_dime = expA["S1"][metric_name]
        d_crisp = expA["S4"][metric_name]
        d_s3 = expA["S3"][metric_name]
        r_crisp = float((d_dime - d_crisp) / d_dime) if abs(d_dime) > 1e-9 else None
        r_oracle = float((d_dime - d_s3) / d_dime) if abs(d_dime) > 1e-9 else None
        return {"dime": d_dime, "crisp": d_crisp, "s3_oracle": d_s3,
                 "R_D_crisp_vs_dime": r_crisp, "R_D_oracle_s3_vs_dime": r_oracle}
    expC = {"TV": relative_improvement("mean_tv"), "JS": relative_improvement("mean_js")}
    print(f"  {expC}")
    report["experiment_C_relative_improvement"] = expC
    save_results(os.path.join(OUT_DIR, "crisp_r5_summary.json"), report)

    # ============================================================
    # Experiment D -- correct-token behavior
    # ============================================================
    print("\nExperiment D: correct-token behavior (coverage, probability, retrieval-only NLL with documented floor)...")
    expD = {}
    for s in SYSTEMS:
        p_true = np.array(per_query_p_true[s])
        nll = retrieval_only_nll(p_true)
        rel_improve = None
        if s != "S1":
            p_dime = np.array(per_query_p_true["S1"])
            denom = p_dime.mean()
            rel_improve = float((p_true.mean() - denom) / denom) if abs(denom) > 1e-9 else None
        expD[s] = {"coverage": retrieval_coverage(p_true), "mean_p_true": float(p_true.mean()),
                     "median_p_true": float(np.median(p_true)), "mean_retrieval_only_nll_with_floor": float(nll.mean()),
                     "epsilon_floor": EPSILON_FLOOR, "relative_improvement_in_p_true_vs_dime": rel_improve}
    print(f"  {expD}")
    report["experiment_D_correct_token"] = expD
    save_results(os.path.join(OUT_DIR, "crisp_r5_summary.json"), report)

    # ============================================================
    # Experiment F -- full storage accounting (all N_CLUSTERS clusters, real bytes)
    # ============================================================
    print("\nExperiment F: full storage accounting (measured, all clusters, not extrapolated)...")
    centroid_bytes = N_CLUSTERS * DIM * DTYPE_BYTES
    hist_bytes_total = sum(models[c]["n_tokens"] * (4 + 4) for c in range(N_CLUSTERS))
    crisp_coef_bytes_total = sum(models[c]["n_tokens"] * CRISP_RANK * DTYPE_BYTES for c in range(N_CLUSTERS) if models[c]["type"] == "fitted")
    crisp_anchor_bytes_total = sum(CRISP_RANK * 4 for c in range(N_CLUSTERS) if models[c]["type"] == "fitted")
    raw_bytes_total = len(ds_values) * (DIM * DTYPE_BYTES + VALUE_BYTES)
    dime_v1_total = centroid_bytes + hist_bytes_total
    crisp_total = centroid_bytes + hist_bytes_total + crisp_coef_bytes_total + crisp_anchor_bytes_total
    storage = {
        "raw_kNN_datastore_bytes": int(raw_bytes_total),
        "dime_v1_bytes": {"centroid_bytes": int(centroid_bytes), "histogram_bytes": int(hist_bytes_total), "total": int(dime_v1_total)},
        "crisp_r5_rank8_bytes": {"centroid_bytes": int(centroid_bytes), "histogram_bytes": int(hist_bytes_total),
                                    "coefficient_bytes": int(crisp_coef_bytes_total), "anchor_bytes": int(crisp_anchor_bytes_total),
                                    "total": int(crisp_total)},
        "s5_s6_storage": "N/A -- diagnostic-only, require raw vectors at inference time, not deployable compressed systems.",
        "compression_ratio_dime_v1": float(raw_bytes_total / dime_v1_total) if dime_v1_total else None,
        "compression_ratio_crisp": float(raw_bytes_total / crisp_total) if crisp_total else None,
        "crisp_expansion_over_dime_v1": float(crisp_total / dime_v1_total) if dime_v1_total else None,
    }
    print(f"  {storage}")
    report["experiment_F_storage"] = storage
    save_results(os.path.join(OUT_DIR, "crisp_r5_summary.json"), report)

    with open(os.path.join(OUT_DIR, "storage_accounting.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["component", "bytes", "method"])
        w.writerow(["raw_kNN_datastore", raw_bytes_total, "raw"])
        w.writerow(["centroids", centroid_bytes, "dime_v1_and_crisp_shared"])
        w.writerow(["histograms", hist_bytes_total, "dime_v1_and_crisp_shared"])
        w.writerow(["crisp_coefficients", crisp_coef_bytes_total, "crisp"])
        w.writerow(["crisp_anchor_indices", crisp_anchor_bytes_total, "crisp"])
        w.writerow(["dime_v1_total", dime_v1_total, "dime_v1"])
        w.writerow(["crisp_total", crisp_total, "crisp"])

    # ============================================================
    # Experiment G -- light latency capture (in-loop, not an isolated microbenchmark)
    # ============================================================
    expG = {s: {"total_seconds": timing[s], "mean_seconds_per_query": timing[s] / n_eval_actual} for s in SYSTEMS}
    print(f"\nExperiment G (light, in-loop timing -- NOT an isolated deployment benchmark): {expG}")
    report["experiment_G_latency_light"] = expG
    save_results(os.path.join(OUT_DIR, "crisp_r5_summary.json"), report)

    # ============================================================
    # Deliverables: global_results.csv, routing_diagnostics.csv, error_attribution.csv
    # ============================================================
    print("\nWriting global_results.csv, routing_diagnostics.csv, error_attribution.csv...")
    with open(os.path.join(OUT_DIR, "global_results.csv"), "w", newline="") as f:
        fieldnames = ["system", "n_queries", "mean_tv", "median_tv", "mean_js", "median_js", "top1_agreement",
                       "top5_overlap", "top10_overlap", "coverage", "mean_p_true", "median_p_true",
                       "mean_retrieval_only_nll", "total_bytes", "mean_latency_seconds"]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        bytes_by_system = {"S0": raw_bytes_total, "S1": dime_v1_total, "S2": raw_bytes_total, "S3": raw_bytes_total,
                              "S4": crisp_total, "S5": raw_bytes_total, "S6": raw_bytes_total}
        for s in SYSTEMS:
            w.writerow({"system": s, "n_queries": n_eval_actual, "mean_tv": expA[s]["mean_tv"], "median_tv": expA[s]["median_tv"],
                         "mean_js": expA[s]["mean_js"], "median_js": expA[s]["median_js"], "top1_agreement": expA[s]["top1_agreement_with_S0"],
                         "top5_overlap": expA[s]["top5_overlap_with_S0"], "top10_overlap": expA[s]["top10_overlap_with_S0"],
                         "coverage": expD[s]["coverage"], "mean_p_true": expD[s]["mean_p_true"], "median_p_true": expD[s]["median_p_true"],
                         "mean_retrieval_only_nll": expD[s]["mean_retrieval_only_nll_with_floor"], "total_bytes": bytes_by_system[s],
                         "mean_latency_seconds": expG[s]["mean_seconds_per_query"]})

    with open(os.path.join(OUT_DIR, "routing_diagnostics.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(routing_rows[0].keys()))
        w.writeheader()
        for row in routing_rows:
            w.writerow(row)

    with open(os.path.join(OUT_DIR, "error_attribution.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(error_attr_rows[0].keys()))
        w.writeheader()
        for row in error_attr_rows:
            w.writerow(row)

    # ============================================================
    # Deliverable: query_level_results.csv (bounded top-10 sparse sample)
    # ============================================================
    print("Writing query_level_results.csv (bounded sparse sample)...")
    with open(os.path.join(OUT_DIR, "query_level_results.csv"), "w", newline="") as f:
        fieldnames = ["query_idx", "system", "rank_in_top10", "token_id", "prob"]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        n_q_detail = min(200, n_eval_actual)
        for i in range(n_q_detail):
            qi = int(sample_idx[i])
            for s in SYSTEMS:
                d = per_query_dists[s][i]
                top = sorted(d.items(), key=lambda kv: -kv[1])[:10]
                for rank_pos, (tok, p) in enumerate(top):
                    w.writerow({"query_idx": qi, "system": s, "rank_in_top10": rank_pos, "token_id": tok, "prob": round(p, 6)})

    # ============================================================
    # Experiment H -- failure analysis
    # ============================================================
    print("\nExperiment H: failure analysis...")
    tv_s0_s1 = attr_arrays["tv_S0_S1"]
    tv_s0_s4 = attr_arrays["tv_S0_S4"]
    tv_s0_s3 = attr_arrays["tv_S0_S3"]
    tv_s3_s4 = attr_arrays["tv_S3_S4"]
    tv_s3_s5 = attr_arrays["tv_S3_S5"]
    crisp_minus_dime = tv_s0_s4 - tv_s0_s1
    categories = {
        "1_crisp_substantially_improves_over_dime": np.argsort(crisp_minus_dime)[:N_FAILURE_PER_CATEGORY],
        "2_crisp_worse_than_dime": np.argsort(-crisp_minus_dime)[:N_FAILURE_PER_CATEGORY],
        "3_both_differ_substantially_from_raw": np.argsort(-(tv_s0_s1 + tv_s0_s4))[:N_FAILURE_PER_CATEGORY],
        # Smallest (most negative) tv_S0_S3 - tv_S3_S4 = S3 close to raw (small tv_S0_S3)
        # AND CRISP far from S3 (large tv_S3_S4) -- ascending argsort already puts these
        # first; a trailing [::-1] here would select the OPPOSITE condition (caught via
        # self-review before running).
        "4_s3_accurate_but_crisp_inaccurate": np.argsort(tv_s0_s3 - tv_s3_s4)[:N_FAILURE_PER_CATEGORY],
        "5_s2_inaccurate_routing_misses_neighbors": np.argsort(-np.array([r["s0_vs_s2_tv"] for r in routing_rows]))[:N_FAILURE_PER_CATEGORY],
        "6_s3_s5_differ_due_to_mass": np.argsort(-tv_s3_s5)[:N_FAILURE_PER_CATEGORY],
    }
    failure_lines = ["# CRISP-R5 Failure Analysis", "",
                       "All examples use the SAME query identifiers across systems. Selected by the stated "
                       "ranking rule per category, not cherry-picked.", ""]
    for cat_name, local_idxs in categories.items():
        failure_lines.append(f"## Category {cat_name}")
        for li in local_idxs:
            qi = int(sample_idx[li])
            true_tok = int(val_values[qi])
            decoded_true = gpt.tokenizer.decode([true_tok])
            row = {
                "query_idx": qi, "true_token": decoded_true,
                "selected_centroids": idx_dime_val[qi].tolist(),
                "candidate_set_size": routing_rows[li]["candidate_set_size"],
                "raw_top20_recall_in_candidates": routing_rows[li]["raw_top20_neighbor_recall_in_candidate_set"],
                "p_true_S0": round(per_query_p_true["S0"][li], 5), "p_true_S1": round(per_query_p_true["S1"][li], 5),
                "p_true_S2": round(per_query_p_true["S2"][li], 5), "p_true_S3": round(per_query_p_true["S3"][li], 5),
                "p_true_S4": round(per_query_p_true["S4"][li], 5),
                "tv_S0_S1": round(float(tv_s0_s1[li]), 5), "tv_S0_S4": round(float(tv_s0_s4[li]), 5),
                "tv_S0_S2": round(float(attr_arrays['tv_S0_S2'][li]), 5), "tv_S3_S4": round(float(tv_s3_s4[li]), 5),
            }
            failure_lines.append(f"- {row}")
        failure_lines.append("")
    with open(os.path.join(OUT_DIR, "failure_cases.md"), "w") as f:
        f.write("\n".join(failure_lines))

    # ============================================================
    # Plots
    # ============================================================
    print("\nGenerating plots...")
    fig, ax = plt.subplots(figsize=(8, 5))
    names = SYSTEMS
    tvs_plot = [expA[s]["mean_tv"] for s in names]
    ax.bar(names, tvs_plot)
    ax.set_ylabel("mean TV vs. S0 (raw kNN)")
    ax.set_title("Experiment A: global TV by system")
    fig.tight_layout(); fig.savefig(os.path.join(PLOT_DIR, "A_tv_by_system.png"), dpi=130); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(names, [expD[s]["coverage"] for s in names])
    ax.set_ylabel("correct-token coverage")
    ax.set_title("Experiment D: correct-token coverage by system")
    fig.tight_layout(); fig.savefig(os.path.join(PLOT_DIR, "D_coverage_by_system.png"), dpi=130); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    keys_b = ["tv_S0_S2", "tv_S2_S3", "tv_S3_S4", "tv_S0_S4"]
    ax.bar(keys_b, [expB["mean_pairwise_errors"][k] for k in keys_b])
    ax.set_ylabel("mean TV"); ax.set_title("Experiment B: error attribution (pairwise, not additive)")
    fig.tight_layout(); fig.savefig(os.path.join(PLOT_DIR, "B_error_attribution.png"), dpi=130); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(attr_arrays["tv_S0_S2"], attr_arrays["tv_S0_S4"], s=6, alpha=0.3)
    ax.set_xlabel("routing error (TV S0 vs S2)"); ax.set_ylabel("CRISP global error (TV S0 vs S4)")
    ax.set_title("Routing error vs. final CRISP error")
    fig.tight_layout(); fig.savefig(os.path.join(PLOT_DIR, "routing_vs_final_error.png"), dpi=130); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    storage_x = [bytes_by_system[s] for s in ["S1", "S4"]]
    storage_y = [expA["S1"]["mean_tv"], expA["S4"]["mean_tv"]]
    ax.scatter(storage_x, storage_y, s=80)
    for s, x, y in zip(["S1", "S4"], storage_x, storage_y):
        ax.annotate(s, (x, y))
    ax.set_xlabel("deployable compressed bytes"); ax.set_ylabel("mean TV vs. S0")
    ax.set_title("Compression vs. fidelity: DIME-V1 vs. CRISP")
    fig.tight_layout(); fig.savefig(os.path.join(PLOT_DIR, "storage_vs_fidelity.png"), dpi=130); plt.close(fig)

    # ============================================================
    # mathematical_observations.md / research_questions.md
    # ============================================================
    print("Writing mathematical_observations.md and research_questions.md...")
    math_obs = f"""# CRISP-R5 Mathematical Observations

## Observation 1 -- Local fidelity does not mechanically imply global fidelity

**Claim:** R4 showed B4 substantially improves WITHIN-cluster TV over DIME-V1 (reconstructed fresh here across all
{N_CLUSTERS} clusters, coverage statistics in crisp_fitting_coverage). R5 tests whether this survives centroid
mixing and global aggregation.

**Evidence:** compare experiment_A_global_fidelity's S4 (CRISP) vs. S1 (DIME) mean_tv/mean_js directly against the
within-cluster improvement reconstructed during Phase 3 fitting (tune-split TV during fitting vs. S1's rank-0
TV). A smaller global gap than the within-cluster gap would be DIRECT evidence that aggregation partially masks
local improvement; a comparable or larger global gap would support local-to-global transfer.

**Status:** empirically evaluated in this script's own E2/E3/E4-equivalent (Experiment A/C), not assumed.

**Technique:** this is a question about how distributional error propagates through a convex combination
(DIME's beta_j(q)-weighted mixture) -- relevant: data-processing-style inequalities for divergences under mixing.

## Observation 2 -- Routing error, aggregation error, and compression error are NOT additive

**Claim:** tv_S0_S2 (routing) + tv_S2_S3 (aggregation) + tv_S3_S4 (compression) does not equal tv_S0_S4 in general,
because TV satisfies the triangle inequality (an upper bound), not an equality, and errors can partially cancel.

**Evidence:** see experiment_B_error_attribution's component_correlation_with_final_error -- if all three
correlate positively but the SUM of means exceeds the mean of tv_S0_S4, this is direct confirmation of
sub-additivity (cancellation), reported explicitly rather than assumed.

**Status:** mathematically guaranteed as an upper bound (triangle inequality); whether cancellation is SUBSTANTIAL
in practice is an empirical question answered by the numbers above.

**Technique:** this is exactly the content of the triangle-inequality slack; a tighter mathematical treatment would
bound the gap via the specific structure of the convex combination rather than generic TV subadditivity.

## Observation 3 -- Candidate-set recall structurally upper-bounds attainable raw-kNN fidelity

**Claim:** no local function (exact or compressed) evaluated only on members of the top-20 DIME-selected clusters
can ever recover the raw top-20 neighbors that routing excludes.

**Evidence:** routing_overlap_full_scale's frac_queries_with_full_top20_recall gives the fraction of queries where
NO excluded neighbor exists (an exact ceiling); on the remaining queries, S2/S3/S4/S5/S6 all inherit whatever
error routing_diagnostics.csv's raw_top20_neighbor_recall_in_candidate_set shows directly.

**Status:** proven as a structural ceiling (S2 by definition cannot see outside U(q)); the empirical MAGNITUDE of
the ceiling's cost is given by the routing-overlap numbers above.

**Technique:** this is a candidate-generation recall question, analogous to retrieval-augmented generation's
"retriever recall upper-bounds generator accuracy" framing.

## Observation 4 -- Evidence-mass-aware aggregation's effect size, holding local functions fixed

**Claim:** comparing S3 (centroid-softmax-weighted exact oracle) against S5 (mass-weighted exact oracle) isolates
exactly how much DIME's beta_j(q) choice (vs. the real evidence mass m_j(q)) matters, independent of any
compression question.

**Evidence:** see experiment_B_error_attribution's mean tv_S3_S5. A near-zero value would mean centroid-softmax
weighting is already a good proxy for real evidence mass; a large value motivates investigating S7 (compressed
mass predictor) as a genuinely separate, worthwhile follow-up -- not assumed in advance.

**Status:** empirically evaluated; see Q7 in report.md for the explicit S7-justification verdict.

**Technique:** this is a question of how well a DISTANCE-based proxy (softmax over centroid distance) approximates
an INTEGRAL quantity (the total kernel mass over all members) -- relevant to kernel density estimation literature
on bandwidth/proxy mismatch.
"""
    with open(os.path.join(OUT_DIR, "mathematical_observations.md"), "w") as f:
        f.write(math_obs)

    research_q = """# CRISP-R5 Research Questions for Subsequent Study

## Functional analysis / approximation theory
- Does local conditional-distribution fidelity (as measured in R1-R4) imply any FORMAL bound on globally
  aggregated distributional error under DIME's convex-combination mixing, or only an empirical correlation?
  Keywords: data-processing inequality for f-divergences, convexity of TV/KL under mixtures.
- What determines the rank needed for accurate GLOBAL (not just within-cluster) reconstruction, given that
  different clusters are selected with different frequencies and different beta_j(q) weights across queries?

## Geometry / information theory
- How does centroid candidate-set recall (Observation 3) constrain attainable raw-kNN fidelity as a function of
  the number of selected centroids k and the within-cluster radius distribution? Is there a closed-form
  relationship, or only an empirical one?
- How does evidence-mass approximation (S3 vs. S5's tv_S3_S5) affect cross-cluster probability weighting --
  specifically, is the centroid-softmax-vs-real-mass gap systematically related to cluster population or
  within-cluster variance?

## RKHS and kernel methods
- Can RKHS approximation methods (e.g. random features, Nystrom) efficiently represent the histogram
  log-correction functions delta_{j,y}(q) with provable approximation guarantees, rather than the anchor-difference
  basis used here (which has no such guarantee)?

## Statistical estimation / compression architecture
- How should approximation capacity (CRISP's rank r) be allocated across clusters under a FIXED total memory
  budget, given R4's finding that required rank varies by cluster -- and does the ANSWER change once global
  retrieval frequency (not just within-cluster difficulty) is accounted for, since a rarely-selected cluster's
  compression error matters less to the global result?
- Is a compressed mass predictor (S7, not implemented here) worth its own storage cost, given the measured
  tv_S3_S5 / tv_S5_S6 gaps? This script's own numbers are the evidence to judge this with.

## Experimental design
- R5's main comparison is confirmatory-but-not-fully-independent (rank=8 and the logit-correction representation
  were chosen partly by looking at R1-R4's own val-set results). What would a genuinely held-out confirmatory
  test require, given this project's only two splits (controller_train, val)?
"""
    with open(os.path.join(OUT_DIR, "research_questions.md"), "w") as f:
        f.write(research_q)

    # ============================================================
    # Final verdict (Q1-Q10)
    # ============================================================
    print("\nComputing final verdict (Q1-Q10)...")
    q1 = {"metric": "TV/JS of S1 vs S0", "tv": expA["S1"]["mean_tv"], "js": expA["S1"]["mean_js"]}
    q2 = {"metric": "TV/JS of S4 vs S0", "tv": expA["S4"]["mean_tv"], "js": expA["S4"]["mean_js"]}
    q3 = {"R_D_tv": expC["TV"]["R_D_crisp_vs_dime"], "R_D_js": expC["JS"]["R_D_crisp_vs_dime"]}
    q4 = {"p_true_relative_improvement": expD["S4"]["relative_improvement_in_p_true_vs_dime"],
           "coverage_s4": expD["S4"]["coverage"], "coverage_s1": expD["S1"]["coverage"]}
    q5 = {"tv_S3_S4_mean_compression_gap": expB["mean_pairwise_errors"]["tv_S3_S4"]}
    q6 = {"tv_S0_S2_routing": expB["mean_pairwise_errors"]["tv_S0_S2"], "tv_S2_S3_aggregation": expB["mean_pairwise_errors"]["tv_S2_S3"]}
    q7 = {"tv_S3_S5_mass_effect": expB["mean_pairwise_errors"]["tv_S3_S5"],
           "s7_justified": bool(expB["mean_pairwise_errors"]["tv_S3_S5"] > 0.02)}
    q8 = {"compression_ratio_crisp": storage["compression_ratio_crisp"], "crisp_mean_latency_s": expG["S4"]["mean_seconds_per_query"]}

    tv_dime_arr = attr_arrays["tv_S0_S1"]
    tv_crisp_arr = attr_arrays["tv_S0_S4"]
    diffs = tv_dime_arr - tv_crisp_arr
    n_boot = 2000
    boot_rng = np.random.default_rng(SEED)
    boot_means = np.array([diffs[boot_rng.integers(0, len(diffs), size=len(diffs))].mean() for _ in range(n_boot)])
    ci_lo, ci_hi = np.percentile(boot_means, [2.5, 97.5])
    frac_crisp_wins = float((diffs > 0).mean())
    q9 = {"mean_tv_improvement_dime_minus_crisp": float(diffs.mean()), "ci_95_low": float(ci_lo), "ci_95_high": float(ci_hi),
           "frac_queries_crisp_improves_over_dime": frac_crisp_wins, "significant": bool(ci_lo > 0 or ci_hi < 0)}
    q10 = "See research_questions.md; the most load-bearing single question is how approximation capacity should be " \
          "allocated given both within-cluster difficulty AND global selection frequency jointly (neither alone, per Observation 1)."

    # Thresholds declared once, here, before being applied -- not tuned after
    # seeing which bucket the real numbers would fall into.
    r_d_strong_threshold, r_d_partial_threshold = 0.3, 0.05
    compression_strong_threshold, compression_partial_threshold = 5.0, 1.0
    strong = (q9["significant"] and q3["R_D_tv"] is not None and q3["R_D_tv"] > r_d_strong_threshold
              and q8["compression_ratio_crisp"] is not None and q8["compression_ratio_crisp"] > compression_strong_threshold
              and q4["p_true_relative_improvement"] is not None and q4["p_true_relative_improvement"] > 0)
    partial = (q9["significant"] and q3["R_D_tv"] is not None and q3["R_D_tv"] > r_d_partial_threshold
               and q8["compression_ratio_crisp"] is not None and q8["compression_ratio_crisp"] > compression_partial_threshold)
    no_transfer = q3["R_D_tv"] is not None and q3["R_D_tv"] <= 0
    if strong:
        verdict = "Strong evidence of useful global compression"
    elif partial:
        verdict = "Partial evidence; global improvement with important limitations"
    elif no_transfer:
        verdict = "Local improvement does not translate into global retrieval improvement"
    else:
        verdict = "Inconclusive due to implementation, coverage, or evaluation limitations"

    verdict_block = {"Q1_dime_distance_from_raw": q1, "Q2_crisp_distance_from_raw": q2, "Q3_relative_change_vs_dime": q3,
                       "Q4_correct_token_behavior": q4, "Q5_crisp_vs_s3_oracle_gap": q5, "Q6_routing_and_aggregation_discrepancy": q6,
                       "Q7_mass_aware_aggregation_effect": q7, "Q8_compressed_size_and_cost": q8,
                       "Q9_significance_and_reliability": q9, "Q10_limiting_mathematical_problem": q10,
                       "final_classification": verdict}
    print(f"  Verdict: {verdict}")
    print(f"  {verdict_block}")
    report["final_verdict"] = verdict_block
    save_results(os.path.join(OUT_DIR, "crisp_r5_summary.json"), report)

    # ============================================================
    # report.md
    # ============================================================
    print("\nGenerating report.md...")
    lines = [
        "# CRISP-R5: Global Retrieval Fidelity and Compression Evaluation", "",
        "## 1. Research question", "See module docstring -- primary + secondary questions.", "",
        "## 2. System definitions", "S0-S6 as defined in the module docstring; S7 not implemented (see Limitations).", "",
        "## 3. Reproducibility and baseline verification",
        f"Phase 1 full-scale baseline: {report['phase1_baseline']}",
        f"Phase 2 sanity checks (S2==S0 whenever candidate recall is complete): {report['phase2_sanity_checks']}", "",
        "## 4. Evaluation dataset and query count", f"{report['evaluation_sample']}", "",
        "## 5. Global TV/JS comparison", f"{expA}", "",
        "## 6. Correct-token behavior", f"{expD}", "",
        "## 7. Routing and approximation diagnostics",
        f"Full-scale routing overlap: {report['routing_overlap_full_scale']}",
        f"Error attribution (pairwise, not additive): {expB}",
        f"CRISP fitting coverage: {report['crisp_fitting_coverage']}", "",
        "## 8. Actual storage and latency", f"Storage: {storage}", f"Latency (light, in-loop): {expG}", "",
        "## 9. Statistical uncertainty", f"Q9: {q9}", "",
        "## 10. Failure cases", "See failure_cases.md for all 6 categories with shared query identifiers.", "",
        "## 11. Limitations",
        "- Secondary tuned-config comparison (raw k=300/tau=1, DIME k=100/tau=2) NOT re-run -- deferred (Phase 5).",
        "- CRISP rank sensitivity (4, 16) and alternative centroid counts NOT run -- deferred (Phase 5).",
        "- S7 (compressed mass predictor) NOT implemented -- see Q7 for whether it would be justified.",
        f"- Evaluation sample bounded to {n_eval_actual} of {len(val_values)} val queries for the expensive systems "
        "(S2-S6); S0/S1 statistics and routing overlap ARE full-scale (cheap).",
        "- R5's main comparison is confirmatory-but-not-fully-independent: val was never used to FIT CRISP "
        "coefficients, but rank=8/the logit-correction representation were chosen partly from R1-R4's own "
        "val-set results across this project's development history.",
        "- Experiment G's latency is a light in-loop measurement, not an isolated deployment benchmark.",
        "- Q9's bootstrap resamples query INDICES i.i.d., not a sequence-aware or block bootstrap -- val queries from "
        "the same underlying document/sequence context may be correlated, which an i.i.d. bootstrap does not "
        "account for. Stated explicitly as a scope simplification, per the spec's own 'where possible' qualifier "
        "on the block-bootstrap instruction, not silently assumed away.",
        "- Retrieval-only NLL uses a single documented epsilon floor (1e-6) identically across all systems; "
        "coverage (fraction nonzero) is always reported alongside, separately.", "",
        "## 12. Final verdict", f"**{verdict}**", f"{verdict_block}",
    ]
    with open(os.path.join(OUT_DIR, "report.md"), "w") as f:
        f.write("\n".join(lines))

    report["sections_completed"] = ["phase1_baseline", "routing_overlap_full_scale", "phase3_global_fitting",
                                      "phase2_sanity_checks", "evaluation_sample", "phase4_main_loop",
                                      "experiment_A_global_fidelity", "experiment_B_error_attribution",
                                      "experiment_C_relative_improvement", "experiment_D_correct_token",
                                      "experiment_F_storage", "experiment_G_latency_light", "failure_cases.md",
                                      "mathematical_observations.md", "research_questions.md", "final_verdict",
                                      "report.md", "plots"]
    report["total_runtime_seconds"] = time.perf_counter() - t_start
    save_results(os.path.join(OUT_DIR, "crisp_r5_summary.json"), report)
    print(f"\nAll results saved under {OUT_DIR}")
    print(f"Total runtime: {report['total_runtime_seconds']/60:.1f} min")
    print("No existing E1/E2/TRI1/NC1/GEOM1/KM1/R1/R2/R3/R4 outputs were modified.")


if __name__ == "__main__":
    main()
