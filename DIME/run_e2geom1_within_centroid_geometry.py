"""E2-GEOM1 -- Diagnosing Within-Centroid Predictive Geometry. Extends the
E1/E2/TRI1/NC1 series without modifying any existing output (writes only to
results/stage3_e2geom1_within_centroid/). This is an information-loss
DIAGNOSTIC, not an architecture search -- Systems B and C intentionally
retain the original or anchor-projected datastore points per query and are
explicitly NOT proposed as deployable, consistent with the spec's own
"do not describe this as a low-memory deployment solution" instruction.

Reuses exactly, not reimplemented: cache loading, seed=42 MiniBatchKMeans
clustering, matched config (k=20, tau=2.0, alpha=0.25), knn.py/mixing.py
retrieval, e1_metrics.py helpers, run_e2_information_allocation's
build_cluster_to_members, and run_e2tri1_anchor_conditioned_compression's
anchor-selection/feature/leave-one-out-accuracy/cluster-stats functions
(imported directly). System D reconstructs E2-TRI1's real persisted K=8
config using the same reconstruct-and-verify precedent E2.1 and E2-NC1 used.

Note on `compute_sys3`: E2's own full-member local-retrieval function (the
real precedent for System B, already proven at full val scale -- its
persisted real result, mean_L_S3_val=3.9282, beats even raw kNN) is a
CLOSURE nested inside run_e2_information_allocation.main(), not an
importable top-level function. System B below reimplements its exact
formula (identical kernel: softmax(-d_local/tau) vote over a cluster's
members) as a standalone, reusable function, and cross-checks against that
real persisted number at tau_local=2.0 before trusting anything built on it.

No scikit-learn anywhere in this script, including PCA (via NumPy SVD) and
the multimodality check (a hand-rolled few-iteration 2-means on NumPy only,
not a new clustering architecture -- used solely as a diagnostic, per the
spec's own "avoid expanding into spectral clustering" instruction). The
only sklearn usage project-wide remains the pre-existing MiniBatchKMeans
clustering, reused unchanged.

=====================================================================
Scoping decisions (stated up front, per established practice -- this spec
is the largest yet, spanning 4 systems and 5 sub-experiments):
=====================================================================
  - System B: FULL-cluster kernel-weighted local retrieval (every original
    member of each retrieved centroid, no capping) at FULL val scale --
    tau_local tuned via a small grid on controller_train, which includes
    2.0 specifically to cross-check against E2's real persisted S3 result.
    The spec's secondary bounded-neighbor-count variant (nearest-M-to-QUERY
    members) is scoped OUT of this delivery entirely, since the full-
    cluster primary variant already has a real cross-checkable precedent
    (E2's S3) and System C below already answers the main "does limiting
    local evidence lose information" question via projection dimension.
  - System C: anchor-projected (K=8, nearest anchors -- TRI1's real chosen
    config) individual-point local retrieval, full-cluster, full val scale,
    tau_z_local tuned on controller_train with the same rescaling
    convention TRI1 used.
  - System D: TRI1's real K=8 config, RECONSTRUCTED and verified against
    its persisted result (same precedent as E2-NC1's N7 reference row).
  - Experiment A (spatial structure) is bounded to the 15 largest-
    population clusters throughout (A1-A5).
  - Experiment B's query-level analyses (B1, B2, B4) run on a 5,000-query
    val subsample, not full val -- B3 reuses System B's full-val NLL
    directly, no separate computation needed.
  - Experiment E's K-sweep (K in {1,2,4,8,16,32} x 3 anchor-selection
    methods, 18 point-level System-C configurations) runs on a 3,000-query
    val subsample, not full val -- the full-val K=8/nearest point is
    already covered by System C's main-experiment number.
  - Both local-retrieval engines (System B and C) memoize each cluster's
    projected member array across all queries in a call -- a cluster
    retrieved by thousands of different queries is projected exactly once,
    not once per query.

Smoke testing: set E2GEOM1_SMOKE=1 with E2GEOM1_CACHE_DIR / E2GEOM1_OUT_DIR /
E2GEOM1_N_CLUSTERS / E2GEOM1_TRI1_SUMMARY before running -- same convention
as TRI1/NC1. Never set it for the real run.
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
from scipy import stats as scistats
from sklearn.cluster import MiniBatchKMeans

from cache_io import load_cache
from knn import build_datastore, query_knn_indices, softmax_np
from results_io import save_results
from e1_metrics import (raw_standalone_prob, dime_standalone_prob, raw_full_distribution, dime_full_distribution,
                          js_divergence_dict, retrieval_coverage, distribution_shape_stats)
from run_e2_information_allocation import build_cluster_to_members
from run_e2tri1_anchor_conditioned_compression import (
    build_centroid_distance_matrix, select_anchors_nearest, select_anchors_random, select_anchors_diverse,
    compute_diff_features, compute_cluster_token_stats, build_cluster_stat_arrays, anchor_standalone_prob,
    leave_one_out_accuracy, nll_from_standalone as tri1_nll_from_standalone)
from gpt_class import FrozenGPT2

SMOKE = os.environ.get("E2GEOM1_SMOKE") == "1"
CACHE_DIR = os.environ.get("E2GEOM1_CACHE_DIR", os.path.join(SCRIPT_DIR, "..", "GPT_Module", "cache"))
OUT_DIR = os.environ.get("E2GEOM1_OUT_DIR", os.path.join(SCRIPT_DIR, "results", "stage3_e2geom1_within_centroid"))
PLOT_DIR = os.path.join(OUT_DIR, "plots")
os.makedirs(PLOT_DIR, exist_ok=True)
TRI1_SUMMARY_PATH = os.environ.get("E2GEOM1_TRI1_SUMMARY",
                                     os.path.join(SCRIPT_DIR, "results", "stage3_e2tri1_anchor_conditioned", "e2tri1_summary.json"))
E2_SUMMARY_PATH = os.environ.get("E2GEOM1_E2_SUMMARY",
                                   os.path.join(SCRIPT_DIR, "results", "stage3_e2_information_allocation", "e2_summary.json"))

N_CLUSTERS = int(os.environ.get("E2GEOM1_N_CLUSTERS", "3811"))
MATCHED = {"k": 20, "tau": 2.0, "alpha": 0.25}
SEED = 42
DIM, DTYPE_BYTES = 768, 4

TAU_LOCAL_GRID = [0.5, 1.0, 2.0, 4.0]
TAU_Z_LOCAL_FACTOR_GRID = [0.5, 1.0, 2.0]
# SYSTEM_C_K is NOT a fixed constant -- it is set inside main() to TRI1's real
# persisted best_K, so Systems C and D always share the exact same anchors.
N_SAMPLE_CLUSTERS = 15
N_SUBSAMPLE_B = 5000
N_SUBSAMPLE_E = 3000
K_SWEEP = [1, 2, 4, 8, 16, 32]
PURITY_NEIGHBOR_SIZES = [5, 10, 20]
WORST_PCTS = [1, 5, 10]
N_SHUFFLE_REPEATS_B4 = 3
VERBOSE_EVERY = 20000 if not SMOKE else 200


# ============================================================
# Generalized local-retrieval engine (Systems B and C share this)
# ============================================================
def local_retrieval_prob(query_keys, true_values, idx_dime, d_dime, tau_outer, cluster_to_members, ds_values,
                           project_query_fn, project_members_fn, tau_local, verbose_every=VERBOSE_EVERY):
    """project_*_fn(indices_or_rows, c) -> projected features. Each cluster's
    projected member array is memoized across the whole call -- a cluster
    retrieved by many queries is projected exactly once."""
    N, k = idx_dime.shape
    w = softmax_np(-d_dime / tau_outer, axis=1)
    p = np.zeros(N, dtype=np.float64)
    member_cache = {}
    t0 = time.perf_counter()
    for i in range(N):
        true_tok = int(true_values[i])
        acc = 0.0
        for j in range(k):
            c = int(idx_dime[i, j])
            members = cluster_to_members[c]
            if len(members) == 0:
                continue
            if c not in member_cache:
                member_cache[c] = project_members_fn(members, c)
            z_m = member_cache[c]
            z_q = project_query_fn(query_keys[i:i + 1], c)[0]
            d_local = np.linalg.norm(z_m - z_q[None, :], axis=1)
            w_local = softmax_np(-d_local / tau_local, axis=-1)
            match = ds_values[members] == true_tok
            if match.any():
                acc += float(w[i, j]) * float(w_local[match].sum())
        p[i] = acc
        if verbose_every and (i + 1) % verbose_every == 0:
            elapsed = time.perf_counter() - t0
            print(f"      progress: {i+1}/{N}  ({(i+1)/elapsed:.1f} q/s, ETA {(N-i-1)/((i+1)/elapsed)/60:.1f} min)")
    return p


def nll_from_standalone(p_true, p_lm_true, alpha):
    return -np.log(alpha * p_true + (1 - alpha) * p_lm_true + 1e-12)


# ============================================================
# Experiment A -- spatial-structure diagnostics (NumPy only)
# ============================================================
def local_label_purity(ds_keys, ds_values, members, neighbor_sizes, rng, max_probes=300):
    """For a random sample of members, fraction of its k-nearest same-cluster
    neighbors sharing its label, vs. the expected rate under random mixing
    (the cluster's own label-frequency entropy-equivalent baseline)."""
    n = len(members)
    if n < 10:
        return None
    probe_idx = rng.choice(n, size=min(max_probes, n), replace=False)
    H = ds_keys[members]
    values = ds_values[members]
    freq = Counter(values.tolist())
    total = sum(freq.values())
    expected_match_rate = sum((c / total) ** 2 for c in freq.values())  # P(same label | random pairing)
    results = {}
    for ksz in neighbor_sizes:
        k_eff = min(ksz, n - 1)
        if k_eff < 1:
            continue
        match_rates = []
        for i in probe_idx:
            d = np.linalg.norm(H - H[i][None, :], axis=1)
            d[i] = np.inf
            nn_idx = np.argpartition(d, k_eff - 1)[:k_eff]
            match_rates.append(float(np.mean(values[nn_idx] == values[i])))
        results[ksz] = {"observed_mean_match_rate": float(np.mean(match_rates)), "expected_under_random_mixing": expected_match_rate,
                          "n_probes": len(probe_idx)}
    return results


def cohesion_vs_shuffled_null(ds_keys, ds_values, members, rng, min_members=5, max_pairs=50, n_shuffles=3):
    values = ds_values[members]
    freq = Counter(values.tolist())
    eligible = [y for y, c in freq.items() if c >= min_members]
    if not eligible:
        return None
    real_cohesions, shuffled_cohesions = [], []
    for y in eligible:
        tok_members = members[values == y]
        n = len(tok_members)
        n_pairs = min(max_pairs, n * (n - 1) // 2)
        idx_a = rng.integers(0, n, size=n_pairs)
        idx_b = rng.integers(0, n, size=n_pairs)
        valid = idx_a != idx_b
        if not valid.any():
            continue
        pa, pb = tok_members[idx_a[valid]], tok_members[idx_b[valid]]
        real_cohesions.append(float(np.mean(np.linalg.norm(ds_keys[pa] - ds_keys[pb], axis=1))))
    for _ in range(n_shuffles):
        shuffled_values = rng.permutation(values)
        for y in eligible:
            tok_members = members[shuffled_values == y]
            n = len(tok_members)
            if n < 2:
                continue
            n_pairs = min(max_pairs, n * (n - 1) // 2)
            idx_a = rng.integers(0, n, size=n_pairs)
            idx_b = rng.integers(0, n, size=n_pairs)
            valid = idx_a != idx_b
            if not valid.any():
                continue
            pa, pb = tok_members[idx_a[valid]], tok_members[idx_b[valid]]
            shuffled_cohesions.append(float(np.mean(np.linalg.norm(ds_keys[pa] - ds_keys[pb], axis=1))))
    if not real_cohesions or not shuffled_cohesions:
        return None
    return {"mean_real_pair_distance": float(np.mean(real_cohesions)), "mean_shuffled_pair_distance": float(np.mean(shuffled_cohesions)),
             "n_tokens_eligible": len(eligible)}


def simple_2means(feats, n_iter=10, seed=0):
    """Hand-rolled 2-means (Lloyd's algorithm), NumPy only -- a diagnostic
    heuristic for multimodality, not a new clustering architecture."""
    n = feats.shape[0]
    if n < 4:
        return None
    rng = np.random.default_rng(seed)
    init_idx = rng.choice(n, size=2, replace=False)
    centers = feats[init_idx].copy()
    assign = np.zeros(n, dtype=int)
    for _ in range(n_iter):
        d0 = np.linalg.norm(feats - centers[0][None, :], axis=1)
        d1 = np.linalg.norm(feats - centers[1][None, :], axis=1)
        new_assign = (d1 < d0).astype(int)
        if np.array_equal(new_assign, assign) and _ > 0:
            break
        assign = new_assign
        for c in (0, 1):
            if (assign == c).any():
                centers[c] = feats[assign == c].mean(axis=0)
    overall_var = float(np.sum((feats - feats.mean(axis=0)) ** 2))
    within_2means_var = float(sum(np.sum((feats[assign == c] - centers[c][None, :]) ** 2) for c in (0, 1) if (assign == c).any()))
    reduction = (overall_var - within_2means_var) / overall_var if overall_var > 0 else 0.0
    return {"variance_reduction_fraction": reduction, "cluster_sizes": [int((assign == 0).sum()), int((assign == 1).sum())]}


def pca_2d(feats):
    """2-component PCA via NumPy SVD -- no scikit-learn."""
    centered = feats - feats.mean(axis=0, keepdims=True)
    try:
        _, _, Vt = np.linalg.svd(centered, full_matrices=False)
    except np.linalg.LinAlgError:
        return None
    return centered @ Vt[:2].T


# ============================================================
# Main
# ============================================================
def main():
    t_start = time.perf_counter()
    print(f"E2-GEOM1 job started. SMOKE={SMOKE}, N_CLUSTERS={N_CLUSTERS}")
    rng = np.random.default_rng(SEED)

    print("Loading cache + TRI1/E2 persisted summaries...")
    ds = load_cache(os.path.join(CACHE_DIR, "wikitext103_gpt2_datastore.npz"))
    ct = load_cache(os.path.join(CACHE_DIR, "wikitext103_gpt2_controller_train.npz"))
    val = load_cache(os.path.join(CACHE_DIR, "wikitext103_gpt2_val.npz"))
    ds_keys, ds_values = ds["keys"], ds["values"]
    ct_keys, ct_values, ct_p_lm = ct["keys"], ct["values"], ct["p_lm_true"]
    val_keys, val_values, val_p_lm = val["keys"], val["values"], val["p_lm_true"]
    with open(TRI1_SUMMARY_PATH) as f:
        tri1_summary = json.load(f)
    e2_summary = None
    if os.path.exists(E2_SUMMARY_PATH):
        with open(E2_SUMMARY_PATH) as f:
            e2_summary = json.load(f)

    gpt = FrozenGPT2(model_name="gpt2")

    print("Re-deriving DIME clusters (seed=42, deterministic, identical to every prior E1/E2/TRI1/NC1 script)...")
    km = MiniBatchKMeans(n_clusters=N_CLUSTERS, random_state=SEED)
    assignment = km.fit_predict(ds_keys)
    dime_centroids = km.cluster_centers_.astype(np.float32)
    compressed_dists = []
    for c in range(N_CLUSTERS):
        member_values = ds_values[assignment == c]
        compressed_dists.append(Counter(member_values.tolist()) if len(member_values) > 0 else Counter())
    cluster_to_members = build_cluster_to_members(assignment, N_CLUSTERS)

    raw_index, raw_values = build_datastore(ds_keys, ds_values)
    dime_index, dime_values = build_datastore(dime_centroids, np.array(compressed_dists, dtype=object))

    kM, tauM, alphaM = MATCHED["k"], MATCHED["tau"], MATCHED["alpha"]
    print("Querying val/controller_train (matched config) -- System A (DIME-V1) + raw baselines...")
    d_raw_val, idx_raw_val = query_knn_indices(raw_index, val_keys, k=kM)
    d_dime_val, idx_dime_val = query_knn_indices(dime_index, val_keys, k=kM)
    r_raw_val, r_dime_val = raw_values[idx_raw_val], dime_values[idx_dime_val]
    p_raw_true_val = raw_standalone_prob(d_raw_val, r_raw_val, val_values, tau=tauM)
    p_A_true_val = dime_standalone_prob(d_dime_val, r_dime_val, val_values, tau=tauM)
    nll_raw_val = nll_from_standalone(p_raw_true_val, val_p_lm, alphaM)
    nll_A_val = nll_from_standalone(p_A_true_val, val_p_lm, alphaM)
    delta_L_val = nll_A_val - nll_raw_val

    d_dime_ct, idx_dime_ct = query_knn_indices(dime_index, ct_keys, k=kM)

    report = {"n_val": int(len(val_values)), "n_ct": int(len(ct_values)), "n_ds": int(len(ds_values)),
              "matched_config": MATCHED, "n_clusters": N_CLUSTERS, "seed": SEED, "smoke": SMOKE,
              "systemA_raw_mean_nll": float(np.mean(nll_raw_val)), "systemA_dime_mean_nll": float(np.mean(nll_A_val))}
    save_results(os.path.join(OUT_DIR, "e2geom1_summary.json"), report)

    # ============================================================
    # System B -- full original within-centroid local retrieval
    # ============================================================
    print("\nSystem B: tuning tau_local on controller_train (full-cluster, raw 768-dim space)...")
    identity_query_fn = lambda q_row, c: q_row
    identity_members_fn = lambda members, c: ds_keys[members]
    tau_local_nll = {}
    for tau_local in TAU_LOCAL_GRID:
        p_ct = local_retrieval_prob(ct_keys, ct_values, idx_dime_ct, d_dime_ct, tauM, cluster_to_members, ds_values,
                                      identity_query_fn, identity_members_fn, tau_local, verbose_every=0)
        nll_ct = float(np.mean(nll_from_standalone(p_ct, ct_p_lm, alphaM)))
        tau_local_nll[tau_local] = nll_ct
        print(f"  tau_local={tau_local}: controller_train mean NLL = {nll_ct:.4f}")
    tau_local_B = min(tau_local_nll, key=tau_local_nll.get)
    print(f"  chosen tau_local for System B = {tau_local_B}")

    print("System B: full-cluster local retrieval at full val scale...")
    p_B_val = local_retrieval_prob(val_keys, val_values, idx_dime_val, d_dime_val, tauM, cluster_to_members, ds_values,
                                     identity_query_fn, identity_members_fn, tau_local_B)
    nll_B_val = nll_from_standalone(p_B_val, val_p_lm, alphaM)
    print(f"  System B mean NLL (val, full scale) = {np.mean(nll_B_val):.4f}")

    if tau_local_B == 2.0 or 2.0 in TAU_LOCAL_GRID:
        p_B_val_tau2 = p_B_val if tau_local_B == 2.0 else local_retrieval_prob(
            val_keys, val_values, idx_dime_val, d_dime_val, tauM, cluster_to_members, ds_values,
            identity_query_fn, identity_members_fn, 2.0)
        nll_B_tau2 = float(np.mean(nll_from_standalone(p_B_val_tau2, val_p_lm, alphaM)))
        print(f"  CORRECTNESS CHECK: System B at tau_local=2.0 gives mean NLL={nll_B_tau2:.6f}")
        if e2_summary is not None and not SMOKE:
            persisted_s3 = e2_summary["baselines"]["mean_L_S3_val"]
            diff = abs(nll_B_tau2 - persisted_s3)
            print(f"    vs. E2's real persisted S3 NLL={persisted_s3:.6f} (abs diff={diff:.6f})")
            assert diff < 1e-3, "System B's tau_local=2.0 does not reproduce E2's real persisted S3 result -- investigate before trusting System B further"
    report["systemB"] = {"tau_local_tuning": tau_local_nll, "chosen_tau_local": tau_local_B,
                           "mean_nll_val": float(np.mean(nll_B_val))}
    save_results(os.path.join(OUT_DIR, "e2geom1_summary.json"), report)

    # ============================================================
    # Anchor set shared by Systems C and D -- MUST be identical (same method,
    # same K) to TRI1's real persisted config, otherwise the C-vs-D
    # comparison (how much per-token aggregation loses) would be comparing
    # two different geometric representations, not isolating aggregation.
    # ============================================================
    print("\nResolving TRI1's real anchor-selection method/K for Systems C and D (must match exactly)...")
    dist_matrix = build_centroid_distance_matrix(dime_centroids)
    tri1_method = tri1_summary["chosen_anchor_method"]
    tri1_best_K = tri1_summary["best_K_by_ct_subsample"]
    SYSTEM_C_K = tri1_best_K
    if tri1_method == "nearest":
        anchors_fixed = {c: select_anchors_nearest(dist_matrix, c, SYSTEM_C_K) for c in range(N_CLUSTERS)}
    elif tri1_method == "diverse":
        anchors_fixed = {c: select_anchors_diverse(dist_matrix, c, SYSTEM_C_K) for c in range(N_CLUSTERS)}
    else:
        print("  WARNING: TRI1's chosen method was 'random' -- best-effort reconstruction, not guaranteed exact; "
              "System D's verification check below will catch a mismatch.")
        tri1_rng = np.random.default_rng(SEED)
        anchors_fixed = {c: select_anchors_random(N_CLUSTERS, c, SYSTEM_C_K, tri1_rng) for c in range(N_CLUSTERS)}
    print(f"  using TRI1's real method='{tri1_method}', K={SYSTEM_C_K} for both System C and System D")

    # ============================================================
    # System C -- anchor-projected individual-point local retrieval
    # ============================================================
    print(f"\nSystem C: tuning tau_z_local on controller_train (K={SYSTEM_C_K}, method={tri1_method})...")
    anchor_query_fn = lambda q_row, c: compute_diff_features(q_row, c, anchors_fixed[c], dime_centroids)
    anchor_members_fn = lambda members, c: compute_diff_features(ds_keys[members], c, anchors_fixed[c], dime_centroids)

    z_sample = []
    for c in list(cluster_to_members.keys())[:200]:
        members = cluster_to_members[c]
        if len(members) > 0:
            z_sample.append(anchor_members_fn(members, c))
    z_std_ref = float(np.std(np.concatenate(z_sample, axis=0))) if z_sample else 1.0
    print(f"  empirical z std (K={SYSTEM_C_K} reference sample): {z_std_ref:.4f}")

    tau_z_local_nll = {}
    for factor in TAU_Z_LOCAL_FACTOR_GRID:
        tau_z_local = factor * max(z_std_ref, 1e-6)
        p_ct = local_retrieval_prob(ct_keys, ct_values, idx_dime_ct, d_dime_ct, tauM, cluster_to_members, ds_values,
                                      anchor_query_fn, anchor_members_fn, tau_z_local, verbose_every=0)
        nll_ct = float(np.mean(nll_from_standalone(p_ct, ct_p_lm, alphaM)))
        tau_z_local_nll[factor] = {"tau_z_local": tau_z_local, "ct_mean_nll": nll_ct}
        print(f"  tau_z_local_factor={factor} (tau_z_local={tau_z_local:.4f}): controller_train mean NLL = {nll_ct:.4f}")
    factor_C = min(tau_z_local_nll, key=lambda f: tau_z_local_nll[f]["ct_mean_nll"])
    tau_z_local_C = tau_z_local_nll[factor_C]["tau_z_local"]
    print(f"  chosen tau_z_local for System C = {tau_z_local_C:.4f}")

    print("System C: anchor-space full-cluster local retrieval at full val scale...")
    p_C_val = local_retrieval_prob(val_keys, val_values, idx_dime_val, d_dime_val, tauM, cluster_to_members, ds_values,
                                     anchor_query_fn, anchor_members_fn, tau_z_local_C)
    nll_C_val = nll_from_standalone(p_C_val, val_p_lm, alphaM)
    print(f"  System C mean NLL (val, full scale) = {np.mean(nll_C_val):.4f}")
    report["systemC"] = {"K": SYSTEM_C_K, "z_std_reference": z_std_ref, "tau_z_local_tuning": tau_z_local_nll,
                           "chosen_tau_z_local": tau_z_local_C, "mean_nll_val": float(np.mean(nll_C_val))}
    save_results(os.path.join(OUT_DIR, "e2geom1_summary.json"), report)

    # ============================================================
    # System D -- reconstruct TRI1's real K=8 config, verify
    # ============================================================
    print("\nSystem D: reconstructing E2-TRI1's real K=8 config (same anchors as System C above), verifying before use...")
    hp = tri1_summary["hyperparameter_tuning"]
    N0_tri1, var_floor_tri1 = hp["chosen_N0"], hp["var_floor"]
    tau_z_tri1 = tri1_summary["experiment_D_grid"][f"A{3 + K_SWEEP[:5].index(tri1_best_K)}_diff_K{tri1_best_K}"]["tau_z"] \
        if tri1_best_K in [1, 2, 4, 8, 16] else None
    tri1_fn = lambda H, c: compute_diff_features(H, c, anchors_fixed[c], dime_centroids)
    tri1_stats = compute_cluster_token_stats(ds_keys, ds_values, cluster_to_members, tri1_fn, N0_tri1, var_floor_tri1)
    tri1_arrays = build_cluster_stat_arrays(tri1_stats, compressed_dists)
    p_D_val = anchor_standalone_prob(val_keys, val_values, idx_dime_val, d_dime_val, tauM, tri1_arrays, tri1_fn,
                                       False, tau_z_tri1, verbose_every=0)
    nll_D_val = tri1_nll_from_standalone(p_D_val, val_p_lm, alphaM)
    reconstructed_nll = float(np.mean(nll_D_val))
    persisted_nll = tri1_summary["experiment_D_grid"][f"A{3 + K_SWEEP[:5].index(tri1_best_K)}_diff_K{tri1_best_K}"]["mean_nll"]
    recon_diff = abs(reconstructed_nll - persisted_nll)
    print(f"  reconstructed System D mean NLL = {reconstructed_nll:.6f} vs. persisted TRI1 = {persisted_nll:.6f} "
          f"(abs diff = {recon_diff:.6f})")
    if not SMOKE:
        assert recon_diff < 1e-3, "System D reconstruction does not match TRI1's persisted real result -- stop"
    report["systemD"] = {"reconstructed_mean_nll_val": reconstructed_nll, "persisted_mean_nll": persisted_nll,
                           "reconstruction_abs_diff": recon_diff}
    save_results(os.path.join(OUT_DIR, "e2geom1_summary.json"), report)

    # ============================================================
    # Main four-system comparison table
    # ============================================================
    main_comparison = {
        "raw_kNN": float(np.mean(nll_raw_val)), "A_dime_v1": float(np.mean(nll_A_val)),
        "B_full_space_local": float(np.mean(nll_B_val)), "C_anchor_space_local": float(np.mean(nll_C_val)),
        "D_tri1_compressed": reconstructed_nll,
    }
    print(f"\nMain comparison (mean val NLL): {main_comparison}")
    report["main_comparison"] = main_comparison
    save_results(os.path.join(OUT_DIR, "e2geom1_summary.json"), report)

    fig, ax = plt.subplots(figsize=(7, 5))
    labels_plot = list(main_comparison.keys())
    ax.bar(range(len(labels_plot)), [main_comparison[l] for l in labels_plot])
    ax.set_xticks(range(len(labels_plot)))
    ax.set_xticklabels(labels_plot, rotation=30, ha="right")
    ax.set_ylabel("mean interpolated NLL (matched config, val)")
    ax.set_title("A vs. B vs. C vs. D -- where does information disappear?")
    fig.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, "main_comparison.png"), dpi=130)
    plt.close(fig)

    # ============================================================
    # Experiment A -- spatial-structure diagnostics (15 largest clusters)
    # ============================================================
    print(f"\nExperiment A: spatial-structure diagnostics on {N_SAMPLE_CLUSTERS} largest-population clusters...")
    populations = np.array([sum(c.values()) for c in compressed_dists])
    sample_clusters = [int(c) for c in np.argsort(-populations)[:N_SAMPLE_CLUSTERS] if len(cluster_to_members[int(c)]) > 10]

    per_cluster_A = {}
    for c in sample_clusters:
        members = cluster_to_members[c]
        purity = local_label_purity(ds_keys, ds_values, members, PURITY_NEIGHBOR_SIZES, rng)
        cohesion = cohesion_vs_shuffled_null(ds_keys, ds_values, members, rng)

        H_raw = ds_keys[members]
        values_c = ds_values[members]
        acc_raw = leave_one_out_accuracy(H_raw, values_c)
        shuffled_values = rng.permutation(values_c)
        acc_shuffled = leave_one_out_accuracy(H_raw, shuffled_values)
        H_anchor = anchor_members_fn(members, c)
        acc_anchor = leave_one_out_accuracy(H_anchor, values_c)
        freq_prior_acc = float(np.max(np.bincount(values_c.astype(np.int64))) / len(values_c)) if len(values_c) > 0 else None

        multimodal_tokens = {}
        for y, cnt in Counter(values_c.tolist()).items():
            if cnt >= 10:
                tok_members_mask = values_c == y
                result = simple_2means(H_raw[tok_members_mask], seed=SEED)
                if result is not None:
                    multimodal_tokens[int(y)] = result
        frac_multimodal = float(np.mean([r["variance_reduction_fraction"] > 0.3 for r in multimodal_tokens.values()])) \
            if multimodal_tokens else None

        per_cluster_A[c] = {"population": int(populations[c]), "purity": purity, "cohesion": cohesion,
                              "separability_accuracy_raw": acc_raw, "separability_accuracy_shuffled": acc_shuffled,
                              "separability_accuracy_anchor_space": acc_anchor, "frequency_prior_accuracy": freq_prior_acc,
                              "n_tokens_checked_for_multimodality": len(multimodal_tokens),
                              "fraction_multimodal_tokens": frac_multimodal}
        print(f"  centroid {c} (pop={int(populations[c])}): raw_acc={acc_raw}, shuffled_acc={acc_shuffled}, "
              f"anchor_acc={acc_anchor}, freq_prior={freq_prior_acc}, frac_multimodal={frac_multimodal}")
    report["experiment_A_per_cluster"] = per_cluster_A
    save_results(os.path.join(OUT_DIR, "e2geom1_summary.json"), report)

    print("Experiment A5: PCA visualizations (strongest-purity, weakest-purity, most-multimodal examples)...")
    purity_scores = {c: per_cluster_A[c]["purity"].get(10, {}).get("observed_mean_match_rate", None)
                       for c in sample_clusters if per_cluster_A[c]["purity"]}
    valid_purity = {c: v for c, v in purity_scores.items() if v is not None}
    pca_examples = []
    if valid_purity:
        best_c = max(valid_purity, key=valid_purity.get)
        worst_c = min(valid_purity, key=valid_purity.get)
        pca_examples = [("highest_purity", best_c), ("lowest_purity_most_mixed", worst_c)]
    for label, c in pca_examples:
        members = cluster_to_members[c]
        coords = pca_2d(ds_keys[members])
        if coords is None:
            continue
        values_c = ds_values[members]
        fig, ax = plt.subplots(figsize=(6, 6))
        top_tokens = [t for t, _ in Counter(values_c.tolist()).most_common(8)]
        for t in top_tokens:
            mask = values_c == t
            ax.scatter(coords[mask, 0], coords[mask, 1], s=10, alpha=0.6, label=gpt.tokenizer.decode([int(t)]))
        ax.set_title(f"centroid {c} ({label}), 2D PCA")
        ax.legend(fontsize=6, loc="best")
        fig.tight_layout()
        fig.savefig(os.path.join(PLOT_DIR, f"pca_{label}_cluster{c}.png"), dpi=130)
        plt.close(fig)
    report["experiment_A5_pca_examples"] = [{"label": l, "centroid": c} for l, c in pca_examples]
    save_results(os.path.join(OUT_DIR, "e2geom1_summary.json"), report)

    # ============================================================
    # Experiment B -- query-level enrichment (bounded val subsample)
    # ============================================================
    print(f"\nExperiment B: query-level enrichment on a {N_SUBSAMPLE_B}-query val subsample...")
    sub_idx = rng.choice(len(val_values), size=min(N_SUBSAMPLE_B, len(val_values)), replace=False)
    p_A_sub, p_B_sub = p_A_true_val[sub_idx], p_B_val[sub_idx]
    enrichment = p_B_sub - p_A_sub
    nonzero_support = p_A_sub > 0
    b1 = {"mean_enrichment_all": float(np.mean(enrichment)), "mean_enrichment_given_nonzero_prior": float(np.mean(enrichment[nonzero_support])) if nonzero_support.any() else None,
           "frac_enriched_positive": float(np.mean(enrichment > 0)), "n_subsample": int(len(sub_idx))}
    print(f"  B1 correct-token enrichment: {b1}")

    raw_dists_sub = raw_full_distribution(d_raw_val[sub_idx], r_raw_val[sub_idx], tau=tauM)
    dime_dists_sub = dime_full_distribution(d_dime_val[sub_idx], r_dime_val[sub_idx], tau=tauM)
    local_dists_sub = []
    for idx_local, i in enumerate(sub_idx):
        d = {}
        for j in range(kM):
            c = int(idx_dime_val[i, j])
            members = cluster_to_members[c]
            if len(members) == 0:
                continue
            z_m = ds_keys[members]
            z_q = val_keys[i]
            d_local = np.linalg.norm(z_m - z_q[None, :], axis=1)
            w_local = softmax_np(-d_local / tau_local_B, axis=-1)
            w_outer = float(softmax_np(-d_dime_val[i:i+1] / tauM, axis=1)[0, j])
            for tok, wl in zip(ds_values[members].tolist(), w_local.tolist()):
                d[tok] = d.get(tok, 0.0) + w_outer * wl
        local_dists_sub.append(d)
    js_b2 = [js_divergence_dict(local_d, dime_d) for local_d, dime_d in zip(local_dists_sub, dime_dists_sub)]
    b2 = {"mean_js_local_vs_dime_prior": float(np.mean(js_b2)), "n_subsample": int(len(sub_idx))}
    print(f"  B2 query-dependent variation (JS local vs. DIME-V1 prior): {b2}")

    print("  B4: shuffled-label control (within-centroid label shuffle, repeated)...")
    b4_results = []
    for rep in range(N_SHUFFLE_REPEATS_B4):
        shuffle_rng = np.random.default_rng(SEED + 100 + rep)
        shuffled_ds_values = ds_values.copy()
        for c in range(N_CLUSTERS):
            members = cluster_to_members[c]
            if len(members) > 1:
                shuffled_ds_values[members] = shuffle_rng.permutation(ds_values[members])
        p_b4 = local_retrieval_prob(val_keys[sub_idx], val_values[sub_idx], idx_dime_val[sub_idx], d_dime_val[sub_idx],
                                      tauM, cluster_to_members, shuffled_ds_values, identity_query_fn, identity_members_fn,
                                      tau_local_B, verbose_every=0)
        nll_b4 = float(np.mean(nll_from_standalone(p_b4, val_p_lm[sub_idx], alphaM)))
        b4_results.append(nll_b4)
        print(f"    shuffle repeat {rep}: mean NLL = {nll_b4:.4f}")
    nll_B_sub = float(np.mean(nll_from_standalone(p_B_sub, val_p_lm[sub_idx], alphaM)))
    b4 = {"real_label_mean_nll_subsample": nll_B_sub, "shuffled_label_mean_nll_repeats": b4_results,
           "shuffled_label_mean_nll_avg": float(np.mean(b4_results))}
    print(f"  B4 shuffled-label control: real={nll_B_sub:.4f} vs. shuffled_avg={b4['shuffled_label_mean_nll_avg']:.4f}")

    report["experiment_B"] = {"B1_enrichment": b1, "B2_query_dependent_variation": b2, "B4_shuffled_control": b4,
                                "B3_predictive_value_full_val_nll": float(np.mean(nll_B_val))}
    save_results(os.path.join(OUT_DIR, "e2geom1_summary.json"), report)

    # ============================================================
    # Experiment C -- information-loss comparison (margins, same sampled clusters)
    # ============================================================
    print("\nExperiment C: preference-margin comparison across raw/A/B/C/D on sampled clusters...")
    margin_examples = []
    for c in sample_clusters:
        top_tokens = [t for t, _ in compressed_dists[c].most_common(2)]
        if len(top_tokens) < 2:
            continue
        ya, yb = top_tokens
        region_idx = np.where(idx_dime_val[:, 0] == c)[0]
        region_idx = region_idx[np.isin(region_idx, sub_idx)]
        if len(region_idx) < 5:
            continue
        m_raw = np.array([raw_dists_sub[np.where(sub_idx == i)[0][0]].get(ya, 0.0) - raw_dists_sub[np.where(sub_idx == i)[0][0]].get(yb, 0.0) for i in region_idx])
        m_A = np.array([dime_dists_sub[np.where(sub_idx == i)[0][0]].get(ya, 0.0) - dime_dists_sub[np.where(sub_idx == i)[0][0]].get(yb, 0.0) for i in region_idx])
        m_B = np.array([local_dists_sub[np.where(sub_idx == i)[0][0]].get(ya, 0.0) - local_dists_sub[np.where(sub_idx == i)[0][0]].get(yb, 0.0) for i in region_idx])
        margin_examples.append({"centroid": c, "token_a": int(ya), "decoded_a": gpt.tokenizer.decode([int(ya)]),
                                   "token_b": int(yb), "decoded_b": gpt.tokenizer.decode([int(yb)]),
                                   "mean_margin_raw": float(np.mean(m_raw)), "mean_margin_A_dime": float(np.mean(m_A)),
                                   "mean_margin_B_local": float(np.mean(m_B)), "n_queries": len(region_idx)})
    report["experiment_C_margins"] = margin_examples
    print(f"  recorded {len(margin_examples)} margin comparisons")
    save_results(os.path.join(OUT_DIR, "e2geom1_summary.json"), report)

    # ============================================================
    # Experiment D -- difficult-query failure attribution
    # ============================================================
    print("\nExperiment D: difficult-query failure attribution (worst-1/5/10% by matched delta_L)...")
    order = np.argsort(-delta_L_val, kind="stable")
    tail_results = {}
    for pct in WORST_PCTS:
        n_worst = max(1, int(len(order) * pct / 100))
        worst_idx = order[:n_worst]
        no_support = np.array([compressed_dists[int(idx_dime_val[i, j])].get(int(val_values[i]), 0) == 0
                                  for i in worst_idx for j in range(kM)]).reshape(len(worst_idx), kM).all(axis=1)
        b_recovers = nll_B_val[worst_idx] < nll_A_val[worst_idx] - 0.05
        c_fails_where_b_succeeds = (nll_C_val[worst_idx] >= nll_A_val[worst_idx] - 0.05) & b_recovers
        d_fails_where_c_succeeds = (nll_D_val[worst_idx] >= nll_A_val[worst_idx] - 0.05) & (nll_C_val[worst_idx] < nll_A_val[worst_idx] - 0.05)
        tail_results[pct] = {
            "n_queries": int(n_worst),
            "mean_nll_raw": float(np.mean(nll_raw_val[worst_idx])), "mean_nll_A": float(np.mean(nll_A_val[worst_idx])),
            "mean_nll_B": float(np.mean(nll_B_val[worst_idx])), "mean_nll_C": float(np.mean(nll_C_val[worst_idx])),
            "mean_nll_D": float(np.mean(nll_D_val[worst_idx])),
            "D1_frac_no_retrieval_support": float(np.mean(no_support)),
            "D2_frac_B_recovers_substantially": float(np.mean(b_recovers)),
            "D3_frac_C_fails_where_B_succeeds": float(np.mean(c_fails_where_b_succeeds)),
            "D4_frac_D_fails_where_C_succeeds": float(np.mean(d_fails_where_c_succeeds)),
        }
        print(f"  worst-{pct}%: {tail_results[pct]}")
    report["experiment_D_failure_attribution"] = tail_results
    save_results(os.path.join(OUT_DIR, "e2geom1_summary.json"), report)

    worst1_idx = order[:max(1, int(len(order) * 0.01))]
    d5_examples = []
    for i in worst1_idx[:10]:
        d5_examples.append({"query_index": int(i), "true_token": int(val_values[i]),
                               "decoded_true_token": gpt.tokenizer.decode([int(val_values[i])]),
                               "nll_raw": float(nll_raw_val[i]), "nll_A": float(nll_A_val[i]), "nll_B": float(nll_B_val[i]),
                               "nll_C": float(nll_C_val[i]), "nll_D": float(nll_D_val[i])})
    report["experiment_D5_representative_cases"] = d5_examples
    save_results(os.path.join(OUT_DIR, "e2geom1_summary.json"), report)

    # ============================================================
    # Experiment E -- anchor-dimension sweep (bounded val subsample, point-level System C)
    # ============================================================
    print(f"\nExperiment E: K-sweep for point-level System C on a {N_SUBSAMPLE_E}-query val subsample...")
    e_idx = rng.choice(len(val_values), size=min(N_SUBSAMPLE_E, len(val_values)), replace=False)
    k_sweep_results = {}
    for method_name, select_fn in [("nearest", lambda c, k: select_anchors_nearest(dist_matrix, c, k)),
                                      ("diverse", lambda c, k: select_anchors_diverse(dist_matrix, c, k))]:
        k_sweep_results[method_name] = {}
        for K in K_SWEEP:
            anchors_K = {c: select_fn(c, K) for c in range(N_CLUSTERS)}
            qfn = lambda q_row, c, a=anchors_K: compute_diff_features(q_row, c, a[c], dime_centroids)
            mfn = lambda members, c, a=anchors_K: compute_diff_features(ds_keys[members], c, a[c], dime_centroids)
            z_sub = [mfn(cluster_to_members[c], c) for c in list(cluster_to_members.keys())[:100] if len(cluster_to_members[c]) > 0]
            tau_e = 1.0 * max(float(np.std(np.concatenate(z_sub, axis=0))), 1e-6) if z_sub else 1.0
            p_e = local_retrieval_prob(val_keys[e_idx], val_values[e_idx], idx_dime_val[e_idx], d_dime_val[e_idx],
                                         tauM, cluster_to_members, ds_values, qfn, mfn, tau_e, verbose_every=0)
            nll_e = float(np.mean(nll_from_standalone(p_e, val_p_lm[e_idx], alphaM)))
            k_sweep_results[method_name][K] = nll_e
            print(f"  method={method_name}, K={K}: subsample mean NLL = {nll_e:.4f}")

    rand_dirs_e = {}
    k_sweep_results["random_projection"] = {}
    for K in K_SWEEP:
        rd = rng.normal(size=(K, DIM)).astype(np.float32)
        rd /= np.linalg.norm(rd, axis=1, keepdims=True)
        qfn = lambda q_row, c, rd=rd: q_row.astype(np.float64) @ rd.T.astype(np.float64)
        mfn = lambda members, c, rd=rd: ds_keys[members].astype(np.float64) @ rd.T.astype(np.float64)
        z_sub = [mfn(cluster_to_members[c], c) for c in list(cluster_to_members.keys())[:100] if len(cluster_to_members[c]) > 0]
        tau_e = 1.0 * max(float(np.std(np.concatenate(z_sub, axis=0))), 1e-6) if z_sub else 1.0
        p_e = local_retrieval_prob(val_keys[e_idx], val_values[e_idx], idx_dime_val[e_idx], d_dime_val[e_idx],
                                     tauM, cluster_to_members, ds_values, qfn, mfn, tau_e, verbose_every=0)
        nll_e = float(np.mean(nll_from_standalone(p_e, val_p_lm[e_idx], alphaM)))
        k_sweep_results["random_projection"][K] = nll_e
        print(f"  method=random_projection, K={K}: subsample mean NLL = {nll_e:.4f}")

    report["experiment_E_k_sweep"] = {"n_subsample": int(len(e_idx)), "results": k_sweep_results}
    save_results(os.path.join(OUT_DIR, "e2geom1_summary.json"), report)

    fig, ax = plt.subplots(figsize=(7, 5))
    for method_name, by_k in k_sweep_results.items():
        ks = sorted(by_k.keys())
        ax.plot(ks, [by_k[k] for k in ks], marker="o", label=method_name)
    ax.set_xlabel("K (number of anchors / projection dimensions)")
    ax.set_ylabel(f"mean NLL ({len(e_idx)}-query val subsample)")
    ax.set_title("Experiment E: point-level System C vs. K")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, "experiment_E_k_sweep.png"), dpi=130)
    plt.close(fig)

    # ============================================================
    # Markdown report
    # ============================================================
    print("\nGenerating report.md...")
    lines = [
        "# DIME Stage 3 -- E2-GEOM1: Diagnosing Within-Centroid Predictive Geometry",
        "",
        f"Matched config (k={MATCHED['k']}, tau={MATCHED['tau']}, alpha={MATCHED['alpha']}), {len(val_values)} val queries, "
        f"{N_CLUSTERS} clusters.",
        "",
        "## Main comparison (mean val NLL, full scale)",
    ]
    for label, v in main_comparison.items():
        lines.append(f"- {label}: {v:.4f}")
    lines += [
        "",
        "## Experiment A -- spatial structure (sampled clusters)",
        f"Per-cluster separability: raw-space vs. shuffled-label control vs. anchor-space, and multimodality "
        "fraction -- see experiment_A_per_cluster in the JSON summary for the full per-cluster table.",
        "",
        "## Experiment B -- query-level enrichment (subsample)",
        f"B1 (correct-token enrichment): {b1}",
        f"B2 (query-dependent variation vs. DIME-V1 prior): {b2}",
        f"B4 (shuffled-label control): real={b4['real_label_mean_nll_subsample']:.4f} vs. "
        f"shuffled_avg={b4['shuffled_label_mean_nll_avg']:.4f}",
        "",
        "## Experiment D -- difficult-query failure attribution",
    ]
    for pct, d in tail_results.items():
        lines.append(f"- worst-{pct}%: raw={d['mean_nll_raw']:.4f}, A={d['mean_nll_A']:.4f}, B={d['mean_nll_B']:.4f}, "
                      f"C={d['mean_nll_C']:.4f}, D={d['mean_nll_D']:.4f}; D1(no support)={d['D1_frac_no_retrieval_support']:.3f}, "
                      f"D2(B recovers)={d['D2_frac_B_recovers_substantially']:.3f}, "
                      f"D3(C fails where B succeeds)={d['D3_frac_C_fails_where_B_succeeds']:.3f}, "
                      f"D4(D fails where C succeeds)={d['D4_frac_D_fails_where_C_succeeds']:.3f}")
    lines += [
        "",
        "## Experiment E -- anchor-dimension sweep (point-level System C, subsample)",
        f"{k_sweep_results}",
        "",
        "## Answers to the research questions (from the measurements above only)",
        f"**Q1/Q2** (real spatial organization vs. the synthetic Paris/Berlin assumption): see Experiment A's "
        "per-cluster purity/cohesion/separability table -- judge directly from those numbers, not asserted here.",
        f"**Q3** (does full-space local retrieval improve over DIME-V1 under fixed routing?): "
        f"System B={main_comparison['B_full_space_local']:.4f} vs. System A={main_comparison['A_dime_v1']:.4f} "
        f"(raw kNN={main_comparison['raw_kNN']:.4f} for reference).",
        f"**Q4** (information lost to anchor projection, B vs. C): "
        f"System C={main_comparison['C_anchor_space_local']:.4f} vs. System B={main_comparison['B_full_space_local']:.4f}.",
        f"**Q5** (information lost to per-token aggregation, C vs. D): "
        f"System D={main_comparison['D_tri1_compressed']:.4f} vs. System C={main_comparison['C_anchor_space_local']:.4f}.",
        "**Q6** (which failure mechanism dominates)/**Q7** (fraction of the DIME-vs-raw gap System B recovers)/"
        "**Q8** (does the evidence justify further anchor-conditioned development?): judge directly from the main "
        "comparison, Experiment D's failure attribution, and Experiment E's K-sweep above -- not pre-decided here.",
        "",
        "## Limitations",
        "- This is a diagnostic: Systems B and C retain original or anchor-projected individual datastore points "
        "per query and are explicitly NOT a proposed deployment architecture.",
        f"- Experiment A bounded to {len(sample_clusters)} largest-population clusters.",
        f"- Experiment B's query-level analyses (B1/B2/B4) run on a {len(sub_idx)}-query val subsample, not full val "
        "-- B3 reuses System B's full-val NLL directly.",
        f"- Experiment E's K-sweep run on a {len(e_idx)}-query val subsample, not full val; the full-val "
        f"K={SYSTEM_C_K}/nearest point is covered by System C's main-experiment number above.",
        "- System B's secondary bounded-neighbor-count variant was scoped out of this delivery entirely in favor "
        "of the full-cluster primary variant, given the latter already has a real cross-checkable precedent (E2's S3).",
        "- Each system uses a different estimator; differences are empirical performance comparisons, not an "
        "exact additive decomposition of information loss.",
    ]
    with open(os.path.join(OUT_DIR, "report.md"), "w") as f:
        f.write("\n".join(lines))

    report["sections_completed"] = ["systemA", "systemB", "systemC", "systemD", "main_comparison", "experiment_A",
                                      "experiment_B", "experiment_C_margins", "experiment_D", "experiment_E", "report.md"]
    report["total_runtime_seconds"] = time.perf_counter() - t_start
    save_results(os.path.join(OUT_DIR, "e2geom1_summary.json"), report)
    print(f"\nAll results saved under {OUT_DIR}")
    print(f"Total runtime: {report['total_runtime_seconds']/60:.1f} min")
    print("No existing E1/E2/TRI1/NC1 outputs were modified.")


if __name__ == "__main__":
    main()
