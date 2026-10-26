"""E2-NC1 -- Neighbor-Centroid Conditional Compression. Builds directly on
E2-TRI1 (run_e2tri1_anchor_conditioned_compression.py); extends the E1/E2
series without modifying any existing output (writes only to
results/stage3_e2nc1_neighbor_centroid/).

Central hypothesis: a datastore entry's SECOND-nearest DIME centroid identity
(discrete, not a distance) may carry query-conditioned token information
DIME-V1's static P(y|c) discards -- testable with far cheaper storage than
E2-TRI1's continuous anchor-distance statistics, since it needs only sparse
integer-keyed count tables, no new floats per (cluster, token).

Reuses exactly, not reimplemented: cache loading, seed=42 MiniBatchKMeans
clustering, matched config (k=20, tau=2.0, alpha=0.25), knn.py/mixing.py
retrieval, e1_metrics.py helpers, run_e2_information_allocation's
build_cluster_to_members, AND -- new in this script -- E2-TRI1's own anchor-
conditioning functions (imported directly from run_e2tri1_anchor_conditioned_compression.py)
to reconstruct its real K=8 result for Experiment F's discrete-vs-continuous
comparison, verified against the real persisted e2tri1_summary.json before
being trusted for anything downstream (same precedent E2.1 used for
reconstructing E2's retained-member sets).

No scikit-learn in any NEW computation (count-table construction, smoothing,
held-out log-likelihood, conditional-MI estimate are all NumPy/Counter). The
only sklearn usage is the pre-existing MiniBatchKMeans clustering, reused
unchanged.

=====================================================================
Scoping decisions (stated up front, per established practice):
=====================================================================
  - Smoothing strength lambda and hybrid-blend gamma are each tuned via a
    small grid on the FULL controller_train split (90,170 queries) --
    unlike E2-TRI1's anchor statistics, these are O(1) dict lookups per
    query, cheap enough that a subsample isn't needed here.
  - Experiment C's conditional-MI estimate is reported two ways: a naive
    pooled plug-in estimate (explicitly flagged as upward-biased, built from
    the same data used to fit the count table) AND a 50/50 held-out
    log-likelihood-improvement check (P_lambda(y|c1,c2) vs. the P(y|c1)
    frequency-only baseline, fit and evaluated on disjoint halves) -- the
    spec's own "avoid trusting naive mutual-information estimates... use
    held-out estimates" instruction. Per-cluster detail is reported for the
    15 largest-population clusters; the pooled estimate covers everything.
  - Ablation grid (Experiment D): N0 (raw, existing), N1 (DIME-V1,
    existing), N2 (primary+c2, gamma=1), N3 (primary+c3 alternative,
    gamma=1), N4 (primary+c2,c3 joint, gamma=1), N5 (primary+c2, 10x
    stronger smoothing than N2's tuned lambda), N6 (primary+c2, hybrid
    fallback with tuned gamma<1), N7 (E2-TRI1 K=8, RECONSTRUCTED from its
    persisted hyperparameters and verified to reproduce its real mean NLL
    before use), plus a shuffled-c2-identity negative control (c2 permuted
    within each c1 group, breaking the (c2,y) association while preserving
    c1's and c2's marginal frequencies -- the same permutation convention
    used for Experiment C's held-out check's negative-control intuition).
  - Difficult-query tail reuses the exact worst-1/5/10% val-query convention
    (stable argsort of matched L_DIME - L_raw on val itself) used by every
    prior E1C.2/E2/E2.1/E2-TRI1 script.
  - Decoded token examples (where c2 identity flips the predicted
    preference) are drawn from the same 15 largest-population clusters used
    throughout, bounded to 20 examples.

Smoke testing: set E2NC1_SMOKE=1 with E2NC1_CACHE_DIR / E2NC1_OUT_DIR /
E2NC1_N_CLUSTERS / E2NC1_TRI1_SUMMARY pointing at a tiny synthetic cache and
a matching tiny TRI1 summary before running -- same convention E2.1/E2-TRI1
established. Never set it for the real run.
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
from e1_metrics import raw_standalone_prob, dime_standalone_prob, js_divergence_dict, retrieval_coverage
from run_e2_information_allocation import build_cluster_to_members
from run_e2tri1_anchor_conditioned_compression import (
    build_centroid_distance_matrix, select_anchors_nearest, select_anchors_random, select_anchors_diverse,
    compute_diff_features, compute_cluster_token_stats, build_cluster_stat_arrays, anchor_standalone_prob,
    nll_from_standalone as tri1_nll_from_standalone)
from gpt_class import FrozenGPT2

SMOKE = os.environ.get("E2NC1_SMOKE") == "1"
CACHE_DIR = os.environ.get("E2NC1_CACHE_DIR", os.path.join(SCRIPT_DIR, "..", "GPT_Module", "cache"))
OUT_DIR = os.environ.get("E2NC1_OUT_DIR", os.path.join(SCRIPT_DIR, "results", "stage3_e2nc1_neighbor_centroid"))
PLOT_DIR = os.path.join(OUT_DIR, "plots")
os.makedirs(PLOT_DIR, exist_ok=True)
TRI1_SUMMARY_PATH = os.environ.get("E2NC1_TRI1_SUMMARY",
                                     os.path.join(SCRIPT_DIR, "results", "stage3_e2tri1_anchor_conditioned", "e2tri1_summary.json"))

N_CLUSTERS = int(os.environ.get("E2NC1_N_CLUSTERS", "3811"))
MATCHED = {"k": 20, "tau": 2.0, "alpha": 0.25}
SEED = 42
DIM, DTYPE_BYTES, VALUE_BYTES = 768, 4, 4

LAMBDA_GRID = [1.0, 10.0, 50.0, 200.0]
STRONGER_SMOOTH_MULTIPLIER = 10.0
GAMMA_GRID = [0.25, 0.5, 0.75, 1.0]
N_SAMPLE_CLUSTERS = 15
HELD_OUT_SEED = 43
WORST_PCTS = [1, 5, 10]
N_PREF_EXAMPLES = 20
VERBOSE_EVERY = 20000 if not SMOKE else 200


# ============================================================
# Count-table construction (generalized to 1 or 2 "other" columns)
# ============================================================
def build_cj_table(c1_arr, other_cols, values):
    """other_cols: list of int arrays, each same length as c1_arr. Returns
    dict {(c1, *others): Counter({y: count})}."""
    table = {}
    n = len(c1_arr)
    for i in range(n):
        key = (int(c1_arr[i]),) + tuple(int(col[i]) for col in other_cols)
        y = int(values[i])
        bucket = table.get(key)
        if bucket is None:
            bucket = Counter()
            table[key] = bucket
        bucket[y] += 1
    return table


def shuffle_within_c1(c1_arr, c2_arr, n_clusters, rng):
    """Permutes c2 among entries sharing the same c1 -- breaks the (c2,y)
    association while preserving c1's and c2's marginal frequencies exactly
    (the required negative control)."""
    shuffled = c2_arr.copy()
    order = np.argsort(c1_arr, kind="stable")
    sorted_c1 = c1_arr[order]
    boundaries = np.searchsorted(sorted_c1, np.arange(n_clusters + 1))
    for c in range(n_clusters):
        lo, hi = boundaries[c], boundaries[c + 1]
        if hi - lo > 1:
            positions = order[lo:hi]
            shuffled[positions] = rng.permutation(shuffled[positions])
    return shuffled


# ============================================================
# Scoring
# ============================================================
def nc1_only_standalone_prob(true_values, idx_dime, d_dime, tau, cj_table, cluster_totals, compressed_dists,
                               lam, neighbor_ranks, verbose_every=VERBOSE_EVERY):
    """P_NC1(y*|q): the primary centroid's contribution replaced by
    P_lambda(y*|c1, *neighbors), every other retrieved centroid's
    contribution left as plain DIME-V1. NOT yet blended with DIME-V1 via
    gamma -- that's the caller's job (p_final = (1-gamma)*p_dime + gamma*p_nc1)."""
    N, k = idx_dime.shape
    w = softmax_np(-d_dime / tau, axis=1)
    p = np.zeros(N, dtype=np.float64)
    t0 = time.perf_counter()
    for i in range(N):
        true_tok = int(true_values[i])
        c1 = int(idx_dime[i, 0])
        total_c1 = cluster_totals[c1]
        p_dime_c1_true = (compressed_dists[c1].get(true_tok, 0) / total_c1) if total_c1 > 0 else 0.0
        key = (c1,) + tuple(int(idx_dime[i, r]) for r in neighbor_ranks)
        cj_counter = cj_table.get(key)
        N_cjy = cj_counter.get(true_tok, 0) if cj_counter else 0
        N_cj = sum(cj_counter.values()) if cj_counter else 0
        p_lam = (N_cjy + lam * p_dime_c1_true) / (N_cj + lam)
        acc = w[i, 0] * p_lam
        for j in range(1, k):
            c = int(idx_dime[i, j])
            total_c = cluster_totals[c]
            p_c_true = (compressed_dists[c].get(true_tok, 0) / total_c) if total_c > 0 else 0.0
            acc += w[i, j] * p_c_true
        p[i] = acc
        if verbose_every and (i + 1) % verbose_every == 0:
            elapsed = time.perf_counter() - t0
            print(f"      progress: {i+1}/{N}  ({(i+1)/elapsed:.1f} q/s, ETA {(N-i-1)/((i+1)/elapsed)/60:.1f} min)")
    return p


def nll_from_standalone(p_true, p_lm_true, alpha):
    p_mixed = alpha * p_true + (1 - alpha) * p_lm_true
    return -np.log(p_mixed + 1e-12)


# ============================================================
# Experiment C -- held-out log-likelihood improvement + naive plug-in MI
# ============================================================
def held_out_nll_improvement(c1_arr, c2_arr, values, lam, sample_clusters, rng):
    n = len(c1_arr)
    is_eval = rng.random(n) < 0.5
    train_idx, eval_idx = np.where(~is_eval)[0], np.where(is_eval)[0]
    cj_table_train = build_cj_table(c1_arr[train_idx], [c2_arr[train_idx]], values[train_idx])
    c1_table_train = {}
    for i in train_idx:
        c1 = int(c1_arr[i])
        bucket = c1_table_train.get(c1)
        if bucket is None:
            bucket = Counter()
            c1_table_train[c1] = bucket
        bucket[int(values[i])] += 1

    per_cluster_records = {}
    ll_lambda_sum, ll_c1only_sum = 0.0, 0.0
    for i in eval_idx:
        c1, c2, y = int(c1_arr[i]), int(c2_arr[i]), int(values[i])
        c1_counter = c1_table_train.get(c1, Counter())
        total_c1 = sum(c1_counter.values())
        p_c1 = (c1_counter.get(y, 0) / total_c1) if total_c1 > 0 else 1e-6
        cj_counter = cj_table_train.get((c1, c2))
        N_cjy = cj_counter.get(y, 0) if cj_counter else 0
        N_cj = sum(cj_counter.values()) if cj_counter else 0
        p_lam = (N_cjy + lam * p_c1) / (N_cj + lam)
        ll_lam, ll_c1 = np.log(max(p_lam, 1e-12)), np.log(max(p_c1, 1e-12))
        ll_lambda_sum += ll_lam
        ll_c1only_sum += ll_c1
        per_cluster_records.setdefault(c1, []).append((ll_lam, ll_c1))

    n_eval = len(eval_idx)
    pooled = {"mean_loglik_lambda": float(ll_lambda_sum / n_eval), "mean_loglik_c1_only": float(ll_c1only_sum / n_eval),
              "improvement_nats": float((ll_lambda_sum - ll_c1only_sum) / n_eval), "n_eval": int(n_eval)}
    per_cluster = {}
    for c in sample_clusters:
        recs = per_cluster_records.get(c, [])
        if len(recs) >= 10:
            lam_ll = float(np.mean([r[0] for r in recs]))
            c1_ll = float(np.mean([r[1] for r in recs]))
            per_cluster[c] = {"mean_loglik_lambda": lam_ll, "mean_loglik_c1_only": c1_ll,
                                "improvement_nats": lam_ll - c1_ll, "n_eval": len(recs)}
        else:
            per_cluster[c] = {"n_eval": len(recs), "reason": "insufficient held-out support (<10)"}
    return pooled, per_cluster


def naive_plugin_conditional_mi(cj_table, c1_table, total_n):
    """I(Y;C2|C1), naive plug-in from the SAME data used to fit the table --
    explicitly flagged as upward-biased (not held-out), per the spec's own
    warning against trusting this alone."""
    mi = 0.0
    for c1, c1_counter in c1_table.items():
        n_c1 = sum(c1_counter.values())
        p_c1 = n_c1 / total_n
        h_y_given_c1 = -sum((cnt / n_c1) * np.log(cnt / n_c1) for cnt in c1_counter.values() if cnt > 0)
        h_y_given_c1c2 = 0.0
        for (c1k, c2k), counter in cj_table.items():
            if c1k != c1:
                continue
            n_cj = sum(counter.values())
            p_c2_given_c1 = n_cj / n_c1
            h_this = -sum((cnt / n_cj) * np.log(cnt / n_cj) for cnt in counter.values() if cnt > 0)
            h_y_given_c1c2 += p_c2_given_c1 * h_this
        mi += p_c1 * (h_y_given_c1 - h_y_given_c1c2)
    return float(mi)


# ============================================================
# Storage accounting
# ============================================================
def storage_bytes_nc1(n_clusters, compressed_dists, cj_table, use_joint=False):
    import pickle
    centroid_bytes = n_clusters * DIM * DTYPE_BYTES
    counter_bytes = len(pickle.dumps(compressed_dists))
    n_entries = sum(len(v) for v in cj_table.values())
    key_bytes = len(cj_table) * (3 if use_joint else 2) * 4
    count_bytes = n_entries * (4 + 4)  # token id (int32) + count (int32) per entry
    total = centroid_bytes + counter_bytes + key_bytes + count_bytes
    return {"centroid_bytes": int(centroid_bytes), "token_dist_bytes": int(counter_bytes),
            "cj_key_bytes": int(key_bytes), "cj_count_bytes": int(count_bytes),
            "n_cj_keys": int(len(cj_table)), "n_cj_token_entries": int(n_entries), "total_bytes": int(total)}


# ============================================================
# Main
# ============================================================
def main():
    t_start = time.perf_counter()
    print(f"E2-NC1 job started. SMOKE={SMOKE}, N_CLUSTERS={N_CLUSTERS}")
    rng = np.random.default_rng(SEED)

    print("Loading cache (datastore/controller_train/val) + TRI1's persisted summary...")
    ds = load_cache(os.path.join(CACHE_DIR, "wikitext103_gpt2_datastore.npz"))
    ct = load_cache(os.path.join(CACHE_DIR, "wikitext103_gpt2_controller_train.npz"))
    val = load_cache(os.path.join(CACHE_DIR, "wikitext103_gpt2_val.npz"))
    ds_keys, ds_values = ds["keys"], ds["values"]
    ct_keys, ct_values, ct_p_lm = ct["keys"], ct["values"], ct["p_lm_true"]
    val_keys, val_values, val_p_lm = val["keys"], val["values"], val["p_lm_true"]
    with open(TRI1_SUMMARY_PATH) as f:
        tri1_summary = json.load(f)

    gpt = FrozenGPT2(model_name="gpt2")

    print("Re-deriving DIME clusters (seed=42, deterministic, identical to every prior E1/E2/TRI1 script)...")
    km = MiniBatchKMeans(n_clusters=N_CLUSTERS, random_state=SEED)
    assignment = km.fit_predict(ds_keys)
    dime_centroids = km.cluster_centers_.astype(np.float32)
    compressed_dists = []
    for c in range(N_CLUSTERS):
        member_values = ds_values[assignment == c]
        compressed_dists.append(Counter(member_values.tolist()) if len(member_values) > 0 else Counter())
    cluster_to_members = build_cluster_to_members(assignment, N_CLUSTERS)
    cluster_totals = np.array([sum(c.values()) for c in compressed_dists])

    raw_index, raw_values = build_datastore(ds_keys, ds_values)
    dime_index, dime_values = build_datastore(dime_centroids, np.array(compressed_dists, dtype=object))

    kM, tauM, alphaM = MATCHED["k"], MATCHED["tau"], MATCHED["alpha"]
    print("Querying val/controller_train/datastore (matched config, k=20) -- raw+DIME retrieval, c1/c2/c3 assignment...")
    d_raw_val, idx_raw_val = query_knn_indices(raw_index, val_keys, k=kM)
    d_dime_val, idx_dime_val = query_knn_indices(dime_index, val_keys, k=kM)
    r_raw_val = raw_values[idx_raw_val]
    r_dime_val = dime_values[idx_dime_val]
    p_raw_true_val = raw_standalone_prob(d_raw_val, r_raw_val, val_values, tau=tauM)
    p_dime_true_val = dime_standalone_prob(d_dime_val, r_dime_val, val_values, tau=tauM)
    nll_raw_val = nll_from_standalone(p_raw_true_val, val_p_lm, alphaM)
    nll_dime_val = nll_from_standalone(p_dime_true_val, val_p_lm, alphaM)
    delta_L_val = nll_dime_val - nll_raw_val

    d_dime_ct, idx_dime_ct = query_knn_indices(dime_index, ct_keys, k=kM)
    p_dime_true_ct = dime_standalone_prob(d_dime_ct, dime_values[idx_dime_ct], ct_values, tau=tauM)

    d_dime_ds, idx_dime_ds = query_knn_indices(dime_index, ds_keys, k=3)
    mismatch_frac = float(np.mean(idx_dime_ds[:, 0] != assignment))
    print(f"  datastore top-1 (fresh query) vs. k-means fit_predict assignment mismatch: {mismatch_frac*100:.3f}%")
    assert mismatch_frac < 0.02, f"unexpectedly high mismatch ({mismatch_frac}) between k-means assignment and a fresh nearest-centroid query -- investigate before trusting c1 below"
    c1_ds, c2_ds, c3_ds = idx_dime_ds[:, 0], idx_dime_ds[:, 1], idx_dime_ds[:, 2]

    report = {"n_val": int(len(val_values)), "n_ct": int(len(ct_values)), "n_ds": int(len(ds_values)),
              "matched_config": MATCHED, "n_clusters": N_CLUSTERS, "seed": SEED, "smoke": SMOKE,
              "ds_top1_mismatch_fraction": mismatch_frac,
              "N0_raw_mean_nll": float(np.mean(nll_raw_val)), "N1_dime_mean_nll": float(np.mean(nll_dime_val))}
    save_results(os.path.join(OUT_DIR, "e2nc1_summary.json"), report)

    # ============================================================
    # Experiment A -- count tables + smoothing lambda tuning (controller_train)
    # ============================================================
    print("\nExperiment A: building (c1,c2)->y and (c1,c2,c3)->y count tables...")
    cj_table_c2 = build_cj_table(c1_ds, [c2_ds], ds_values)
    cj_table_c3 = build_cj_table(c1_ds, [c3_ds], ds_values)
    cj_table_joint = build_cj_table(c1_ds, [c2_ds, c3_ds], ds_values)
    c2_shuffled = shuffle_within_c1(c1_ds, c2_ds, N_CLUSTERS, rng)
    cj_table_shuffle = build_cj_table(c1_ds, [c2_shuffled], ds_values)
    print(f"  (c1,c2) table: {len(cj_table_c2)} distinct keys, {sum(len(v) for v in cj_table_c2.values())} (key,token) entries")
    print(f"  (c1,c3) table: {len(cj_table_c3)} distinct keys")
    print(f"  (c1,c2,c3) joint table: {len(cj_table_joint)} distinct keys")

    print("Tuning smoothing lambda on full controller_train (N2-style primary+c2, gamma=1)...")
    lam_nll = {}
    for lam in LAMBDA_GRID:
        p_ct = nc1_only_standalone_prob(ct_values, idx_dime_ct, d_dime_ct, tauM, cj_table_c2, cluster_totals,
                                          compressed_dists, lam, (1,), verbose_every=0)
        nll_ct = float(np.mean(nll_from_standalone(p_ct, ct_p_lm, alphaM)))
        lam_nll[lam] = nll_ct
        print(f"  lambda={lam}: controller_train mean NLL = {nll_ct:.4f}")
    lam_chosen = min(lam_nll, key=lam_nll.get)
    lam_stronger = lam_chosen * STRONGER_SMOOTH_MULTIPLIER
    print(f"  chosen lambda = {lam_chosen} (CT mean NLL = {lam_nll[lam_chosen]:.4f}); N5 stronger-smoothing lambda = {lam_stronger}")
    report["lambda_tuning"] = {"grid": LAMBDA_GRID, "ct_mean_nll_by_lambda": lam_nll, "chosen_lambda": lam_chosen,
                                 "n5_stronger_lambda": lam_stronger}
    save_results(os.path.join(OUT_DIR, "e2nc1_summary.json"), report)

    # ============================================================
    # Experiment B -- gamma (hybrid fallback) tuning for N6
    # ============================================================
    print("\nTuning hybrid-blend gamma on full controller_train (N6)...")
    p_nc1_ct = nc1_only_standalone_prob(ct_values, idx_dime_ct, d_dime_ct, tauM, cj_table_c2, cluster_totals,
                                          compressed_dists, lam_chosen, (1,), verbose_every=0)
    gamma_nll = {}
    for gamma in GAMMA_GRID:
        p_hybrid = (1 - gamma) * p_dime_true_ct + gamma * p_nc1_ct
        gamma_nll[gamma] = float(np.mean(nll_from_standalone(p_hybrid, ct_p_lm, alphaM)))
        print(f"  gamma={gamma}: controller_train mean NLL = {gamma_nll[gamma]:.4f}")
    gamma_chosen = min(gamma_nll, key=gamma_nll.get)
    print(f"  chosen gamma = {gamma_chosen} (CT mean NLL = {gamma_nll[gamma_chosen]:.4f})")
    report["gamma_tuning"] = {"grid": GAMMA_GRID, "ct_mean_nll_by_gamma": gamma_nll, "chosen_gamma": gamma_chosen}
    save_results(os.path.join(OUT_DIR, "e2nc1_summary.json"), report)

    # ============================================================
    # Experiment D -- main ablation grid N0-N7 + shuffled control, full val scale
    # ============================================================
    print("\nExperiment D: ablation grid (N2-N6 + shuffle), full val...")
    grid = {"N0_raw": {"mean_nll": float(np.mean(nll_raw_val))}, "N1_dime": {"mean_nll": float(np.mean(nll_dime_val))}}

    p_n2 = nc1_only_standalone_prob(val_values, idx_dime_val, d_dime_val, tauM, cj_table_c2, cluster_totals,
                                      compressed_dists, lam_chosen, (1,))
    grid["N2_primary_c2"] = {"mean_nll": float(np.mean(nll_from_standalone(p_n2, val_p_lm, alphaM)))}
    print(f"  N2 (primary+c2, gamma=1): val mean NLL = {grid['N2_primary_c2']['mean_nll']:.4f}")
    save_results(os.path.join(OUT_DIR, "e2nc1_summary.json"), {**report, "experiment_D_grid": grid})

    p_n3 = nc1_only_standalone_prob(val_values, idx_dime_val, d_dime_val, tauM, cj_table_c3, cluster_totals,
                                      compressed_dists, lam_chosen, (2,))
    grid["N3_primary_c3"] = {"mean_nll": float(np.mean(nll_from_standalone(p_n3, val_p_lm, alphaM)))}
    print(f"  N3 (primary+c3 alternative): val mean NLL = {grid['N3_primary_c3']['mean_nll']:.4f}")

    p_n4 = nc1_only_standalone_prob(val_values, idx_dime_val, d_dime_val, tauM, cj_table_joint, cluster_totals,
                                      compressed_dists, lam_chosen, (1, 2))
    grid["N4_primary_c2_c3_joint"] = {"mean_nll": float(np.mean(nll_from_standalone(p_n4, val_p_lm, alphaM)))}
    print(f"  N4 (primary+c2,c3 joint): val mean NLL = {grid['N4_primary_c2_c3_joint']['mean_nll']:.4f}")
    save_results(os.path.join(OUT_DIR, "e2nc1_summary.json"), {**report, "experiment_D_grid": grid})

    p_n5 = nc1_only_standalone_prob(val_values, idx_dime_val, d_dime_val, tauM, cj_table_c2, cluster_totals,
                                      compressed_dists, lam_stronger, (1,))
    grid["N5_primary_c2_stronger_smoothing"] = {"mean_nll": float(np.mean(nll_from_standalone(p_n5, val_p_lm, alphaM))),
                                                  "lambda": lam_stronger}
    print(f"  N5 (primary+c2, stronger smoothing, lambda={lam_stronger}): val mean NLL = "
          f"{grid['N5_primary_c2_stronger_smoothing']['mean_nll']:.4f}")

    p_hybrid_val = (1 - gamma_chosen) * p_dime_true_val + gamma_chosen * p_n2
    grid["N6_primary_c2_hybrid"] = {"mean_nll": float(np.mean(nll_from_standalone(p_hybrid_val, val_p_lm, alphaM))),
                                      "gamma": gamma_chosen}
    print(f"  N6 (primary+c2, hybrid gamma={gamma_chosen}): val mean NLL = {grid['N6_primary_c2_hybrid']['mean_nll']:.4f}")

    p_shuffle = nc1_only_standalone_prob(val_values, idx_dime_val, d_dime_val, tauM, cj_table_shuffle, cluster_totals,
                                           compressed_dists, lam_chosen, (1,))
    grid["N_shuffle_control"] = {"mean_nll": float(np.mean(nll_from_standalone(p_shuffle, val_p_lm, alphaM)))}
    print(f"  shuffled-c2 negative control: val mean NLL = {grid['N_shuffle_control']['mean_nll']:.4f} "
          f"(should sit close to N1_dime if the formula isn't just an artifact of smoothing itself)")
    save_results(os.path.join(OUT_DIR, "e2nc1_summary.json"), {**report, "experiment_D_grid": grid})

    # ============================================================
    # N7 -- reconstruct E2-TRI1's real K=8 config, verify, then evaluate on val
    # ============================================================
    print("\nReconstructing E2-TRI1's K=8 config from its persisted hyperparameters, verifying before use...")
    tri1_method = tri1_summary["chosen_anchor_method"]
    tri1_best_K = tri1_summary["best_K_by_ct_subsample"]
    hp = tri1_summary["hyperparameter_tuning"]
    base_K = hp["z_std_reference_K"]
    N0_tri1, var_floor_tri1 = hp["chosen_N0"], hp["var_floor"]
    tau_z_tri1 = tri1_summary["experiment_D_grid"][f"A{3 + [1,2,4,8,16].index(tri1_best_K)}_diff_K{tri1_best_K}"]["tau_z"]

    dist_matrix = build_centroid_distance_matrix(dime_centroids)
    if tri1_method == "nearest":
        tri1_anchors = {c: select_anchors_nearest(dist_matrix, c, tri1_best_K) for c in range(N_CLUSTERS)}
    elif tri1_method == "diverse":
        tri1_anchors = {c: select_anchors_diverse(dist_matrix, c, tri1_best_K) for c in range(N_CLUSTERS)}
    else:
        print(f"  WARNING: TRI1's chosen method was 'random' -- this reconstruction replays "
              f"select_anchors_random with a freshly-seeded rng in cluster-ID order to match TRI1's own call "
              f"sequence as closely as possible, but exact RNG-sequence reproduction across separate script runs "
              f"is NOT guaranteed (unlike 'nearest'/'diverse', which are fully deterministic from dist_matrix "
              f"alone). N7's reconstruction-verification check below will catch a mismatch if this occurs.")
        tri1_rng = np.random.default_rng(SEED)
        tri1_anchors = {c: select_anchors_random(N_CLUSTERS, c, tri1_best_K, tri1_rng) for c in range(N_CLUSTERS)}
    tri1_fn = lambda H, c: compute_diff_features(H, c, tri1_anchors[c], dime_centroids)
    tri1_stats = compute_cluster_token_stats(ds_keys, ds_values, cluster_to_members, tri1_fn, N0_tri1, var_floor_tri1)
    tri1_arrays = build_cluster_stat_arrays(tri1_stats, compressed_dists)
    p_tri1_val = anchor_standalone_prob(val_keys, val_values, idx_dime_val, d_dime_val, tauM, tri1_arrays, tri1_fn,
                                          False, tau_z_tri1, verbose_every=0)
    nll_tri1_val = tri1_nll_from_standalone(p_tri1_val, val_p_lm, alphaM)
    reconstructed_mean_nll = float(np.mean(nll_tri1_val))
    persisted_mean_nll = tri1_summary["experiment_D_grid"][f"A{3 + [1,2,4,8,16].index(tri1_best_K)}_diff_K{tri1_best_K}"]["mean_nll"]
    recon_diff = abs(reconstructed_mean_nll - persisted_mean_nll)
    print(f"  reconstructed TRI1 K={tri1_best_K} mean NLL = {reconstructed_mean_nll:.6f} vs. persisted = {persisted_mean_nll:.6f} "
          f"(abs diff = {recon_diff:.6f})")
    if not SMOKE:
        assert recon_diff < 1e-3, "TRI1 reconstruction does not match its persisted real result -- do not trust N7/Experiment F below"
    grid["N7_tri1_reference"] = {"mean_nll": reconstructed_mean_nll, "persisted_mean_nll": persisted_mean_nll,
                                   "reconstruction_abs_diff": recon_diff}
    report["experiment_D_grid"] = grid
    save_results(os.path.join(OUT_DIR, "e2nc1_summary.json"), report)

    fig, ax = plt.subplots(figsize=(8, 5))
    labels_plot = ["N0_raw", "N1_dime", "N2_primary_c2", "N3_primary_c3", "N4_primary_c2_c3_joint",
                   "N5_primary_c2_stronger_smoothing", "N6_primary_c2_hybrid", "N_shuffle_control", "N7_tri1_reference"]
    vals_plot = [grid[l]["mean_nll"] for l in labels_plot]
    ax.bar(range(len(labels_plot)), vals_plot)
    ax.set_xticks(range(len(labels_plot)))
    ax.set_xticklabels(labels_plot, rotation=55, ha="right", fontsize=7)
    ax.set_ylabel("mean interpolated NLL (matched config, val)")
    ax.set_title("Experiment D: ablation grid")
    fig.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, "experiment_D_ablation_grid.png"), dpi=130)
    plt.close(fig)

    # ============================================================
    # Experiment C -- conditional-MI diagnostics (naive + held-out)
    # ============================================================
    print("\nExperiment C: does c2 identity carry information about y beyond c1?...")
    populations = cluster_totals.copy()
    sample_clusters = [int(c) for c in np.argsort(-populations)[:N_SAMPLE_CLUSTERS]]

    c1_table_full = {}
    for i in range(len(c1_ds)):
        c1 = int(c1_ds[i])
        bucket = c1_table_full.get(c1)
        if bucket is None:
            bucket = Counter()
            c1_table_full[c1] = bucket
        bucket[int(ds_values[i])] += 1
    naive_mi = naive_plugin_conditional_mi(cj_table_c2, c1_table_full, len(ds_values))
    print(f"  naive pooled plug-in I(Y;C2|C1) = {naive_mi:.5f} nats (UPWARD-BIASED -- same data used to fit the table)")

    rng_ho = np.random.default_rng(HELD_OUT_SEED)
    pooled_ho, per_cluster_ho = held_out_nll_improvement(c1_ds, c2_ds, ds_values, lam_chosen, sample_clusters, rng_ho)
    print(f"  held-out check: mean log-lik with c2 = {pooled_ho['mean_loglik_lambda']:.4f}, "
          f"c1-only = {pooled_ho['mean_loglik_c1_only']:.4f}, improvement = {pooled_ho['improvement_nats']:.5f} nats/token "
          f"(n_eval={pooled_ho['n_eval']})")
    report["experiment_C_conditional_mi"] = {"naive_pooled_plugin_mi_nats": naive_mi, "held_out_pooled": pooled_ho,
                                                "held_out_per_sampled_cluster": per_cluster_ho,
                                                "sampled_clusters": sample_clusters}
    save_results(os.path.join(OUT_DIR, "e2nc1_summary.json"), report)

    # ============================================================
    # Section 9 -- distributional metrics + preference-change examples
    # ============================================================
    print("\nDistributional metrics (correct-token coverage) for N0/N1/N2...")
    cov_raw = retrieval_coverage(p_raw_true_val)
    cov_dime = retrieval_coverage(p_dime_true_val)
    cov_n2 = retrieval_coverage(p_n2)
    dist_metrics = {"correct_token_coverage_raw": cov_raw, "correct_token_coverage_dime": cov_dime,
                      "correct_token_coverage_n2": cov_n2}
    print(f"  coverage: raw={cov_raw:.4f}, DIME-V1={cov_dime:.4f}, N2={cov_n2:.4f}")
    report["distributional_metrics"] = dist_metrics

    print(f"Decoded-token examples where c2 identity changes the preferred token (up to {N_PREF_EXAMPLES})...")
    pref_examples = []
    for c in sample_clusters:
        top_tokens = [t for t, _ in compressed_dists[c].most_common(2)]
        if len(top_tokens) < 2:
            continue
        yi, yj = top_tokens
        total_c1 = cluster_totals[c]
        p_dime_i, p_dime_j = compressed_dists[c].get(yi, 0) / total_c1, compressed_dists[c].get(yj, 0) / total_c1
        for (c1k, c2k), counter in cj_table_c2.items():
            if c1k != c or (c2k) == c:
                continue
            total_cj = sum(counter.values())
            if total_cj < 5:
                continue
            p_i = (counter.get(yi, 0) + lam_chosen * p_dime_i) / (total_cj + lam_chosen)
            p_j = (counter.get(yj, 0) + lam_chosen * p_dime_j) / (total_cj + lam_chosen)
            dime_prefers_i = p_dime_i > p_dime_j
            nc1_prefers_i = p_i > p_j
            if dime_prefers_i != nc1_prefers_i:
                pref_examples.append({
                    "centroid": c, "second_nearest_centroid": int(c2k),
                    "token_i": int(yi), "decoded_i": gpt.tokenizer.decode([int(yi)]),
                    "token_j": int(yj), "decoded_j": gpt.tokenizer.decode([int(yj)]),
                    "p_dime_i": float(p_dime_i), "p_dime_j": float(p_dime_j),
                    "p_nc1_given_c2_i": float(p_i), "p_nc1_given_c2_j": float(p_j), "n_cj_observations": int(total_cj),
                })
                if len(pref_examples) >= N_PREF_EXAMPLES:
                    break
        if len(pref_examples) >= N_PREF_EXAMPLES:
            break
    report["preference_change_examples"] = pref_examples
    print(f"  found {len(pref_examples)} preference-reversal examples")
    save_results(os.path.join(OUT_DIR, "e2nc1_summary.json"), report)

    # ============================================================
    # Difficult-query tail (reuses E1C.2/E2/TRI1 worst-decile convention on val)
    # ============================================================
    print("\nDifficult-query tail: worst-1/5/10% val queries by matched delta_L (DIME - raw)...")
    order = np.argsort(-delta_L_val, kind="stable")
    tail_results = {}
    for pct in WORST_PCTS:
        n_worst = max(1, int(len(order) * pct / 100))
        worst_idx = order[:n_worst]
        tail_results[pct] = {
            "n_queries": int(n_worst),
            "mean_nll_raw": float(np.mean(nll_raw_val[worst_idx])),
            "mean_nll_dime": float(np.mean(nll_dime_val[worst_idx])),
            "mean_nll_n2": float(np.mean(nll_from_standalone(p_n2[worst_idx], val_p_lm[worst_idx], alphaM))),
            "mean_nll_n7_tri1": float(np.mean(nll_tri1_val[worst_idx])),
        }
        print(f"  worst-{pct}%: raw={tail_results[pct]['mean_nll_raw']:.4f}, dime={tail_results[pct]['mean_nll_dime']:.4f}, "
              f"N2={tail_results[pct]['mean_nll_n2']:.4f}, N7(TRI1)={tail_results[pct]['mean_nll_n7_tri1']:.4f}")
    report["difficult_query_tail"] = tail_results
    save_results(os.path.join(OUT_DIR, "e2nc1_summary.json"), report)

    # ============================================================
    # Experiment F -- discrete (N2) vs. continuous (N7/TRI1) comparison
    # ============================================================
    print("\nExperiment F: discrete (N2) vs. continuous (N7/TRI1 K=8) comparison...")
    nll_n2_val = nll_from_standalone(p_n2, val_p_lm, alphaM)
    improved_by_n2 = nll_n2_val < nll_dime_val - 1e-9
    improved_by_n7 = nll_tri1_val < nll_dime_val - 1e-9
    both = improved_by_n2 & improved_by_n7
    only_n2 = improved_by_n2 & ~improved_by_n7
    only_n7 = improved_by_n7 & ~improved_by_n2
    neither = ~improved_by_n2 & ~improved_by_n7
    overlap = {"n_improved_by_both": int(both.sum()), "n_improved_by_n2_only": int(only_n2.sum()),
                "n_improved_by_n7_only": int(only_n7.sum()), "n_improved_by_neither": int(neither.sum()),
                "n_total": int(len(val_values)),
                "jaccard_overlap": float(both.sum() / max(1, (improved_by_n2 | improved_by_n7).sum()))}
    print(f"  {overlap}")
    storage_n2 = storage_bytes_nc1(N_CLUSTERS, compressed_dists, cj_table_c2, use_joint=False)
    tri1_storage = tri1_summary["compression_efficiency"][f"anchor_mean_only_K{tri1_best_K}_bytes"]
    dime_v1_storage = tri1_summary["compression_efficiency"]["dime_v1_bytes"]
    raw_storage = tri1_summary["compression_efficiency"]["raw_kNN_bytes"]
    experiment_F = {"per_query_overlap": overlap, "n2_storage_bytes": storage_n2,
                      "tri1_k8_storage_bytes": tri1_storage, "dime_v1_storage_bytes": dime_v1_storage,
                      "raw_storage_bytes": raw_storage,
                      "n2_mean_nll": float(np.mean(nll_n2_val)), "n7_tri1_mean_nll": reconstructed_mean_nll}
    report["experiment_F_discrete_vs_continuous"] = experiment_F
    print(f"  N2 storage: {storage_n2['total_bytes']} bytes vs. TRI1 K=8: {tri1_storage['total_bytes']} bytes "
          f"vs. DIME-V1: {dime_v1_storage} bytes")
    save_results(os.path.join(OUT_DIR, "e2nc1_summary.json"), report)

    # ============================================================
    # Experiment E -- storage efficiency (all configs)
    # ============================================================
    print("\nExperiment E: storage accounting for all NC1 configs...")
    storage_all = {
        "N2_primary_c2": storage_n2,
        "N3_primary_c3": storage_bytes_nc1(N_CLUSTERS, compressed_dists, cj_table_c3, use_joint=False),
        "N4_primary_c2_c3_joint": storage_bytes_nc1(N_CLUSTERS, compressed_dists, cj_table_joint, use_joint=True),
        "dime_v1_bytes": dime_v1_storage, "tri1_k8_bytes": tri1_storage["total_bytes"], "raw_bytes": raw_storage,
    }
    print(f"  {storage_all}")
    report["storage_efficiency"] = storage_all
    save_results(os.path.join(OUT_DIR, "e2nc1_summary.json"), report)

    # ============================================================
    # Failure mechanisms (Section 10)
    # ============================================================
    print("\nFailure mechanism evidence...")
    cj_entry_sizes = [sum(c.values()) for c in cj_table_c2.values()]
    sparse_frac = float(np.mean([s < 5 for s in cj_entry_sizes])) if cj_entry_sizes else None
    failure_evidence = {
        "fraction_cj_keys_with_fewer_than_5_observations": sparse_frac,
        "n2_vs_shuffle_mean_nll_diff": float(grid["N2_primary_c2"]["mean_nll"] - grid["N_shuffle_control"]["mean_nll"]),
        "n2_vs_dime_mean_nll_diff": float(grid["N2_primary_c2"]["mean_nll"] - grid["N1_dime"]["mean_nll"]),
        "note": "a near-zero n2_vs_shuffle_mean_nll_diff would indicate N2's apparent gain over DIME-V1 (if any) is "
                "an artifact of the smoothing formula itself, not genuine (c2,y) association information.",
    }
    print(f"  {failure_evidence}")
    report["failure_mechanism_evidence"] = failure_evidence
    save_results(os.path.join(OUT_DIR, "e2nc1_summary.json"), report)

    # ============================================================
    # Markdown report
    # ============================================================
    print("\nGenerating report.md...")
    lines = [
        "# DIME Stage 3 -- E2-NC1: Neighbor-Centroid Conditional Compression",
        "",
        f"Matched config (k={MATCHED['k']}, tau={MATCHED['tau']}, alpha={MATCHED['alpha']}), "
        f"{len(val_values)} val queries, {N_CLUSTERS} clusters. Chosen lambda={lam_chosen}, chosen gamma={gamma_chosen}.",
        "",
        "## Experiment D -- ablation grid (mean NLL, matched config, val)",
    ]
    for label, d in grid.items():
        lines.append(f"- {label}: {d['mean_nll']:.4f}")
    lines += [
        "",
        "## Experiment C -- does c2 carry information about y beyond c1?",
        f"Naive pooled plug-in I(Y;C2|C1) = {naive_mi:.5f} nats (upward-biased, not held-out).",
        f"Held-out check: mean log-lik with c2 = {pooled_ho['mean_loglik_lambda']:.4f} vs. c1-only = "
        f"{pooled_ho['mean_loglik_c1_only']:.4f} -- improvement = {pooled_ho['improvement_nats']:.5f} nats/token "
        f"(n_eval={pooled_ho['n_eval']}).",
        "",
        "## Experiment F -- discrete (N2) vs. continuous (N7/TRI1 K=8)",
        f"Per-query improvement overlap vs. DIME-V1: {overlap}",
        f"Storage: N2={storage_n2['total_bytes']} bytes, TRI1 K=8={tri1_storage['total_bytes']} bytes, "
        f"DIME-V1={dime_v1_storage} bytes, raw={raw_storage} bytes.",
        "",
        "## Difficult-query tail",
    ]
    for pct, d in tail_results.items():
        lines.append(f"- worst-{pct}%: raw={d['mean_nll_raw']:.4f}, DIME-V1={d['mean_nll_dime']:.4f}, "
                      f"N2={d['mean_nll_n2']:.4f}, N7(TRI1)={d['mean_nll_n7_tri1']:.4f}")
    lines += [
        "",
        "## Failure mechanism evidence",
        f"{failure_evidence}",
        "",
        "## Answers to the research questions (from the measurements above only)",
        f"**Q1** (does c2 identity carry predictive information beyond c1?): held-out improvement = "
        f"{pooled_ho['improvement_nats']:.5f} nats/token; naive plug-in MI = {naive_mi:.5f} nats.",
        "**Q2** (does this recover lost context-token associations?): compare N1 (DIME-V1) against N2-N6 above.",
        f"**Q3** (how much NLL improvement from discrete identity alone?): N2 mean NLL = {grid['N2_primary_c2']['mean_nll']:.4f} "
        f"vs. N1 = {grid['N1_dime']['mean_nll']:.4f}.",
        f"**Q4** (better memory/fidelity trade-off than TRI1?): see Experiment F's storage and NLL comparison above.",
        f"**Q5** (does c2 add info beyond DIME's existing multi-centroid weights?): compare N2 against the shuffled "
        f"control (N_shuffle_control, mean NLL={grid['N_shuffle_control']['mean_nll']:.4f}) -- shuffle breaks the "
        f"(c2,y) association while preserving marginals, isolating what the existing weighting alone could achieve.",
        "**Q6** (are improvements concentrated in specific regions?): see the difficult-query tail and per-cluster "
        "held-out table in the JSON summary.",
        "**Q7** (would finer neighbor rankings/distances be justified?): compare N2/N3/N4 against N7(TRI1) above -- "
        "judge directly from the measured numbers, not pre-decided here.",
        "",
        "## Recommendation",
        "Judge directly from the Experiment D grid, Experiment F's storage/overlap comparison, and the held-out "
        "Experiment C check above whether E2-NC1 should replace, complement, or be discarded relative to E2-TRI1 -- "
        "not asserted here in advance of the real numbers.",
        "",
        "## Limitations",
        "- Smoothing lambda and hybrid gamma tuned via a small grid, not a continuous search.",
        f"- Experiment C's per-cluster held-out detail limited to the {N_SAMPLE_CLUSTERS} largest-population clusters; "
        "the pooled estimate covers all clusters.",
        f"- Preference-change examples bounded to {N_PREF_EXAMPLES} cases across the same sampled clusters.",
        "- N7's TRI1 reference is a RECONSTRUCTION from TRI1's persisted hyperparameters, verified to reproduce its "
        "real mean NLL within tolerance before use, not a re-run of the original script.",
        "- Correlation/overlap analyses in Experiment F do not establish which mechanism causes any given query's change.",
    ]
    with open(os.path.join(OUT_DIR, "report.md"), "w") as f:
        f.write("\n".join(lines))

    report["sections_completed"] = ["experiment_A", "experiment_B_gamma", "experiment_D", "experiment_C",
                                      "distributional_metrics", "preference_change_examples", "difficult_query_tail",
                                      "experiment_F", "experiment_E_storage", "failure_mechanisms", "report.md"]
    report["total_runtime_seconds"] = time.perf_counter() - t_start
    save_results(os.path.join(OUT_DIR, "e2nc1_summary.json"), report)
    print(f"\nAll results saved under {OUT_DIR}")
    print(f"Total runtime: {report['total_runtime_seconds']/60:.1f} min")
    print("No existing E1/E2/TRI1 outputs were modified.")


if __name__ == "__main__":
    main()
