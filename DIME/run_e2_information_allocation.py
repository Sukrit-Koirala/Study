"""E2 -- Information Allocation Under a Fixed Compression Budget. Extends the
E1 series without modifying any existing output (writes only to
results/stage3_e2_information_allocation/).

Scoping decisions (stated here, not silently applied -- the spec explicitly
permits this: "if the full budget/strategy/seed grid is computationally
excessive, prioritize a complete, controlled matched-configuration comparison
over a large but incomplete parameter sweep"):
  - Budgets: 1%, 5%, 10% of raw embedding storage (not the full 0.5/1/2/5/10%
    set) -- still spans a meaningful tradeoff curve.
  - Within-cluster selection: `nearest_centroid` (deterministic) is the PRIMARY
    method used across all three strategies for the headline comparison, per
    the spec's own "use the same within-cluster member-selection method...in
    the primary comparison" instruction. `random` (3 seeds) is compared
    against it only within Strategy A, as a secondary analysis. Geometrically-
    diverse and token-label-diverse selection are not implemented this round.
  - Selective criteria: cluster-usage-frequency (via controller_train retrieval
    counts) and controller-train predictive-sensitivity -- 2 of the 4 listed
    criteria. Entropy and geometric-dispersion criteria are deferred.
  - Matched config (k=20,tau=2.0,alpha=0.25) gets the full 77,089-query
    evaluation (the declared primary comparison) -- this is ALL that's
    implemented in this delivery. A tuned-config secondary pass was planned
    but is NOT implemented here, stated plainly rather than left as a
    misleading claim or silent omission; it's a reasonable, scoped follow-up
    given how much the matched-config primary comparison alone already covers.

Part A -- common member-aware prediction rule:
  P_c^member(y|q) = [sum_{i in R_c} w_i(q) * 1[v_i=y]] / [sum_{i in R_c} w_i(q)]
  P_c^hybrid(y|q) = (1-beta)*P_c^DIME(y) + beta*P_c^member(y|q)
  where w_i(q) uses the SAME distance kernel as S3 (softmax(-d/tau) over just
  the retained members R_c). beta is a SINGLE GLOBAL constant, tuned once on
  controller_train (grid over {0.25,0.5,0.75,1.0}, picking whichever minimizes
  controller_train NLL at a small reference budget), then held FIXED and
  common across every strategy/budget combination in the primary comparison --
  per the explicit "keep the same estimator across allocation strategies"
  instruction. When R_c is empty, P_c^member is undefined (0/0); the code
  falls back to beta_effective=0 automatically in that case, which is exactly
  the required "zero retained members reproduces original DIME" property, not
  a special case bolted on afterward.
"""
import sys, os, json, time, pickle
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "GPT_Module"))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.cluster import MiniBatchKMeans

from cache_io import load_cache
from knn import build_datastore, query_knn_indices, softmax_np, mix_knn_and_lm
from mixing import mix_dime_and_lm
from results_io import save_results
from e1_metrics import raw_standalone_prob, dime_standalone_prob, js_divergence_dict, retrieval_coverage, distribution_shape_stats

CACHE_DIR = os.path.join(os.path.dirname(__file__), "..", "GPT_Module", "cache")
OUT_DIR = os.path.join(os.path.dirname(__file__), "results", "stage3_e2_information_allocation")
PLOT_DIR = os.path.join(OUT_DIR, "plots")
os.makedirs(PLOT_DIR, exist_ok=True)

N_CLUSTERS = 3811
MATCHED = {"k": 20, "tau": 2.0, "alpha": 0.25}
BUDGET_FRACTIONS = [0.01, 0.05, 0.10]  # of raw embedding storage
BETA_GRID = [0.25, 0.5, 0.75, 1.0]
RANDOM_SEEDS = [42, 43, 44]
N_TUNED_SAMPLE = 3000
N_BOOT = 2000
DIM, DTYPE_BYTES, VALUE_BYTES = 768, 4, 4
WORST_PCTS = [1, 5, 10]


def quantiles(x):
    x = np.asarray(x, dtype=np.float64)
    if len(x) == 0:
        return None
    return {"mean": float(np.mean(x)), "median": float(np.median(x)),
            "p90": float(np.percentile(x, 90)), "p95": float(np.percentile(x, 95)), "p99": float(np.percentile(x, 99))}


def build_cluster_to_members(assignment, n_clusters):
    order = np.argsort(assignment, kind="stable")
    sorted_assign = assignment[order]
    boundaries = np.searchsorted(sorted_assign, np.arange(n_clusters + 1))
    return {c: order[boundaries[c]:boundaries[c + 1]] for c in range(n_clusters)}


def ratio_with_stability_guard(numerator, denominator, min_denom=0.01):
    """Same lesson learned in E1C.2: guard both sign and NLL-appropriate scale,
    not just a tiny epsilon -- a non-positive or too-small denominator makes
    the ratio uninterpretable, not merely noisy."""
    stable = denominator > min_denom
    return (float(numerator / denominator * 100) if stable else None,
            "stable" if stable else f"UNDEFINED: denominator ({denominator:.6f}) non-positive or too small relative to NLL-gap scale")


def select_members_nearest_centroid(members, centroid, ds_keys, m):
    if len(members) <= m:
        return members
    d = np.linalg.norm(ds_keys[members] - centroid[None, :], axis=1)
    return members[np.argsort(d)[:m]]


def select_members_random(members, m, rng):
    if len(members) <= m:
        return members
    return rng.choice(members, size=m, replace=False)


def hybrid_standalone_prob(query_keys, true_values, idx_dime, d_dime, tau, beta, retained_members_per_cluster,
                            ds_keys, ds_values, compressed_dists, verbose_every=20000):
    """P_c^hybrid, true-token probability only (this is all NLL needs). Falls
    back to pure DIME (beta_effective=0) whenever a selected cluster has no
    retained members -- the required zero-budget-reproduces-DIME property."""
    N, k = idx_dime.shape
    w_dime = softmax_np(-d_dime / tau, axis=1)
    p_hybrid = np.zeros(N, dtype=np.float64)
    t0 = time.perf_counter()
    for i in range(N):
        q_key = query_keys[i]
        true_tok = int(true_values[i])
        acc = 0.0
        for j in range(k):
            c = idx_dime[i, j]
            w_c = float(w_dime[i, j])
            counter = compressed_dists[c]
            total = sum(counter.values())
            p_dime_c = (counter.get(true_tok, 0) / total) if total > 0 else 0.0

            retained = retained_members_per_cluster.get(c)
            if retained is None or len(retained) == 0:
                p_c = p_dime_c  # beta_effective = 0, exactly reproduces DIME for this cluster
            else:
                d_members = np.linalg.norm(ds_keys[retained] - q_key[None, :], axis=1)
                w_members = softmax_np(-d_members / tau, axis=-1)
                match = ds_values[retained] == true_tok
                p_member_c = float(w_members[match].sum()) if match.any() else 0.0
                p_c = (1 - beta) * p_dime_c + beta * p_member_c
            acc += w_c * p_c
        p_hybrid[i] = acc
        if verbose_every and (i + 1) % verbose_every == 0:
            elapsed = time.perf_counter() - t0
            print(f"      progress: {i+1}/{N}  ({(i+1)/elapsed:.1f} q/s, ETA {(N-i-1)/((i+1)/elapsed)/60:.1f} min)")
    return p_hybrid


def allocate_uniform(cluster_to_members, dime_centroids, ds_keys, budget_vectors, selection_fn):
    nonempty = [c for c in cluster_to_members if len(cluster_to_members[c]) > 0]
    m_per_cluster = max(1, budget_vectors // max(1, len(nonempty)))
    retained = {}
    total = 0
    for c in nonempty:
        if total >= budget_vectors:
            break
        sel = selection_fn(cluster_to_members[c], c, m_per_cluster)
        retained[c] = sel
        total += len(sel)
    return retained, total


def allocate_selective(cluster_to_members, priority_order, budget_vectors, selection_fn, max_per_cluster=50):
    retained = {}
    total = 0
    for c in priority_order:
        if total >= budget_vectors:
            break
        members = cluster_to_members[c]
        if len(members) == 0:
            continue
        take = min(max_per_cluster, len(members), budget_vectors - total)
        sel = selection_fn(members, c, take)
        retained[c] = sel
        total += len(sel)
    return retained, total


def allocate_hybrid(cluster_to_members, priority_order, dime_centroids, ds_keys, budget_vectors, selection_fn, baseline_per_cluster=1):
    nonempty = [c for c in cluster_to_members if len(cluster_to_members[c]) > 0]
    retained = {}
    total = 0
    for c in nonempty:
        sel = selection_fn(cluster_to_members[c], c, baseline_per_cluster)
        retained[c] = sel
        total += len(sel)
    remaining = max(0, budget_vectors - total)
    if remaining > 0:
        for c in priority_order:
            if remaining <= 0:
                break
            members = cluster_to_members[c]
            already = len(retained.get(c, []))
            if len(members) <= already:
                continue
            extra_available = len(members) - already
            take_extra = min(extra_available, remaining, 50)
            extra_sel = selection_fn(members, c, already + take_extra)
            merged = np.unique(np.concatenate([retained.get(c, np.array([], dtype=int)), extra_sel]))
            added = len(merged) - already
            retained[c] = merged
            total += added
            remaining -= added
    return retained, total


def storage_bytes_hybrid_method(n_clusters, compressed_dists, dime_centroids_bytes_per, n_retained_total):
    centroid_bytes = n_clusters * dime_centroids_bytes_per
    counter_bytes = len(pickle.dumps(compressed_dists))
    retained_embedding_bytes = n_retained_total * DIM * DTYPE_BYTES
    retained_token_bytes = n_retained_total * VALUE_BYTES
    routing_bytes = n_retained_total * 8  # lookup/metadata estimate, stated assumption
    total = centroid_bytes + counter_bytes + retained_embedding_bytes + retained_token_bytes + routing_bytes
    return {"centroid_bytes": int(centroid_bytes), "token_dist_bytes": int(counter_bytes),
            "retained_embedding_bytes": int(retained_embedding_bytes), "retained_token_bytes": int(retained_token_bytes),
            "routing_metadata_bytes": int(routing_bytes), "n_retained_total": int(n_retained_total),
            "total_bytes": int(total)}


def main():
    print("Loading cache (datastore/controller_train/val) + real persisted E1C tuned per-query data...")
    ds = load_cache(os.path.join(CACHE_DIR, "wikitext103_gpt2_datastore.npz"))
    ct = load_cache(os.path.join(CACHE_DIR, "wikitext103_gpt2_controller_train.npz"))
    val = load_cache(os.path.join(CACHE_DIR, "wikitext103_gpt2_val.npz"))
    with open("results/wikitext103_gpt2_tier1.json") as f:
        tier1 = json.load(f)
    raw_tuned, dime_tuned = tier1["raw_knn"]["best_config"], tier1["dime_minibatch_kmeans"]["best_config"]
    assert tier1["n_clusters"] == N_CLUSTERS
    ds_keys, ds_values = ds["keys"], ds["values"]
    ct_keys, ct_values, ct_p_lm = ct["keys"], ct["values"], ct["p_lm_true"]
    val_keys, val_values, val_p_lm = val["keys"], val["values"], val["p_lm_true"]
    print(f"datastore={len(ds_values)}  controller_train={len(ct_values)}  val={len(val_values)}")

    print("Re-deriving DIME clusters (seed=42, deterministic, identical to every prior E1-series script)...")
    km = MiniBatchKMeans(n_clusters=N_CLUSTERS, random_state=42)
    assignment = km.fit_predict(ds_keys)
    dime_centroids = km.cluster_centers_.astype(np.float32)
    compressed_dists = []
    for c in range(N_CLUSTERS):
        member_values = ds_values[assignment == c]
        compressed_dists.append(Counter(member_values.tolist()) if len(member_values) > 0 else Counter())
    compressed_dists = np.array(compressed_dists, dtype=object)
    cluster_to_members = build_cluster_to_members(assignment, N_CLUSTERS)
    bytes_raw_total = len(ds_values) * DIM * DTYPE_BYTES
    bytes_dime_total = N_CLUSTERS * DIM * DTYPE_BYTES + len(pickle.dumps(compressed_dists))

    raw_index, raw_values = build_datastore(ds_keys, ds_values)
    dime_index, dime_values = build_datastore(dime_centroids, compressed_dists)

    kr, taur, alphar = MATCHED["k"], MATCHED["tau"], MATCHED["alpha"]
    kd, taud, alphad = MATCHED["k"], MATCHED["tau"], MATCHED["alpha"]

    print("Computing S1/S2 baselines (val, full scale)...")
    d_raw_v, idx_raw_v = query_knn_indices(raw_index, val_keys, k=kr)
    d_dime_v, idx_dime_v = query_knn_indices(dime_index, val_keys, k=kd)
    r_raw_v, r_dime_v = raw_values[idx_raw_v], dime_values[idx_dime_v]
    _, L_raw_val = mix_knn_and_lm(d_raw_v, r_raw_v, val_values, val_p_lm, tau=taur, alpha=alphar)
    _, L_dime_val = mix_dime_and_lm(d_dime_v, r_dime_v, val_values, val_p_lm, tau=taud, alpha=alphad)

    print("Computing S1/S2 baselines (controller_train, for beta-tuning and cluster-priority ranking)...")
    d_raw_ct, idx_raw_ct = query_knn_indices(raw_index, ct_keys, k=kr)
    d_dime_ct, idx_dime_ct = query_knn_indices(dime_index, ct_keys, k=kd)
    r_raw_ct, r_dime_ct = raw_values[idx_raw_ct], dime_values[idx_dime_ct]
    _, L_raw_ct = mix_knn_and_lm(d_raw_ct, r_raw_ct, ct_values, ct_p_lm, tau=taur, alpha=alphar)
    _, L_dime_ct = mix_dime_and_lm(d_dime_ct, r_dime_ct, ct_values, ct_p_lm, tau=taud, alpha=alphad)

    print("Computing S3 (full member-aware) on val, matched config -- the uncompressed reference for R_S3...")
    def compute_sys3(idx_dime_arr, d_dime_arr, keys, values, tau):
        w = softmax_np(-d_dime_arr / tau, axis=1)
        N, k = idx_dime_arr.shape
        p = np.zeros(N, dtype=np.float64)
        for i in range(N):
            q_key, true_tok = keys[i], int(values[i])
            acc = 0.0
            for j in range(k):
                c = idx_dime_arr[i, j]
                members = cluster_to_members[c]
                if len(members) == 0:
                    continue
                d_m = np.linalg.norm(ds_keys[members] - q_key[None, :], axis=1)
                w_m = softmax_np(-d_m / tau, axis=-1)
                match = ds_values[members] == true_tok
                if match.any():
                    acc += float(w[i, j]) * float(w_m[match].sum())
            p[i] = acc
        return p
    p_s3_val = compute_sys3(idx_dime_v, d_dime_v, val_keys, val_values, taud)
    p_mixed_s3 = alphad * p_s3_val + (1 - alphad) * val_p_lm
    L_S3_val = -np.log(p_mixed_s3 + 1e-12)
    print(f"  mean L_S1={L_raw_val.mean():.4f}  L_S2={L_dime_val.mean():.4f}  L_S3={L_S3_val.mean():.4f}")

    # ============================================================
    # Part A regression checks
    # ============================================================
    print("Validation: zero-budget hybrid == DIME; full-budget (beta=1, all members) hybrid == S3...")
    p_hybrid_zero = hybrid_standalone_prob(val_keys[:200], val_values[:200], idx_dime_v[:200], d_dime_v[:200],
                                             taud, beta=0.5, retained_members_per_cluster={}, ds_keys=ds_keys,
                                             ds_values=ds_values, compressed_dists=compressed_dists, verbose_every=0)
    p_dime_check = dime_standalone_prob(d_dime_v[:200], r_dime_v[:200], val_values[:200], tau=taud)
    zero_budget_ok = np.allclose(p_hybrid_zero, p_dime_check, atol=1e-9)
    print(f"  zero-budget == DIME: {zero_budget_ok} (max abs diff {np.abs(p_hybrid_zero - p_dime_check).max():.2e})")

    full_members = {c: cluster_to_members[c] for c in np.unique(idx_dime_v[:50].reshape(-1))}
    p_hybrid_full = hybrid_standalone_prob(val_keys[:50], val_values[:50], idx_dime_v[:50], d_dime_v[:50],
                                             taud, beta=1.0, retained_members_per_cluster=full_members, ds_keys=ds_keys,
                                             ds_values=ds_values, compressed_dists=compressed_dists, verbose_every=0)
    p_s3_check = compute_sys3(idx_dime_v[:50], d_dime_v[:50], val_keys[:50], val_values[:50], taud)
    full_budget_ok = np.allclose(p_hybrid_full, p_s3_check, atol=1e-9)
    print(f"  full-budget (beta=1) == S3: {full_budget_ok} (max abs diff {np.abs(p_hybrid_full - p_s3_check).max():.2e})")
    assert zero_budget_ok and full_budget_ok, "common prediction rule failed a required reconstruction property -- stop and investigate"

    print("Tuning beta on controller_train (small reference budget: 1 nearest-centroid member/cluster)...")
    ref_budget_vectors = int(0.01 * len(ds_values))
    ref_retained, _ = allocate_uniform(cluster_to_members, dime_centroids, ds_keys, ref_budget_vectors,
                                         lambda members, c, m: select_members_nearest_centroid(members, dime_centroids[c], ds_keys, m))
    beta_search = []
    for beta in BETA_GRID:
        p_h_ct = hybrid_standalone_prob(ct_keys, ct_values, idx_dime_ct, d_dime_ct, taud, beta, ref_retained,
                                          ds_keys, ds_values, compressed_dists, verbose_every=30000)
        p_mixed_ct = alphad * p_h_ct + (1 - alphad) * ct_p_lm
        nll_ct = float((-np.log(p_mixed_ct + 1e-12)).mean())
        beta_search.append({"beta": beta, "controller_train_nll": nll_ct})
        print(f"  beta={beta}: controller_train NLL={nll_ct:.4f}")
    BETA = min(beta_search, key=lambda r: r["controller_train_nll"])["beta"]
    print(f"Selected FIXED beta={BETA} (common across every strategy/budget below)")

    report = {"n_val": int(len(val_values)), "n_controller_train": int(len(ct_values)), "matched_config": MATCHED,
              "budget_fractions": BUDGET_FRACTIONS, "beta_tuning": {"grid": beta_search, "selected_beta": BETA},
              "validation": {"zero_budget_equals_dime": bool(zero_budget_ok), "full_budget_equals_s3": bool(full_budget_ok)},
              "baselines": {"mean_L_raw_val": float(L_raw_val.mean()), "mean_L_dime_val": float(L_dime_val.mean()),
                             "mean_L_S3_val": float(L_S3_val.mean())}}
    save_results(os.path.join(OUT_DIR, "e2_summary.json"), report)

    # ============================================================
    # Cluster priority rankings for Strategy B/C (controller_train ONLY, no val leakage)
    # ============================================================
    print("Computing cluster priority rankings (controller_train only)...")
    usage_counts = np.bincount(idx_dime_ct.reshape(-1), minlength=N_CLUSTERS)
    priority_usage = np.argsort(-usage_counts).tolist()

    cluster_sens = np.zeros(N_CLUSTERS)
    cluster_hit_counts = np.zeros(N_CLUSTERS)
    with np.errstate(invalid="ignore"):
        delta_L_ct = L_dime_ct - L_raw_ct
    for i in range(len(ct_values)):
        top_c = idx_dime_ct[i, 0]
        cluster_sens[top_c] += max(0.0, delta_L_ct[i])
        cluster_hit_counts[top_c] += 1
    with np.errstate(invalid="ignore", divide="ignore"):
        cluster_sens_mean = np.where(cluster_hit_counts > 0, cluster_sens / np.maximum(cluster_hit_counts, 1), 0.0)
    priority_sensitivity = np.argsort(-cluster_sens_mean).tolist()
    print(f"  usage-based and sensitivity-based priority rankings computed "
          f"(attribution: positive controller_train delta_L assigned to each query's TOP-1 nearest cluster only -- "
          f"a simplified, stated attribution rule, not a causal decomposition across all k retrieved clusters)")

    def nearest_centroid_fn(members, c, m):
        return select_members_nearest_centroid(members, dime_centroids[c], ds_keys, m)

    def eval_method(name, retained, n_total_vectors, eval_keys, eval_values, eval_p_lm, idx_dime_arr, d_dime_arr, tau, beta, label):
        print(f"    evaluating {label} ({len(eval_keys)} queries)...")
        p_h = hybrid_standalone_prob(eval_keys, eval_values, idx_dime_arr, d_dime_arr, tau, beta, retained,
                                       ds_keys, ds_values, compressed_dists)
        p_mixed = alphad * p_h + (1 - alphad) * eval_p_lm
        L_h = -np.log(p_mixed + 1e-12)
        return p_h, L_h

    results_grid = {}
    for budget_frac in BUDGET_FRACTIONS:
        budget_vectors = int(budget_frac * len(ds_values))
        print(f"\n################ BUDGET: {budget_frac*100:.0f}% of raw storage (~{budget_vectors} vectors) ################")
        budget_report = {}

        strategies = {
            "uniform": lambda: allocate_uniform(cluster_to_members, dime_centroids, ds_keys, budget_vectors, nearest_centroid_fn),
            "selective_usage": lambda: allocate_selective(cluster_to_members, priority_usage, budget_vectors, nearest_centroid_fn),
            "selective_sensitivity": lambda: allocate_selective(cluster_to_members, priority_sensitivity, budget_vectors, nearest_centroid_fn),
            "hybrid": lambda: allocate_hybrid(cluster_to_members, priority_sensitivity, dime_centroids, ds_keys, budget_vectors, nearest_centroid_fn),
        }
        for strat_name, alloc_fn in strategies.items():
            print(f"  --- strategy: {strat_name} ---")
            retained, n_total = alloc_fn()
            storage = storage_bytes_hybrid_method(N_CLUSTERS, compressed_dists, DIM * DTYPE_BYTES, n_total)
            p_h_val, L_h_val = eval_method(strat_name, retained, n_total, val_keys, val_values, val_p_lm,
                                             idx_dime_v, d_dime_v, taud, BETA, f"{strat_name}@{budget_frac}")

            coverage = retrieval_coverage(p_h_val)
            with np.errstate(invalid="ignore"):
                delta_vs_dime = L_h_val - L_dime_val
            both_finite = np.isfinite(L_h_val) & np.isfinite(L_dime_val)
            improves = both_finite & (delta_vs_dime < 0)
            degrades = both_finite & (delta_vs_dime > 0)

            r_s3_num = L_dime_val.mean() - L_h_val.mean()
            r_s3_denom = L_dime_val.mean() - L_S3_val.mean()
            r_s3_pct, r_s3_interp = ratio_with_stability_guard(r_s3_num, r_s3_denom)
            r_s3_boot = None
            rng_b = np.random.default_rng(42)
            n = len(val_values)
            boot_vals = []
            for _ in range(N_BOOT):
                idx_b = rng_b.integers(0, n, size=n)
                denom_b = L_dime_val[idx_b].mean() - L_S3_val[idx_b].mean()
                if denom_b > 0.01:
                    boot_vals.append((L_dime_val[idx_b].mean() - L_h_val[idx_b].mean()) / denom_b * 100)
            if len(boot_vals) > N_BOOT * 0.5:
                r_s3_boot = {"ci_95_low": float(np.percentile(boot_vals, 2.5)), "ci_95_high": float(np.percentile(boot_vals, 97.5)),
                             "n_stable_draws": len(boot_vals)}

            method_report = {
                "n_retained_vectors": int(n_total), "storage": storage,
                "compression_ratio_vs_raw": bytes_raw_total / storage["total_bytes"],
                "overhead_vs_dime_pct": (storage["total_bytes"] - bytes_dime_total) / bytes_dime_total * 100,
                "mean_interpolated_nll": float(L_h_val.mean()), "coverage": coverage,
                "p_true": quantiles(p_h_val),
                "frac_improves_vs_dime": float(improves.sum() / both_finite.sum()) if both_finite.sum() else None,
                "frac_degrades_vs_dime": float(degrades.sum() / both_finite.sum()) if both_finite.sum() else None,
                "mean_improvement_magnitude": float(-delta_vs_dime[improves].mean()) if improves.sum() else None,
                "mean_degradation_magnitude": float(delta_vs_dime[degrades].mean()) if degrades.sum() else None,
                "r_s3_recovery_pct": r_s3_pct, "r_s3_interpretability": r_s3_interp, "r_s3_bootstrap_ci": r_s3_boot,
            }
            print(f"    n_retained={n_total}  bytes={storage['total_bytes']/1e6:.3f}MB  "
                  f"comp_ratio={method_report['compression_ratio_vs_raw']:.1f}x  mean_nll={method_report['mean_interpolated_nll']:.4f}  "
                  f"R_S3={r_s3_pct}  ({r_s3_interp})")
            budget_report[strat_name] = method_report

        results_grid[budget_frac] = budget_report
        report["results_grid"] = results_grid
        save_results(os.path.join(OUT_DIR, "e2_summary.json"), report)

    # ============================================================
    # Strategy A secondary: random (3 seeds) vs nearest_centroid, at each budget
    # ============================================================
    print("\n################ Strategy A secondary: random vs nearest_centroid ################")
    secondary = {}
    for budget_frac in BUDGET_FRACTIONS:
        budget_vectors = int(budget_frac * len(ds_values))
        random_results = []
        for seed in RANDOM_SEEDS:
            rng_sel = np.random.default_rng(seed)
            retained, n_total = allocate_uniform(cluster_to_members, dime_centroids, ds_keys, budget_vectors,
                                                    lambda members, c, m: select_members_random(members, m, rng_sel))
            p_h, L_h = eval_method("uniform_random", retained, n_total, val_keys, val_values, val_p_lm,
                                     idx_dime_v, d_dime_v, taud, BETA, f"uniform_random seed={seed}@{budget_frac}")
            random_results.append({"seed": seed, "mean_nll": float(L_h.mean()), "n_retained": int(n_total)})
        mean_nlls = [r["mean_nll"] for r in random_results]
        secondary[budget_frac] = {
            "random_across_seeds": random_results,
            "random_mean_nll_mean": float(np.mean(mean_nlls)), "random_mean_nll_std": float(np.std(mean_nlls)),
            "nearest_centroid_mean_nll": results_grid[budget_frac]["uniform"]["mean_interpolated_nll"],
        }
        print(f"  budget {budget_frac*100:.0f}%: random mean_nll={secondary[budget_frac]['random_mean_nll_mean']:.4f}"
              f"+/-{secondary[budget_frac]['random_mean_nll_std']:.4f}  "
              f"nearest_centroid={secondary[budget_frac]['nearest_centroid_mean_nll']:.4f}")
    report["strategyA_secondary_selection_comparison"] = secondary
    save_results(os.path.join(OUT_DIR, "e2_summary.json"), report)

    # ============================================================
    # Part E -- worst-query-group analysis (reuse E1C's established groups, not re-ranked)
    # ============================================================
    print("\n################ Part E: worst-query-group recovery ################")
    with np.errstate(invalid="ignore"):
        delta_L_matched_baseline = L_dime_val - L_raw_val
    order_matched = np.argsort(-delta_L_matched_baseline, kind="stable")
    worst_masks = {}
    for pct in WORST_PCTS:
        n = max(1, round(len(val_values) * pct / 100))
        mask = np.zeros(len(val_values), dtype=bool)
        mask[order_matched[:n]] = True
        worst_masks[pct] = mask

    worst_query_report = {}
    for budget_frac, budget_report in results_grid.items():
        worst_query_report[budget_frac] = {}
        for strat_name in budget_report:
            retained_fn = {
                "uniform": lambda: allocate_uniform(cluster_to_members, dime_centroids, ds_keys, int(budget_frac * len(ds_values)), nearest_centroid_fn),
                "selective_usage": lambda: allocate_selective(cluster_to_members, priority_usage, int(budget_frac * len(ds_values)), nearest_centroid_fn),
                "selective_sensitivity": lambda: allocate_selective(cluster_to_members, priority_sensitivity, int(budget_frac * len(ds_values)), nearest_centroid_fn),
                "hybrid": lambda: allocate_hybrid(cluster_to_members, priority_sensitivity, dime_centroids, ds_keys, int(budget_frac * len(ds_values)), nearest_centroid_fn),
            }[strat_name]
            retained, n_total = retained_fn()
            p_h, L_h = eval_method(strat_name, retained, n_total, val_keys, val_values, val_p_lm, idx_dime_v, d_dime_v, taud, BETA, "worst-query-recompute")
            per_pct = {}
            with np.errstate(invalid="ignore"):
                dL_vs_raw_method = L_h - L_raw_val
                dL_vs_raw_dime = L_dime_val - L_raw_val
            for pct, mask in worst_masks.items():
                pos_loss_dime = float(dL_vs_raw_dime[mask][dL_vs_raw_dime[mask] > 0].sum())
                pos_loss_method = float(dL_vs_raw_method[mask][dL_vs_raw_method[mask] > 0].sum())
                per_pct[pct] = {
                    "n": int(mask.sum()), "mean_nll_method": float(L_h[mask].mean()), "mean_nll_dime": float(L_dime_val[mask].mean()),
                    "frac_positive_loss_repaired": float((pos_loss_dime - pos_loss_method) / pos_loss_dime) if pos_loss_dime > 0 else None,
                    "frac_improved_vs_dime": float((L_h[mask] < L_dime_val[mask]).mean()),
                    "remaining_positive_degradation_p95_p99": {
                        95: float(np.percentile(dL_vs_raw_method[mask][dL_vs_raw_method[mask] > 0], 95)) if (dL_vs_raw_method[mask] > 0).any() else None,
                        99: float(np.percentile(dL_vs_raw_method[mask][dL_vs_raw_method[mask] > 0], 99)) if (dL_vs_raw_method[mask] > 0).any() else None,
                    },
                }
            worst_query_report[budget_frac][strat_name] = per_pct
            print(f"  budget={budget_frac*100:.0f}% strat={strat_name}: "
                  f"worst-10% repaired={per_pct[10]['frac_positive_loss_repaired']}")
    report["partE_worst_query_recovery"] = worst_query_report
    save_results(os.path.join(OUT_DIR, "e2_summary.json"), report)

    # ============================================================
    # Plots
    # ============================================================
    print("\nGenerating plots...")
    fig, ax = plt.subplots(figsize=(8, 6))
    for strat_name in ["uniform", "selective_usage", "selective_sensitivity", "hybrid"]:
        xs = [results_grid[b][strat_name]["storage"]["total_bytes"] / 1e6 for b in BUDGET_FRACTIONS]
        ys = [results_grid[b][strat_name]["mean_interpolated_nll"] for b in BUDGET_FRACTIONS]
        ax.plot(xs, ys, "o-", label=strat_name)
    ax.axhline(L_dime_val.mean(), color="gray", linestyle="--", label="DIME (0 extra bytes)")
    ax.axhline(L_S3_val.mean(), color="black", linestyle=":", label="S3 (uncompressed reference)")
    ax.set_xlabel("total stored bytes (MB)"); ax.set_ylabel("mean interpolated NLL")
    ax.set_title("NLL vs. actual stored bytes")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, "nll_vs_bytes.png"), dpi=130)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 6))
    for strat_name in ["uniform", "selective_usage", "selective_sensitivity", "hybrid"]:
        xs = [results_grid[b][strat_name]["compression_ratio_vs_raw"] for b in BUDGET_FRACTIONS]
        ys = [results_grid[b][strat_name]["r_s3_recovery_pct"] for b in BUDGET_FRACTIONS]
        ax.plot(xs, ys, "o-", label=strat_name)
    ax.set_xlabel("compression ratio vs. raw"); ax.set_ylabel("% of S3 improvement recovered")
    ax.set_title("Fraction of S3 recovery vs. compression ratio")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, "s3_recovery_vs_compression.png"), dpi=130)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 6))
    for strat_name in ["uniform", "selective_usage", "selective_sensitivity", "hybrid"]:
        xs = [b * 100 for b in BUDGET_FRACTIONS]
        ys = [worst_query_report[b][strat_name][10]["frac_positive_loss_repaired"] for b in BUDGET_FRACTIONS]
        ax.plot(xs, ys, "o-", label=strat_name)
    ax.set_xlabel("storage budget (% of raw)"); ax.set_ylabel("fraction of worst-10% positive loss repaired")
    ax.set_title("Worst-10% positive-loss repaired vs. storage budget")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, "worst10_repair_vs_budget.png"), dpi=130)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 6))
    budgets_pct = [b * 100 for b in BUDGET_FRACTIONS]
    for name, label in [("random_mean_nll_mean", "random"), ("nearest_centroid_mean_nll", "nearest_centroid")]:
        ys = [secondary[b][name] for b in BUDGET_FRACTIONS]
        if name == "random_mean_nll_mean":
            errs = [secondary[b]["random_mean_nll_std"] for b in BUDGET_FRACTIONS]
            ax.errorbar(budgets_pct, ys, yerr=errs, fmt="o-", label=label)
        else:
            ax.plot(budgets_pct, ys, "s--", label=label)
    ax.set_xlabel("storage budget (% of raw)"); ax.set_ylabel("mean interpolated NLL")
    ax.set_title("Strategy A: random (3 seeds, +/-1 std) vs. nearest-centroid selection")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, "strategyA_selection_comparison.png"), dpi=130)
    plt.close(fig)

    print(f"\nAll E2 results saved under {OUT_DIR}")
    print("No existing E1-series outputs were modified.")


if __name__ == "__main__":
    main()
