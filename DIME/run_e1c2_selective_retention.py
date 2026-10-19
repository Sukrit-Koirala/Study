"""E1C.2 -- Selective Retention and Severe-Failure Recovery. Extends E1/E1C
without modifying any existing E1-series outputs (writes only to
results/stage3_e1c2_selective_retention/).

Data provenance, stated plainly:
  - TUNED-config per-query delta_L: loaded DIRECTLY from the real, already-
    persisted results/stage3_e1_wikitext103/metrics/per_query.csv -- no
    recomputation, exact reuse of the original E1C run's own numbers.
  - MATCHED-config per-query delta_L: that column was never persisted by the
    original E1C run (confirmed by reading its CSV header) -- recomputed here
    under the IDENTICAL deterministic pipeline (same cache, same seed=42
    clustering, same k=20/tau=2.0/alpha=0.25), which reproduces exactly what
    that run would have saved, had it saved it. Not a different ranking --
    the same deterministic computation, just not previously written to disk.
  - A third, explicitly separate "common-alpha tuned" ranking is also computed
    (tuned k/tau, but alpha=0.25 instead of each system's own tuned alpha,
    matching E1B.4's "controlled-alpha" convention) and reported only as a
    sensitivity check, never substituted for the original tuned ranking.

Part A (selective retention) and Part B (does S3 repair the worst 10%) share
setup: cache load, deterministic re-clustering (seed=42, identical to every
prior E1-series script), and -- the one new efficiency requirement here --
a precomputed cluster-to-member map (same technique as E1B.4) since Part A's
Section 8 builds several RESTRICTED raw indices and Part B needs S3's
per-cluster-member computation again (scoped this time to the worst-query
groups for the primary analysis, full-scale once for the worst-10%-vs-
remaining-90% comparison in Section 13, which genuinely needs it).

Scoping decisions, stated rather than silently applied:
  - Section 8/9's restricted-index diagnostics are evaluated on their own
    target query group in full (that's the point), plus a fixed representative
    sample of 3,000 queries (stratified by JS decile, same method as E1B.3) for
    the "additional queries" / fixed-budget-baseline generalization checks --
    not the full 77,089 for every one of the ~16 restricted indices this
    section builds, which would multiply an already large cost.
  - Storage accounting follows the same measured-bytes methodology established
    in GPT_Module's efficiency_analysis.json (`.nbytes`, `pickle.dumps` for
    Counters) wherever an exact measurement is possible; the one estimate is
    "index bytes" for brute-force NearestNeighbors, which does not build a
    separate tree structure at this scale -- so index bytes are counted as
    already included in the embedding array, not double-counted as a separate
    line item. Stated explicitly, not silently assumed.
"""
import sys, os, json, csv, time, pickle
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "GPT_Module"))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.cluster import MiniBatchKMeans
from sklearn.neighbors import NearestNeighbors

from cache_io import load_cache
from knn import build_datastore, query_knn, query_knn_indices, softmax_np, mix_knn_and_lm
from mixing import mix_dime_and_lm
from results_io import save_results
from e1_metrics import raw_standalone_prob, dime_standalone_prob, js_divergence_dict, retrieval_coverage

CACHE_DIR = os.path.join(os.path.dirname(__file__), "..", "GPT_Module", "cache")
OUT_DIR = os.path.join(os.path.dirname(__file__), "results", "stage3_e1c2_selective_retention")
PLOT_DIR = os.path.join(OUT_DIR, "plots")
os.makedirs(PLOT_DIR, exist_ok=True)

N_CLUSTERS = 3811
MATCHED = {"k": 20, "tau": 2.0, "alpha": 0.25}
WORST_PCTS = [1, 5, 10, 20]
N_GENERALIZATION_SAMPLE = 3000
N_FIXED_BUDGET_SAMPLE = 2000
N_BOOT = 2000


def quantiles(x):
    x = np.asarray(x, dtype=np.float64)
    if len(x) == 0:
        return None
    return {"mean": float(np.mean(x)), "median": float(np.median(x)),
            "p90": float(np.percentile(x, 90)), "p95": float(np.percentile(x, 95)), "p99": float(np.percentile(x, 99))}


def load_tuned_per_query():
    """Direct reuse of the real, already-persisted E1C tuned-config per-query data."""
    rows = {"position_index": [], "true_token_id": [], "p_lm_true": [], "delta_d": [], "js_divergence": [],
            "p_raw_standalone_true": [], "p_dime_standalone_true": [], "nll_lm": [],
            "nll_raw_mixed_tuned": [], "nll_dime_mixed_tuned": [], "delta_L_tuned": []}
    path = "results/stage3_e1_wikitext103/metrics/per_query.csv"
    with open(path) as f:
        reader = csv.DictReader(f)
        for r in reader:
            for k in rows:
                rows[k].append(float(r[k]))
    return {k: np.asarray(v) for k, v in rows.items()}


def paired_bootstrap_diff(a, b, n_boot=N_BOOT, seed=42):
    rng = np.random.default_rng(seed)
    n = len(a)
    boot = np.zeros(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boot[i] = (a[idx] - b[idx]).mean()
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return {"mean_diff": float((a - b).mean()), "ci_95_low": float(lo), "ci_95_high": float(hi)}


def storage_bytes_raw(n_entries, dim=768, dtype_bytes=4, value_bytes=4):
    """System A. Index bytes = embedding array itself (brute-force NearestNeighbors
    builds no separate tree structure at this scale) -- stated assumption, not
    double-counted as a separate line item."""
    embedding_bytes = n_entries * dim * dtype_bytes
    token_bytes = n_entries * value_bytes
    return {"n_entries": int(n_entries), "embedding_bytes": int(embedding_bytes), "token_label_bytes": int(token_bytes),
            "index_bytes": 0, "index_bytes_note": "brute-force NearestNeighbors stores no separate structure; "
                                                     "already counted in embedding_bytes",
            "metadata_bytes": 0, "total_bytes": int(embedding_bytes + token_bytes)}


def storage_bytes_dime(n_clusters, compressed_dists, dim=768, dtype_bytes=4):
    """System B. Measured via pickle.dumps for the Counters, same methodology as
    GPT_Module/efficiency_analysis.json."""
    centroid_bytes = n_clusters * dim * dtype_bytes
    counter_bytes = len(pickle.dumps(compressed_dists))
    return {"n_clusters": int(n_clusters), "embedding_bytes": int(centroid_bytes),
            "token_label_bytes": int(counter_bytes), "index_bytes": 0, "metadata_bytes": 0,
            "total_bytes": int(centroid_bytes + counter_bytes)}


def storage_bytes_hybrid(dime_bytes, n_retained, dim=768, dtype_bytes=4, value_bytes=4, routing_bytes_per_vector=8):
    retained_embedding_bytes = n_retained * dim * dtype_bytes
    retained_token_bytes = n_retained * value_bytes
    routing_metadata_bytes = n_retained * routing_bytes_per_vector  # supplementary index/lookup overhead, stated estimate
    total = dime_bytes["total_bytes"] + retained_embedding_bytes + retained_token_bytes + routing_metadata_bytes
    return {"n_retained_vectors": int(n_retained), "retained_embedding_bytes": int(retained_embedding_bytes),
            "retained_token_bytes": int(retained_token_bytes),
            "routing_metadata_bytes": int(routing_metadata_bytes),
            "routing_metadata_note": f"estimated at {routing_bytes_per_vector} bytes/vector for a lookup table, stated assumption",
            "dime_component_bytes": dime_bytes["total_bytes"], "total_bytes": int(total)}


def evaluate_restricted_index(restricted_keys, restricted_values, eval_keys, eval_values, eval_p_lm,
                               k, tau, alpha, full_raw_top_k_values=None):
    """Builds a NearestNeighbors index from ONLY the retained vectors and evaluates
    it -- never falls back to the full raw datastore. Returns coverage/p_true/NLL/
    reconstruction-rate stats."""
    k_eff = min(k, len(restricted_values))
    idx = NearestNeighbors(metric="euclidean").fit(restricted_keys)
    distances, neighbor_idx = idx.kneighbors(eval_keys, n_neighbors=k_eff)
    retrieved_values = restricted_values[neighbor_idx]
    p_true = raw_standalone_prob(distances, retrieved_values, eval_values, tau=tau)
    _, nll_mixed = mix_knn_and_lm(distances, retrieved_values, eval_values, eval_p_lm, tau=tau, alpha=alpha)
    result = {"coverage": retrieval_coverage(p_true), "p_true": quantiles(p_true), "mean_interpolated_nll": float(nll_mixed.mean())}
    if full_raw_top_k_values is not None:
        reconstruction = np.array([len(set(retrieved_values[i].tolist()) & set(full_raw_top_k_values[i].tolist())) / k_eff
                                    for i in range(len(eval_values))])
        result["neighbor_reconstruction_rate"] = quantiles(reconstruction)
    return result


def main():
    print("Loading cache (no extraction) + the real, persisted E1C tuned per-query data...")
    ds = load_cache(os.path.join(CACHE_DIR, "wikitext103_gpt2_datastore.npz"))
    val = load_cache(os.path.join(CACHE_DIR, "wikitext103_gpt2_val.npz"))
    with open("results/wikitext103_gpt2_tier1.json") as f:
        tier1 = json.load(f)
    raw_tuned, dime_tuned = tier1["raw_knn"]["best_config"], tier1["dime_minibatch_kmeans"]["best_config"]
    assert tier1["n_clusters"] == N_CLUSTERS
    ds_keys, ds_values = ds["keys"], ds["values"]
    val_keys, val_values, val_p_lm = val["keys"], val["values"], val["p_lm_true"]
    tuned_csv = load_tuned_per_query()
    assert len(tuned_csv["position_index"]) == len(val_values), "E1C CSV row count mismatch vs. current val cache"
    print(f"val={len(val_values)}  raw_tuned={raw_tuned}  dime_tuned={dime_tuned}  matched={MATCHED}")

    print("Re-deriving DIME clusters (seed=42, deterministic, identical to every prior E1-series script)...")
    t0 = time.perf_counter()
    km = MiniBatchKMeans(n_clusters=N_CLUSTERS, random_state=42)
    assignment = km.fit_predict(ds_keys)
    dime_centroids = km.cluster_centers_.astype(np.float32)
    compressed_dists = []
    for c in range(N_CLUSTERS):
        member_values = ds_values[assignment == c]
        compressed_dists.append(Counter(member_values.tolist()) if len(member_values) > 0 else Counter())
    compressed_dists = np.array(compressed_dists, dtype=object)
    print(f"clustering done in {time.perf_counter()-t0:.1f}s")

    raw_index, raw_values = build_datastore(ds_keys, ds_values)
    dime_index, dime_values = build_datastore(dime_centroids, compressed_dists)

    # ============================================================
    # Matched-config S1/S2 interpolated NLL -- fresh but deterministic recomputation
    # ============================================================
    print("Computing matched-config S1/S2 interpolated NLL (fresh, deterministic -- see docstring)...")
    kr, taur, alphar = MATCHED["k"], MATCHED["tau"], MATCHED["alpha"]
    kd, taud, alphad = MATCHED["k"], MATCHED["tau"], MATCHED["alpha"]
    d_raw_m, idx_raw_m = query_knn_indices(raw_index, val_keys, k=kr)
    d_dime_m, idx_dime_m = query_knn_indices(dime_index, val_keys, k=kd)
    r_raw_m = raw_values[idx_raw_m]
    r_dime_m = dime_values[idx_dime_m]
    _, L_raw_matched = mix_knn_and_lm(d_raw_m, r_raw_m, val_values, val_p_lm, tau=taur, alpha=alphar)
    _, L_dime_matched = mix_dime_and_lm(d_dime_m, r_dime_m, val_values, val_p_lm, tau=taud, alpha=alphad)
    delta_L_matched = L_dime_matched - L_raw_matched

    # ============================================================
    # Tuned-config data: DIRECTLY from the persisted CSV, no recomputation
    # ============================================================
    L_raw_tuned, L_dime_tuned, delta_L_tuned = tuned_csv["nll_raw_mixed_tuned"], tuned_csv["nll_dime_mixed_tuned"], tuned_csv["delta_L_tuned"]

    # Common-alpha sensitivity ranking: tuned k/tau, but alpha=0.25 for both, matching
    # E1B.4's controlled-alpha convention -- a SEPARATE, explicitly-labeled ranking.
    print("Computing the common-alpha (tuned k/tau) sensitivity ranking...")
    kr_t, taur_t = raw_tuned["k"], raw_tuned["tau"]
    kd_t, taud_t = dime_tuned["k"], dime_tuned["tau"]
    d_raw_t, idx_raw_t = query_knn_indices(raw_index, val_keys, k=kr_t)
    d_dime_t, idx_dime_t = query_knn_indices(dime_index, val_keys, k=kd_t)
    r_raw_t = raw_values[idx_raw_t]
    r_dime_t = dime_values[idx_dime_t]
    _, L_raw_commonalpha = mix_knn_and_lm(d_raw_t, r_raw_t, val_values, val_p_lm, tau=taur_t, alpha=0.25)
    _, L_dime_commonalpha = mix_dime_and_lm(d_dime_t, r_dime_t, val_values, val_p_lm, tau=taud_t, alpha=0.25)
    delta_L_commonalpha = L_dime_commonalpha - L_raw_commonalpha

    report = {"n_total_queries": int(len(val_values)), "matched_config": MATCHED,
              "tuned_config": {"raw": raw_tuned, "dime": dime_tuned}, "worst_percentages": WORST_PCTS,
              "data_provenance": {"tuned": "loaded directly from results/stage3_e1_wikitext103/metrics/per_query.csv",
                                   "matched": "recomputed fresh under the identical deterministic pipeline (never persisted before)",
                                   "common_alpha_sensitivity": "tuned k/tau with alpha=0.25 for both systems, separate from the original tuned ranking"}}

    configs = {
        "matched": {"delta_L": delta_L_matched, "L_raw": L_raw_matched, "L_dime": L_dime_matched,
                    "k_raw": kr, "tau_raw": taur, "alpha_raw": alphar, "k_dime": kd, "tau_dime": taud, "alpha_dime": alphad,
                    "idx_raw_full": idx_raw_m, "d_raw_full": d_raw_m},
        "tuned": {"delta_L": delta_L_tuned, "L_raw": L_raw_tuned, "L_dime": L_dime_tuned,
                  "k_raw": kr_t, "tau_raw": taur_t, "alpha_raw": raw_tuned["alpha"], "k_dime": kd_t, "tau_dime": taud_t, "alpha_dime": dime_tuned["alpha"],
                  "idx_raw_full": idx_raw_t, "d_raw_full": d_raw_t},
    }
    sensitivity = {"delta_L": delta_L_commonalpha}

    # Representative sample for restricted-index generalization checks (Section 8/9)
    js_full = tuned_csv["js_divergence"]  # only available at tuned config, used as the stratification key for both
    js_deciles = np.digitize(js_full, np.quantile(js_full, np.linspace(0, 1, 11))[1:-1])
    rng_sample = np.random.default_rng(42)
    gen_sample_idx = np.concatenate([
        rng_sample.choice(np.where(js_deciles == d)[0], size=min(N_GENERALIZATION_SAMPLE // 10, (js_deciles == d).sum()), replace=False)
        for d in range(10)
    ])
    fixed_budget_sample_idx = rng_sample.choice(len(val_values), size=N_FIXED_BUDGET_SAMPLE, replace=False)
    print(f"Restricted-index generalization sample: {len(gen_sample_idx)} queries (JS-stratified). "
          f"Fixed-budget-comparison sample: {len(fixed_budget_sample_idx)} queries (random).")

    for config_name, cfg in configs.items():
        print(f"\n################ CONFIG: {config_name} ################")
        cfg_report = {}
        delta_L, L_raw, L_dime = cfg["delta_L"], cfg["L_raw"], cfg["L_dime"]
        kr_c, taur_c, alphar_c = cfg["k_raw"], cfg["tau_raw"], cfg["alpha_raw"]
        total_positive_loss = float(delta_L[delta_L > 0].sum())

        # ============================================================
        # Section 3 -- worst-query identification (nested, deterministic, stable)
        # ============================================================
        order = np.argsort(-delta_L, kind="stable")
        n_total = len(delta_L)
        worst_groups = {}
        for pct in WORST_PCTS:
            n = max(1, round(n_total * pct / 100))
            worst_groups[pct] = order[:n]

        # ============================================================
        # Section 4 -- original vectors needed by each worst-query group
        # ============================================================
        print("Section 4: identifying retained vectors per worst-query group...")
        retention_report = {}
        retained_sets = {}
        for pct in WORST_PCTS:
            q_idx = worst_groups[pct]
            neighbor_idx_for_group = cfg["idx_raw_full"][q_idx][:, :kr_c]  # [n_q, kr_c]
            flat = neighbor_idx_for_group.reshape(-1)
            unique, counts = np.unique(flat, return_counts=True)
            retained_sets[pct] = unique
            retention_report[pct] = {
                "n_queries": int(len(q_idx)), "total_raw_neighbor_refs": int(len(flat)),
                "n_unique_vectors": int(len(unique)), "frac_of_datastore_retained": float(len(unique) / len(ds_values)),
                "dedup_ratio": float(len(flat) / len(unique)) if len(unique) else None,
                "reuse_freq": quantiles(counts.astype(np.float64)),
            }
            print(f"  worst {pct}%: n_queries={retention_report[pct]['n_queries']}  "
                  f"unique_vectors={retention_report[pct]['n_unique_vectors']}  "
                  f"frac_of_datastore={retention_report[pct]['frac_of_datastore_retained']:.5f}")
        cfg_report["section4_vector_identification"] = retention_report

        # ============================================================
        # Section 5 -- storage accounting
        # ============================================================
        bytes_raw = storage_bytes_raw(len(ds_values))
        bytes_dime = storage_bytes_dime(N_CLUSTERS, compressed_dists)
        storage_report = {"system_A_raw": bytes_raw, "system_B_dime": bytes_dime, "system_C_hybrid_per_budget": {}}
        for pct in WORST_PCTS:
            hybrid = storage_bytes_hybrid(bytes_dime, len(retained_sets[pct]))
            hybrid["compression_ratio_vs_raw"] = bytes_raw["total_bytes"] / hybrid["total_bytes"]
            hybrid["overhead_vs_dime_pct"] = (hybrid["total_bytes"] - bytes_dime["total_bytes"]) / bytes_dime["total_bytes"] * 100
            storage_report["system_C_hybrid_per_budget"][pct] = hybrid
            print(f"  worst {pct}% hybrid storage: {hybrid['total_bytes']/1e6:.3f} MB  "
                  f"compression_vs_raw={hybrid['compression_ratio_vs_raw']:.2f}x  overhead_vs_dime={hybrid['overhead_vs_dime_pct']:.2f}%")
        cfg_report["section5_storage_accounting"] = storage_report

        # ============================================================
        # Section 6 -- reuse/cluster concentration + random-group baseline
        # ============================================================
        print("Section 6: reuse frequency, cluster concentration, random-group baseline...")
        cluster_concentration = {}
        random_baseline = {}
        for pct in WORST_PCTS:
            unique = retained_sets[pct]
            clusters_of_retained = assignment[unique]
            cluster_counts = Counter(clusters_of_retained.tolist())
            cluster_sizes = np.array([len(np.where(assignment == c)[0]) for c in cluster_counts.keys()])
            retained_counts = np.array(list(cluster_counts.values()))
            cluster_concentration[pct] = {
                "n_clusters_touched": len(cluster_counts),
                "frac_of_clusters_touched": len(cluster_counts) / N_CLUSTERS,
                "mean_frac_of_cluster_retained": float((retained_counts / cluster_sizes).mean()),
                "top10_clusters_share_of_retained": float(np.sort(retained_counts)[::-1][:10].sum() / retained_counts.sum()),
            }

            # Random query group of the same size, same raw-neighbor-union logic
            rand_q_idx = rng_sample.choice(n_total, size=len(worst_groups[pct]), replace=False)
            rand_neighbor_idx = cfg["idx_raw_full"][rand_q_idx][:, :kr_c].reshape(-1)
            rand_unique = np.unique(rand_neighbor_idx)
            random_baseline[pct] = {"n_unique_vectors": int(len(rand_unique)),
                                     "frac_of_datastore_retained": float(len(rand_unique) / len(ds_values))}
            print(f"  worst {pct}%: clusters_touched={cluster_concentration[pct]['n_clusters_touched']}  "
                  f"top10_share={cluster_concentration[pct]['top10_clusters_share_of_retained']:.3f}  |  "
                  f"random-group unique_vectors={random_baseline[pct]['n_unique_vectors']} "
                  f"(worst-group: {retention_report[pct]['n_unique_vectors']})")
        cfg_report["section6_cluster_concentration"] = cluster_concentration
        cfg_report["section6_random_group_baseline"] = random_baseline

        unique_all, counts_all = np.unique(cfg["idx_raw_full"][worst_groups[20]][:, :kr_c].reshape(-1), return_counts=True)
        sorted_counts = np.sort(counts_all)[::-1]
        cumulative = np.cumsum(sorted_counts) / sorted_counts.sum()
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        axes[0].hist(counts_all, bins=100)
        axes[0].set_xlabel("reuse frequency (worst-20%% group)"); axes[0].set_ylabel("count")
        axes[0].set_title(f"[{config_name}] retained-vector reuse-frequency histogram")
        axes[1].plot(np.arange(1, len(cumulative) + 1) / len(cumulative), cumulative)
        axes[1].set_xlabel("fraction of unique retained vectors (sorted by reuse, descending)")
        axes[1].set_ylabel("cumulative fraction of neighbor references covered")
        axes[1].set_title(f"[{config_name}] cumulative reuse coverage")
        fig.tight_layout()
        fig.savefig(os.path.join(PLOT_DIR, f"{config_name}_reuse_and_cumulative.png"), dpi=130)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(7, 5))
        pcts = WORST_PCTS
        ax.plot(pcts, [retention_report[p]["frac_of_datastore_retained"] * 100 for p in pcts], "o-", label="worst-query-driven")
        ax.plot(pcts, [random_baseline[p]["frac_of_datastore_retained"] * 100 for p in pcts], "s--", label="random-query-group")
        ax.set_xlabel("worst-query percentage"); ax.set_ylabel("% of datastore retained")
        ax.set_title(f"[{config_name}] retained-datastore fraction vs. worst-query %")
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(PLOT_DIR, f"{config_name}_retained_fraction_vs_pct.png"), dpi=130)
        plt.close(fig)

        # ============================================================
        # Section 7 -- oracle predictive recovery
        # ============================================================
        print("Section 7: oracle predictive recovery...")
        oracle_report = {}
        for pct in WORST_PCTS:
            mask = np.zeros(n_total, dtype=bool)
            mask[worst_groups[pct]] = True
            L_oracle = np.where(mask, L_raw, L_dime)
            remaining_pos_loss = float(delta_L[mask & (delta_L > 0)].clip(min=0).sum())
            # oracle's own degradation relative to raw on the SWITCHED queries is 0 by construction;
            # "remaining positive degradation" below refers to the NON-switched queries' own delta_L.
            pos_loss_outside = float(delta_L[~mask][delta_L[~mask] > 0].sum())
            denom_gap = L_dime.mean() - L_raw.mean()
            # Two separate failure modes, both make the ratio uninterpretable, not just noisy:
            # (1) denom_gap <= 0 means DIME wasn't actually worse than raw overall -- there's no
            #     "gap toward raw" to recover in the first place, regardless of its magnitude.
            # (2) denom_gap small-but-positive relative to typical NLL-gap scale in this whole
            #     project (~0.03-0.35 nats) still inflates the ratio into nonsense -- 1e-4 (an
            #     E1B.4 threshold carried over without rechecking) was too small to catch this;
            #     0.01 nats is a more appropriate floor at this metric's actual scale.
            min_denom = 0.01
            denom_stable = denom_gap > min_denom
            gap_recovered_pct = float((L_dime.mean() - L_oracle.mean()) / denom_gap * 100) if denom_stable else None
            oracle_report[pct] = {
                "mean_nll_oracle": float(L_oracle.mean()),
                "improvement_vs_dime": float(L_dime.mean() - L_oracle.mean()),
                "gap_recovered_vs_raw_pct": gap_recovered_pct,
                "gap_recovered_interpretability": "stable" if denom_stable else
                    f"UNDEFINED: denominator (mean L_DIME - mean L_raw = {denom_gap:.6f}) is non-positive "
                    "or too small relative to typical NLL-gap scale to interpret as a percentage",
                "frac_queries_improved": float(mask.mean()),
                "remaining_positive_degradation_outside_group": pos_loss_outside,
                "pct_of_total_positive_loss_eliminated": float((total_positive_loss - pos_loss_outside) / total_positive_loss * 100) if total_positive_loss > 0 else None,
            }
            print(f"  oracle worst {pct}%: mean_nll={oracle_report[pct]['mean_nll_oracle']:.4f}  "
                  f"gap_recovered={oracle_report[pct]['gap_recovered_vs_raw_pct']}%  "
                  f"pos_loss_eliminated={oracle_report[pct]['pct_of_total_positive_loss_eliminated']}%")
        cfg_report["section7_oracle_recovery"] = oracle_report

        # ============================================================
        # Section 8/9 -- actual retained-subset retrieval diagnostic + fixed-budget baselines
        # ============================================================
        print("Section 8/9: restricted-index diagnostics (worst-query-driven budgets)...")
        restricted_report = {}
        gen_sample_keys, gen_sample_values, gen_sample_p_lm = val_keys[gen_sample_idx], val_values[gen_sample_idx], val_p_lm[gen_sample_idx]
        for pct in WORST_PCTS:
            unique = retained_sets[pct]
            restricted_keys, restricted_values = ds_keys[unique], ds_values[unique]
            q_idx = worst_groups[pct]
            full_top_k_on_target = cfg["idx_raw_full"][q_idx][:, :kr_c]
            full_top_k_values_on_target = raw_values[full_top_k_on_target]
            on_target = evaluate_restricted_index(restricted_keys, restricted_values, val_keys[q_idx], val_values[q_idx],
                                                    val_p_lm[q_idx], kr_c, taur_c, alphar_c, full_top_k_values_on_target)
            on_generalization = evaluate_restricted_index(restricted_keys, restricted_values, gen_sample_keys, gen_sample_values,
                                                             gen_sample_p_lm, kr_c, taur_c, alphar_c)
            restricted_report[pct] = {"on_target_queries": on_target, "on_generalization_sample": on_generalization,
                                        "n_retained": int(len(unique))}
            print(f"  worst {pct}% restricted index: on-target coverage={on_target['coverage']:.4f}  "
                  f"reconstruction_rate_mean={on_target.get('neighbor_reconstruction_rate', {}).get('mean')}  |  "
                  f"generalization-sample coverage={on_generalization['coverage']:.4f}")
        cfg_report["section8_restricted_index_diagnostic"] = restricted_report

        print("Section 9: fixed-budget baselines (random / frequency / oracle-derived selection)...")
        fixed_budget_report = {}
        fb_keys, fb_values, fb_p_lm = val_keys[fixed_budget_sample_idx], val_values[fixed_budget_sample_idx], val_p_lm[fixed_budget_sample_idx]
        # "Frequency-based" = most-retrieved across ALL queries (general retrieval popularity),
        # computed once from the full idx_raw_full array -- distinct from the oracle/worst-query selection.
        all_neighbor_idx = cfg["idx_raw_full"][:, :kr_c].reshape(-1)
        freq_unique, freq_counts = np.unique(all_neighbor_idx, return_counts=True)
        freq_order = freq_unique[np.argsort(-freq_counts)]
        for pct in WORST_PCTS:
            n_budget = len(retained_sets[pct])  # match the oracle-derived budget's vector COUNT exactly
            oracle_sel = retained_sets[pct]
            rand_sel = rng_sample.choice(len(ds_values), size=n_budget, replace=False)
            freq_sel = freq_order[:n_budget]
            budget_result = {}
            for name, sel in {"random": rand_sel, "frequency_based": freq_sel, "oracle_derived": oracle_sel}.items():
                res = evaluate_restricted_index(ds_keys[sel], ds_values[sel], fb_keys, fb_values, fb_p_lm, kr_c, taur_c, alphar_c)
                budget_result[name] = res
            fixed_budget_report[pct] = {"n_budget_vectors": int(n_budget), "selection_methods": budget_result}
            print(f"  budget matching worst-{pct}% ({n_budget} vectors): "
                  f"random_nll={budget_result['random']['mean_interpolated_nll']:.4f}  "
                  f"freq_nll={budget_result['frequency_based']['mean_interpolated_nll']:.4f}  "
                  f"oracle_nll={budget_result['oracle_derived']['mean_interpolated_nll']:.4f}  (eval sample, n={len(fb_keys)})")
        cfg_report["section9_fixed_budget_comparison"] = fixed_budget_report

        fig, ax = plt.subplots(figsize=(8, 5))
        for name in ["random", "frequency_based", "oracle_derived"]:
            vals = [fixed_budget_report[p]["selection_methods"][name]["mean_interpolated_nll"] for p in WORST_PCTS]
            ax.plot(WORST_PCTS, vals, "o-", label=name)
        ax.axhline(L_dime.mean(), color="gray", linestyle="--", label="DIME (no retention)")
        ax.axhline(L_raw.mean(), color="black", linestyle=":", label="full raw kNN")
        ax.set_xlabel("worst-query percentage (budget sizing reference)")
        ax.set_ylabel(f"mean interpolated NLL (evaluated on {len(fb_keys)}-query fixed sample)")
        ax.set_title(f"[{config_name}] storage-performance tradeoff: selection method comparison")
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(PLOT_DIR, f"{config_name}_storage_performance_tradeoff.png"), dpi=130)
        plt.close(fig)

        report[config_name] = cfg_report
        save_results(os.path.join(OUT_DIR, "e1c2_summary.json"), report)

    report["sensitivity_common_alpha_ranking"] = {
        "note": "tuned k/tau, alpha=0.25 for both systems -- separate from the original tuned ranking",
        "mean_delta_L": float(sensitivity["delta_L"].mean()),
        "rank_correlation_with_original_tuned_ranking": float(
            np.corrcoef(np.argsort(np.argsort(-delta_L_tuned)), np.argsort(np.argsort(-sensitivity["delta_L"])))[0, 1]
        ),
    }
    save_results(os.path.join(OUT_DIR, "e1c2_summary.json"), report)

    # ============================================================
    # PART B -- does S3 repair the worst 10%? (and worst 1%/5%, plus remaining 90%)
    # ============================================================
    print("\n################ PART B: S3 recovery on E1C's worst queries ################")
    print("Computing S3 (query-conditioned) standalone probability for ALL queries, per config -- "
          "needed for both the worst-group analysis and the worst-vs-remaining-90% comparison.")

    def build_cluster_to_members(assignment, n_clusters):
        order = np.argsort(assignment, kind="stable")
        sorted_assign = assignment[order]
        boundaries = np.searchsorted(sorted_assign, np.arange(n_clusters + 1))
        return {c: order[boundaries[c]:boundaries[c + 1]] for c in range(n_clusters)}

    cluster_to_members = build_cluster_to_members(assignment, N_CLUSTERS)

    def compute_sys3_full(idx_dime_full, d_dime_full, tau, verbose_every=10000):
        N, k = idx_dime_full.shape
        w_dime = softmax_np(-d_dime_full / tau, axis=1)
        p_sys3 = np.zeros(N, dtype=np.float64)
        t0 = time.perf_counter()
        for i in range(N):
            q_key = val_keys[i]
            true_tok = int(val_values[i])
            acc = 0.0
            for j in range(k):
                c = idx_dime_full[i, j]
                members = cluster_to_members[c]
                if len(members) == 0:
                    continue
                d_members = np.linalg.norm(ds_keys[members] - q_key[None, :], axis=1)
                w_members = softmax_np(-d_members / tau, axis=-1)
                match = ds_values[members] == true_tok
                if match.any():
                    acc += float(w_dime[i, j]) * float(w_members[match].sum())
            p_sys3[i] = acc
            if verbose_every and (i + 1) % verbose_every == 0:
                elapsed = time.perf_counter() - t0
                print(f"    sys3 progress: {i+1}/{N}  ({(i+1)/elapsed:.1f} q/s, ETA {(N-i-1)/((i+1)/elapsed)/60:.1f} min)")
        return p_sys3

    partB_report = {}
    for config_name, cfg in configs.items():
        print(f"\n--- Part B, config: {config_name} ---")
        kd_c, taud_c, alphad_c = cfg["k_dime"], cfg["tau_dime"], cfg["alpha_dime"]
        d_dime_full = d_dime_m if config_name == "matched" else d_dime_t
        idx_dime_full = idx_dime_m if config_name == "matched" else idx_dime_t
        p_sys3_full = compute_sys3_full(idx_dime_full[:, :kd_c], d_dime_full[:, :kd_c], taud_c)
        p_mixed_s3 = alphad_c * p_sys3_full + (1 - alphad_c) * val_p_lm
        L_S3_full = -np.log(p_mixed_s3 + 1e-12)

        delta_L, L_raw, L_dime = cfg["delta_L"], cfg["L_raw"], cfg["L_dime"]
        order = np.argsort(-delta_L, kind="stable")
        cfgB_report = {}
        for pct in [1, 5, 10]:
            n = max(1, round(len(delta_L) * pct / 100))
            q_idx = order[:n]
            dL_S2 = (L_dime - L_raw)[q_idx]
            dL_S3 = (L_S3_full - L_raw)[q_idx]
            R_S3 = (L_dime - L_S3_full)[q_idx]
            improves = R_S3 > 0
            total_pos_s2 = float(dL_S2.clip(min=0).sum())
            total_pos_s3 = float(dL_S3.clip(min=0).sum())
            cfgB_report[pct] = {
                "n": int(n), "S1_nll": quantiles(L_raw[q_idx]), "S2_nll": quantiles(L_dime[q_idx]), "S3_nll": quantiles(L_S3_full[q_idx]),
                "S2_minus_S3_improvement": quantiles(R_S3),
                "frac_improved_by_S3": float(improves.mean()), "frac_worsened_by_S3": float((R_S3 < 0).mean()),
                "remaining_positive_degradation_percentiles": {p: float(np.percentile(dL_S3[dL_S3 > 0], p)) for p in (90, 95, 99)} if (dL_S3 > 0).any() else None,
                "total_positive_loss_S2_vs_S1": total_pos_s2, "total_positive_loss_S3_vs_S1": total_pos_s3,
                "fraction_positive_loss_repaired": float((total_pos_s2 - total_pos_s3) / total_pos_s2) if total_pos_s2 > 0 else None,
                "signed_net_nll_recovery_S2_minus_S3": float((L_dime[q_idx] - L_S3_full[q_idx]).mean()),
            }
            print(f"  worst {pct}%: frac_improved_by_S3={cfgB_report[pct]['frac_improved_by_S3']:.3f}  "
                  f"fraction_positive_loss_repaired={cfgB_report[pct]['fraction_positive_loss_repaired']}")

        mask10 = np.zeros(len(delta_L), dtype=bool)
        mask10[order[:max(1, round(len(delta_L) * 0.10))]] = True
        for name, mask in [("A_worst_10pct", mask10), ("B_remaining_90pct", ~mask10)]:
            s2_minus_s1 = float((L_dime[mask] - L_raw[mask]).mean())
            s3_minus_s1 = float((L_S3_full[mask] - L_raw[mask]).mean())
            s2_minus_s3 = float((L_dime[mask] - L_S3_full[mask]).mean())
            total_s3_improvement = float((L_dime - L_S3_full).sum())
            contribution = float((L_dime[mask] - L_S3_full[mask]).sum() / total_s3_improvement) if total_s3_improvement != 0 else None
            cfgB_report.setdefault("worst_vs_remaining", {})[name] = {
                "n": int(mask.sum()), "mean_S2_minus_S1": s2_minus_s1, "mean_S3_minus_S1": s3_minus_s1,
                "mean_S2_minus_S3_recovery": s2_minus_s3, "frac_helped_by_S3": float((L_dime[mask] > L_S3_full[mask]).mean()),
                "contribution_to_total_S3_improvement": contribution,
            }
            print(f"  [{name}] n={mask.sum()}  S2-S3 recovery={s2_minus_s3:.4f}  contribution_to_total={contribution}")

        partB_report[config_name] = cfgB_report

    report["partB_s3_recovery_on_worst_queries"] = partB_report
    save_results(os.path.join(OUT_DIR, "e1c2_summary.json"), report)

    print(f"\nAll E1C.2 results saved under {OUT_DIR}")
    print("No existing E1-series outputs were modified.")


if __name__ == "__main__":
    main()
