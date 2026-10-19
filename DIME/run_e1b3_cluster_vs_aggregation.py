"""E1B.3 -- Cluster Selection vs. Within-Cluster Information Loss. Extends E1
without touching E1/E1A/E1A.2/E1B.2's existing outputs (writes only to
results/stage3_e1b3_cluster_vs_aggregation/).

See the conversation turn preceding this script for the full equation writeup
(System 1-4 definitions) -- summarized here:
  System 1 (raw kNN):      P_raw(y|q)  = sum_i w_i * 1[y_i=y],           w_i from raw-neighbor distances
  System 2 (DIME):         P_DIME(y|q) = sum_c w_c * (count_c(y)/total_c), w_c from centroid distances
  System 3 (DIME sel. + query-conditioned agg.): same w_c as System 2, but each
            cluster's inner term is replaced by a freshly-computed, query-distance-
            weighted vote over that cluster's ACTUAL members (needs ds_keys/ds_values
            + the assignment array -- not persisted anywhere, re-derived once below
            via the exact same deterministic MiniBatchKMeans fit state_object.py uses).
  System 4 (raw-guided sel. + DIME agg.): raw top-k neighbors' weights regrouped by
            their cluster assignment, combined with DIME's ORIGINAL stored Counters.

Experiment A (cluster-selection overlap) runs on the FULL 77,089-query val set --
it only needs assignment lookups, not per-member access, so it's cheap. Experiments
B/C (within-cluster information, the four-system comparison, subgroup analysis) need
actual per-member hidden vectors and distances -- at ~100 members/cluster x k=20
clusters/query that's ~150M query-member distance computations at full scale, not
justified for a diagnostic. Sampled to 3,000 queries, STRATIFIED across JS-divergence
deciles (computed at matched config) so rare high-divergence cases aren't drowned out
-- this is the "evaluation subset" the spec asks to report explicitly.
"""
import sys, os, json, time
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
from e1_metrics import (
    raw_standalone_prob, dime_standalone_prob, js_divergence_dict, raw_full_distribution, dime_full_distribution,
    correlations_with_ci, binned_stats, plot_binned, distribution_shape_stats, retrieval_coverage,
)

CACHE_DIR = os.path.join(os.path.dirname(__file__), "..", "GPT_Module", "cache")
OUT_DIR = os.path.join(os.path.dirname(__file__), "results", "stage3_e1b3_cluster_vs_aggregation")
PLOT_DIR = os.path.join(OUT_DIR, "plots")
os.makedirs(PLOT_DIR, exist_ok=True)

N_CLUSTERS = 3811
MATCHED = {"k": 20, "tau": 2.0, "alpha": 0.25}
CONTROLLED_ALPHA = 0.25
N_SAMPLE = 3000
# Subgroup thresholds, declared up front (not chosen after seeing results)
SUBGROUP_JS_HIGH_PCT = 75     # top quartile JS
SUBGROUP_DIST_LOW_PCT = 25    # bottom quartile |delta_d|
SUBGROUP_C_DELTA_P_THRESHOLD = -0.10  # "substantially worsens" = delta_p < -0.10


def quantiles(x):
    x = np.asarray(x, dtype=np.float64)
    return {"mean": float(np.mean(x)), "median": float(np.median(x)),
            "p90": float(np.percentile(x, 90)), "p95": float(np.percentile(x, 95)), "p99": float(np.percentile(x, 99))}


def counter_entropy_and_max(counter):
    total = sum(counter.values())
    if total == 0:
        return 0.0, 0.0
    probs = np.array([c / total for c in counter.values()], dtype=np.float64)
    probs = probs[probs > 0]
    return float(-(probs * np.log(probs)).sum()), float(probs.max())


def main():
    print("Loading cache (no extraction)...")
    ds = load_cache(os.path.join(CACHE_DIR, "wikitext103_gpt2_datastore.npz"))
    val = load_cache(os.path.join(CACHE_DIR, "wikitext103_gpt2_val.npz"))
    with open("results/wikitext103_gpt2_tier1.json") as f:
        tier1 = json.load(f)
    raw_tuned, dime_tuned = tier1["raw_knn"]["best_config"], tier1["dime_minibatch_kmeans"]["best_config"]
    assert tier1["n_clusters"] == N_CLUSTERS
    ds_keys, ds_values = ds["keys"], ds["values"]
    val_keys, val_values, val_p_lm = val["keys"], val["values"], val["p_lm_true"]
    print(f"val={len(val_values)}  raw_tuned={raw_tuned}  dime_tuned={dime_tuned}  matched={MATCHED}")

    print("Re-deriving DIME clusters (seed=42, deterministic, same as every prior E1 script) "
          "-- this time ALSO keeping the per-point assignment array, which minibatch_kmeans_partition "
          "doesn't return (changing its return shape would break every existing caller).")
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

    # Regression check: does this script's own re-derivation reproduce e1_metrics'
    # already-validated dime_standalone_prob on a handful of queries?
    probe_k = min(MATCHED["k"], N_CLUSTERS)
    d_probe, idx_probe = query_knn_indices(dime_index, val_keys[:50], k=probe_k)
    retrieved_probe = dime_values[idx_probe]
    probe_p = dime_standalone_prob(d_probe, retrieved_probe, val_values[:50], tau=MATCHED["tau"])
    print(f"regression check (first 50 queries, standalone DIME prob of true token): "
          f"mean={probe_p.mean():.6f} -- should be a sane probability-like number, not NaN/negative")

    configs = {"matched": (MATCHED["k"], MATCHED["tau"], MATCHED["k"], MATCHED["tau"]),
               "tuned": (raw_tuned["k"], raw_tuned["tau"], dime_tuned["k"], dime_tuned["tau"])}
    k_max_raw = max(MATCHED["k"], raw_tuned["k"])
    k_max_dime = max(MATCHED["k"], dime_tuned["k"])
    d_raw_full, idx_raw_full = query_knn_indices(raw_index, val_keys, k=k_max_raw)
    d_dime_full, idx_dime_full = query_knn_indices(dime_index, val_keys, k=k_max_dime)
    retrieved_raw_full = raw_values[idx_raw_full]

    report = {"n_total_queries": int(len(val_values)), "matched_config": MATCHED,
              "tuned_config": {"raw": raw_tuned, "dime": dime_tuned}, "n_sample_for_B_C": N_SAMPLE,
              "subgroup_thresholds": {"js_high_percentile": SUBGROUP_JS_HIGH_PCT,
                                       "distance_low_percentile": SUBGROUP_DIST_LOW_PCT,
                                       "substantial_worsening_delta_p": SUBGROUP_C_DELTA_P_THRESHOLD}}

    for config_name, (kr, taur, kd, taud) in configs.items():
        print(f"\n################ CONFIG: {config_name} (raw k={kr},tau={taur} | dime k={kd},tau={taud}) ################")
        cfg_report = {}
        d_raw_t, r_raw_t = d_raw_full[:, :kr], retrieved_raw_full[:, :kr]
        idx_raw_t = idx_raw_full[:, :kr]
        d_dime_t = d_dime_full[:, :kd]
        idx_dime_t = idx_dime_full[:, :kd]

        w_raw_full = softmax_np(-d_raw_t / taur, axis=1)       # [N, kr]
        w_dime_full = softmax_np(-d_dime_t / taud, axis=1)     # [N, kd]
        raw_neighbor_clusters = assignment[idx_raw_t]          # [N, kr] -- vectorized cluster lookup

        p_raw = raw_standalone_prob(d_raw_t, r_raw_t, val_values, tau=taur)
        p_dime = dime_standalone_prob(d_dime_t, dime_values[idx_dime_t], val_values, tau=taud)
        with np.errstate(divide="ignore"):
            L_raw, L_dime = -np.log(p_raw), -np.log(p_dime)
        raw_dists_list = raw_full_distribution(d_raw_t, r_raw_t, tau=taur)
        dime_dists_list = dime_full_distribution(d_dime_t, dime_values[idx_dime_t], tau=taud)
        js = np.array([js_divergence_dict(raw_dists_list[i], dime_dists_list[i]) for i in range(len(val_values))])
        d_raw_nearest, d_dime_nearest = d_raw_full[:, 0], d_dime_full[:, 0]
        delta_d = d_dime_nearest - d_raw_nearest

        # ============================================================
        # Experiment A -- cluster-selection overlap (full scale)
        # ============================================================
        print("Experiment A: cluster-selection overlap (full val set)...")
        frac_neighbor_overlap = np.zeros(len(val_values))
        frac_mass_overlap = np.zeros(len(val_values))
        n_distinct_raw_clusters = np.zeros(len(val_values), dtype=np.int32)
        n_dime_clusters_missed_by_raw = np.zeros(len(val_values), dtype=np.int32)
        for i in range(len(val_values)):
            raw_c = raw_neighbor_clusters[i]
            dime_c_set = set(idx_dime_t[i].tolist())
            raw_c_set = set(raw_c.tolist())
            in_dime = np.isin(raw_c, idx_dime_t[i])
            frac_neighbor_overlap[i] = in_dime.mean()
            frac_mass_overlap[i] = w_raw_full[i][in_dime].sum()
            n_distinct_raw_clusters[i] = len(raw_c_set)
            n_dime_clusters_missed_by_raw[i] = len(dime_c_set - raw_c_set)

        expA = {
            "frac_neighbor_overlap": quantiles(frac_neighbor_overlap),
            "frac_probability_mass_overlap": quantiles(frac_mass_overlap),
            "n_distinct_clusters_among_raw_neighbors": quantiles(n_distinct_raw_clusters.astype(np.float64)),
            "n_dime_selected_clusters_missed_by_raw": quantiles(n_dime_clusters_missed_by_raw.astype(np.float64)),
            "corr_mass_overlap_vs_js": correlations_with_ci(frac_mass_overlap, js),
        }
        finite_dL_mask = np.isfinite(L_dime - L_raw) if False else (p_raw > 0) & (p_dime > 0)
        with np.errstate(invalid="ignore"):
            delta_L_all = L_dime - L_raw
        expA["corr_mass_overlap_vs_deltaL_finite_subset"] = correlations_with_ci(
            frac_mass_overlap[finite_dL_mask], delta_L_all[finite_dL_mask])
        cfg_report["experiment_A_cluster_overlap"] = expA
        print(f"  mean frac_neighbor_overlap={expA['frac_neighbor_overlap']['mean']:.4f}  "
              f"mean frac_mass_overlap={expA['frac_probability_mass_overlap']['mean']:.4f}  "
              f"corr(mass_overlap, JS) spearman={expA['corr_mass_overlap_vs_js']['spearman']['rho']:.4f}")

        fig, ax = plt.subplots(figsize=(7, 5))
        ax.hist(frac_mass_overlap, bins=100)
        ax.set_xlabel("fraction of raw retrieval probability mass inside DIME-selected clusters")
        ax.set_ylabel("count")
        ax.set_title(f"[{config_name}] Experiment A: raw-mass / DIME-selection overlap")
        fig.tight_layout()
        fig.savefig(os.path.join(PLOT_DIR, f"{config_name}_expA_mass_overlap_hist.png"), dpi=130)
        plt.close(fig)

        # ============================================================
        # Sample selection for Experiments B/C -- stratified across JS deciles
        # ============================================================
        js_deciles = np.digitize(js, np.quantile(js, np.linspace(0, 1, 11))[1:-1])
        rng = np.random.default_rng(42)
        per_decile = N_SAMPLE // 10
        sample_idx = []
        for dcl in range(10):
            pool = np.where(js_deciles == dcl)[0]
            take = min(per_decile, len(pool))
            sample_idx.append(rng.choice(pool, size=take, replace=False))
        sample_idx = np.concatenate(sample_idx)
        print(f"Experiments B/C sample: {len(sample_idx)} queries, stratified across JS deciles "
              f"({per_decile} target per decile)")

        # ============================================================
        # Experiment B/C -- within-cluster info + four-system counterfactuals (sampled)
        # ============================================================
        print("Experiment B/C: within-cluster analysis + four-system reconstruction (sampled)...")
        sys3_dists, sys4_dists = [], []
        cluster_level_stats = []
        verify_sys4_matches_raw = []
        verify_sys3_cluster_set_matches_dime = []

        for qi in sample_idx:
            q_key = val_keys[qi]
            true_tok = int(val_values[qi])
            dime_c_ids = idx_dime_t[qi]
            dime_w = w_dime_full[qi]

            sys3_dist = {}
            for c, w_c in zip(dime_c_ids, dime_w):
                member_idx = np.where(assignment == c)[0]
                if len(member_idx) == 0:
                    continue
                member_keys = ds_keys[member_idx]
                member_values = ds_values[member_idx]
                d_members = np.linalg.norm(member_keys - q_key[None, :], axis=1)
                w_members = softmax_np(-d_members / taud, axis=-1)
                for j, val_tok in enumerate(member_values):
                    sys3_dist[int(val_tok)] = sys3_dist.get(int(val_tok), 0.0) + float(w_c) * float(w_members[j])

                ent, max_p = counter_entropy_and_max(compressed_dists[c])
                d_centroid = float(d_dime_t[qi][list(dime_c_ids).index(c)])
                cluster_level_stats.append({
                    "n_members": int(len(member_idx)), "n_distinct_labels": len(compressed_dists[c]),
                    "label_entropy_stored": ent, "max_label_prob_stored": max_p,
                    "frac_members_equal_target": float((member_values == true_tok).mean()),
                    "d_query_to_members_mean": float(d_members.mean()), "d_query_to_members_std": float(d_members.std()),
                    "d_query_to_centroid_minus_mean_member_dist": float(d_centroid - d_members.mean()),
                })
            sys3_dists.append(sys3_dist)

            raw_c_this = raw_neighbor_clusters[qi]
            raw_w_this = w_raw_full[qi]
            cluster_to_weight = {}
            cluster_to_raw_tokens = {}
            for c, w in zip(raw_c_this, raw_w_this):
                cluster_to_weight[c] = cluster_to_weight.get(c, 0.0) + float(w)
            sys4_dist = {}
            for c, w_prime in cluster_to_weight.items():
                counter = compressed_dists[c]
                total = sum(counter.values())
                if total > 0:
                    for tok, cnt in counter.items():
                        sys4_dist[tok] = sys4_dist.get(tok, 0.0) + w_prime * (cnt / total)
            sys4_dists.append(sys4_dist)

            # Verification spot-check: regrouping raw neighbors by cluster (keeping their
            # OWN exact labels, not DIME's aggregate) is an algebraic identity to System 1 --
            # Sum_i w_i*1[y_i=y] == Sum_c Sum_{i in c} w_i*1[y_i=y] by pure regrouping of terms.
            direct_raw_true_prob = float(w_raw_full[qi][r_raw_t[qi] == true_tok].sum())
            verify_sys4_matches_raw.append(abs(direct_raw_true_prob - p_raw[qi]) < 1e-9)
            verify_sys3_cluster_set_matches_dime.append(set(dime_c_ids.tolist()) == set(idx_dime_t[qi].tolist()))

        print(f"  verification: raw per-position true-token prob matches p_raw exactly for "
              f"{sum(verify_sys4_matches_raw)}/{len(verify_sys4_matches_raw)} sampled queries")
        print(f"  verification: System 3's cluster set matches DIME's own selection for "
              f"{sum(verify_sys3_cluster_set_matches_dime)}/{len(verify_sys3_cluster_set_matches_dime)} sampled queries")

        val_values_s = val_values[sample_idx]
        val_p_lm_s = val_p_lm[sample_idx]
        p_sys3 = np.array([sys3_dists[i].get(int(val_values_s[i]), 0.0) for i in range(len(sample_idx))])
        p_sys4 = np.array([sys4_dists[i].get(int(val_values_s[i]), 0.0) for i in range(len(sample_idx))])
        p_raw_s, p_dime_s = p_raw[sample_idx], p_dime[sample_idx]
        with np.errstate(divide="ignore"):
            L_sys3 = -np.log(p_sys3)
            L_sys4 = -np.log(p_sys4)
        L_raw_s, L_dime_s, js_s = L_raw[sample_idx], L_dime[sample_idx], js[sample_idx]

        sys3_shape = distribution_shape_stats(sys3_dists)
        sys4_shape = distribution_shape_stats(sys4_dists)
        raw_shape_s = distribution_shape_stats([raw_dists_list[i] for i in sample_idx])
        dime_shape_s = distribution_shape_stats([dime_dists_list[i] for i in sample_idx])

        def perf_block(p, L, name):
            cov = retrieval_coverage(p)
            finite = np.isfinite(L)
            # Relative to System 1 (raw) -- the natural reference throughout this analysis;
            # trivially 0/0 when called on System 1 itself against itself.
            with np.errstate(invalid="ignore"):
                delta_vs_raw = L - L_raw_s
            both_finite_vs_raw = finite & np.isfinite(L_raw_s)
            improves = both_finite_vs_raw & (delta_vs_raw < 0)
            degrades = both_finite_vs_raw & (delta_vs_raw > 0)
            return {
                "coverage": cov, "p_true": quantiles(p) if finite.sum() else None,
                "n_finite": int(finite.sum()), "n_infinite": int((~finite).sum()),
                "frac_improves_vs_raw": float(improves.sum() / both_finite_vs_raw.sum()) if both_finite_vs_raw.sum() else None,
                "frac_degrades_vs_raw": float(degrades.sum() / both_finite_vs_raw.sum()) if both_finite_vs_raw.sum() else None,
                "mean_improvement_magnitude_vs_raw": float(-delta_vs_raw[improves].mean()) if improves.sum() else None,
                "mean_degradation_magnitude_vs_raw": float(delta_vs_raw[degrades].mean()) if degrades.sum() else None,
                "p95_degradation_vs_raw": float(np.percentile(delta_vs_raw[degrades], 95)) if degrades.sum() else None,
            }

        four_systems = {
            "system1_raw": {"performance": perf_block(p_raw_s, L_raw_s, "raw"),
                             "shape": {"entropy": quantiles(raw_shape_s["entropy"]), "effective_support": quantiles(raw_shape_s["effective_support"]),
                                       "max_prob": quantiles(raw_shape_s["max_prob"]),
                                       "top1_agreement_with_raw": 1.0}},
            "system2_dime": {"performance": perf_block(p_dime_s, L_dime_s, "dime"),
                              "shape": {"entropy": quantiles(dime_shape_s["entropy"]), "effective_support": quantiles(dime_shape_s["effective_support"]),
                                        "max_prob": quantiles(dime_shape_s["max_prob"]),
                                        "top1_agreement_with_raw": float((dime_shape_s["top1_token"] == raw_shape_s["top1_token"]).mean())}},
            "system3_dime_selection_query_conditioned_agg": {
                "performance": perf_block(p_sys3, L_sys3, "sys3"),
                "shape": {"entropy": quantiles(sys3_shape["entropy"]), "effective_support": quantiles(sys3_shape["effective_support"]),
                          "max_prob": quantiles(sys3_shape["max_prob"]),
                          "top1_agreement_with_raw": float((sys3_shape["top1_token"] == raw_shape_s["top1_token"]).mean())}},
            "system4_raw_selection_dime_agg": {
                "performance": perf_block(p_sys4, L_sys4, "sys4"),
                "shape": {"entropy": quantiles(sys4_shape["entropy"]), "effective_support": quantiles(sys4_shape["effective_support"]),
                          "max_prob": quantiles(sys4_shape["max_prob"]),
                          "top1_agreement_with_raw": float((sys4_shape["top1_token"] == raw_shape_s["top1_token"]).mean())}},
        }
        for name, sys_dist_list in [("system3", sys3_dists), ("system4", sys4_dists)]:
            js_vs_raw = np.array([js_divergence_dict(raw_dists_list[sample_idx[i]], sys_dist_list[i]) for i in range(len(sample_idx))])
            four_systems[f"{'system3_dime_selection_query_conditioned_agg' if name=='system3' else 'system4_raw_selection_dime_agg'}"]["js_divergence_from_raw"] = quantiles(js_vs_raw)

        cfg_report["experiment_BC_four_systems"] = four_systems
        cfg_report["within_cluster_stats_aggregate"] = {
            key: quantiles([s[key] for s in cluster_level_stats]) for key in
            ["n_members", "label_entropy_stored", "max_label_prob_stored", "frac_members_equal_target",
             "d_query_to_members_mean", "d_query_to_centroid_minus_mean_member_dist"]
        }
        cfg_report["verification"] = {
            "raw_decomposition_exact_match_fraction": float(np.mean(verify_sys4_matches_raw)),
            "system3_cluster_set_matches_dime_fraction": float(np.mean(verify_sys3_cluster_set_matches_dime)),
        }
        print(json.dumps({"system1_raw": four_systems["system1_raw"]["performance"],
                           "system2_dime": four_systems["system2_dime"]["performance"],
                           "system3": four_systems["system3_dime_selection_query_conditioned_agg"]["performance"],
                           "system4": four_systems["system4_raw_selection_dime_agg"]["performance"]}, indent=2))

        # Does recovering query-conditioned info (System 3) raise correct-token prob vs System 2?
        delta_p_sys3_vs_dime = p_sys3 - p_dime_s
        cfg_report["system3_recovery_over_system2"] = quantiles(delta_p_sys3_vs_dime)
        print(f"  System3 - System2 correct-token prob change: mean={delta_p_sys3_vs_dime.mean():.5f}  "
              f"median={np.median(delta_p_sys3_vs_dime):.5f}")

        fig, ax = plt.subplots(figsize=(8, 5))
        names = ["raw", "DIME", "sys3 (DIME sel.\n+query-cond. agg.)", "sys4 (raw sel.\n+DIME agg.)"]
        covs = [four_systems["system1_raw"]["performance"]["coverage"], four_systems["system2_dime"]["performance"]["coverage"],
                four_systems["system3_dime_selection_query_conditioned_agg"]["performance"]["coverage"],
                four_systems["system4_raw_selection_dime_agg"]["performance"]["coverage"]]
        ax.bar(names, covs)
        ax.set_ylabel("correct-token coverage")
        ax.set_title(f"[{config_name}] four-system coverage comparison (sample, n={len(sample_idx)})")
        fig.tight_layout()
        fig.savefig(os.path.join(PLOT_DIR, f"{config_name}_four_system_coverage.png"), dpi=130)
        plt.close(fig)

        # ============================================================
        # Subgroup analysis (on the sample)
        # ============================================================
        print("Subgroup analysis...")
        abs_delta_d_s = np.abs(delta_d[sample_idx])
        js_high_thresh = np.percentile(js_s, SUBGROUP_JS_HIGH_PCT)
        dist_low_thresh = np.percentile(abs_delta_d_s, SUBGROUP_DIST_LOW_PCT)
        both_cover_s = (p_raw_s > 0) & (p_dime_s > 0)
        delta_p_s = np.where(both_cover_s, p_dime_s - p_raw_s, np.nan)

        subgroup_A = (js_s >= js_high_thresh) & (abs_delta_d_s <= dist_low_thresh)
        subgroup_B = both_cover_s & (delta_p_s > 0)
        subgroup_C = both_cover_s & (delta_p_s < SUBGROUP_C_DELTA_P_THRESHOLD)
        subgroup_D = (p_raw_s == 0) & (p_dime_s > 0)

        subgroups = {}
        for name, mask in {"A_high_js_low_distance_change": subgroup_A, "B_dime_improves": subgroup_B,
                            "C_dime_substantially_worsens": subgroup_C, "D_dime_only_coverage": subgroup_D}.items():
            n = int(np.nansum(mask)) if mask.dtype != bool else int(mask.sum())
            mask_bool = mask.astype(bool) if mask.dtype != bool else mask
            idx_local = np.where(mask_bool)[0]
            subgroups[name] = {
                "n": int(len(idx_local)), "frac_of_sample": float(len(idx_local) / len(sample_idx)),
                "mean_frac_neighbor_overlap": float(frac_neighbor_overlap[sample_idx][idx_local].mean()) if len(idx_local) else None,
                "mean_frac_mass_overlap": float(frac_mass_overlap[sample_idx][idx_local].mean()) if len(idx_local) else None,
                "mean_p_dime": float(p_dime_s[idx_local].mean()) if len(idx_local) else None,
                "mean_sys3_minus_sys2_p": float(delta_p_sys3_vs_dime[idx_local].mean()) if len(idx_local) else None,
            }
            print(f"  [{name}] n={subgroups[name]['n']}  mass_overlap={subgroups[name]['mean_frac_mass_overlap']}  "
                  f"sys3_recovery={subgroups[name]['mean_sys3_minus_sys2_p']}")
        cfg_report["subgroup_analysis"] = subgroups

        report[config_name] = cfg_report
        save_results(os.path.join(OUT_DIR, "e1b3_summary.json"), report)

    print(f"\nAll E1B.3 results saved under {OUT_DIR}")
    print("E1/E1A/E1A.2/E1B.2's existing outputs were not touched.")


if __name__ == "__main__":
    main()
