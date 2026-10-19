"""E1B.2 -- Coverage-Concentration Tradeoff and Predictive Performance. Extends
E1B without touching E1, E1A, E1A.2, or E1-followup's existing outputs (writes
only to results/stage3_e1b2_coverage_concentration/).

Reuses the same cache, the same seed=42 DIME re-clustering (necessary -- centroids
were never persisted), and the same tuned (k,tau,alpha) from
results/wikitext103_gpt2_tier1.json. The initial E1B measurements (quoted in the
motivation) were computed by run_e1_wikitext103.py using the TUNED config only
(k_raw_t=raw_tuned['k'], k_dime_t=dime_tuned['k'] -- confirmed by reading that
script). Per the instruction not to compare JS/coverage across configurations
without flagging the difference, this script computes BOTH "matched"
(k=20, tau=2.0, alpha=0.25 for both systems) and "tuned" (each system's own
best config) throughout, every result keyed and labeled by which one it is --
never silently mixed.

Standalone NLL (L = -log(p_standalone)) is used for RQ1/RQ2/RQ4/RQ5, with no
epsilon -- infinite losses (zero standalone probability) are counted and
reported, never silently replaced. For RQ3's "does coverage translate to overall
performance" question, a CONTROLLED interpolation weight (alpha=0.25, same value
applied identically to both raw and DIME at each config's own (k,tau)) is used to
get finite LM-interpolated NLL for every query, including zero-coverage ones --
explicitly labeled as a separate, sensitivity-style analysis, not a replacement
for the standalone/coverage numbers.
"""
import sys, os, json, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "GPT_Module"))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from cache_io import load_cache
from knn import build_datastore, query_knn_indices, mix_knn_and_lm
from mixing import mix_dime_and_lm
from state_object import minibatch_kmeans_partition
from results_io import save_results
from e1_metrics import (
    raw_standalone_prob, dime_standalone_prob, js_divergence_dict, raw_full_distribution, dime_full_distribution,
    correlations_with_ci, binned_stats, plot_binned, distribution_shape_stats,
)

CACHE_DIR = os.path.join(os.path.dirname(__file__), "..", "GPT_Module", "cache")
OUT_DIR = os.path.join(os.path.dirname(__file__), "results", "stage3_e1b2_coverage_concentration")
PLOT_DIR = os.path.join(OUT_DIR, "plots")
os.makedirs(PLOT_DIR, exist_ok=True)

N_CLUSTERS = 3811
MATCHED = {"k": 20, "tau": 2.0, "alpha": 0.25}
CONTROLLED_ALPHA = 0.25  # RQ3's sensitivity analysis: same interpolation weight applied to both systems


def quantiles(x):
    return {"mean": float(np.mean(x)), "median": float(np.median(x)),
            "p90": float(np.percentile(x, 90)), "p95": float(np.percentile(x, 95)), "p99": float(np.percentile(x, 99))}


def main():
    print("Loading cache (no extraction)...")
    ds = load_cache(os.path.join(CACHE_DIR, "wikitext103_gpt2_datastore.npz"))
    val = load_cache(os.path.join(CACHE_DIR, "wikitext103_gpt2_val.npz"))
    with open("results/wikitext103_gpt2_tier1.json") as f:
        tier1 = json.load(f)
    raw_tuned, dime_tuned = tier1["raw_knn"]["best_config"], tier1["dime_minibatch_kmeans"]["best_config"]
    assert tier1["n_clusters"] == N_CLUSTERS
    val_values, val_p_lm = val["values"], val["p_lm_true"]
    print(f"val={len(val_values)}  raw_tuned={raw_tuned}  dime_tuned={dime_tuned}  matched={MATCHED}")

    print("Building raw index, re-deriving DIME clusters (seed=42, deterministic, same as E1/E1A.2)...")
    raw_index, raw_values = build_datastore(ds["keys"], ds["values"])
    t0 = time.perf_counter()
    dime_centroids, dime_dists = minibatch_kmeans_partition(ds["keys"], ds["values"], n_clusters=N_CLUSTERS, seed=42)
    dime_dists = np.array(dime_dists, dtype=object)
    dime_index, dime_values = build_datastore(dime_centroids, dime_dists)
    print(f"clustering done in {time.perf_counter()-t0:.1f}s")

    configs = {
        "matched": (MATCHED["k"], MATCHED["tau"], MATCHED["alpha"], MATCHED["k"], MATCHED["tau"], MATCHED["alpha"]),
        "tuned": (raw_tuned["k"], raw_tuned["tau"], raw_tuned["alpha"], dime_tuned["k"], dime_tuned["tau"], dime_tuned["alpha"]),
    }
    k_max_raw = max(MATCHED["k"], raw_tuned["k"])
    k_max_dime = max(MATCHED["k"], dime_tuned["k"])
    d_raw_full, idx_raw_full = query_knn_indices(raw_index, val["keys"], k=k_max_raw)
    d_dime_full, idx_dime_full = query_knn_indices(dime_index, val["keys"], k=k_max_dime)
    retrieved_raw_full = raw_values[idx_raw_full]
    retrieved_dime_full = dime_values[idx_dime_full]

    report = {"n_total_queries": int(len(val_values)), "matched_config": MATCHED,
              "tuned_config": {"raw": raw_tuned, "dime": dime_tuned}, "controlled_alpha_for_RQ3": CONTROLLED_ALPHA,
              "note_initial_E1B_used": "tuned config only (confirmed by reading run_e1_wikitext103.py)"}
    sample_idx = np.random.default_rng(0).choice(len(val_values), size=min(20000, len(val_values)), replace=False)

    for config_name, (kr, taur, alphar, kd, taud, alphad) in configs.items():
        print(f"\n################ CONFIG: {config_name} (raw k={kr},tau={taur},alpha={alphar} | "
              f"dime k={kd},tau={taud},alpha={alphad}) ################")
        cfg_report = {}
        d_raw_t, r_raw_t = d_raw_full[:, :kr], retrieved_raw_full[:, :kr]
        d_dime_t, r_dime_t = d_dime_full[:, :kd], retrieved_dime_full[:, :kd]

        p_raw = raw_standalone_prob(d_raw_t, r_raw_t, val_values, tau=taur)
        p_dime = dime_standalone_prob(d_dime_t, r_dime_t, val_values, tau=taud)
        with np.errstate(divide="ignore"):
            L_raw, L_dime = -np.log(p_raw), -np.log(p_dime)
        raw_covers, dime_covers = p_raw > 0, p_dime > 0

        print("Computing full standalone distributions (JS, entropy, concentration)...")
        raw_dists_list = raw_full_distribution(d_raw_t, r_raw_t, tau=taur)
        dime_dists_list = dime_full_distribution(d_dime_t, r_dime_t, tau=taud)
        js = np.array([js_divergence_dict(raw_dists_list[i], dime_dists_list[i]) for i in range(len(val_values))])
        raw_shape = distribution_shape_stats(raw_dists_list)
        dime_shape = distribution_shape_stats(dime_dists_list)

        # ============================================================
        # RQ1 -- four-group coverage transition table
        # ============================================================
        groups = {
            "A_both_cover": raw_covers & dime_covers,
            "B_raw_only": raw_covers & ~dime_covers,
            "C_dime_only": ~raw_covers & dime_covers,
            "D_neither": ~raw_covers & ~dime_covers,
        }
        assert sum(m.sum() for m in groups.values()) == len(val_values), "coverage groups must partition all queries"
        rq1 = {}
        for name, mask in groups.items():
            idx = np.where(mask)[0]
            rq1[name] = {
                "n": int(mask.sum()), "pct": float(mask.mean() * 100),
                "p_raw": quantiles(p_raw[idx]) if len(idx) else None,
                "p_dime": quantiles(p_dime[idx]) if len(idx) else None,
                "js": quantiles(js[idx]) if len(idx) else None,
            }
            print(f"  [{name}] n={mask.sum()} ({mask.mean()*100:.2f}%)  "
                  f"mean_p_raw={rq1[name]['p_raw']['mean'] if rq1[name]['p_raw'] else None}  "
                  f"mean_p_dime={rq1[name]['p_dime']['mean'] if rq1[name]['p_dime'] else None}  "
                  f"mean_js={rq1[name]['js']['mean'] if rq1[name]['js'] else None}")
        cfg_report["RQ1_coverage_transitions"] = rq1

        # ============================================================
        # RQ2 -- probability/NLL changes on shared-coverage (group A) queries
        # ============================================================
        idx_A = np.where(groups["A_both_cover"])[0]
        delta_p = p_dime[idx_A] - p_raw[idx_A]
        delta_L_A = L_dime[idx_A] - L_raw[idx_A]
        increases, decreases = delta_p > 0, delta_p < 0
        degraded, improved = delta_L_A > 0, delta_L_A < 0
        prob_ratio = p_dime[idx_A] / p_raw[idx_A]

        rq2 = {
            "n_group_A": int(len(idx_A)),
            "frac_dime_increases_prob": float(increases.mean()),
            "frac_dime_decreases_prob": float(decreases.mean()),
            "mean_delta_p": float(delta_p.mean()), "median_delta_p": float(np.median(delta_p)),
            "mean_degradation_conditional": float(delta_L_A[degraded].mean()) if degraded.sum() else None,
            "median_degradation_conditional": float(np.median(delta_L_A[degraded])) if degraded.sum() else None,
            "mean_improvement_conditional": float(-delta_L_A[improved].mean()) if improved.sum() else None,
            "median_improvement_conditional": float(-np.median(delta_L_A[improved])) if improved.sum() else None,
            "degradation_percentiles": {p: float(np.percentile(delta_L_A[degraded], p)) for p in (90, 95, 99)} if degraded.sum() else None,
            "mean_signed_delta_L": float(delta_L_A.mean()), "median_signed_delta_L": float(np.median(delta_L_A)),
            "n_improved": int(improved.sum()), "n_degraded": int(degraded.sum()), "n_unchanged": int((delta_L_A == 0).sum()),
        }
        cfg_report["RQ2_shared_coverage_changes"] = rq2
        print(f"  RQ2: n_A={rq2['n_group_A']}  frac_increase={rq2['frac_dime_increases_prob']:.3f}  "
              f"mean_signed_deltaL={rq2['mean_signed_delta_L']:.4f}")

        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        axes[0].hist(delta_p, bins=100)
        axes[0].set_xlabel("delta p = P_DIME(y*) - P_raw(y*)"); axes[0].set_title(f"[{config_name}] RQ2: delta p (group A)")
        axes[1].hist(delta_L_A, bins=100)
        axes[1].set_xlabel("delta L (nats)"); axes[1].set_title(f"[{config_name}] RQ2: delta L (group A)")
        axes[2].hist(np.log10(prob_ratio[np.isfinite(prob_ratio) & (prob_ratio > 0)]), bins=100)
        axes[2].set_xlabel("log10(P_DIME / P_raw)"); axes[2].set_title(f"[{config_name}] RQ2: probability ratio (log scale)")
        fig.tight_layout()
        fig.savefig(os.path.join(PLOT_DIR, f"{config_name}_rq2_histograms.png"), dpi=130)
        plt.close(fig)

        # ============================================================
        # RQ3 -- coverage gain vs. probability degradation, plus controlled-alpha
        # interpolated NLL (sensitivity analysis, finite for all queries)
        # ============================================================
        n_effect_A = int(groups["C_dime_only"].sum())  # DIME covers, raw misses entirely
        n_effect_B = int(decreases.sum())  # within group A, DIME assigns LESS probability than raw

        _, nll_raw_interp = mix_knn_and_lm(d_raw_t, r_raw_t, val_values, val_p_lm, tau=taur, alpha=CONTROLLED_ALPHA)
        _, nll_dime_interp = mix_dime_and_lm(d_dime_t, r_dime_t, val_values, val_p_lm, tau=taud, alpha=CONTROLLED_ALPHA)
        delta_L_interp = nll_dime_interp - nll_raw_interp

        rq3 = {
            "effect_A_coverage_gain_n": n_effect_A, "effect_A_coverage_gain_pct": float(n_effect_A / len(val_values) * 100),
            "effect_B_probability_degradation_n": n_effect_B,
            "effect_B_probability_degradation_pct_of_group_A": float(n_effect_B / len(idx_A) * 100) if len(idx_A) else None,
            "group_A_severity": {"mean_signed_delta_L": rq2["mean_signed_delta_L"], "mean_delta_p": rq2["mean_delta_p"]},
            "controlled_alpha_sensitivity_analysis": {
                "alpha_used": CONTROLLED_ALPHA,
                "mean_nll_raw_interp": float(nll_raw_interp.mean()), "mean_nll_dime_interp": float(nll_dime_interp.mean()),
                "mean_delta_L_interp": float(delta_L_interp.mean()),
                "mean_delta_L_interp_by_group": {name: (float(delta_L_interp[mask].mean()) if mask.sum() else None)
                                                   for name, mask in groups.items()},
            },
        }
        cfg_report["RQ3_coverage_vs_degradation"] = rq3
        print(f"  RQ3: effect_A(coverage gain)={n_effect_A}  effect_B(prob degradation in A)={n_effect_B}  "
              f"controlled-alpha mean_delta_L_interp={rq3['controlled_alpha_sensitivity_analysis']['mean_delta_L_interp']:.4f}")

        # ============================================================
        # RQ4 -- distribution shape / concentration
        # ============================================================
        top1_agree = raw_shape["top1_token"] == dime_shape["top1_token"]
        delta_entropy = dime_shape["entropy"] - raw_shape["entropy"]
        rq4 = {
            "entropy": {"raw": quantiles(raw_shape["entropy"]), "dime": quantiles(dime_shape["entropy"])},
            "effective_support": {"raw": quantiles(raw_shape["effective_support"]), "dime": quantiles(dime_shape["effective_support"])},
            "max_prob": {"raw": quantiles(raw_shape["max_prob"]), "dime": quantiles(dime_shape["max_prob"])},
            "support_size": {"raw": quantiles(raw_shape["support_size"].astype(np.float64)), "dime": quantiles(dime_shape["support_size"].astype(np.float64))},
            "top1_agreement_fraction": float(top1_agree.mean()),
        }
        if len(idx_A) > 1:
            rq4["delta_entropy_vs_delta_p_groupA"] = correlations_with_ci(delta_entropy[idx_A], delta_p)
        cfg_report["RQ4_distribution_shape"] = rq4
        print(f"  RQ4: mean entropy raw={rq4['entropy']['raw']['mean']:.3f} dime={rq4['entropy']['dime']['mean']:.3f}  "
              f"top1_agreement={rq4['top1_agreement_fraction']:.3f}")

        # ============================================================
        # RQ5 -- JS divergence vs. NLL change
        # ============================================================
        both_finite = raw_covers & dime_covers  # == group A, same finite-paired subset
        with np.errstate(invalid="ignore"):
            delta_L_full = L_dime - L_raw  # inf-inf=nan where neither finite; only ever used after masking with both_finite
        corr_standalone = correlations_with_ci(js[both_finite], delta_L_full[both_finite])
        corr_standalone_abs = correlations_with_ci(js[both_finite], np.abs(delta_L_full[both_finite]))
        js_bins_standalone = binned_stats(js[both_finite], delta_L_full[both_finite])
        frac_improve_per_bin = []
        edges = np.quantile(js[both_finite], np.linspace(0, 1, 11))
        bin_idx_js = np.clip(np.digitize(js[both_finite], edges[1:-1]), 0, 9)
        dL_bf = delta_L_full[both_finite]
        for b in range(10):
            m = bin_idx_js == b
            if m.sum():
                frac_improve_per_bin.append({"bin": b, "n": int(m.sum()), "frac_improving": float((dL_bf[m] < 0).mean())})

        corr_interp = correlations_with_ci(js, delta_L_interp)  # repeated with interpolated NLL -- finite for all queries
        rq5 = {
            "standalone_finite_subset": {
                "n_used": int(both_finite.sum()), "js_vs_signed_deltaL": corr_standalone,
                "js_vs_abs_deltaL": corr_standalone_abs, "binned": js_bins_standalone, "frac_improving_per_js_bin": frac_improve_per_bin,
            },
            "interpolated_all_queries": {"js_vs_signed_deltaL_interp": corr_interp},
        }
        cfg_report["RQ5_js_vs_nll_change"] = rq5
        print(f"  RQ5 (standalone, finite subset n={rq5['standalone_finite_subset']['n_used']}): "
              f"js_vs_deltaL spearman={corr_standalone['spearman']['rho']:.4f}")
        print(f"  RQ5 (interpolated, all queries): js_vs_deltaL_interp spearman={corr_interp['spearman']['rho']:.4f}")

        plot_binned(js_bins_standalone, "JS divergence (bin mean)", "delta L (standalone, group A)",
                    f"[{config_name}] RQ5: JS vs delta L (standalone, finite subset)",
                    os.path.join(PLOT_DIR, f"{config_name}_rq5_js_vs_deltaL_standalone.png"))

        fig, ax = plt.subplots(figsize=(6, 6))
        m = both_finite[sample_idx]
        ax.scatter(js[sample_idx][m], delta_L_full[sample_idx][m], s=2, alpha=0.15)
        ax.set_xlabel("JS divergence"); ax.set_ylabel("delta L (standalone)")
        ax.set_title(f"[{config_name}] RQ5: JS vs signed delta L (standalone, group A)")
        fig.tight_layout()
        fig.savefig(os.path.join(PLOT_DIR, f"{config_name}_rq5_scatter_standalone.png"), dpi=130)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(6, 6))
        ax.scatter(js[sample_idx], delta_L_interp[sample_idx], s=2, alpha=0.15)
        ax.set_xlabel("JS divergence"); ax.set_ylabel("delta L (controlled-alpha interpolated, all queries)")
        ax.set_title(f"[{config_name}] RQ5: JS vs delta L (interpolated, all queries)")
        fig.tight_layout()
        fig.savefig(os.path.join(PLOT_DIR, f"{config_name}_rq5_scatter_interpolated.png"), dpi=130)
        plt.close(fig)

        report[config_name] = cfg_report
        save_results(os.path.join(OUT_DIR, "e1b2_summary.json"), report)

    print(f"\nAll E1B.2 results saved under {OUT_DIR}")
    print("E1/E1A/E1A.2/E1-followup's existing outputs were not touched.")


if __name__ == "__main__":
    main()
