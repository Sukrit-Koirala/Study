"""E1A.2 — Distance-Prediction Relationship. Extends E1 without touching its
existing outputs (results/stage3_e1_wikitext103/, results/stage3_e1_followup/ are
untouched; this writes to its own results/stage3_e1a2_distance_prediction/).

Reuses exactly the same setup as run_e1_followup.py (same cache, same seed=42
DIME re-clustering -- necessary since centroids were never persisted, same tuned
(k,tau) retrieval hyperparameters) -- no new extraction, no new tuning, no new
data split. The one thing this script does differently from E1/E1-followup: it
uses STANDALONE retrieval NLL (L = -log(p_standalone), the probability mass the
retrieval system itself assigns the true token, no interpolation with the base
LM), per the explicit instruction to use standalone distributions as the primary
measurement here, not the mixed/interpolated predictions E1C analyzed.

Numerical stability: standalone probabilities can be EXACTLY zero (true token
absent from every retrieved neighbor/cluster), making L = -log(0) = +inf. No
epsilon is added -- infinite losses are counted and reported explicitly, then
excluded (with the exclusion count reported) from any correlation that would
otherwise be undefined.
"""
import sys, os, json, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "GPT_Module"))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from cache_io import load_cache
from knn import build_datastore, query_knn_indices
from state_object import minibatch_kmeans_partition
from results_io import save_results
from e1_metrics import (
    raw_standalone_prob, dime_standalone_prob, js_divergence_dict, raw_full_distribution, dime_full_distribution,
    correlations_with_ci, binned_stats, plot_binned,
)

CACHE_DIR = os.path.join(os.path.dirname(__file__), "..", "GPT_Module", "cache")
OUT_DIR = os.path.join(os.path.dirname(__file__), "results", "stage3_e1a2_distance_prediction")
PLOT_DIR = os.path.join(OUT_DIR, "plots")
os.makedirs(PLOT_DIR, exist_ok=True)

N_CLUSTERS = 3811


def main():
    print("Loading cache (no extraction)...")
    ds = load_cache(os.path.join(CACHE_DIR, "wikitext103_gpt2_datastore.npz"))
    val = load_cache(os.path.join(CACHE_DIR, "wikitext103_gpt2_val.npz"))
    with open("results/wikitext103_gpt2_tier1.json") as f:
        tier1 = json.load(f)
    raw_tuned, dime_tuned = tier1["raw_knn"]["best_config"], tier1["dime_minibatch_kmeans"]["best_config"]
    assert tier1["n_clusters"] == N_CLUSTERS
    val_values = val["values"]
    print(f"val={len(val_values)}  raw_tuned={raw_tuned}  dime_tuned={dime_tuned}")

    print("Building raw index, re-deriving DIME clusters (seed=42, deterministic, same as E1/follow-up)...")
    raw_index, raw_values = build_datastore(ds["keys"], ds["values"])
    t0 = time.perf_counter()
    dime_centroids, dime_dists = minibatch_kmeans_partition(ds["keys"], ds["values"], n_clusters=N_CLUSTERS, seed=42)
    dime_dists = np.array(dime_dists, dtype=object)
    dime_index, dime_values = build_datastore(dime_centroids, dime_dists)
    print(f"clustering done in {time.perf_counter()-t0:.1f}s")

    kr, taur = raw_tuned["k"], raw_tuned["tau"]
    kd, taud = dime_tuned["k"], dime_tuned["tau"]
    d_raw_full, idx_raw_full = query_knn_indices(raw_index, val["keys"], k=max(kr, 1))
    d_dime_full, idx_dime_full = query_knn_indices(dime_index, val["keys"], k=max(kd, 1))
    retrieved_raw = raw_values[idx_raw_full][:, :kr]
    retrieved_dime = dime_values[idx_dime_full][:, :kd]
    d_raw_t, d_dime_t = d_raw_full[:, :kr], d_dime_full[:, :kd]

    d_raw_nearest, d_dime_nearest = d_raw_full[:, 0], d_dime_full[:, 0]
    delta_d = d_dime_nearest - d_raw_nearest

    print("Computing standalone retrieval probabilities and NLL (no epsilon)...")
    p_raw = raw_standalone_prob(d_raw_t, retrieved_raw, val_values, tau=taur)
    p_dime = dime_standalone_prob(d_dime_t, retrieved_dime, val_values, tau=taud)
    with np.errstate(divide="ignore"):
        L_raw = -np.log(p_raw)
        L_dime = -np.log(p_dime)

    n_raw_inf, n_dime_inf = int(np.isinf(L_raw).sum()), int(np.isinf(L_dime).sum())
    print(f"non-finite L_raw: {n_raw_inf} ({n_raw_inf/len(L_raw)*100:.2f}%)  "
          f"non-finite L_dime: {n_dime_inf} ({n_dime_inf/len(L_dime)*100:.2f}%)")

    report = {"n_total_queries": int(len(val_values)), "tuned_config": {"raw": raw_tuned, "dime": dime_tuned},
              "n_L_raw_infinite": n_raw_inf, "n_L_dime_infinite": n_dime_inf}

    # ============================================================
    # RQ1 — distance vs. standalone NLL, raw and DIME separately
    # ============================================================
    print("\n=== RQ1: distance vs. predictive quality ===")
    finite_raw = np.isfinite(L_raw)
    finite_dime = np.isfinite(L_dime)
    rq1_raw = correlations_with_ci(d_raw_nearest[finite_raw], L_raw[finite_raw])
    rq1_dime = correlations_with_ci(d_dime_nearest[finite_dime], L_dime[finite_dime])
    report["RQ1"] = {
        "raw": {"correlations": rq1_raw, "n_excluded_infinite": int((~finite_raw).sum())},
        "dime": {"correlations": rq1_dime, "n_excluded_infinite": int((~finite_dime).sum())},
    }
    print(f"raw: d_raw vs L_raw -> {rq1_raw}")
    print(f"dime: d_dime vs L_dime -> {rq1_dime}")

    sample_idx = np.random.default_rng(0).choice(len(val_values), size=min(20000, len(val_values)), replace=False)
    fig, ax = plt.subplots(figsize=(6, 6))
    m = finite_raw[sample_idx]
    ax.scatter(d_raw_nearest[sample_idx][m], L_raw[sample_idx][m], s=2, alpha=0.15)
    ax.set_xlabel("raw nearest-neighbor distance")
    ax.set_ylabel("L_raw = -log P_raw_standalone(true token)")
    ax.set_title("RQ1: raw kNN-LM distance vs. standalone NLL")
    fig.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, "raw_distance_vs_nll.png"), dpi=130)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 6))
    m = finite_dime[sample_idx]
    ax.scatter(d_dime_nearest[sample_idx][m], L_dime[sample_idx][m], s=2, alpha=0.15)
    ax.set_xlabel("DIME nearest-centroid distance")
    ax.set_ylabel("L_DIME = -log P_DIME_standalone(true token)")
    ax.set_title("RQ1: DIME distance vs. standalone NLL")
    fig.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, "dime_distance_vs_nll.png"), dpi=130)
    plt.close(fig)

    raw_bins = binned_stats(d_raw_nearest[finite_raw], L_raw[finite_raw])
    dime_bins = binned_stats(d_dime_nearest[finite_dime], L_dime[finite_dime])
    report["RQ1"]["raw"]["binned"] = raw_bins
    report["RQ1"]["dime"]["binned"] = dime_bins
    plot_binned(raw_bins, "raw nearest distance (bin mean)", "L_raw", "RQ1: binned raw distance vs. NLL",
                os.path.join(PLOT_DIR, "raw_distance_vs_nll_binned.png"))
    plot_binned(dime_bins, "DIME nearest-centroid distance (bin mean)", "L_DIME", "RQ1: binned DIME distance vs. NLL",
                os.path.join(PLOT_DIR, "dime_distance_vs_nll_binned.png"))

    # ============================================================
    # RQ2 — signed delta_d vs. signed delta_L (standalone)
    # ============================================================
    print("\n=== RQ2: distance change vs. prediction change ===")
    both_finite = finite_raw & finite_dime
    with np.errstate(invalid="ignore"):
        delta_L = L_dime - L_raw  # inf-inf=nan where neither is finite; only ever used after masking with both_finite
    n_excluded_rq2 = int((~both_finite).sum())
    rq2_corr = correlations_with_ci(delta_d[both_finite], delta_L[both_finite])
    report["RQ2"] = {"correlations": rq2_corr, "n_excluded_nonfinite": n_excluded_rq2,
                      "n_used": int(both_finite.sum())}
    print(f"delta_d vs delta_L -> {rq2_corr}  (excluded {n_excluded_rq2} non-finite pairs)")

    fig, ax = plt.subplots(figsize=(6, 6))
    m2 = both_finite[sample_idx]
    ax.scatter(delta_d[sample_idx][m2], delta_L[sample_idx][m2], s=2, alpha=0.15)
    ax.axhline(0, color="gray", linewidth=0.5)
    ax.axvline(0, color="gray", linewidth=0.5)
    ax.set_xlabel("signed delta d = d_DIME - d_raw")
    ax.set_ylabel("signed delta L = L_DIME - L_raw (standalone)")
    ax.set_title("RQ2: distance change vs. prediction change")
    fig.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, "delta_d_vs_delta_L.png"), dpi=130)
    plt.close(fig)

    rq2_bins = binned_stats(delta_d[both_finite], delta_L[both_finite])
    report["RQ2"]["binned"] = rq2_bins
    plot_binned(rq2_bins, "signed delta d (bin mean)", "delta L", "RQ2: binned delta_d vs. delta_L (standalone)",
                os.path.join(PLOT_DIR, "delta_d_vs_delta_L_binned.png"))

    # ============================================================
    # RQ3 — the 29.38% phenomenon: closer / farther / approximately-equal groups
    # ============================================================
    print("\n=== RQ3: closer-centroid vs. farther-centroid groups ===")
    tol = 0.01 * float(np.median(d_raw_nearest))  # explicit, scale-aware tolerance: 1% of median raw distance
    closer = delta_d < -tol
    farther = delta_d > tol
    equal = ~closer & ~farther
    print(f"tolerance used for 'approximately equal': {tol:.6f} (1% of median raw nearest distance)")

    raw_dists_list = raw_full_distribution(d_raw_t, retrieved_raw, tau=taur)
    dime_dists_list = dime_full_distribution(d_dime_t, retrieved_dime, tau=taud)

    rq3 = {"tolerance_used": tol}
    for name, mask in {"closer_centroid": closer, "farther_centroid": farther, "approximately_equal": equal}.items():
        idx = np.where(mask & both_finite)[0]
        if len(idx) == 0:
            rq3[name] = {"n": 0}
            continue
        js_group = np.array([js_divergence_dict(raw_dists_list[i], dime_dists_list[i]) for i in idx])
        rq3[name] = {
            "n": int(mask.sum()), "frac_of_total": float(mask.mean()),
            "mean_delta_d": float(delta_d[mask].mean()), "median_delta_d": float(np.median(delta_d[mask])),
            "mean_delta_L": float(delta_L[idx].mean()), "median_delta_L": float(np.median(delta_L[idx])),
            "frac_dime_improves_standalone": float((L_dime[idx] < L_raw[idx]).mean()),
            "mean_js": float(js_group.mean()), "median_js": float(np.median(js_group)),
            "n_used_for_delta_L_js": int(len(idx)),
        }
        print(f"[{name}] n={mask.sum()} ({mask.mean()*100:.2f}%)  mean_delta_L={rq3[name]['mean_delta_L']:.4f}  "
              f"frac_dime_improves={rq3[name]['frac_dime_improves_standalone']:.4f}  mean_js={rq3[name]['mean_js']:.4f}")

        # Improvement/degradation magnitude comparison (closer vs. farther only), reusing
        # the exact same finite-paired idx/delta_L/tol already computed above -- no new
        # retrieval, no new tuning, no change to any prior E1/E1A.2 field.
        if name in ("closer_centroid", "farther_centroid"):
            dL = delta_L[idx]
            improved = dL < 0
            worsened = dL > 0
            n_improved, n_worsened = int(improved.sum()), int(worsened.sum())
            rq3[name]["magnitude_comparison"] = {
                "mean_improvement_magnitude_mean_neg_deltaL": float(-dL[improved].mean()) if n_improved > 0 else None,
                "mean_degradation_mean_deltaL": float(dL[worsened].mean()) if n_worsened > 0 else None,
                "p95_degradation_among_worsened": float(np.percentile(dL[worsened], 95)) if n_worsened > 0 else None,
                "n_improved": n_improved, "n_worsened": n_worsened,
                "frac_dime_improves_standalone": float(improved.mean()),
                "n_excluded_infinite_in_group": int(mask.sum()) - int(len(idx)),
            }
            consistency_ok = abs(rq3[name]["magnitude_comparison"]["frac_dime_improves_standalone"]
                                  - rq3[name]["frac_dime_improves_standalone"]) < 1e-9
            print(f"  [{name}] magnitude_comparison: {rq3[name]['magnitude_comparison']}  "
                  f"(consistent with existing frac_dime_improves: {consistency_ok})")

    report["RQ3"] = rq3

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    group_names = ["closer_centroid", "farther_centroid", "approximately_equal"]
    delta_L_means = [rq3[g].get("mean_delta_L", 0) for g in group_names]
    improve_fracs = [rq3[g].get("frac_dime_improves_standalone", 0) for g in group_names]
    axes[0].bar(group_names, delta_L_means)
    axes[0].set_ylabel("mean delta L (standalone)")
    axes[0].set_title("RQ3: mean prediction change by group")
    axes[1].bar(group_names, improve_fracs)
    axes[1].set_ylabel("fraction where DIME improves (standalone)")
    axes[1].set_title("RQ3: improvement rate by group")
    fig.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, "group_comparison.png"), dpi=130)
    plt.close(fig)

    save_results(os.path.join(OUT_DIR, "e1a2_summary.json"), report)
    print(f"\nAll E1A.2 results saved under {OUT_DIR}")
    print("E1's existing outputs (stage3_e1_wikitext103/, stage3_e1_followup/) were not touched.")


if __name__ == "__main__":
    main()
