"""E1B.4 -- Quantifying NLL Recovery from Within-Cluster Information. Extends
E1B.3 without modifying E1/E1A/E1A.2/E1B.2/E1B.3's existing outputs (writes only
to results/stage3_e1b4_nll_recovery/).

S1/S2/S3 definitions are UNCHANGED from E1B.3 (see that script's docstring for the
full equation writeup). This script evaluates all three on the FULL 77,089-query
val set, not E1B.3's 3,000-query JS-stratified sample -- the spec explicitly
prefers full scale "if computationally feasible", and it is, with one efficiency
fix E1B.3 didn't need at its smaller sample size: cluster membership is looked up
via np.where(assignment==c) *inside* a per-query, per-cluster loop in E1B.3; at
full scale that's ~1.5M repeated O(N_datastore) scans. Precomputing a
cluster_id -> member_indices dict ONCE (via a single argsort) turns every
subsequent lookup into O(1) dict indexing instead.

Interpolation weight alpha=0.25 is COMMON to all three systems in both the
primary (matched k=20,tau=2.0) and secondary (each system's own tuned k,tau)
analysis -- deliberately not reusing each system's own separately-tuned alpha,
per the explicit "use a common alpha for the paired performance comparison"
instruction. Standalone (no-epsilon) NLL is reported separately, only on the
shared-finite subset, never conflated with the always-finite interpolated NLL.
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
from e1_metrics import raw_standalone_prob, dime_standalone_prob, retrieval_coverage

CACHE_DIR = os.path.join(os.path.dirname(__file__), "..", "GPT_Module", "cache")
OUT_DIR = os.path.join(os.path.dirname(__file__), "results", "stage3_e1b4_nll_recovery")
PLOT_DIR = os.path.join(OUT_DIR, "plots")
os.makedirs(PLOT_DIR, exist_ok=True)

N_CLUSTERS = 3811
MATCHED = {"k": 20, "tau": 2.0}
INTERP_ALPHA = 0.25  # common to S1/S2/S3, both primary and secondary analysis
N_BOOT = 2000


def quantiles(x):
    x = np.asarray(x, dtype=np.float64)
    return {"mean": float(np.mean(x)), "median": float(np.median(x)),
            "p90": float(np.percentile(x, 90)), "p95": float(np.percentile(x, 95)), "p99": float(np.percentile(x, 99))}


def build_cluster_to_members(assignment, n_clusters):
    """O(N log N) once, via a single argsort -- not O(N) per cluster lookup."""
    order = np.argsort(assignment, kind="stable")
    sorted_assign = assignment[order]
    boundaries = np.searchsorted(sorted_assign, np.arange(n_clusters + 1))
    return {c: order[boundaries[c]:boundaries[c + 1]] for c in range(n_clusters)}


def compute_sys3_standalone(val_keys, val_values, idx_dime_t, d_dime_t, tau, cluster_to_members, ds_keys, ds_values, verbose_every=10000):
    """P_sys3(y_true|q) for every query -- full scale, using the precomputed
    cluster membership map. Returns p_sys3 [N] and, for a later JS/shape pass if
    needed, nothing else (kept lean: this script only needs the true-token prob
    for the NLL-recovery question, not full per-query distributions)."""
    N, k = idx_dime_t.shape
    w_dime = softmax_np(-d_dime_t / tau, axis=1)
    p_sys3 = np.zeros(N, dtype=np.float64)
    t0 = time.perf_counter()
    for i in range(N):
        q_key = val_keys[i]
        true_tok = int(val_values[i])
        acc = 0.0
        for j in range(k):
            c = idx_dime_t[i, j]
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
            rate = (i + 1) / elapsed
            eta = (N - i - 1) / rate
            print(f"    sys3 progress: {i+1}/{N}  ({rate:.1f} queries/sec, ETA {eta/60:.1f} min)")
    return p_sys3


def joint_bootstrap(arrays_dict, n_boot=N_BOOT, seed=42):
    """Resamples QUERY INDICES once per draw, applies to every array in
    arrays_dict identically -- paired/joint resampling across systems, as the
    spec requires for the recovery-fraction CI."""
    rng = np.random.default_rng(seed)
    n = len(next(iter(arrays_dict.values())))
    boot_means = {name: np.zeros(n_boot) for name in arrays_dict}
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        for name, arr in arrays_dict.items():
            boot_means[name][b] = arr[idx].mean()
    return boot_means


def recovery_fraction_with_ci(L_S1, L_S2, L_S3, n_boot=N_BOOT, seed=42, min_denom=1e-4):
    mean1, mean2, mean3 = L_S1.mean(), L_S2.mean(), L_S3.mean()
    denom = mean2 - mean1
    point_R = (mean2 - mean3) / denom if abs(denom) > min_denom else None

    boot = joint_bootstrap({"S1": L_S1, "S2": L_S2, "S3": L_S3}, n_boot=n_boot, seed=seed)
    denom_boot = boot["S2"] - boot["S1"]
    valid = np.abs(denom_boot) > min_denom
    R_boot = np.full(n_boot, np.nan)
    R_boot[valid] = (boot["S2"][valid] - boot["S3"][valid]) / denom_boot[valid]
    n_valid = int(valid.sum())

    result = {
        "mean_L_S1": float(mean1), "mean_L_S2": float(mean2), "mean_L_S3": float(mean3),
        "denominator_mean_L_S2_minus_L_S1": float(denom),
        "point_estimate": point_R,
        "n_bootstrap_draws_with_stable_denominator": n_valid, "n_bootstrap_total": n_boot,
    }
    if point_R is None:
        result["interpretability"] = "UNDEFINED/UNSTABLE: denominator (mean L_S2 - mean L_S1) is zero or too close to zero"
    elif n_valid < n_boot * 0.9:
        result["interpretability"] = (f"CAUTION: only {n_valid}/{n_boot} bootstrap draws had a stable denominator -- "
                                       "CI below is computed only over those, treat with care")
        lo, hi = np.percentile(R_boot[valid], [2.5, 97.5])
        result["ci_95_low"], result["ci_95_high"] = float(lo), float(hi)
    else:
        lo, hi = np.percentile(R_boot[valid], [2.5, 97.5])
        result["interpretability"] = "stable"
        result["ci_95_low"], result["ci_95_high"] = float(lo), float(hi)
    return result


def paired_comparison(L_a, L_b, name_a, name_b, n_boot=N_BOOT, seed=42):
    delta = L_a - L_b
    boot = joint_bootstrap({"a": L_a, "b": L_b}, n_boot=n_boot, seed=seed)
    delta_boot = boot["a"] - boot["b"]
    lo, hi = np.percentile(delta_boot, [2.5, 97.5])
    return {
        "comparison": f"{name_a} - {name_b}", "mean_diff": float(delta.mean()), "median_diff": float(np.median(delta)),
        "p90": float(np.percentile(delta, 90)), "p95": float(np.percentile(delta, 95)), "p99": float(np.percentile(delta, 99)),
        "ci_95_low": float(lo), "ci_95_high": float(hi),
        "frac_a_improves_over_b": float((delta < 0).mean()), "frac_a_worsens_vs_b": float((delta > 0).mean()),
    }


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
    print(f"val={len(val_values)}  raw_tuned={raw_tuned}  dime_tuned={dime_tuned}  matched={MATCHED}  "
          f"common interpolation alpha={INTERP_ALPHA}")

    print("Re-deriving DIME clusters (seed=42, deterministic, identical to E1/E1A.2/E1B.2/E1B.3)...")
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

    print("Precomputing cluster -> member-indices map (one argsort, not per-cluster scans)...")
    t0 = time.perf_counter()
    cluster_to_members = build_cluster_to_members(assignment, N_CLUSTERS)
    print(f"done in {time.perf_counter()-t0:.2f}s")

    raw_index, raw_values = build_datastore(ds_keys, ds_values)
    dime_index, dime_values = build_datastore(dime_centroids, compressed_dists)

    configs = {"matched": (MATCHED["k"], MATCHED["tau"], MATCHED["k"], MATCHED["tau"]),
               "tuned": (raw_tuned["k"], raw_tuned["tau"], dime_tuned["k"], dime_tuned["tau"])}
    k_max_raw = max(MATCHED["k"], raw_tuned["k"])
    k_max_dime = max(MATCHED["k"], dime_tuned["k"])
    d_raw_full, idx_raw_full = query_knn_indices(raw_index, val_keys, k=k_max_raw)
    d_dime_full, idx_dime_full = query_knn_indices(dime_index, val_keys, k=k_max_dime)
    retrieved_raw_full = raw_values[idx_raw_full]

    report = {"n_total_queries": int(len(val_values)), "matched_config": MATCHED,
              "tuned_config": {"raw": raw_tuned, "dime": dime_tuned}, "common_interpolation_alpha": INTERP_ALPHA,
              "evaluation_scale": "full 77089-query val set (no subsampling -- confirmed computationally feasible "
                                   "after precomputing cluster membership once)"}

    for config_name, (kr, taur, kd, taud) in configs.items():
        print(f"\n################ CONFIG: {config_name} (raw k={kr},tau={taur} | dime k={kd},tau={taud}) ################")
        cfg_report = {}
        d_raw_t, r_raw_t = d_raw_full[:, :kr], retrieved_raw_full[:, :kr]
        d_dime_t, idx_dime_t = d_dime_full[:, :kd], idx_dime_full[:, :kd]

        print("Computing S1 (raw) and S2 (DIME) standalone probabilities...")
        p_raw = raw_standalone_prob(d_raw_t, r_raw_t, val_values, tau=taur)
        p_dime = dime_standalone_prob(d_dime_t, dime_values[idx_dime_t], val_values, tau=taud)

        # Validation: S1/S2 reproduce established probabilities under an independent
        # recomputation using the SAME already-validated e1_metrics functions (not new
        # code for S1/S2 -- only S3 is new here).
        p_raw_check = raw_standalone_prob(d_raw_t, r_raw_t, val_values, tau=taur)
        p_dime_check = dime_standalone_prob(d_dime_t, dime_values[idx_dime_t], val_values, tau=taud)
        assert np.array_equal(p_raw, p_raw_check) and np.array_equal(p_dime, p_dime_check), \
            "S1/S2 non-deterministic across calls -- investigate before trusting anything downstream"
        print("  validation: S1/S2 reproduce identically across independent calls -- OK")

        if config_name == "matched":
            try:
                with open("results/stage3_e1b2_coverage_concentration/e1b2_summary.json") as f:
                    e1b2 = json.load(f)
                prior_cov = (e1b2["matched"]["RQ1_coverage_transitions"]["A_both_cover"]["n"]
                             + e1b2["matched"]["RQ1_coverage_transitions"]["C_dime_only"]["n"]) / e1b2["n_total_queries"]
                this_cov = retrieval_coverage(p_dime)
                match = abs(prior_cov - this_cov) < 1e-6
                print(f"  cross-check vs. E1B.2's persisted matched-config DIME coverage: "
                      f"prior={prior_cov:.6f}  this_run={this_cov:.6f}  match={match}")
            except FileNotFoundError:
                print("  cross-check vs. E1B.2 skipped: results/stage3_e1b2_coverage_concentration/e1b2_summary.json "
                      "not found locally (not fatal -- only runs if that file has been pulled)")

        print(f"Computing S3 (query-conditioned, full {len(val_values)} queries) -- this is the new, expensive part...")
        p_sys3 = compute_sys3_standalone(val_keys, val_values, idx_dime_t, d_dime_t, taud,
                                          cluster_to_members, ds_keys, ds_values)

        # Validation: S3's cluster selection matches S2's exactly (same idx_dime_t used for both).
        print("  validation: S3 uses the identical idx_dime_t/d_dime_t as S2 by construction (same arrays, not recomputed)")

        with np.errstate(divide="ignore"):
            L_raw_standalone = -np.log(p_raw)
            L_dime_standalone = -np.log(p_dime)
            L_sys3_standalone = -np.log(p_sys3)

        # ============================================================
        # Primary/secondary: interpolated NLL, common alpha for all 3 systems
        # ============================================================
        _, L_S1 = mix_knn_and_lm(d_raw_t, r_raw_t, val_values, val_p_lm, tau=taur, alpha=INTERP_ALPHA)
        _, L_S2 = mix_dime_and_lm(d_dime_t, dime_values[idx_dime_t], val_values, val_p_lm, tau=taud, alpha=INTERP_ALPHA)
        p_mixed_s3 = INTERP_ALPHA * p_sys3 + (1 - INTERP_ALPHA) * val_p_lm
        L_S3 = -np.log(p_mixed_s3 + 1e-12)  # same epsilon convention as mix_knn_and_lm/mix_dime_and_lm

        cfg_report["interpolated_nll"] = {
            "mean_S1": float(L_S1.mean()), "mean_S2": float(L_S2.mean()), "mean_S3": float(L_S3.mean()),
            "S2_minus_S1": paired_comparison(L_S2, L_S1, "S2", "S1"),
            "S3_minus_S1": paired_comparison(L_S3, L_S1, "S3", "S1"),
            "S2_minus_S3": paired_comparison(L_S2, L_S3, "S2", "S3"),
        }
        rec = recovery_fraction_with_ci(L_S1, L_S2, L_S3)
        cfg_report["recovery_fraction"] = rec
        print(f"  interpolated: mean_S1={L_S1.mean():.4f}  mean_S2={L_S2.mean():.4f}  mean_S3={L_S3.mean():.4f}")
        print(f"  recovery fraction R={rec['point_estimate']}  ({rec['interpretability']})")

        # ============================================================
        # Standalone NLL on shared-finite subset (no epsilon, coverage reported explicitly)
        # ============================================================
        raw_covers, dime_covers, sys3_covers = p_raw > 0, p_dime > 0, p_sys3 > 0
        shared_finite = raw_covers & dime_covers & sys3_covers
        cfg_report["standalone_coverage"] = {
            "S1_coverage": retrieval_coverage(p_raw), "S2_coverage": retrieval_coverage(p_dime),
            "S3_coverage": retrieval_coverage(p_sys3), "n_shared_finite_all_three": int(shared_finite.sum()),
            "n_total": int(len(val_values)),
        }
        if shared_finite.sum() > 1:
            cfg_report["standalone_nll_shared_finite_subset"] = {
                "S1": quantiles(L_raw_standalone[shared_finite]), "S2": quantiles(L_dime_standalone[shared_finite]),
                "S3": quantiles(L_sys3_standalone[shared_finite]),
                "S2_minus_S1": paired_comparison(L_dime_standalone[shared_finite], L_raw_standalone[shared_finite], "S2", "S1"),
                "S3_minus_S1": paired_comparison(L_sys3_standalone[shared_finite], L_raw_standalone[shared_finite], "S3", "S1"),
                "S2_minus_S3": paired_comparison(L_dime_standalone[shared_finite], L_sys3_standalone[shared_finite], "S2", "S3"),
            }
        print(f"  standalone coverage: S1={cfg_report['standalone_coverage']['S1_coverage']:.4f}  "
              f"S2={cfg_report['standalone_coverage']['S2_coverage']:.4f}  S3={cfg_report['standalone_coverage']['S3_coverage']:.4f}  "
              f"shared_finite_n={shared_finite.sum()}")

        # ============================================================
        # Coverage-group decomposition (using standalone raw/DIME coverage, as
        # established in E1B.2 -- S3's own coverage is reported per group too)
        # ============================================================
        groups = {"A_both_cover": raw_covers & dime_covers, "B_raw_only": raw_covers & ~dime_covers,
                  "C_dime_only": ~raw_covers & dime_covers, "D_neither": ~raw_covers & ~dime_covers}
        assert sum(m.sum() for m in groups.values()) == len(val_values)
        group_report = {}
        total_gap = float((L_S2 - L_S1).mean())
        total_recovery = float((L_S2 - L_S3).mean())
        for name, mask in groups.items():
            n = int(mask.sum())
            if n == 0:
                group_report[name] = {"n": 0}
                continue
            s2_minus_s1_group = float((L_S2[mask] - L_S1[mask]).mean())
            s2_minus_s3_group = float((L_S2[mask] - L_S3[mask]).mean())
            group_report[name] = {
                "n": n, "frac_of_total": float(mask.mean()),
                "mean_L_S1": float(L_S1[mask].mean()), "mean_L_S2": float(L_S2[mask].mean()), "mean_L_S3": float(L_S3[mask].mean()),
                "mean_S2_minus_S1": s2_minus_s1_group, "mean_S2_minus_S3": s2_minus_s3_group,
                "contribution_to_total_gap": float((mask.sum() / len(val_values)) * s2_minus_s1_group / total_gap) if total_gap != 0 else None,
                "contribution_to_total_recovery": float((mask.sum() / len(val_values)) * s2_minus_s3_group / total_recovery) if total_recovery != 0 else None,
            }
            print(f"  [{name}] n={n} ({mask.mean()*100:.2f}%)  S2-S1={s2_minus_s1_group:.4f}  S2-S3={s2_minus_s3_group:.4f}")
        cfg_report["coverage_group_decomposition"] = group_report

        # ============================================================
        # Plots
        # ============================================================
        sample_idx = np.random.default_rng(0).choice(len(val_values), size=min(20000, len(val_values)), replace=False)
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        axes[0].scatter((L_S2 - L_S1)[sample_idx], (L_S2 - L_S3)[sample_idx], s=2, alpha=0.1)
        axes[0].axhline(0, color="gray", linewidth=0.5); axes[0].axvline(0, color="gray", linewidth=0.5)
        axes[0].set_xlabel("S2 - S1 (interpolated NLL gap vs. raw)")
        axes[0].set_ylabel("S2 - S3 (interpolated NLL recovered)")
        axes[0].set_title(f"[{config_name}] gap vs. recovery (interpolated)")
        axes[1].bar(["S1 (raw)", "S2 (DIME)", "S3 (DIME sel.\n+query-cond.)"],
                    [L_S1.mean(), L_S2.mean(), L_S3.mean()])
        axes[1].set_ylabel("mean interpolated NLL")
        axes[1].set_title(f"[{config_name}] mean NLL by system")
        fig.tight_layout()
        fig.savefig(os.path.join(PLOT_DIR, f"{config_name}_recovery_overview.png"), dpi=130)
        plt.close(fig)

        # Empty groups (n=0, possible at small/sparse scales) have no mean_S2_minus_S1/S3
        # fields -- excluded from the bar chart entirely rather than plotted as a crash
        # or a misleading zero-height bar.
        plot_names = [n for n in groups if group_report[n]["n"] > 0]
        if len(plot_names) != len(groups):
            print(f"  note: {len(groups) - len(plot_names)} empty coverage group(s) excluded from the bar chart: "
                  f"{[n for n in groups if n not in plot_names]}")
        fig, ax = plt.subplots(figsize=(8, 5))
        gaps = [group_report[n]["mean_S2_minus_S1"] for n in plot_names]
        recov = [group_report[n]["mean_S2_minus_S3"] for n in plot_names]
        x = np.arange(len(plot_names))
        ax.bar(x - 0.2, gaps, width=0.4, label="S2-S1 (gap vs raw)")
        ax.bar(x + 0.2, recov, width=0.4, label="S2-S3 (recovered)")
        ax.set_xticks(x); ax.set_xticklabels(plot_names, rotation=20)
        ax.set_ylabel("mean interpolated NLL difference")
        ax.set_title(f"[{config_name}] gap and recovery by coverage group")
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(PLOT_DIR, f"{config_name}_recovery_by_group.png"), dpi=130)
        plt.close(fig)

        report[config_name] = cfg_report
        save_results(os.path.join(OUT_DIR, "e1b4_summary.json"), report)

    print(f"\nAll E1B.4 results saved under {OUT_DIR}")
    print("E1/E1A/E1A.2/E1B.2/E1B.3's existing outputs were not touched.")


if __name__ == "__main__":
    main()
