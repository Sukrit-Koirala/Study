"""E2-KM1-DENSE -- Dense Token-Conditioned Kernel Mean Experiment. Extends
the E1/E2/TRI1/NC1/GEOM1 series without modifying any existing output
(writes only to results/stage3_e2km1_dense_kernel_mean/). Deliberately
dense/uncompressed by design -- no sparse approximation, prototype
selection, or learned compression here, per the spec's own explicit
instruction.

=====================================================================
Equivalence identification (required by the spec before any other work,
Section 5's "If E2-GEOM1 already implements essentially the same
kernel-weighted computation, explicitly identify the equivalence instead of
presenting the method as new"):
=====================================================================
This spec's System A / baseline B4 ("Dense token-conditioned kernel
estimator", Section 3) is, term for term, THE SAME computation as
E2-GEOM1's System B ("full original within-centroid local retrieval"):
  - S_{c,y}(q) = sum over same-cluster, same-label original hidden states of
    K(q,h_i), normalized within the centroid -- GEOM1's
    local_retrieval_prob's inner softmax-weighted vote over a cluster's
    FULL original membership (no capping), normalized the same way.
  - Combined via the SAME outer DIME routing weights w_c(q) -- GEOM1's
    local_retrieval_prob does exactly this (softmax_np(-d_dime/tau_outer)).
  - Kernel Variant A here, K1(q,h)=exp(-||q-h||_2/tau), is EXACTLY GEOM1's
    System B kernel (softmax_np(-d_local/tau_local) is, term by term, the
    same normalized exponential-distance kernel).
  - Interpolated with the base LM the same way (alpha * P_kernel + (1-alpha)
    * P_LM), matched config alpha=0.25.
There is no independent "B3 vs. B4(Kernel A)" distinction to draw: this
script computes that ONE full-validation pass once (reusing GEOM1's own
local_retrieval_prob, imported directly) and reports it as both B3 (E2-GEOM1
reproduction) and B4-KernelA simultaneously, rather than wastefully
recomputing the identical ~30-40-minute pass twice. Kernel Variant B
(Gaussian squared-distance) is the one genuinely NEW estimator this script
adds beyond what GEOM1 already computed.

Per the spec's own "verify whether E2-GEOM1 used exactly the same retrieval
and interpolation settings before making direct claims" instruction: this
script does NOT cite GEOM1's previously-reported numbers from outside this
repository (its real persisted results/stage3_e2geom1_within_centroid/
summary was not available locally when this script was written). It
reruns System B's exact computation itself, under the SAME matched config
(k=20, tau=2.0, alpha=0.25) and the SAME seed=42 clustering, so every number
in this script's own output is independently, freshly verified here -- not
assumed from an external citation.

Reuses exactly, not reimplemented: cache loading, seed=42 MiniBatchKMeans
clustering, matched config, knn.py/mixing.py retrieval, e1_metrics.py
helpers, run_e2_information_allocation's build_cluster_to_members, and
run_e2geom1_within_centroid_geometry's local_retrieval_prob (imported
directly for Kernel Variant A / System B / B3).

No scikit-learn in any new computation here; the only sklearn usage
project-wide remains the pre-existing MiniBatchKMeans clustering.

=====================================================================
Scoping decisions (stated up front):
=====================================================================
  - B0 (base LM), B1 (raw kNN), B2 (DIME-V1): recomputed fresh (cheap),
    matched config, same convention as every prior script in this series.
  - B3 = B4-KernelA (exponential kernel, full-cluster, no capping): ONE
    full-val pass, tau_local tuned on controller_train via a grid informed
    by the ACTUAL empirical local-distance distribution (not reused blindly
    from GEOM1's own grid, even though the kernel formula is identical --
    satisfies the spec's explicit "grid informed by the actual distance
    distributions" instruction).
  - B4-KernelB (Gaussian squared-distance kernel): ONE new full-val pass,
    sigma tuned the same way.
  - B5 (within-centroid label-shuffle control): computed on a bounded
    val subsample (not full val), using Kernel A's chosen tau_local -- the
    full-cluster local-retrieval computation is identical in cost structure
    to B3/B4-KernelA itself, so a third full-scale pass is avoided here.
  - B6 (uniform-kernel identity check): computed on a bounded verification
    sample and asserted (not merely reported) to reproduce DIME-V1's real
    standalone probability within floating-point tolerance -- the spec
    calls this check "essential."
  - Diagnostic B (query-conditioning vs. improvement) and Diagnostic C
    (entropy/correct-prob/NLL jointly) need FULL per-query retrieval
    distributions (not just the true-token probability), which is
    meaningfully more expensive per query than the scalar NLL grid --
    computed on a bounded val subsample, stated explicitly, per the spec's
    own "do not substitute subsample results for full-validation metrics
    without explicitly stating the limitation."
  - Diagnostic D (grouped failure analysis) uses only scalar per-query
    covariates (centroid population/entropy/token-frequency/local
    concentration) that are cheap at full scale, so it runs on the FULL
    77,089-query validation set.
  - GEOM1's own separately-reported "anchor-space retrieval" number is
    cited in this script's report ONLY as contextual background (clearly
    labeled as not independently reverified here), since it is not one of
    this spec's required B0-B6 baselines.

Smoke testing: set E2KM1_SMOKE=1 with E2KM1_CACHE_DIR / E2KM1_OUT_DIR /
E2KM1_N_CLUSTERS before running -- same convention as TRI1/NC1/GEOM1. Never
set it for the real run.
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
from e1_metrics import (raw_standalone_prob, dime_standalone_prob, raw_full_distribution, dime_full_distribution,
                          js_divergence_dict, retrieval_coverage, distribution_shape_stats)
from run_e2_information_allocation import build_cluster_to_members
from run_e2geom1_within_centroid_geometry import local_retrieval_prob

SMOKE = os.environ.get("E2KM1_SMOKE") == "1"
CACHE_DIR = os.environ.get("E2KM1_CACHE_DIR", os.path.join(SCRIPT_DIR, "..", "GPT_Module", "cache"))
OUT_DIR = os.environ.get("E2KM1_OUT_DIR", os.path.join(SCRIPT_DIR, "results", "stage3_e2km1_dense_kernel_mean"))
PLOT_DIR = os.path.join(OUT_DIR, "plots")
os.makedirs(PLOT_DIR, exist_ok=True)

N_CLUSTERS = int(os.environ.get("E2KM1_N_CLUSTERS", "3811"))
MATCHED = {"k": 20, "tau": 2.0, "alpha": 0.25}
SEED = 42
DIM, DTYPE_BYTES = 768, 4

N_PROBE_FOR_GRID = 500  # queries used to empirically probe local distance scale before building the tau/sigma grids
N_SUBSAMPLE_B5 = 5000   # B5 shuffle control
N_SUBSAMPLE_DIAG = 3000  # Diagnostics B/C (need full distributions)
N_VERIFY_B6 = 1000       # B6 identity-check verification sample
N_BOOT = 2000
WORST_PCTS = [1, 5, 10]
VERBOSE_EVERY = 20000 if not SMOKE else 200


def nll_from_standalone(p_true, p_lm_true, alpha):
    return -np.log(alpha * p_true + (1 - alpha) * p_lm_true + 1e-12)


def paired_bootstrap_mean_diff_ci(a, b, n_boot=N_BOOT, seed=SEED):
    """95% CI for mean(a) - mean(b), paired (same queries), via the standard
    paired bootstrap (resample query indices jointly for both arrays)."""
    rng = np.random.default_rng(seed)
    n = len(a)
    diffs = a - b
    point = float(np.mean(diffs))
    boot = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boot[i] = np.mean(diffs[idx])
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return {"mean_diff": point, "ci_95_low": float(lo), "ci_95_high": float(hi), "n": int(n)}


def local_retrieval_prob_gaussian(query_keys, true_values, idx_dime, d_dime, tau_outer, cluster_to_members, ds_values,
                                    ds_keys, sigma, verbose_every=VERBOSE_EVERY):
    """Kernel Variant B: K2(q,h)=exp(-||q-h||_2^2/(2*sigma^2)). Same
    full-cluster, no-capping, member-memoized structure as GEOM1's
    local_retrieval_prob (identity projection), but squared distance in the
    exponent instead of linear -- the one genuinely new estimator here."""
    N, k = idx_dime.shape
    w = softmax_np(-d_dime / tau_outer, axis=1)
    p = np.zeros(N, dtype=np.float64)
    member_cache_keys = {}
    t0 = time.perf_counter()
    for i in range(N):
        true_tok = int(true_values[i])
        acc = 0.0
        for j in range(k):
            c = int(idx_dime[i, j])
            members = cluster_to_members[c]
            if len(members) == 0:
                continue
            if c not in member_cache_keys:
                member_cache_keys[c] = ds_keys[members]
            z_m = member_cache_keys[c]
            q_row = query_keys[i]
            d2_local = np.sum((z_m - q_row[None, :]) ** 2, axis=1)
            w_local = softmax_np(-d2_local / (2 * sigma ** 2), axis=-1)
            match = ds_values[members] == true_tok
            if match.any():
                acc += float(w[i, j]) * float(w_local[match].sum())
        p[i] = acc
        if verbose_every and (i + 1) % verbose_every == 0:
            elapsed = time.perf_counter() - t0
            print(f"      progress: {i+1}/{N}  ({(i+1)/elapsed:.1f} q/s, ETA {(N-i-1)/((i+1)/elapsed)/60:.1f} min)")
    return p


def local_retrieval_full_distribution(query_keys, idx_dime, d_dime, tau_outer, cluster_to_members, ds_values, ds_keys,
                                        tau_or_sigma, squared):
    """Full sparse {token: prob} distribution per query -- needed only for
    Diagnostics B/C (entropy, JS divergence), hence bounded to a subsample."""
    N, k = idx_dime.shape
    w = softmax_np(-d_dime / tau_outer, axis=1)
    out = []
    for i in range(N):
        d_out = {}
        for j in range(k):
            c = int(idx_dime[i, j])
            members = cluster_to_members[c]
            if len(members) == 0:
                continue
            z_m = ds_keys[members]
            q_row = query_keys[i]
            if squared:
                d_local = np.sum((z_m - q_row[None, :]) ** 2, axis=1)
                w_local = softmax_np(-d_local / (2 * tau_or_sigma ** 2), axis=-1)
            else:
                d_local = np.linalg.norm(z_m - q_row[None, :], axis=1)
                w_local = softmax_np(-d_local / tau_or_sigma, axis=-1)
            wj = float(w[i, j])
            for tok, wl in zip(ds_values[members].tolist(), w_local.tolist()):
                d_out[tok] = d_out.get(tok, 0.0) + wj * wl
        out.append(d_out)
    return out


def uniform_kernel_prob(query_keys, true_values, idx_dime, d_dime, tau_outer, cluster_to_members, ds_values):
    """B6: replace all local similarities with a constant within each
    centroid -- must reduce EXACTLY to DIME-V1's static P(y|c), verified by
    direct comparison against dime_standalone_prob, not merely asserted
    structurally."""
    N, k = idx_dime.shape
    w = softmax_np(-d_dime / tau_outer, axis=1)
    p = np.zeros(N, dtype=np.float64)
    for i in range(N):
        true_tok = int(true_values[i])
        acc = 0.0
        for j in range(k):
            c = int(idx_dime[i, j])
            members = cluster_to_members[c]
            if len(members) == 0:
                continue
            match = ds_values[members] == true_tok
            p_uniform = float(match.sum()) / len(members)
            acc += float(w[i, j]) * p_uniform
        p[i] = acc
    return p


def main():
    t_start = time.perf_counter()
    print(f"E2-KM1-DENSE job started. SMOKE={SMOKE}, N_CLUSTERS={N_CLUSTERS}")
    rng = np.random.default_rng(SEED)

    print("Loading cache (datastore/controller_train/val)...")
    ds = load_cache(os.path.join(CACHE_DIR, "wikitext103_gpt2_datastore.npz"))
    ct = load_cache(os.path.join(CACHE_DIR, "wikitext103_gpt2_controller_train.npz"))
    val = load_cache(os.path.join(CACHE_DIR, "wikitext103_gpt2_val.npz"))
    ds_keys, ds_values = ds["keys"], ds["values"]
    ct_keys, ct_values, ct_p_lm = ct["keys"], ct["values"], ct["p_lm_true"]
    val_keys, val_values, val_p_lm = val["keys"], val["values"], val["p_lm_true"]

    print("Re-deriving DIME clusters (seed=42, deterministic, identical to every prior E1/E2/TRI1/NC1/GEOM1 script)...")
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
    print("Querying val/controller_train (matched config) -- B0/B1/B2 baselines...")
    d_raw_val, idx_raw_val = query_knn_indices(raw_index, val_keys, k=kM)
    d_dime_val, idx_dime_val = query_knn_indices(dime_index, val_keys, k=kM)
    r_raw_val, r_dime_val = raw_values[idx_raw_val], dime_values[idx_dime_val]
    p_raw_true_val = raw_standalone_prob(d_raw_val, r_raw_val, val_values, tau=tauM)
    p_dime_true_val = dime_standalone_prob(d_dime_val, r_dime_val, val_values, tau=tauM)
    nll_B0_val = -np.log(val_p_lm + 1e-12)
    nll_B1_val = nll_from_standalone(p_raw_true_val, val_p_lm, alphaM)
    nll_B2_val = nll_from_standalone(p_dime_true_val, val_p_lm, alphaM)

    d_dime_ct, idx_dime_ct = query_knn_indices(dime_index, ct_keys, k=kM)

    report = {"n_val": int(len(val_values)), "n_ct": int(len(ct_values)), "n_ds": int(len(ds_values)),
              "matched_config": MATCHED, "n_clusters": N_CLUSTERS, "seed": SEED, "smoke": SMOKE,
              "B0_base_lm_mean_nll": float(np.mean(nll_B0_val)), "B1_raw_knn_mean_nll": float(np.mean(nll_B1_val)),
              "B2_dime_v1_mean_nll": float(np.mean(nll_B2_val))}
    save_results(os.path.join(OUT_DIR, "e2km1_summary.json"), report)

    # ============================================================
    # Empirically probe the local-distance scale before building grids
    # ============================================================
    print(f"\nProbing empirical local-distance distribution ({N_PROBE_FOR_GRID} queries x their top-1 centroid)...")
    probe_idx = rng.choice(len(ct_values), size=min(N_PROBE_FOR_GRID, len(ct_values)), replace=False)
    probe_dists = []
    for i in probe_idx:
        c = int(idx_dime_ct[i, 0])
        members = cluster_to_members[c]
        if len(members) == 0:
            continue
        d = np.linalg.norm(ds_keys[members] - ct_keys[i][None, :], axis=1)
        probe_dists.append(d)
    probe_dists = np.concatenate(probe_dists)
    dist_median, dist_std = float(np.median(probe_dists)), float(np.std(probe_dists))
    print(f"  empirical local-distance median={dist_median:.4f}, std={dist_std:.4f} (n={len(probe_dists)})")
    tau_grid = sorted(set(round(x, 4) for x in [dist_median * 0.25, dist_median * 0.5, dist_median, dist_median * 2.0]))
    sigma_grid = sorted(set(round(x, 4) for x in [dist_std * 0.5, dist_std, dist_std * 2.0, dist_std * 4.0]))
    print(f"  tau_local grid (Kernel A): {tau_grid}")
    print(f"  sigma grid (Kernel B): {sigma_grid}")
    report["distance_probe"] = {"median": dist_median, "std": dist_std, "n": len(probe_dists),
                                  "tau_grid": tau_grid, "sigma_grid": sigma_grid}
    save_results(os.path.join(OUT_DIR, "e2km1_summary.json"), report)

    # ============================================================
    # B3 = B4-KernelA -- GEOM1's System B, reproduced and tuned fresh here
    # ============================================================
    print("\nB3/B4-KernelA: tuning tau_local on controller_train (exponential kernel, full-cluster)...")
    identity_query_fn = lambda q_row, c: q_row
    identity_members_fn = lambda members, c: ds_keys[members]
    tauA_nll = {}
    for tau_local in tau_grid:
        p_ct = local_retrieval_prob(ct_keys, ct_values, idx_dime_ct, d_dime_ct, tauM, cluster_to_members, ds_values,
                                      identity_query_fn, identity_members_fn, tau_local, verbose_every=0)
        nll_ct = float(np.mean(nll_from_standalone(p_ct, ct_p_lm, alphaM)))
        tauA_nll[tau_local] = nll_ct
        print(f"  tau_local={tau_local}: controller_train mean NLL = {nll_ct:.4f}")
    tau_local_A = min(tauA_nll, key=tauA_nll.get)
    print(f"  chosen tau_local (Kernel A) = {tau_local_A}")

    print("B3/B4-KernelA: full-cluster exponential-kernel local retrieval at full val scale...")
    p_A_val = local_retrieval_prob(val_keys, val_values, idx_dime_val, d_dime_val, tauM, cluster_to_members, ds_values,
                                     identity_query_fn, identity_members_fn, tau_local_A)
    nll_A_val = nll_from_standalone(p_A_val, val_p_lm, alphaM)
    print(f"  B3/B4-KernelA mean NLL (val, full scale) = {np.mean(nll_A_val):.4f}")
    report["B3_B4_kernelA"] = {"tau_local_tuning": tauA_nll, "chosen_tau_local": tau_local_A,
                                 "mean_nll_val": float(np.mean(nll_A_val))}
    save_results(os.path.join(OUT_DIR, "e2km1_summary.json"), report)

    # ============================================================
    # B4-KernelB -- Gaussian squared-distance kernel, genuinely new
    # ============================================================
    print("\nB4-KernelB: tuning sigma on controller_train (Gaussian squared-distance kernel, full-cluster)...")
    sigmaB_nll = {}
    for sigma in sigma_grid:
        p_ct = local_retrieval_prob_gaussian(ct_keys, ct_values, idx_dime_ct, d_dime_ct, tauM, cluster_to_members,
                                               ds_values, ds_keys, sigma, verbose_every=0)
        nll_ct = float(np.mean(nll_from_standalone(p_ct, ct_p_lm, alphaM)))
        sigmaB_nll[sigma] = nll_ct
        print(f"  sigma={sigma}: controller_train mean NLL = {nll_ct:.4f}")
    sigma_B = min(sigmaB_nll, key=sigmaB_nll.get)
    print(f"  chosen sigma (Kernel B) = {sigma_B}")

    print("B4-KernelB: full-cluster Gaussian-kernel local retrieval at full val scale...")
    p_B_val = local_retrieval_prob_gaussian(val_keys, val_values, idx_dime_val, d_dime_val, tauM, cluster_to_members,
                                              ds_values, ds_keys, sigma_B)
    nll_B_val = nll_from_standalone(p_B_val, val_p_lm, alphaM)
    print(f"  B4-KernelB mean NLL (val, full scale) = {np.mean(nll_B_val):.4f}")
    report["B4_kernelB"] = {"sigma_tuning": sigmaB_nll, "chosen_sigma": sigma_B, "mean_nll_val": float(np.mean(nll_B_val))}
    save_results(os.path.join(OUT_DIR, "e2km1_summary.json"), report)

    # ============================================================
    # B6 -- uniform-kernel identity check (verified, not just asserted structurally)
    # ============================================================
    print(f"\nB6: uniform-kernel identity check on {N_VERIFY_B6} verification queries...")
    verify_idx = rng.choice(len(val_values), size=min(N_VERIFY_B6, len(val_values)), replace=False)
    p_uniform = uniform_kernel_prob(val_keys[verify_idx], val_values[verify_idx], idx_dime_val[verify_idx],
                                      d_dime_val[verify_idx], tauM, cluster_to_members, ds_values)
    max_abs_diff_b6 = float(np.max(np.abs(p_uniform - p_dime_true_val[verify_idx])))
    print(f"  max abs diff (B6 uniform-kernel vs. real DIME-V1 standalone prob) = {max_abs_diff_b6:.2e}")
    assert max_abs_diff_b6 < 1e-9, "B6 identity check FAILED -- uniform-kernel local retrieval does not reproduce DIME-V1 exactly, stop and investigate"
    report["B6_identity_check"] = {"n_verification_queries": int(len(verify_idx)), "max_abs_diff": max_abs_diff_b6,
                                      "passed": True}
    save_results(os.path.join(OUT_DIR, "e2km1_summary.json"), report)

    # ============================================================
    # B5 -- within-centroid label-shuffle control (bounded subsample, Kernel A's tau)
    # ============================================================
    print(f"\nB5: within-centroid label-shuffle control on a {N_SUBSAMPLE_B5}-query val subsample (Kernel A tau)...")
    sub5_idx = rng.choice(len(val_values), size=min(N_SUBSAMPLE_B5, len(val_values)), replace=False)
    shuffle_rng = np.random.default_rng(SEED + 500)
    shuffled_ds_values = ds_values.copy()
    for c in range(N_CLUSTERS):
        members = cluster_to_members[c]
        if len(members) > 1:
            shuffled_ds_values[members] = shuffle_rng.permutation(ds_values[members])
    p_B5 = local_retrieval_prob(val_keys[sub5_idx], val_values[sub5_idx], idx_dime_val[sub5_idx], d_dime_val[sub5_idx],
                                  tauM, cluster_to_members, shuffled_ds_values, identity_query_fn, identity_members_fn,
                                  tau_local_A, verbose_every=0)
    nll_B5_sub = float(np.mean(nll_from_standalone(p_B5, val_p_lm[sub5_idx], alphaM)))
    nll_A_sub = float(np.mean(nll_A_val[sub5_idx]))
    print(f"  B5 shuffled-label mean NLL = {nll_B5_sub:.4f} vs. B4-KernelA real-label mean NLL (same subsample) = {nll_A_sub:.4f}")
    report["B5_shuffle_control"] = {"n_subsample": int(len(sub5_idx)), "shuffled_label_mean_nll": nll_B5_sub,
                                      "real_label_mean_nll_same_subsample": nll_A_sub}
    save_results(os.path.join(OUT_DIR, "e2km1_summary.json"), report)

    # ============================================================
    # Main comparison table + bootstrap CIs
    # ============================================================
    print("\nMain comparison table (full val scale, matched config)...")
    main_table = {"B0_base_lm": float(np.mean(nll_B0_val)), "B1_raw_knn": float(np.mean(nll_B1_val)),
                   "B2_dime_v1": float(np.mean(nll_B2_val)), "B3_B4_kernelA_geom1_equivalent": float(np.mean(nll_A_val)),
                   "B4_kernelB_gaussian": float(np.mean(nll_B_val))}
    print(f"  {main_table}")

    ci_results = {
        "kernelA_vs_dime": paired_bootstrap_mean_diff_ci(nll_A_val, nll_B2_val),
        "kernelB_vs_dime": paired_bootstrap_mean_diff_ci(nll_B_val, nll_B2_val),
        "kernelA_vs_raw": paired_bootstrap_mean_diff_ci(nll_A_val, nll_B1_val),
        "kernelB_vs_raw": paired_bootstrap_mean_diff_ci(nll_B_val, nll_B1_val),
        "kernelB_vs_kernelA": paired_bootstrap_mean_diff_ci(nll_B_val, nll_A_val),
    }
    for name, r in ci_results.items():
        print(f"  {name}: mean_diff={r['mean_diff']:.4f}, 95% CI=[{r['ci_95_low']:.4f}, {r['ci_95_high']:.4f}]")
    frac_improved_A = float(np.mean(nll_A_val < nll_B2_val - 1e-9))
    frac_improved_B = float(np.mean(nll_B_val < nll_B2_val - 1e-9))
    cov_A = retrieval_coverage(p_A_val)
    cov_B = retrieval_coverage(p_B_val)
    cov_dime = retrieval_coverage(p_dime_true_val)
    main_metrics = {"mean_nll": main_table, "bootstrap_ci_nll_diffs": ci_results,
                      "frac_queries_improved_vs_dime_kernelA": frac_improved_A,
                      "frac_queries_improved_vs_dime_kernelB": frac_improved_B,
                      "correct_token_coverage": {"kernelA": cov_A, "kernelB": cov_B, "dime_v1": cov_dime}}
    report["main_comparison"] = main_metrics
    save_results(os.path.join(OUT_DIR, "e2km1_summary.json"), report)

    fig, ax = plt.subplots(figsize=(7, 5))
    labels_plot = list(main_table.keys())
    ax.bar(range(len(labels_plot)), [main_table[l] for l in labels_plot])
    ax.set_xticks(range(len(labels_plot)))
    ax.set_xticklabels(labels_plot, rotation=25, ha="right")
    ax.set_ylabel("mean interpolated NLL (matched config, val)")
    ax.set_title("E2-KM1-DENSE: baseline comparison")
    fig.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, "main_comparison.png"), dpi=130)
    plt.close(fig)

    # ============================================================
    # Diagnostic A -- real vs. shuffled (reuses B5 above)
    # ============================================================
    diagnostic_A = {"real_label_mean_nll": nll_A_sub, "shuffled_label_mean_nll": nll_B5_sub,
                      "real_outperforms_shuffled_by_nats": nll_B5_sub - nll_A_sub}
    print(f"\nDiagnostic A (context-target association): {diagnostic_A}")
    report["diagnostic_A"] = diagnostic_A

    # ============================================================
    # Diagnostics B & C -- need full distributions, bounded subsample
    # ============================================================
    print(f"\nDiagnostics B & C: full-distribution analysis on a {N_SUBSAMPLE_DIAG}-query val subsample...")
    diag_idx = rng.choice(len(val_values), size=min(N_SUBSAMPLE_DIAG, len(val_values)), replace=False)
    dime_dists_diag = dime_full_distribution(d_dime_val[diag_idx], r_dime_val[diag_idx], tau=tauM)
    kernelA_dists_diag = local_retrieval_full_distribution(val_keys[diag_idx], idx_dime_val[diag_idx], d_dime_val[diag_idx],
                                                              tauM, cluster_to_members, ds_values, ds_keys, tau_local_A, False)
    js_diag = np.array([js_divergence_dict(ka, dd) for ka, dd in zip(kernelA_dists_diag, dime_dists_diag)])
    nll_improve_diag = nll_B2_val[diag_idx] - nll_A_val[diag_idx]
    corr_js_improve = float(np.corrcoef(js_diag, nll_improve_diag)[0, 1]) if len(js_diag) > 1 else None
    diagnostic_B = {"n_subsample": int(len(diag_idx)), "mean_js_kernelA_vs_dime_prior": float(np.mean(js_diag)),
                      "pearson_corr_js_vs_nll_improvement": corr_js_improve}
    print(f"  Diagnostic B (query conditioning vs. improvement): {diagnostic_B}")

    shape_dime_diag = distribution_shape_stats(dime_dists_diag)
    shape_kernelA_diag = distribution_shape_stats(kernelA_dists_diag)
    diagnostic_C = {"n_subsample": int(len(diag_idx)),
                      "mean_entropy_dime": float(np.mean(shape_dime_diag["entropy"])),
                      "mean_entropy_kernelA": float(np.mean(shape_kernelA_diag["entropy"])),
                      "mean_correct_token_prob_dime": float(np.mean(p_dime_true_val[diag_idx])),
                      "mean_correct_token_prob_kernelA": float(np.mean(p_A_val[diag_idx])),
                      "mean_nll_dime_subsample": float(np.mean(nll_B2_val[diag_idx])),
                      "mean_nll_kernelA_subsample": float(np.mean(nll_A_val[diag_idx]))}
    print(f"  Diagnostic C (distribution fidelity): {diagnostic_C}")
    report["diagnostic_B"] = diagnostic_B
    report["diagnostic_C"] = diagnostic_C
    save_results(os.path.join(OUT_DIR, "e2km1_summary.json"), report)

    # ============================================================
    # Diagnostic D -- grouped failure analysis (full val scale, scalar covariates only)
    # ============================================================
    print("\nDiagnostic D: grouped analysis (full val scale)...")
    populations = np.array([sum(c.values()) for c in compressed_dists])
    entropies = np.zeros(N_CLUSTERS)
    max_freq_ratio = np.zeros(N_CLUSTERS)
    for c in range(N_CLUSTERS):
        total = populations[c]
        if total > 0:
            probs = np.array([v / total for v in compressed_dists[c].values()])
            entropies[c] = float(-(probs * np.log(probs)).sum())
            max_freq_ratio[c] = float(probs.max())
    top1_cluster = idx_dime_val[:, 0]
    query_pop = populations[top1_cluster]
    query_entropy = entropies[top1_cluster]
    query_concentration = max_freq_ratio[top1_cluster]
    delta_dime_vs_raw = nll_B2_val - nll_B1_val
    kernelA_improvement = nll_B2_val - nll_A_val
    kernelB_improvement = nll_B2_val - nll_B_val

    def bucket_stats(covariate, n_buckets=4):
        edges = np.quantile(covariate, np.linspace(0, 1, n_buckets + 1))
        edges[0] -= 1e-9
        buckets = np.digitize(covariate, edges[1:-1])
        out = {}
        for b in range(n_buckets):
            mask = buckets == b
            if mask.sum() == 0:
                continue
            out[f"bucket_{b}"] = {"n": int(mask.sum()), "covariate_range": [float(edges[b] if b > 0 else covariate.min()), float(edges[b + 1])],
                                     "mean_kernelA_improvement": float(np.mean(kernelA_improvement[mask])),
                                     "mean_kernelB_improvement": float(np.mean(kernelB_improvement[mask])),
                                     "mean_dime_degradation_vs_raw": float(np.mean(delta_dime_vs_raw[mask]))}
        return out

    diagnostic_D = {
        "by_dime_degradation_vs_raw": bucket_stats(delta_dime_vs_raw),
        "by_centroid_population": bucket_stats(query_pop),
        "by_centroid_entropy": bucket_stats(query_entropy),
        "by_local_concentration": bucket_stats(query_concentration),
    }
    print(f"  Diagnostic D computed across {len(diagnostic_D)} grouping dimensions (4 buckets each, full val scale)")
    report["diagnostic_D"] = diagnostic_D
    save_results(os.path.join(OUT_DIR, "e2km1_summary.json"), report)

    order = np.argsort(-delta_dime_vs_raw, kind="stable")
    tail_results = {}
    for pct in WORST_PCTS:
        n_worst = max(1, int(len(order) * pct / 100))
        worst_idx = order[:n_worst]
        tail_results[pct] = {"n_queries": int(n_worst), "mean_nll_raw": float(np.mean(nll_B1_val[worst_idx])),
                               "mean_nll_dime": float(np.mean(nll_B2_val[worst_idx])),
                               "mean_nll_kernelA": float(np.mean(nll_A_val[worst_idx])),
                               "mean_nll_kernelB": float(np.mean(nll_B_val[worst_idx]))}
        print(f"  worst-{pct}%: raw={tail_results[pct]['mean_nll_raw']:.4f}, dime={tail_results[pct]['mean_nll_dime']:.4f}, "
              f"kernelA={tail_results[pct]['mean_nll_kernelA']:.4f}, kernelB={tail_results[pct]['mean_nll_kernelB']:.4f}")
    report["difficult_query_tail"] = tail_results
    save_results(os.path.join(OUT_DIR, "e2km1_summary.json"), report)

    # ============================================================
    # Decision criteria
    # ============================================================
    print("\nApplying decision criteria (Outcome A / B / C)...")
    kernelA_beats_dime = ci_results["kernelA_vs_dime"]["ci_95_high"] < 0
    kernelB_beats_dime = ci_results["kernelB_vs_dime"]["ci_95_high"] < 0
    kernelB_beats_kernelA = ci_results["kernelB_vs_kernelA"]["ci_95_high"] < 0
    real_beats_shuffled = nll_A_sub < nll_B5_sub
    if (kernelA_beats_dime or kernelB_beats_dime) and real_beats_shuffled and kernelB_beats_kernelA:
        outcome = "A"
        recommendation = ("Dense kernel estimation shows a meaningful, statistically-supported advantage over "
                           "DIME-V1 AND the Gaussian-kernel variant meaningfully improves on the exponential/GEOM1 "
                           "variant -- investigate sparse kernel mean approximation and prediction-aware "
                           "representative selection.")
    elif (kernelA_beats_dime or kernelB_beats_dime) and real_beats_shuffled:
        outcome = "B"
        recommendation = ("The estimator improves over DIME-V1 with genuine context-target signal (real beats "
                           "shuffled), but is essentially equivalent to E2-GEOM1's full-space local retrieval -- "
                           "sparsification may still be worth studying as a way to compress that EXISTING "
                           "retrieval behavior, but this is not evidence of a novel predictive mechanism.")
    else:
        outcome = "C"
        recommendation = ("The estimator does not show a consistent, statistically-supported benefit over DIME-V1 "
                           "and/or does not clearly beat the shuffled-label control -- deprioritize sparse kernel "
                           "mean approximation for now.")
    print(f"  Outcome: {outcome}. {recommendation}")
    decision = {"outcome": outcome, "recommendation": recommendation,
                 "kernelA_significantly_beats_dime": bool(kernelA_beats_dime),
                 "kernelB_significantly_beats_dime": bool(kernelB_beats_dime),
                 "kernelB_significantly_beats_kernelA": bool(kernelB_beats_kernelA),
                 "real_label_beats_shuffled_control": bool(real_beats_shuffled)}
    report["decision"] = decision
    save_results(os.path.join(OUT_DIR, "e2km1_summary.json"), report)

    # ============================================================
    # Markdown report
    # ============================================================
    print("\nGenerating report.md...")
    lines = [
        "# E2-KM1-DENSE: Dense Token-Conditioned Kernel Mean Experiment",
        "",
        f"Matched config (k={MATCHED['k']}, tau={MATCHED['tau']}, alpha={MATCHED['alpha']}), {len(val_values)} val queries, "
        f"{N_CLUSTERS} clusters.",
        "",
        "## Equivalence identification",
        "This spec's System A / B4-KernelA (exponential kernel, full-cluster, DIME routing weights, LM-interpolated) "
        "is the SAME computation as E2-GEOM1's System B. B3 and B4-KernelA are reported as a single reused number "
        "below, computed once, not twice. Kernel B (Gaussian squared-distance) is the genuinely new estimator.",
        "",
        "## Main comparison (mean val NLL, full scale)",
    ]
    for label, v in main_table.items():
        lines.append(f"- {label}: {v:.4f}")
    lines += ["", "## Bootstrap 95% CIs (paired, mean NLL difference)"]
    for name, r in ci_results.items():
        lines.append(f"- {name}: {r['mean_diff']:.4f} [{r['ci_95_low']:.4f}, {r['ci_95_high']:.4f}] (n={r['n']})")
    lines += [
        "",
        f"Fraction of queries improved vs. DIME-V1: Kernel A = {frac_improved_A:.4f}, Kernel B = {frac_improved_B:.4f}.",
        f"Correct-token coverage: Kernel A = {cov_A:.4f}, Kernel B = {cov_B:.4f}, DIME-V1 = {cov_dime:.4f}.",
        "",
        "## B6 -- uniform-kernel identity check",
        f"Max abs diff vs. real DIME-V1 standalone probability: {max_abs_diff_b6:.2e} on {len(verify_idx)} queries "
        f"(PASSED, < 1e-9 required).",
        "",
        "## Diagnostic A -- context-target association (real vs. shuffled labels)",
        f"{diagnostic_A}",
        "",
        "## Diagnostic B -- query conditioning vs. improvement",
        f"{diagnostic_B}",
        "",
        "## Diagnostic C -- predictive distribution fidelity",
        f"{diagnostic_C}",
        "",
        "## Diagnostic D -- grouped failure analysis (full val scale)",
        "See diagnostic_D in the JSON summary for the full 4-bucket breakdown by DIME-vs-raw degradation, centroid "
        "population, centroid entropy, and local label concentration.",
        "",
        "## Difficult-query tail",
    ]
    for pct, d in tail_results.items():
        lines.append(f"- worst-{pct}%: raw={d['mean_nll_raw']:.4f}, DIME-V1={d['mean_nll_dime']:.4f}, "
                      f"KernelA={d['mean_nll_kernelA']:.4f}, KernelB={d['mean_nll_kernelB']:.4f}")
    lines += [
        "",
        f"## Decision: Outcome {outcome}",
        recommendation,
        "",
        "## Context (not independently reverified in this script)",
        "E2-GEOM1's separately-reported anchor-space retrieval result is background context only -- this script's "
        "own B3/B4-KernelA number above is the independently, freshly verified reproduction of GEOM1's System B, "
        "not a citation of an external number.",
        "",
        "## Limitations",
        f"- B5 (shuffle control) run on a {len(sub5_idx)}-query val subsample, not full val.",
        f"- Diagnostics B and C run on a {len(diag_idx)}-query val subsample (full retrieval distributions are "
        "meaningfully more expensive per query than the scalar NLL grid).",
        f"- B6 verified on {len(verify_idx)} queries, not full val -- it is a pure mathematical identity check, "
        "not an independently interesting reported metric.",
        "- Each estimator uses a different approximation; differences are empirical performance comparisons.",
    ]
    with open(os.path.join(OUT_DIR, "report.md"), "w") as f:
        f.write("\n".join(lines))

    report["sections_completed"] = ["B0_B1_B2", "distance_probe", "B3_B4_kernelA", "B4_kernelB", "B6_identity_check",
                                      "B5_shuffle", "main_comparison", "diagnostic_A", "diagnostic_B", "diagnostic_C",
                                      "diagnostic_D", "difficult_query_tail", "decision", "report.md"]
    report["total_runtime_seconds"] = time.perf_counter() - t_start
    save_results(os.path.join(OUT_DIR, "e2km1_summary.json"), report)
    print(f"\nAll results saved under {OUT_DIR}")
    print(f"Total runtime: {report['total_runtime_seconds']/60:.1f} min")
    print("No existing E1/E2/TRI1/NC1/GEOM1 outputs were modified.")


if __name__ == "__main__":
    main()
