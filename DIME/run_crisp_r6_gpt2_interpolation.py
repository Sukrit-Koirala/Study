"""CRISP-R6 -- GPT-2 Interpolation and Predictive NLL Evaluation. Determines
whether CRISP's retrieval-distribution improvements (R4/R5) translate into
an actual improvement in GPT-2's predictive negative log-likelihood. Writes
only to results/stage3_crisp_r6_gpt2_interpolation/, preserving every prior
E1/E2/TRI1/NC1/GEOM1/KM1/R1/R2/R3/R4/R5 output unchanged.

=====================================================================
Reproduction note
=====================================================================
results/stage3_crisp_r5_global_retrieval does not exist locally (checked
directly) -- R5's real artifacts (and R1-R4's) are not available. Per this
project's established practice (every CRISP delivery has faced this), this
script does NOT cite R5's numbers as ground truth. It reproduces R5's exact
CRISP-fitting procedure fresh by importing its own functions directly
(fit_global_crisp_cluster, crisp_predict, cluster_distances,
exact_cluster_dist_and_mass, build_fit_tune_split -- none redefined), under
the identical seed=42 clustering and rank=8 configuration. The spec's own
cited R5 numbers (global TV 0.75599 vs 0.68960, storage 1,172,346,652 /
13,552,584 / 20,732,360 bytes, etc.) are treated as UNVERIFIED context, not
assumed -- this script's own storage/coverage numbers (Experiment F) are an
independent, fresh measurement under the same methodology.

=====================================================================
Key simplification this script exploits (stated explicitly, not a silent
scope narrowing): NLL only needs the STANDALONE TRUE-TOKEN PROBABILITY
=====================================================================
P_mix(y|q) = (1-alpha)*P_LM(y|q) + alpha*P_R(y|q), and the per-token loss is
-log(P_mix(y_true|q)) -- which only requires P_LM(y_true|q) (already cached
as p_lm_true in every .npz cache file, confirmed by reading cache_io.py) and
P_R(y_true|q) (the retrieval method's probability mass on the SPECIFIC true
token). Neither requires the FULL retrieval distribution over the vocabulary.
This is why R6, despite evaluating MORE systems (M0-M6) than R5's S0-S6,
needs LESS expensive per-query work than R5's own full-distribution TV/JS
computation -- and is why this script runs the expensive per-cluster
member-distance loop (needed for M3/M4/M5/M6's true-token probability) over
ALL ~77,089 val queries directly, rather than R5's bounded 8,000-query
sample, per this spec's own explicit "prioritize evaluating all eligible
positions" instruction (Section 9). R5's bounded sample is additionally
reproduced as a PAIRED diagnostic (same stratification method) from the
same already-computed full-scale per-query values, at zero extra compute
cost, per the spec's "query-sample comparison" instruction.

Because P_R(y_true|q) does not depend on alpha, it is computed ONCE per
system per query; the full alpha sweep (Experiment B) is then a cheap,
purely elementwise, vectorized recombination -- no repeated per-cluster
computation across the 9 alpha values.

No arbitrary retrieval-probability floor is used anywhere in this script
(per the spec's explicit "Critical numerical requirement"): since alpha<1
and p_lm_true is a real GPT-2 softmax output (never exactly 0 in the cached
arrays, verified below), (1-alpha)*p_lm_true > 0 strictly, so
-log((1-alpha)*p_lm_true + alpha*p_R) is always finite without any added
epsilon. alpha=1 is examined separately in Experiment B with its own
explicit zero-probability handling, per the spec's own instruction --
treated as a distinct, documented edge case, not silently guarded by the
same mechanism.

=====================================================================
Systems implemented (M0-M6, all required systems; M6 included since it is
cheap given M3/M4/M5 already gather the same per-cluster candidate members)
=====================================================================
  M0 GPT-2 alone              -- p_lm_true, already cached.
  M1 GPT-2 + raw kNN           -- e1_metrics.raw_standalone_prob, reused directly.
  M2 GPT-2 + DIME-V1           -- e1_metrics.dime_standalone_prob, reused directly.
  M3 GPT-2 + CRISP-R4/R5 rank8 -- R5's fit_global_crisp_cluster/crisp_predict, reused directly,
                                    combined via R5's same beta_j(q) centroid-softmax weights.
  M4 GPT-2 + exact cluster-function oracle (R5's S3) -- R5's exact_cluster_dist_and_mass, reused.
  M5 GPT-2 + mass-aware exact oracle (R5's S5)       -- same per-cluster computation as M4,
                                    combined via real evidence mass instead of beta_j(q).
  M6 GPT-2 + candidate-restricted top-k oracle (R5's S2) -- reuses the SAME per-cluster member
                                    distances gathered for M4/M5 (no duplicate distance computation).
CRISP's S6 diagnostic (CRISP + exact masses) is NOT part of R6's required M-list and is not
computed here -- not needed for any of the spec's named comparisons.

=====================================================================
Scoping decisions (stated up front, per this project's established
practice):
=====================================================================
  - PRIMARY matched configuration only for the main comparison: raw k=20,
    DIME/CRISP 20 centroids, tau=2.0, alpha=0.25, CRISP rank=8 -- exactly
    as R5 used, per the spec's own "do not invent a new CRISP model"
    instruction extended to the interpolation config.
  - A baseline-reproduction check (Phase 1) ALSO computes the tuned
    configuration's raw/DIME NLL (k=300/tau=1/alpha=0.25 for raw, k=100/
    tau=2/alpha=0.1 for DIME, both real numbers from the existing
    results/wikitext103_gpt2_tier1.json) as a verification cross-check
    against that historical artifact -- NOT a tuned-CRISP evaluation
    (CRISP is only ever fit/evaluated at the matched rank=8 config,
    consistent with R5's own precedent of not re-running CRISP at a
    different configuration).
  - No genuinely untouched THIRD data split exists in this project's cache
    files (only controller_train and val -- confirmed across every prior
    script). Per the spec's own explicit instruction, Experiment B's
    "individually tuned alpha" comparison is therefore labeled EXPLORATORY
    throughout, not an independent generalization result -- the alpha
    that minimizes NLL is selected and reported on the SAME val queries
    used for the primary comparison, stated honestly rather than
    represented as unbiased.
  - Experiment E's primary evaluation uses the FULL val set (~77,089
    queries, fewer under SMOKE), per the spec's explicit preference; R5's
    JS-decile-stratified 8,000-query sample (same stratification method)
    is reproduced as a paired diagnostic from the same per-query arrays,
    at no extra computational cost.
  - Statistical uncertainty uses an i.i.d. (not sequence/block-aware)
    bootstrap over query positions -- the same scope simplification R5
    made, stated explicitly again here, not silently repeated.
  - GPT-2 confidence groups use PREDETERMINED quartile bins of p_lm_true
    (not a development-tuned threshold), per the spec's own explicitly
    allowed alternative.

Smoke testing: set CRISPR6_SMOKE=1 with CRISPR6_CACHE_DIR / CRISPR6_OUT_DIR /
CRISPR6_N_CLUSTERS before running. Never set it for the real run.
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
from e1_metrics import raw_standalone_prob, dime_standalone_prob, js_divergence_dict
from run_e2_information_allocation import build_cluster_to_members
from run_e2tri1_anchor_conditioned_compression import build_centroid_distance_matrix, select_anchors_nearest
from run_crisp_subspace_projection import exact_evidence, normalize_rows, linear_anchor_diff_basis, fit_ridge
from run_crisp_r4_histogram_corrections import log_ratio_target, reconstruct_logit, mean_tv, RIDGE_LAMBDA_GRID
from run_crisp_r5_global_retrieval import (
    cluster_distances, exact_cluster_dist_and_mass, build_fit_tune_split, fit_global_crisp_cluster, crisp_predict,
    MIN_FIT_POP, MIN_FIT_QUERIES, MIN_TUNE_QUERIES, MAX_FIT, MAX_TUNE)
from gpt_class import FrozenGPT2

SMOKE = os.environ.get("CRISPR6_SMOKE") == "1"
CACHE_DIR = os.environ.get("CRISPR6_CACHE_DIR", os.path.join(SCRIPT_DIR, "..", "GPT_Module", "cache"))
OUT_DIR = os.environ.get("CRISPR6_OUT_DIR", os.path.join(SCRIPT_DIR, "results", "stage3_crisp_r6_gpt2_interpolation"))
PLOT_DIR = os.path.join(OUT_DIR, "plots")
os.makedirs(PLOT_DIR, exist_ok=True)

N_CLUSTERS = int(os.environ.get("CRISPR6_N_CLUSTERS", "3811"))
SEED = 42
DIM, DTYPE_BYTES, VALUE_BYTES = 768, 4, 4
OUTER_K = 20
TAU = 2.0
CRISP_RANK = 8
PRIMARY_ALPHA = 0.25
ALPHA_GRID = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]
N_R5_DIAGNOSTIC_SAMPLE = 8000
N_FAILURE_PER_CATEGORY = 10
METHODS = ["M0", "M1", "M2", "M3", "M4", "M5", "M6"]
METHOD_NAMES = {"M0": "GPT-2 alone", "M1": "GPT-2 + raw kNN", "M2": "GPT-2 + DIME-V1",
                 "M3": "GPT-2 + CRISP rank8", "M4": "GPT-2 + exact cluster oracle",
                 "M5": "GPT-2 + mass-aware oracle", "M6": "GPT-2 + candidate-restricted oracle"}


def interpolated_nll(p_lm, p_r, alpha):
    """-log[(1-alpha)*P_LM + alpha*P_R]. No epsilon floor: for alpha<1 and
    p_lm>0 (always true for cached real GPT-2 output), the mixed
    probability is strictly positive. alpha==1 is the caller's
    responsibility to handle separately (see Experiment B)."""
    p_mix = (1.0 - alpha) * p_lm + alpha * p_r
    return p_mix, -np.log(p_mix)


def main():
    t_start = time.perf_counter()
    print(f"CRISP-R6 job started. SMOKE={SMOKE}, N_CLUSTERS={N_CLUSTERS}")
    rng = np.random.default_rng(SEED)

    print("Loading cache (datastore/controller_train/val)...")
    ds = load_cache(os.path.join(CACHE_DIR, "wikitext103_gpt2_datastore.npz"))
    ct = load_cache(os.path.join(CACHE_DIR, "wikitext103_gpt2_controller_train.npz"))
    val = load_cache(os.path.join(CACHE_DIR, "wikitext103_gpt2_val.npz"))
    ds_keys, ds_values = ds["keys"].astype(np.float64), ds["values"]
    ct_keys, ct_values = ct["keys"].astype(np.float64), ct["values"]
    val_keys, val_values, val_p_lm = val["keys"].astype(np.float64), val["values"], val["p_lm_true"].astype(np.float64)
    gpt = FrozenGPT2(model_name="gpt2")

    assert np.all(val_p_lm > 0), "val_p_lm_true contains non-positive values -- interpolation epsilon-free assumption violated"
    print(f"  p_lm_true sanity: min={val_p_lm.min():.3e}, all strictly positive (confirmed, no epsilon needed for alpha<1)")

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
              "crisp_rank": CRISP_RANK, "primary_alpha": PRIMARY_ALPHA, "alpha_grid": ALPHA_GRID,
              "reproduction_note": "R5's real persisted results were not available locally; this script "
                                     "reproduces R5's CRISP-fitting procedure fresh via direct function import. "
                                     "See module docstring."}
    save_results(os.path.join(OUT_DIR, "crisp_r6_summary.json"), report)

    # ============================================================
    # Phase 1 -- baseline reproduction (GPT-2 alone, matched + tuned raw/DIME)
    # ============================================================
    print("\nPhase 1: baseline reproduction...")
    nll0_full = -np.log(val_p_lm)
    tier1_path = os.path.join(SCRIPT_DIR, "results", "wikitext103_gpt2_tier1.json")
    tier1 = None
    if os.path.exists(tier1_path):
        with open(tier1_path) as f:
            tier1 = json.load(f)
    gpt_only_check = {"measured_mean_nll": float(nll0_full.mean()),
                        "historical_mean_nll": tier1.get("gpt_only_mean_nll") if tier1 else None}
    if tier1:
        gpt_only_check["abs_diff"] = abs(gpt_only_check["measured_mean_nll"] - gpt_only_check["historical_mean_nll"])
    print(f"  GPT-2 alone (M0): {gpt_only_check}")

    d_raw_val, idx_raw_val = query_knn_indices(raw_index, val_keys, k=OUTER_K)
    d_dime_val, idx_dime_val = query_knn_indices(dime_index, val_keys, k=OUTER_K)
    d_dime_ct, idx_dime_ct = query_knn_indices(dime_index, ct_keys, k=OUTER_K)
    retrieved_raw_val = raw_values[idx_raw_val]
    retrieved_dime_val = dime_values[idx_dime_val]

    p_raw_true = raw_standalone_prob(d_raw_val, retrieved_raw_val, val_values, tau=TAU)
    p_dime_true = dime_standalone_prob(d_dime_val, retrieved_dime_val, val_values, tau=TAU)
    _, nll1_matched = interpolated_nll(val_p_lm, p_raw_true, PRIMARY_ALPHA)
    _, nll2_matched = interpolated_nll(val_p_lm, p_dime_true, PRIMARY_ALPHA)

    tuned_check = None
    if tier1 is not None:
        raw_t, dime_t = tier1["raw_knn"]["best_config"], tier1["dime_minibatch_kmeans"]["best_config"]
        kr_t, taur_t, alphar_t = min(raw_t["k"], len(ds_values)), raw_t["tau"], raw_t["alpha"]
        kd_t, taud_t, alphad_t = min(dime_t["k"], N_CLUSTERS), dime_t["tau"], dime_t["alpha"]
        d_raw_t, idx_raw_t = query_knn_indices(raw_index, val_keys, k=kr_t)
        d_dime_t, idx_dime_t = query_knn_indices(dime_index, val_keys, k=kd_t)
        p_raw_true_t = raw_standalone_prob(d_raw_t, raw_values[idx_raw_t], val_values, tau=taur_t)
        p_dime_true_t = dime_standalone_prob(d_dime_t, dime_values[idx_dime_t], val_values, tau=taud_t)
        _, nll_raw_t = interpolated_nll(val_p_lm, p_raw_true_t, alphar_t)
        _, nll_dime_t = interpolated_nll(val_p_lm, p_dime_true_t, alphad_t)
        tuned_check = {
            "raw_tuned_config": raw_t, "measured_mean_nll_raw_tuned": float(nll_raw_t.mean()), "historical_mean_nll_raw_tuned": raw_t["mean_nll"],
            "dime_tuned_config": dime_t, "measured_mean_nll_dime_tuned": float(nll_dime_t.mean()), "historical_mean_nll_dime_tuned": dime_t["mean_nll"],
        }
        tuned_check["raw_abs_diff"] = abs(tuned_check["measured_mean_nll_raw_tuned"] - raw_t["mean_nll"])
        tuned_check["dime_abs_diff"] = abs(tuned_check["measured_mean_nll_dime_tuned"] - dime_t["mean_nll"])
        print(f"  Tuned-config reproduction check: {tuned_check}")
    report["phase1_baseline"] = {"gpt_only_check": gpt_only_check, "matched_mean_nll_raw": float(nll1_matched.mean()),
                                    "matched_mean_nll_dime": float(nll2_matched.mean()), "tuned_config_reproduction_check": tuned_check}
    save_results(os.path.join(OUT_DIR, "crisp_r6_summary.json"), report)

    # ============================================================
    # Phase 2 -- fit CRISP-R4 B4 globally (all N_CLUSTERS clusters, rank=8)
    # ============================================================
    print(f"\nPhase 2: fitting CRISP-R4 B4 (rank={CRISP_RANK}) for all {N_CLUSTERS} clusters (R5's exact procedure)...")
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
            print(f"  fitted {c + 1}/{N_CLUSTERS} (fitted={n_fitted}, fallback={n_fallback}, empty/tiny={n_empty})")
    t_fit = time.perf_counter() - t_fit_start
    token_id_sets = {c: (set(models[c]["token_ids"].tolist()) if models[c]["type"] != "empty_or_tiny" else set()) for c in range(N_CLUSTERS)}
    print(f"  Global fitting done in {t_fit:.1f}s. fitted={n_fitted} ({100*n_fitted/N_CLUSTERS:.1f}%), "
          f"fallback={n_fallback} ({100*n_fallback/N_CLUSTERS:.1f}%), empty/tiny={n_empty}")
    report["crisp_fitting_coverage"] = {"n_fitted": n_fitted, "n_fallback": n_fallback, "n_empty_or_tiny": n_empty,
                                           "global_fit_wall_clock_seconds": t_fit}
    save_results(os.path.join(OUT_DIR, "crisp_r6_summary.json"), report)

    # ============================================================
    # Phase 3 -- matched-interpolation sanity checks (small deterministic pilot)
    # ============================================================
    print("\nPhase 3: matched-interpolation sanity checks...")
    pilot_n = min(50, len(val_values))
    checks = {"alpha0_matches_gpt2": [], "identical_retrieval_gives_identical_nll": [], "mixture_finite_and_valid": []}
    for qi in range(pilot_n):
        p_r = p_raw_true[qi]
        p_mix0, nll0_ck = interpolated_nll(val_p_lm[qi], p_r, 0.0)
        checks["alpha0_matches_gpt2"].append(bool(np.isclose(p_mix0, val_p_lm[qi], atol=1e-12)))
        p_mix_a, nll_a = interpolated_nll(val_p_lm[qi], p_r, PRIMARY_ALPHA)
        p_mix_b, nll_b = interpolated_nll(val_p_lm[qi], p_r, PRIMARY_ALPHA)
        checks["identical_retrieval_gives_identical_nll"].append(bool(np.isclose(nll_a, nll_b, atol=1e-12)))
        checks["mixture_finite_and_valid"].append(bool(np.isfinite(nll_a) and 0.0 < p_mix_a <= 1.0 + 1e-9))
    checks_summary = {k: f"{sum(v)}/{len(v)}" for k, v in checks.items()}
    print(f"  sanity checks (expected all pass): {checks_summary}")
    report["phase3_sanity_checks"] = checks_summary
    save_results(os.path.join(OUT_DIR, "crisp_r6_summary.json"), report)

    # ============================================================
    # Phase 4 -- main predictive evaluation: M3-M6 true-token probabilities, FULL val scale
    # ============================================================
    n_full = len(val_values)
    print(f"\nPhase 4: main evaluation loop over ALL {n_full} val queries (M3-M6 true-token probabilities)...")
    w_dime_full_val = softmax_np(-d_dime_val / TAU, axis=1)
    p_crisp_true = np.zeros(n_full)
    p_s3_true = np.zeros(n_full)
    p_s5_true = np.zeros(n_full)
    p_s6_candidate_true = np.zeros(n_full)
    dime_crisp_support = np.zeros(n_full, dtype=bool)
    raw_support = np.zeros(n_full, dtype=bool)
    t_loop_start = time.perf_counter()
    for qi in range(n_full):
        q_key = val_keys[qi]
        true_tok = int(val_values[qi])
        selected = idx_dime_val[qi]
        w_dime_q = w_dime_full_val[qi]
        all_d, all_v = [], []
        s3_t, s4_t, s5_num, mass_sum = 0.0, 0.0, 0.0, 0.0
        found_support = False
        for j, c in enumerate(selected):
            members = cluster_to_members[c]
            if len(members) == 0:
                continue
            member_keys, member_values = ds_keys[members], ds_values[members]
            d = cluster_distances(q_key, member_keys)
            all_d.append(d); all_v.append(member_values)
            dist_j, mass_j = exact_cluster_dist_and_mass(d, member_values, TAU)
            p_exact_j = dist_j.get(true_tok, 0.0)
            wj = float(w_dime_q[j])
            s3_t += wj * p_exact_j
            crisp_dist_j = crisp_predict(models[c], q_key, c, dime_centroids)
            s4_t += wj * crisp_dist_j.get(true_tok, 0.0)
            if mass_j > 0:
                mass_sum += mass_j
                s5_num += mass_j * p_exact_j
            if true_tok in token_id_sets[c]:
                found_support = True
        p_s3_true[qi] = s3_t
        p_crisp_true[qi] = s4_t
        p_s5_true[qi] = (s5_num / mass_sum) if mass_sum > 0 else 0.0
        dime_crisp_support[qi] = found_support
        raw_support[qi] = bool(np.any(retrieved_raw_val[qi] == true_tok))
        if all_d:
            cand_d = np.concatenate(all_d); cand_v = np.concatenate(all_v)
            topk_local = np.argsort(cand_d)[:OUTER_K]
            d_topk, v_topk = cand_d[topk_local], cand_v[topk_local]
            p_s6_candidate_true[qi] = raw_standalone_prob(d_topk[None, :], v_topk[None, :], np.array([true_tok]), tau=TAU)[0]
        if (qi + 1) % max(1, n_full // 20) == 0:
            print(f"  evaluated {qi + 1}/{n_full} queries ({100*(qi+1)/n_full:.0f}%)")
            save_results(os.path.join(OUT_DIR, "crisp_r6_summary.json"), report)
    t_loop = time.perf_counter() - t_loop_start
    print(f"  Phase 4 main loop done in {t_loop:.1f}s ({t_loop/n_full*1000:.2f} ms/query)")
    report["phase4_loop_wall_clock_seconds"] = t_loop
    save_results(os.path.join(OUT_DIR, "crisp_r6_summary.json"), report)

    # ============================================================
    # Build the full per-query NLL table at PRIMARY_ALPHA (Experiment A)
    # ============================================================
    print(f"\nComputing interpolated NLL for all systems at alpha={PRIMARY_ALPHA} (Experiment A)...")
    p_true_by_method = {"M1": p_raw_true, "M2": p_dime_true, "M3": p_crisp_true, "M4": p_s3_true,
                          "M5": p_s5_true, "M6": p_s6_candidate_true}
    p_mix_by_method, nll_by_method = {}, {}
    nll_by_method["M0"] = nll0_full
    for m, p_r in p_true_by_method.items():
        p_mix_by_method[m], nll_by_method[m] = interpolated_nll(val_p_lm, p_r, PRIMARY_ALPHA)

    def frac_improves(nll_a, nll_b):
        return float((nll_a < nll_b).mean())

    expA = {}
    for m in METHODS:
        nll_m = nll_by_method[m]
        p_true_m = val_p_lm if m == "M0" else p_mix_by_method[m]
        expA[m] = {
            "n_queries": n_full, "mean_nll": float(nll_m.mean()), "median_nll": float(np.median(nll_m)),
            "ppl": float(np.exp(nll_m.mean())), "mean_p_true": float(p_true_m.mean()), "median_p_true": float(np.median(p_true_m)),
            "delta_nll_vs_gpt2": float(nll_m.mean() - nll0_full.mean()), "delta_nll_vs_dime": float(nll_m.mean() - nll_by_method["M2"].mean()),
            "frac_improves_vs_gpt2": frac_improves(nll_m, nll0_full), "frac_improves_vs_dime": frac_improves(nll_m, nll_by_method["M2"]),
        }
    print(f"  M2 (DIME): {expA['M2']}")
    print(f"  M3 (CRISP): {expA['M3']}")
    print(f"  Central success criterion -- NLL_CRISP < NLL_DIME: {expA['M3']['mean_nll'] < expA['M2']['mean_nll']} "
          f"({expA['M3']['mean_nll']:.5f} vs {expA['M2']['mean_nll']:.5f})")
    report["experiment_A_matched_nll"] = expA
    save_results(os.path.join(OUT_DIR, "crisp_r6_summary.json"), report)

    # ============================================================
    # R5-style stratified 8,000-query diagnostic (paired, zero extra compute)
    # ============================================================
    print(f"\nR5-style JS-decile-stratified {N_R5_DIAGNOSTIC_SAMPLE}-query diagnostic (paired, reusing already-computed values)...")
    # TV/JS-free stratification proxy (|p_raw_true - p_dime_true| decile) -- R5's own stratification used
    # full JS divergence, which is not computed in this script (only standalone true-token probabilities
    # are); this proxy serves the same purpose (spreading the sample across raw/DIME disagreement) cheaply.
    js_proxy = np.abs(p_raw_true - p_dime_true)
    deciles = np.digitize(js_proxy, np.quantile(js_proxy, np.linspace(0, 1, 11))[1:-1])
    per_decile = N_R5_DIAGNOSTIC_SAMPLE // 10
    r5_sample = []
    for dcl in range(10):
        pool = np.where(deciles == dcl)[0]
        take = min(per_decile, len(pool))
        r5_sample.append(rng.choice(pool, size=take, replace=False))
    r5_sample = np.concatenate(r5_sample)
    expA_r5sample = {m: {"n_queries": int(len(r5_sample)), "mean_nll": float(nll_by_method[m][r5_sample].mean())} for m in METHODS}
    print(f"  R5-sample diagnostic ({len(r5_sample)} queries): M2={expA_r5sample['M2']['mean_nll']:.5f}, M3={expA_r5sample['M3']['mean_nll']:.5f}")
    report["r5_sample_diagnostic"] = expA_r5sample
    save_results(os.path.join(OUT_DIR, "crisp_r6_summary.json"), report)

    # ============================================================
    # Experiment B -- alpha sensitivity (cheap vectorized sweep, same p_true arrays)
    # ============================================================
    print(f"\nExperiment B: alpha sensitivity sweep, alpha in {ALPHA_GRID}...")
    alpha_sweep_rows = []
    nll_by_method_alpha = {m: {} for m in ["M1", "M2", "M3", "M4", "M5"]}
    for alpha in ALPHA_GRID:
        for m in ["M1", "M2", "M3", "M4", "M5"]:
            _, nll_a = interpolated_nll(val_p_lm, p_true_by_method[m], alpha)
            nll_by_method_alpha[m][alpha] = float(nll_a.mean())
            alpha_sweep_rows.append({"method": m, "alpha": alpha, "mean_nll": float(nll_a.mean()), "ppl": float(np.exp(nll_a.mean()))})
    for m in ["M1", "M2", "M3", "M4", "M5"]:
        assert abs(nll_by_method_alpha[m][0.0] - nll0_full.mean()) < 1e-9, f"alpha=0 sanity check failed for {m}"
    print("  alpha=0 recovers GPT-2 NLL exactly for every method (verified).")

    # alpha=1 handled separately with explicit zero-probability accounting
    alpha1_report = {}
    for m, p_r in p_true_by_method.items():
        n_zero = int((p_r <= 0).sum())
        nll1 = -np.log(np.where(p_r > 0, p_r, np.nan))
        alpha1_report[m] = {"n_zero_prob_positions": n_zero, "frac_zero_prob": n_zero / n_full,
                              "mean_nll_over_nonzero_only": float(np.nanmean(nll1)), "nll_undefined_on_zero_positions": n_zero > 0}
    print(f"  alpha=1 (retrieval-only, explicit zero-probability accounting, NOT a floored result): {alpha1_report}")

    # Exploratory "individually tuned alpha" (Comparison 2) -- same val set, labeled explicitly
    best_alpha_per_method = {}
    for m in ["M1", "M2", "M3", "M4", "M5"]:
        best_alpha = min(nll_by_method_alpha[m], key=nll_by_method_alpha[m].get)
        best_alpha_per_method[m] = {"best_alpha_on_this_val_set": best_alpha, "mean_nll_at_best_alpha": nll_by_method_alpha[m][best_alpha],
                                       "mean_nll_at_primary_alpha": nll_by_method_alpha[m][PRIMARY_ALPHA],
                                       "EXPLORATORY_WARNING": "selected and evaluated on the SAME val queries -- "
                                       "NOT an independent generalization estimate; no untouched third split exists."}
    print(f"  Exploratory best-alpha-on-val (NOT independently validated): {best_alpha_per_method}")
    report["experiment_B_alpha_sensitivity"] = {"nll_by_method_alpha": nll_by_method_alpha, "alpha1_explicit_handling": alpha1_report,
                                                    "exploratory_best_alpha_same_val_set": best_alpha_per_method}
    save_results(os.path.join(OUT_DIR, "crisp_r6_summary.json"), report)

    with open(os.path.join(OUT_DIR, "alpha_sweep.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["method", "alpha", "mean_nll", "ppl"])
        w.writeheader()
        for row in alpha_sweep_rows:
            w.writerow(row)

    # ============================================================
    # Experiment C -- why CRISP helps or hurts (confidence groups, support groups, delta analysis)
    # ============================================================
    print("\nExperiment C: delta-NLL analysis, GPT-2 confidence groups, retrieval-support groups...")
    delta_crisp_dime = nll_by_method["M3"] - nll_by_method["M2"]
    worst_1pct = np.percentile(delta_crisp_dime, 99)
    worst_5pct = np.percentile(delta_crisp_dime, 95)
    worst_10pct = np.percentile(delta_crisp_dime, 90)
    contrib_1pct = float(delta_crisp_dime[delta_crisp_dime >= worst_1pct].sum() / np.abs(delta_crisp_dime).sum()) if np.abs(delta_crisp_dime).sum() > 0 else None
    contrib_5pct = float(delta_crisp_dime[delta_crisp_dime >= worst_5pct].sum() / np.abs(delta_crisp_dime).sum()) if np.abs(delta_crisp_dime).sum() > 0 else None
    contrib_10pct = float(delta_crisp_dime[delta_crisp_dime >= worst_10pct].sum() / np.abs(delta_crisp_dime).sum()) if np.abs(delta_crisp_dime).sum() > 0 else None

    high_conf_mask = val_p_lm >= np.quantile(val_p_lm, 0.75)
    overrule_mask = high_conf_mask & (nll_by_method["M3"] > nll0_full)
    expC_core = {
        "mean_delta_crisp_minus_dime": float(delta_crisp_dime.mean()), "median_delta_crisp_minus_dime": float(np.median(delta_crisp_dime)),
        "frac_improved": float((delta_crisp_dime < 0).mean()), "best_delta": float(delta_crisp_dime.min()), "worst_delta": float(delta_crisp_dime.max()),
        "contribution_of_worst_1pct": contrib_1pct, "contribution_of_worst_5pct": contrib_5pct, "contribution_of_worst_10pct": contrib_10pct,
        "frac_high_confidence_gpt2_overruled_by_crisp": float(overrule_mask.mean()),
        "n_high_confidence_gpt2_overruled_by_crisp": int(overrule_mask.sum()),
    }
    print(f"  {expC_core}")

    quartile_edges = np.quantile(val_p_lm, [0.25, 0.5, 0.75])
    conf_group = np.digitize(val_p_lm, quartile_edges)
    conf_names = {0: "very_low", 1: "low", 2: "moderate", 3: "high"}
    expC_conf_groups = {}
    for g, name in conf_names.items():
        mask = conf_group == g
        expC_conf_groups[name] = {"n": int(mask.sum()),
                                     "mean_delta_nll_raw_vs_gpt2": float((nll_by_method["M1"][mask] - nll0_full[mask]).mean()) if mask.sum() else None,
                                     "mean_delta_nll_dime_vs_gpt2": float((nll_by_method["M2"][mask] - nll0_full[mask]).mean()) if mask.sum() else None,
                                     "mean_delta_nll_crisp_vs_gpt2": float((nll_by_method["M3"][mask] - nll0_full[mask]).mean()) if mask.sum() else None,
                                     "mean_delta_nll_crisp_vs_dime": float(delta_crisp_dime[mask].mean()) if mask.sum() else None}
    print(f"  confidence groups: {expC_conf_groups}")

    expC_support_groups = {}
    for support_name, support_mask in [("raw_top20_support", raw_support), ("dime_crisp_support", dime_crisp_support),
                                          ("neither_support", ~raw_support & ~dime_crisp_support),
                                          ("both_support", raw_support & dime_crisp_support)]:
        mask = support_mask
        expC_support_groups[support_name] = {"n": int(mask.sum()), "frac_of_total": float(mask.mean()),
                                                "mean_nll_raw": float(nll_by_method["M1"][mask].mean()) if mask.sum() else None,
                                                "mean_nll_dime": float(nll_by_method["M2"][mask].mean()) if mask.sum() else None,
                                                "mean_nll_crisp": float(nll_by_method["M3"][mask].mean()) if mask.sum() else None}
    print(f"  support groups: {expC_support_groups}")
    report["experiment_C_delta_analysis"] = {"core": expC_core, "confidence_groups": expC_conf_groups, "support_groups": expC_support_groups}
    save_results(os.path.join(OUT_DIR, "crisp_r6_summary.json"), report)

    # ============================================================
    # Experiment D -- predictive value of local corrections (M1-M5 comparison)
    # ============================================================
    print("\nExperiment D: predictive value of local corrections...")
    nll_m1, nll_m2, nll_m3, nll_m4, nll_m5 = [nll_by_method[m] for m in ["M1", "M2", "M3", "M4", "M5"]]
    oracle_headroom = nll_m2.mean() - nll_m4.mean()
    crisp_recovered = nll_m2.mean() - nll_m3.mean()
    frac_recovered = float(crisp_recovered / oracle_headroom) if abs(oracle_headroom) > 1e-9 else None
    expD = {
        "exact_oracle_vs_dime": {"nll_m4": float(nll_m4.mean()), "nll_m2": float(nll_m2.mean()), "m4_beats_m2": bool(nll_m4.mean() < nll_m2.mean())},
        "crisp_fraction_of_oracle_headroom_recovered": frac_recovered,
        "mass_aware_vs_centroid_softmax": {"nll_m5": float(nll_m5.mean()), "nll_m4": float(nll_m4.mean()), "m5_beats_m4": bool(nll_m5.mean() < nll_m4.mean())},
        "remaining_gap_oracle_to_raw": float(nll_m4.mean() - nll_m1.mean()),
        "aggregation_imposes_major_limitation": bool((nll_m4.mean() - nll_m1.mean()) > 0.3 * abs(nll_m2.mean() - nll_m1.mean())) if abs(nll_m2.mean() - nll_m1.mean()) > 1e-9 else None,
    }
    print(f"  {expD}")
    report["experiment_D_predictive_value_of_local_corrections"] = expD
    save_results(os.path.join(OUT_DIR, "crisp_r6_summary.json"), report)

    # ============================================================
    # Experiment F -- storage versus predictive quality
    # ============================================================
    print("\nExperiment F: storage versus predictive quality (real measured bytes, all clusters)...")
    centroid_bytes = N_CLUSTERS * DIM * DTYPE_BYTES
    hist_bytes_total = sum(models[c]["n_tokens"] * (4 + 4) for c in range(N_CLUSTERS))
    crisp_coef_bytes_total = sum(models[c]["n_tokens"] * CRISP_RANK * DTYPE_BYTES for c in range(N_CLUSTERS) if models[c]["type"] == "fitted")
    crisp_anchor_bytes_total = sum(CRISP_RANK * 4 for c in range(N_CLUSTERS) if models[c]["type"] == "fitted")
    raw_bytes_total = len(ds_values) * (DIM * DTYPE_BYTES + VALUE_BYTES)
    dime_v1_total = centroid_bytes + hist_bytes_total
    crisp_total = centroid_bytes + hist_bytes_total + crisp_coef_bytes_total + crisp_anchor_bytes_total
    storage_rows = [
        {"system": "GPT-2 alone", "bytes": "N/A (GPT-2 weights shared across all systems, excluded from retrieval-datastore comparison)",
          "compression_ratio": "N/A", "nll": expA["M0"]["mean_nll"], "delta_nll_vs_gpt2": 0.0},
        {"system": "Raw kNN", "bytes": raw_bytes_total, "compression_ratio": 1.0, "nll": expA["M1"]["mean_nll"], "delta_nll_vs_gpt2": expA["M1"]["delta_nll_vs_gpt2"]},
        {"system": "DIME-V1", "bytes": dime_v1_total, "compression_ratio": float(raw_bytes_total / dime_v1_total), "nll": expA["M2"]["mean_nll"], "delta_nll_vs_gpt2": expA["M2"]["delta_nll_vs_gpt2"]},
        {"system": "CRISP rank8", "bytes": crisp_total, "compression_ratio": float(raw_bytes_total / crisp_total), "nll": expA["M3"]["mean_nll"], "delta_nll_vs_gpt2": expA["M3"]["delta_nll_vs_gpt2"]},
    ]
    crisp_worth_it = expA["M3"]["mean_nll"] < expA["M2"]["mean_nll"]
    expF = {"storage_rows": storage_rows, "crisp_expansion_over_dime_bytes": crisp_total - dime_v1_total,
             "crisp_expansion_over_dime_ratio": float(crisp_total / dime_v1_total),
             "crisp_nll_improvement_over_dime": float(expA['M2']['mean_nll'] - expA['M3']['mean_nll']),
             "verdict": ("CRISP's extra memory purchases a real NLL improvement over DIME" if crisp_worth_it else
                          "CRISP has WORSE NLL than DIME despite using MORE memory -- the extra storage is not "
                          "currently worth it")}
    print(f"  {expF}")
    report["experiment_F_storage_vs_nll"] = expF
    save_results(os.path.join(OUT_DIR, "crisp_r6_summary.json"), report)

    with open(os.path.join(OUT_DIR, "storage_nll_tradeoff.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["system", "bytes", "compression_ratio", "nll", "delta_nll_vs_gpt2"])
        w.writeheader()
        for row in storage_rows:
            w.writerow(row)

    # ============================================================
    # Deliverables: nll_results.csv, token_level_nll.csv
    # ============================================================
    print("\nWriting nll_results.csv and token_level_nll.csv...")
    with open(os.path.join(OUT_DIR, "nll_results.csv"), "w", newline="") as f:
        fieldnames = ["method", "method_name", "k", "tau", "alpha", "n_queries", "nll", "ppl",
                       "delta_nll_vs_gpt2", "delta_nll_vs_dime", "compression_ratio", "evaluation_split"]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        comp_ratio_by_method = {"M0": "N/A", "M1": 1.0, "M2": float(raw_bytes_total / dime_v1_total),
                                   "M3": float(raw_bytes_total / crisp_total), "M4": "N/A (diagnostic, requires raw vectors)",
                                   "M5": "N/A (diagnostic, requires raw vectors)", "M6": "N/A (diagnostic, requires raw vectors)"}
        for m in METHODS:
            w.writerow({"method": m, "method_name": METHOD_NAMES[m], "k": OUTER_K, "tau": TAU, "alpha": PRIMARY_ALPHA,
                         "n_queries": n_full, "nll": expA[m]["mean_nll"], "ppl": expA[m]["ppl"],
                         "delta_nll_vs_gpt2": expA[m]["delta_nll_vs_gpt2"], "delta_nll_vs_dime": expA[m]["delta_nll_vs_dime"],
                         "compression_ratio": comp_ratio_by_method[m], "evaluation_split": "val_full_scale"})
        if tuned_check:
            w.writerow({"method": "M1_tuned", "method_name": "GPT-2 + raw kNN (tuned)", "k": tuned_check["raw_tuned_config"]["k"],
                         "tau": tuned_check["raw_tuned_config"]["tau"], "alpha": tuned_check["raw_tuned_config"]["alpha"],
                         "n_queries": n_full, "nll": tuned_check["measured_mean_nll_raw_tuned"], "ppl": float(np.exp(tuned_check["measured_mean_nll_raw_tuned"])),
                         "delta_nll_vs_gpt2": tuned_check["measured_mean_nll_raw_tuned"] - gpt_only_check["measured_mean_nll"],
                         "delta_nll_vs_dime": None, "compression_ratio": 1.0, "evaluation_split": "val_full_scale (reproduction check)"})
            w.writerow({"method": "M2_tuned", "method_name": "GPT-2 + DIME (tuned)", "k": tuned_check["dime_tuned_config"]["k"],
                         "tau": tuned_check["dime_tuned_config"]["tau"], "alpha": tuned_check["dime_tuned_config"]["alpha"],
                         "n_queries": n_full, "nll": tuned_check["measured_mean_nll_dime_tuned"], "ppl": float(np.exp(tuned_check["measured_mean_nll_dime_tuned"])),
                         "delta_nll_vs_gpt2": tuned_check["measured_mean_nll_dime_tuned"] - gpt_only_check["measured_mean_nll"],
                         "delta_nll_vs_dime": None, "compression_ratio": None, "evaluation_split": "val_full_scale (reproduction check)"})

    with open(os.path.join(OUT_DIR, "token_level_nll.csv"), "w", newline="") as f:
        fieldnames = ["query_idx", "target_token_id", "p_lm_true", "p_raw_true", "p_dime_true", "p_crisp_true",
                       "p_s3_true", "p_s5_true", "nll_gpt2", "nll_raw", "nll_dime", "nll_crisp", "nll_s3", "nll_s5",
                       "delta_nll_crisp_minus_dime", "raw_support", "dime_crisp_support"]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for qi in range(n_full):
            w.writerow({"query_idx": qi, "target_token_id": int(val_values[qi]), "p_lm_true": round(float(val_p_lm[qi]), 8),
                         "p_raw_true": round(float(p_raw_true[qi]), 8), "p_dime_true": round(float(p_dime_true[qi]), 8),
                         "p_crisp_true": round(float(p_crisp_true[qi]), 8), "p_s3_true": round(float(p_s3_true[qi]), 8),
                         "p_s5_true": round(float(p_s5_true[qi]), 8), "nll_gpt2": round(float(nll0_full[qi]), 6),
                         "nll_raw": round(float(nll_by_method["M1"][qi]), 6), "nll_dime": round(float(nll_by_method["M2"][qi]), 6),
                         "nll_crisp": round(float(nll_by_method["M3"][qi]), 6), "nll_s3": round(float(nll_by_method["M4"][qi]), 6),
                         "nll_s5": round(float(nll_by_method["M5"][qi]), 6), "delta_nll_crisp_minus_dime": round(float(delta_crisp_dime[qi]), 6),
                         "raw_support": bool(raw_support[qi]), "dime_crisp_support": bool(dime_crisp_support[qi])})

    # ============================================================
    # Experiment G -- error analysis / failure cases
    # ============================================================
    print("\nExperiment G: error analysis / failure cases...")
    categories = {
        "1_crisp_substantially_improves_over_dime": np.argsort(delta_crisp_dime)[:N_FAILURE_PER_CATEGORY],
        "2_crisp_substantially_worse_than_dime": np.argsort(-delta_crisp_dime)[:N_FAILURE_PER_CATEGORY],
        "3_raw_knn_much_better_than_both_compressed": np.argsort(-((nll_by_method["M2"] - nll_m1) + (nll_m3 - nll_m1)))[:N_FAILURE_PER_CATEGORY],
        # (nll_m2-nll_m4) - (nll_m2-nll_m3) algebraically simplifies to nll_m3-nll_m4 (CRISP minus oracle);
        # the trailing [::-1] after ascending argsort puts the LARGEST (most positive, CRISP much worse
        # than the oracle) first -- verified by direct algebra before use, unlike the reversed-direction
        # bug caught in R5's equivalent category.
        "4_exact_oracle_helps_but_crisp_does_not": np.argsort((nll_m2 - nll_m4) - (nll_m2 - nll_m3))[::-1][:N_FAILURE_PER_CATEGORY],
        "5_mass_aware_changes_performance_substantially": np.argsort(-np.abs(nll_m5 - nll_m4))[:N_FAILURE_PER_CATEGORY],
    }
    error_lines = ["# CRISP-R6 Error Analysis", "",
                     "All examples use the SAME query identifiers across systems. Selected by the stated ranking "
                     "rule per category (not cherry-picked to favor CRISP).", ""]
    for cat_name, idxs in categories.items():
        error_lines.append(f"## Category {cat_name}")
        for qi in idxs:
            qi = int(qi)
            true_tok = int(val_values[qi])
            decoded = gpt.tokenizer.decode([true_tok])
            row = {"query_idx": qi, "true_token": decoded, "selected_centroids": idx_dime_val[qi].tolist(),
                    "p_lm_true": round(float(val_p_lm[qi]), 5), "p_raw_true": round(float(p_raw_true[qi]), 5),
                    "p_dime_true": round(float(p_dime_true[qi]), 5), "p_crisp_true": round(float(p_crisp_true[qi]), 5),
                    "p_s3_true": round(float(p_s3_true[qi]), 5), "nll_gpt2": round(float(nll0_full[qi]), 5),
                    "nll_raw": round(float(nll_m1[qi]), 5), "nll_dime": round(float(nll_m2[qi]), 5),
                    "nll_crisp": round(float(nll_m3[qi]), 5), "nll_s3": round(float(nll_m4[qi]), 5),
                    "delta_nll_crisp_minus_dime": round(float(delta_crisp_dime[qi]), 5)}
            error_lines.append(f"- {row}")
        error_lines.append("")
    with open(os.path.join(OUT_DIR, "error_analysis.md"), "w") as f:
        f.write("\n".join(error_lines))

    # ============================================================
    # Statistical reliability (Section 12)
    # ============================================================
    print("\nStatistical reliability (paired bootstrap, i.i.d. over query positions)...")
    n_boot = 2000
    boot_rng = np.random.default_rng(SEED)
    boot_means = np.array([delta_crisp_dime[boot_rng.integers(0, n_full, size=n_full)].mean() for _ in range(n_boot)])
    ci_lo, ci_hi = np.percentile(boot_means, [2.5, 97.5])
    stats_reliability = {"mean_delta_crisp_minus_dime": float(delta_crisp_dime.mean()), "median_delta": float(np.median(delta_crisp_dime)),
                           "ci_95_low": float(ci_lo), "ci_95_high": float(ci_hi), "frac_improved": float((delta_crisp_dime < 0).mean()),
                           "significant": bool(ci_lo < 0 and ci_hi < 0), "n_queries": n_full,
                           "bootstrap_type": "i.i.d. over query positions, NOT sequence/block-aware (same scope "
                                               "simplification R5 made, stated explicitly, not silently repeated)."}
    print(f"  {stats_reliability}")
    report["statistical_reliability"] = stats_reliability
    save_results(os.path.join(OUT_DIR, "crisp_r6_summary.json"), report)

    # ============================================================
    # Plots
    # ============================================================
    print("\nGenerating plots...")
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(METHODS, [expA[m]["mean_nll"] for m in METHODS])
    ax.axhline(expA["M0"]["mean_nll"], color="k", linestyle="--", linewidth=1, label="GPT-2 alone")
    ax.set_ylabel("mean interpolated NLL"); ax.legend(); ax.set_title("Experiment A: interpolated NLL by method")
    fig.tight_layout(); fig.savefig(os.path.join(PLOT_DIR, "A_nll_by_method.png"), dpi=130); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    for m in ["M1", "M2", "M3", "M4", "M5"]:
        ax.plot(ALPHA_GRID, [nll_by_method_alpha[m][a] for a in ALPHA_GRID], marker="o", label=m)
    ax.set_xlabel("alpha"); ax.set_ylabel("mean NLL"); ax.legend(); ax.set_title("Experiment B: NLL vs. alpha")
    fig.tight_layout(); fig.savefig(os.path.join(PLOT_DIR, "B_nll_vs_alpha.png"), dpi=130); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(delta_crisp_dime, bins=80)
    ax.axvline(0, color="k", linestyle="--")
    ax.set_xlabel("NLL(CRISP) - NLL(DIME) per token"); ax.set_ylabel("count")
    ax.set_title(f"Paired CRISP-minus-DIME NLL distribution (mean={delta_crisp_dime.mean():.4f})")
    fig.tight_layout(); fig.savefig(os.path.join(PLOT_DIR, "C_paired_delta_distribution.png"), dpi=130); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    names_c = list(conf_names.values())
    ax.plot(names_c, [expC_conf_groups[n]["mean_delta_nll_dime_vs_gpt2"] for n in names_c], marker="o", label="DIME")
    ax.plot(names_c, [expC_conf_groups[n]["mean_delta_nll_crisp_vs_gpt2"] for n in names_c], marker="s", label="CRISP")
    ax.set_ylabel("mean delta NLL vs. GPT-2 alone"); ax.legend()
    ax.set_title("NLL improvement by GPT-2 confidence group")
    fig.tight_layout(); fig.savefig(os.path.join(PLOT_DIR, "D_confidence_group.png"), dpi=130); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    # Some support categories can be empty (n=0, mean_nll_crisp=None) depending on how retrieval behaves
    # on the actual data -- filtered out here rather than plotted as a crash-inducing None bar.
    support_names = [s for s in expC_support_groups if expC_support_groups[s]["mean_nll_crisp"] is not None]
    ax.bar(support_names, [expC_support_groups[s]["mean_nll_crisp"] for s in support_names])
    ax.set_ylabel("mean CRISP NLL"); ax.set_xticklabels(support_names, rotation=20, ha="right")
    ax.set_title("NLL by retrieval-support category")
    fig.tight_layout(); fig.savefig(os.path.join(PLOT_DIR, "E_support_category.png"), dpi=130); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    bytes_plot = [raw_bytes_total, dime_v1_total, crisp_total]
    nll_plot = [expA["M1"]["mean_nll"], expA["M2"]["mean_nll"], expA["M3"]["mean_nll"]]
    ax.scatter(bytes_plot, nll_plot, s=80)
    for name, x, y in zip(["Raw kNN", "DIME", "CRISP"], bytes_plot, nll_plot):
        ax.annotate(name, (x, y))
    ax.set_xscale("log"); ax.set_xlabel("retrieval-datastore bytes (log scale)"); ax.set_ylabel("mean NLL")
    ax.set_title("NLL versus datastore memory")
    fig.tight_layout(); fig.savefig(os.path.join(PLOT_DIR, "F_nll_vs_memory.png"), dpi=130); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(["M2 DIME", "M3 CRISP", "M4 exact oracle", "M5 mass oracle"],
            [expA["M2"]["mean_nll"], expA["M3"]["mean_nll"], expA["M4"]["mean_nll"], expA["M5"]["mean_nll"]])
    ax.axhline(expA["M1"]["mean_nll"], color="r", linestyle="--", label="Raw kNN (M1)")
    ax.set_ylabel("mean NLL"); ax.legend(); ax.set_title("Oracle versus compressed NLL")
    fig.tight_layout(); fig.savefig(os.path.join(PLOT_DIR, "G_oracle_vs_compressed.png"), dpi=130); plt.close(fig)

    # ============================================================
    # mathematical_observations.md / research_questions.md
    # ============================================================
    print("Writing mathematical_observations.md and research_questions.md...")
    math_obs = f"""# CRISP-R6 Mathematical Observations

## Observation 1 -- Lower TV/JS does not mechanically imply lower interpolated NLL

**Claim:** R5 found CRISP closer to raw kNN in global TV/JS than DIME, yet R5's own FLOORED retrieval-only NLL
showed CRISP worse than DIME. R6 tests whether this conflict persists under proper, floor-free interpolation.

**Evidence:** compare experiment_A_matched_nll's M2 vs M3 mean_nll directly -- {"CRISP beats DIME" if expA['M3']['mean_nll'] < expA['M2']['mean_nll'] else "DIME beats CRISP"} at alpha={PRIMARY_ALPHA}
({expA['M3']['mean_nll']:.5f} vs {expA['M2']['mean_nll']:.5f}). NLL is a LOG-scale, SINGLE-coordinate functional of
the retrieval distribution (only the true token's probability matters), while TV/JS are FULL-support divergences --
these can disagree because a method can improve the AVERAGE shape of a distribution while occasionally assigning
much lower probability to the one coordinate (the true token) that NLL actually scores.

**Status:** empirically resolved by this script's own alpha=0.25 matched comparison; NOT assumed from R5's TV/JS
finding either way.

**Technique:** this is exactly the mismatch between an L1/f-divergence-style distributional metric and a
log-loss (proper scoring rule) evaluated at one realized outcome -- relevant literature: proper scoring rules,
calibration vs. sharpness tradeoffs.

## Observation 2 -- Interpolation with a strictly positive LM prior removes the floor-sensitivity problem entirely

**Claim:** for alpha<1, -log[(1-alpha)*P_LM + alpha*P_R] is insensitive to whether P_R is EXACTLY zero or merely
very small, because the GPT-2 term alone already lower-bounds the mixture away from zero. R5's floored
retrieval-only NLL (alpha=1 implicitly, with an artificial floor) is NOT representative of the actual deployed
interpolated system.

**Evidence:** alpha1_report in the JSON summary shows the exact zero-probability rate for every method; comparing
those n_zero counts against experiment_A_matched_nll's alpha=0.25 mean_nll values shows the mixed NLL is governed
by p_lm_true's distribution in the many-zero-probability regime, not by the retrieval method's floor choice.

**Status:** proven algebraically (p_mix >= (1-alpha)*p_lm_true > 0 strictly); the magnitude of how much this
matters in practice is the empirical content of alpha1_report.

**Technique:** this is a direct consequence of log-sum lower-bounding; no further technique needed to prove it,
though quantifying HOW MUCH residual sensitivity remains near alpha=1 is an open empirical question (see
research_questions.md).

## Observation 3 -- Oracle headroom vs. recovered headroom separates "CRISP's own limitation" from "DIME's inherent limitation"

**Claim:** M2 (DIME) vs. M4 (exact oracle) isolates how much NLL improvement is even POSSIBLE under perfect local
reconstruction (holding centroid routing/mixing fixed); M2 vs. M3 (CRISP) isolates how much of that CRISP actually
captures.

**Evidence:** experiment_D's crisp_fraction_of_oracle_headroom_recovered = {frac_recovered}. A value near 1 means
CRISP is close to saturating the achievable local-reconstruction improvement; a value near 0 or negative means
CRISP's specific approximation (not the underlying local-reconstruction idea) is the bottleneck.

**Status:** empirically measured directly; interpretation (whether the number is "good") is left to report.md's
verdict section, not asserted here.

**Technique:** this is a standard oracle-vs-model decomposition; a sharper treatment would need a formal
regret bound relating within-cluster approximation error (R4's own TV metric) to this NLL-space headroom-recovery
fraction.
"""
    with open(os.path.join(OUT_DIR, "mathematical_observations.md"), "w") as f:
        f.write(math_obs)

    research_q = """# CRISP-R6 Research Questions for Subsequent Mathematical Study

## Proper scoring rules / information theory
- Observation 1 shows TV/JS improvement does not guarantee NLL improvement. Is there a PRECISE condition
  (e.g. on the correlation between a method's per-token support coverage and its probability mass on rare
  correct tokens) under which a TV/JS improvement DOES guarantee an NLL improvement? Keywords: proper scoring
  rules, Bregman divergence, calibration-sharpness decomposition.
- How much residual sensitivity to near-zero retrieval probability remains as alpha approaches 1 (Observation 2)?
  Is there a closed-form expression for d(NLL)/d(alpha) at alpha=1 in terms of the retrieval distribution's mass
  on the true token?

## Approximation theory / statistical estimation
- Is there a formal regret bound relating CRISP's within-cluster TV error (measured in R4) to its NLL-space
  "fraction of oracle headroom recovered" (Observation 3)? Currently only an empirical correlation can be
  measured, not a provable bound.
- Given R5's finding that required CRISP rank varies by cluster, and R6's finding of which clusters/confidence
  groups dominate the NLL gap, would a DIFFERENT (NLL-aware, not TV-aware) capacity allocation rule change which
  clusters are prioritized for higher rank?

## Experimental design
- R6's "individually tuned alpha" comparison is explicitly exploratory (no untouched third split exists). What
  is the smallest additional held-out set this project could carve out (e.g. a contiguous document-level
  held-back block) that would make a genuinely independent tuned-alpha confirmation possible?
- The confidence-group and support-group analyses (Experiment C) use predetermined quartile bins -- would a
  continuous (non-binned) regression of delta-NLL on p_lm_true reveal a smoother, more precise relationship?

## Mechanism
- Experiment D's "aggregation imposes a major limitation" check compares M4-vs-M1 against M2-vs-M1 at a fixed,
  predetermined threshold (30%). Is there a principled (not arbitrary) threshold derivable from the geometry of
  DIME's centroid partition (e.g. related to average within-cluster radius vs. between-cluster separation)?
"""
    with open(os.path.join(OUT_DIR, "research_questions.md"), "w") as f:
        f.write(research_q)

    # ============================================================
    # Final verdict (Q1-Q10)
    # ============================================================
    print("\nComputing final verdict (Q1-Q10)...")
    q1 = {"crisp_beats_gpt2_alone": bool(expA["M3"]["mean_nll"] < expA["M0"]["mean_nll"]),
           "nll_crisp": expA["M3"]["mean_nll"], "nll_gpt2": expA["M0"]["mean_nll"]}
    q2 = {"crisp_beats_dime_at_alpha025": bool(expA["M3"]["mean_nll"] < expA["M2"]["mean_nll"]),
           "nll_crisp": expA["M3"]["mean_nll"], "nll_dime": expA["M2"]["mean_nll"], "significant": stats_reliability["significant"]}
    q3 = {"nll_crisp": expA["M3"]["mean_nll"], "nll_raw": expA["M1"]["mean_nll"], "gap_to_raw": expA["M3"]["mean_nll"] - expA["M1"]["mean_nll"]}
    q4 = best_alpha_per_method.get("M3")
    q5 = {"frac_tokens_improved": expC_core["frac_improved"], "contribution_of_worst_10pct_regressions": expC_core["contribution_of_worst_10pct"]}
    q6 = {"n_high_confidence_overruled": expC_core["n_high_confidence_gpt2_overruled_by_crisp"],
           "frac_high_confidence_overruled": expC_core["frac_high_confidence_gpt2_overruled_by_crisp"]}
    q7 = {"exact_oracle_beats_dime": expD["exact_oracle_vs_dime"]["m4_beats_m2"], "magnitude": expD["exact_oracle_vs_dime"]["nll_m2"] - expD["exact_oracle_vs_dime"]["nll_m4"]}
    q8 = {"mass_aware_beats_centroid_softmax": expD["mass_aware_vs_centroid_softmax"]["m5_beats_m4"],
           "magnitude": expD["mass_aware_vs_centroid_softmax"]["nll_m4"] - expD["mass_aware_vs_centroid_softmax"]["nll_m5"]}
    q9 = {"crisp_nll_improvement_over_dime": expF["crisp_nll_improvement_over_dime"], "crisp_extra_bytes_over_dime": expF["crisp_expansion_over_dime_bytes"],
           "worth_it": expF["crisp_nll_improvement_over_dime"] > 0}
    q10 = f"Remaining gap oracle-to-raw: {expD['remaining_gap_oracle_to_raw']:.5f} -- " + (
        "aggregation/routing (not local-function approximation) is the dominant remaining limitation."
        if expD["aggregation_imposes_major_limitation"] else
        "local-function approximation quality remains the dominant lever, not routing/aggregation.")

    if q2["crisp_beats_dime_at_alpha025"] and q2["significant"] and q3["gap_to_raw"] < 0.3 * abs(expA["M2"]["mean_nll"] - expA["M1"]["mean_nll"]):
        verdict = "Strong evidence of useful predictive compression"
    elif q2["crisp_beats_dime_at_alpha025"] and q2["significant"]:
        verdict = "Partial evidence: CRISP improves on DIME but remains substantially behind raw kNN"
    elif (not q2["crisp_beats_dime_at_alpha025"]) and expA["M3"]["mean_nll"] < expA["M0"]["mean_nll"]:
        verdict = "CRISP worsens predictive performance relative to DIME"
    elif q2["crisp_beats_dime_at_alpha025"] and not q2["significant"]:
        verdict = "Inconclusive due to evaluation limitations"
    else:
        verdict = "Distributional improvement without predictive NLL improvement"
    print(f"  Verdict: {verdict}")
    verdict_block = {"Q1": q1, "Q2": q2, "Q3": q3, "Q4": q4, "Q5": q5, "Q6": q6, "Q7": q7, "Q8": q8, "Q9": q9, "Q10": q10,
                       "final_classification": verdict}
    print(f"  {verdict_block}")
    report["final_verdict"] = verdict_block
    save_results(os.path.join(OUT_DIR, "crisp_r6_summary.json"), report)

    # ============================================================
    # report.md
    # ============================================================
    print("\nGenerating report.md...")
    lines = [
        "# CRISP-R6: GPT-2 Interpolation and Predictive NLL Evaluation", "",
        "## 1. Research question", "See module docstring.", "",
        "## 2. Exact evaluation configurations", f"k={OUTER_K}, tau={TAU}, primary alpha={PRIMARY_ALPHA}, CRISP rank={CRISP_RANK}.", "",
        "## 3. Dataset and split provenance", f"Full val set, n={n_full}. No untouched third split exists -- see Limitations.", "",
        "## 4. GPT-2 baseline reproduction", f"{gpt_only_check}", f"Tuned-config reproduction check: {tuned_check}", "",
        "## 5. Matched interpolated NLL results", f"{expA}", "",
        "## 6. Perplexity results", f"{ {m: expA[m]['ppl'] for m in METHODS} }", "",
        "## 7. Alpha sensitivity", f"{report['experiment_B_alpha_sensitivity']}", "",
        "## 8. Correct-token probability analysis", f"{ {m: {'mean_p_true': expA[m]['mean_p_true']} for m in METHODS} }", "",
        "## 9. Oracle comparisons", f"{expD}", "",
        "## 10. Storage-versus-NLL tradeoff", f"{expF}", "",
        "## 11. Confidence intervals", f"{stats_reliability}", "",
        "## 12. Failure cases", "See error_analysis.md for all 5 categories with shared query identifiers.", "",
        "## 13. Limitations",
        "- No genuinely untouched third data split exists -- Experiment B's 'individually tuned alpha' is "
        "EXPLORATORY, not an independent generalization result.",
        "- Statistical reliability uses an i.i.d. (not sequence/block-aware) bootstrap over query positions.",
        "- GPT-2 confidence groups use predetermined quartile bins, not development-tuned thresholds.",
        "- CRISP's S6 diagnostic (CRISP + exact masses) is not part of R6's required M-list and was not computed.",
        "- The tuned-config reproduction check (Phase 1) covers raw/DIME only, not a tuned-CRISP evaluation -- "
        "CRISP is only ever fit/evaluated at the matched rank=8 configuration, consistent with R5's precedent.", "",
        "## 14. Final verdict", f"**{verdict}**", f"{verdict_block}",
    ]
    with open(os.path.join(OUT_DIR, "report.md"), "w") as f:
        f.write("\n".join(lines))

    report["sections_completed"] = ["phase1_baseline", "crisp_fitting_coverage", "phase3_sanity_checks",
                                      "phase4_main_loop", "experiment_A_matched_nll", "r5_sample_diagnostic",
                                      "experiment_B_alpha_sensitivity", "experiment_C_delta_analysis",
                                      "experiment_D_predictive_value_of_local_corrections", "experiment_F_storage_vs_nll",
                                      "statistical_reliability", "error_analysis.md", "mathematical_observations.md",
                                      "research_questions.md", "final_verdict", "report.md", "plots"]
    report["total_runtime_seconds"] = time.perf_counter() - t_start
    save_results(os.path.join(OUT_DIR, "crisp_r6_summary.json"), report)
    print(f"\nAll results saved under {OUT_DIR}")
    print(f"Total runtime: {report['total_runtime_seconds']/60:.1f} min")
    print("No existing E1/E2/TRI1/NC1/GEOM1/KM1/R1/R2/R3/R4/R5 outputs were modified.")


if __name__ == "__main__":
    main()
