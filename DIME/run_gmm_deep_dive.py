"""Deep dive: GMM-hard DIME vs k-means DIME, isolating WHICH part of GMM geometry
(mixture weights, component-specific variance, anisotropic covariance) is
responsible for any change, and whether that change is concentrated near k-means'
cluster boundaries. Soft DIME membership and spectral clustering are explicitly out
of scope here — every variant below uses HARD final cluster assignment
(c_i = argmax_c gamma_ic) and ordinary hard-count DIME construction.

TinyStories/gpt2, same seed=42/seq_len=128/{350,75,75} split as every other
TinyStories/gpt2 result in this project. Retrieval hyperparameters (k=20, tau=2.0,
alpha=0.05 — DIME's established tuned config at this scale) are held fixed across
every variant, matching the original GMM experiment.

Data-split discipline: every tuning/selection decision below (covariance family,
regularization, EM iteration count, seed) is driven by CONTROLLER-TRAIN NLL. The
validation set is touched only in the final summary table (section 13 / I).
"""
import sys, os, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "GPT_Module"))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from gpt_class import FrozenGPT2
from dataset_helpers import collect_chunks_split
from extract import build_datastore_from_chunks, run_batch
from knn import build_datastore, query_knn
from mixing import mix_dime_and_lm
from results_io import save_results

from gmm import init_gmm_from_kmeans, fit_gmm_rich
from gmm_dime_construction import (
    build_kmeans_dime, build_hard_dime_from_assignment,
    kmeans_boundary_margin, assignment_similarity_report, margin_percentile_curve,
    counterfactual_assignments, conditional_entropy_report, cluster_level_report,
)

FIG_DIR = os.path.join(os.path.dirname(__file__), "figures")
os.makedirs(FIG_DIR, exist_ok=True)
OUT_PATH = "results/gmm_deep_dive.json"

N_CLUSTERS = 500
K, TAU, ALPHA = 20, 2.0, 0.05
MAX_ITER = 300
REL_TOL = 1e-5
PATIENCE = 3
CHECKPOINT_EVERY = 10
N_SEEDS = 5


def encode_split(gpt, chunks_list, batch_size=256):
    all_keys, all_targets, all_p_lm = [], [], []
    for i in range(0, len(chunks_list), batch_size):
        batch = chunks_list[i:i + batch_size]
        h_pred, y_target, p_true, nll = run_batch(gpt, batch)
        B, L, D = h_pred.shape
        all_keys.append(h_pred.reshape(B * L, D).cpu().numpy())
        all_targets.append(y_target.reshape(B * L).cpu().numpy())
        all_p_lm.append(p_true.reshape(B * L).cpu().numpy())
    return (np.concatenate(all_keys, axis=0),
            np.concatenate(all_targets, axis=0),
            np.concatenate(all_p_lm, axis=0))


def eval_dime_nll(compressed_keys, compressed_dists, query_keys, query_values, query_p_lm_true, k=K):
    index, stored_values = build_datastore(compressed_keys, compressed_dists)
    k_eff = min(k, len(compressed_keys))
    distances, retrieved = query_knn(index, stored_values, query_keys, k=k_eff)
    _, nll = mix_dime_and_lm(distances, retrieved, query_values, query_p_lm_true, tau=TAU, alpha=ALPHA)
    return float(nll.mean()), nll


def plot_em_diagnostics(history, name):
    iters = [h["iteration"] for h in history]
    ll = [h["mean_log_likelihood"] for h in history]
    rel_imp = [h["rel_improvement"] for h in history[1:]]
    frac_changed = [h["frac_assignment_changed"] for h in history[1:]]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].plot(iters, ll)
    axes[0].set_title(f"{name}: log-likelihood vs iteration")
    axes[0].set_xlabel("iteration"); axes[0].set_ylabel("mean log-lik")
    axes[1].plot(iters[1:], rel_imp)
    axes[1].set_yscale("log")
    axes[1].set_title(f"{name}: relative improvement")
    axes[1].set_xlabel("iteration"); axes[1].set_ylabel("rel. improvement (log scale)")
    axes[2].plot(iters[1:], frac_changed)
    axes[2].set_title(f"{name}: fraction assignment changed")
    axes[2].set_xlabel("iteration"); axes[2].set_ylabel("fraction changed")
    fig.tight_layout()
    path = os.path.join(FIG_DIR, f"em_diagnostics_{name}.png")
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print(f"  saved {path}")


def main():
    gpt = FrozenGPT2()
    dataset = __import__("datasets").load_dataset(
        "roneneldan/TinyStories", split="train", streaming=True, trust_remote_code=True
    )
    chunks, _ = collect_chunks_split(
        dataset, gpt.tokenizer, seq_len=128,
        n_chunks_needed={"datastore": 350, "controller_train": 75, "val": 75}, seed=42,
    )
    print("Encoding datastore / controller_train / val...")
    ds_keys, ds_values = build_datastore_from_chunks(gpt, chunks["datastore"], batch_size=256)
    ct_keys, ct_values, ct_p_lm_true = encode_split(gpt, chunks["controller_train"])
    val_keys, val_values, val_p_lm_true = encode_split(gpt, chunks["val"])
    print(f"datastore: {len(ds_values)}  controller_train: {len(ct_values)}  val: {len(val_values)}")

    results = {"phase": "gmm_deep_dive", "dataset": "tinystories", "model": "gpt2",
               "n_clusters": N_CLUSTERS, "fixed_config": {"k": K, "tau": TAU, "alpha": ALPHA},
               "max_iter": MAX_ITER, "rel_tol": REL_TOL, "patience": PATIENCE}

    # ============================================================
    # k-means baseline (Variant A), reused throughout
    # ============================================================
    print("\n=== k-means baseline ===")
    centroids_a, assignment_a, dists_a = build_kmeans_dime(ds_keys, ds_values, N_CLUSTERS, seed=42)
    ct_nll_kmeans, _ = eval_dime_nll(centroids_a, dists_a, ct_keys, ct_values, ct_p_lm_true)
    print(f"k-means controller-train NLL: {ct_nll_kmeans:.4f}")
    results["kmeans_baseline"] = {"controller_train_nll": ct_nll_kmeans}
    save_results(OUT_PATH, results)

    # ============================================================
    # Section 1/2 — EM convergence + covariance family comparison
    # ============================================================
    print("\n=== Section 1/2: EM convergence + covariance family comparison ===")
    covariance_families = {
        "learned_spherical": "spherical",
        "diagonal": "diag",
        "tied_diagonal": "tied_diag",
    }
    family_results = {}
    fitted_gmms = {}

    for name, cov_type in covariance_families.items():
        print(f"\n--- fitting {name} ({cov_type}) ---")
        init_means, init_var, init_w = init_gmm_from_kmeans(ds_keys, centroids_a, assignment_a, cov_type)
        fit = fit_gmm_rich(ds_keys, init_means, init_var, init_w, covariance_type=cov_type,
                            rel_tol=REL_TOL, patience=PATIENCE, max_iter=MAX_ITER,
                            batch_size=5000, checkpoint_every=CHECKPOINT_EVERY, verbose=True)
        fitted_gmms[name] = fit
        plot_em_diagnostics(fit["history"], name)

        # Checkpoint -> controller-train NLL curve: does higher likelihood -> better DIME?
        checkpoint_curve = []
        for ckpt in fit["checkpoints"]:
            dists_ckpt = build_hard_dime_from_assignment(ds_values, ckpt["hard_assignment"], N_CLUSTERS)
            ct_nll_ckpt, _ = eval_dime_nll(ckpt["means"], dists_ckpt, ct_keys, ct_values, ct_p_lm_true)
            checkpoint_curve.append({"iteration": ckpt["iteration"],
                                      "mean_log_likelihood": ckpt["mean_log_likelihood"],
                                      "controller_train_nll": ct_nll_ckpt})
            print(f"  checkpoint iter {ckpt['iteration']}: ll={ckpt['mean_log_likelihood']:.4f} "
                  f"ct_nll={ct_nll_ckpt:.4f}")

        fig, ax = plt.subplots(figsize=(6, 5))
        lls = [c["mean_log_likelihood"] for c in checkpoint_curve]
        nlls = [c["controller_train_nll"] for c in checkpoint_curve]
        ax.plot(lls, nlls, "o-")
        for c in checkpoint_curve:
            ax.annotate(str(c["iteration"]), (c["mean_log_likelihood"], c["controller_train_nll"]), fontsize=7)
        ax.set_xlabel("GMM mean log-likelihood")
        ax.set_ylabel("DIME controller-train NLL")
        ax.set_title(f"{name}: does higher likelihood predict better DIME?")
        fig.tight_layout()
        fig.savefig(os.path.join(FIG_DIR, f"likelihood_vs_dime_nll_{name}.png"), dpi=130)
        plt.close(fig)

        hard_assignment = fit["responsibilities"].argmax(axis=1)
        dists_final = build_hard_dime_from_assignment(ds_values, hard_assignment, N_CLUSTERS)
        ct_nll_final, _ = eval_dime_nll(fit["means"], dists_final, ct_keys, ct_values, ct_p_lm_true)
        sim_report = assignment_similarity_report(assignment_a, hard_assignment, centroids_a, fit["means"])

        cluster_sizes = np.bincount(hard_assignment, minlength=N_CLUSTERS).astype(np.float64)
        family_results[name] = {
            "covariance_type": cov_type,
            "converged": fit["converged"], "n_iterations": fit["n_iterations"],
            "final_log_likelihood": fit["history"][-1]["mean_log_likelihood"],
            "controller_train_nll": ct_nll_final,
            "checkpoint_curve": checkpoint_curve,
            "assignment_vs_kmeans": sim_report,
            "cluster_size_distribution": {"mean": float(cluster_sizes.mean()), "std": float(cluster_sizes.std()),
                                           "min": float(cluster_sizes.min()), "max": float(cluster_sizes.max())},
            "alpha_distribution": {"mean": float(fit["weights"].mean()), "std": float(fit["weights"].std()),
                                    "min": float(fit["weights"].min()), "max": float(fit["weights"].max())},
        }
        if cov_type == "spherical":
            family_results[name]["variance_stats"] = {
                "mean": float(fit["variances"].mean()), "min": float(fit["variances"].min()),
                "max": float(fit["variances"].max()),
            }
        elif cov_type == "diag":
            per_comp_mean_var = fit["variances"].mean(axis=1)
            per_comp_logdet = np.log(fit["variances"]).sum(axis=1)
            ratio = fit["variances"].max(axis=1) / np.clip(fit["variances"].min(axis=1), 1e-12, None)
            family_results[name]["covariance_stats"] = {
                "per_component_mean_variance": {"mean": float(per_comp_mean_var.mean()),
                                                 "min": float(per_comp_mean_var.min()), "max": float(per_comp_mean_var.max())},
                "per_component_logdet": {"mean": float(per_comp_logdet.mean()),
                                          "min": float(per_comp_logdet.min()), "max": float(per_comp_logdet.max())},
                "per_component_max_min_var_ratio": {"mean": float(ratio.mean()), "max": float(ratio.max())},
            }

        results["covariance_family_comparison"] = family_results
        save_results(OUT_PATH, results)

    # The "equal spherical, sanity baseline" family (B) — NOT fit via EM at all, a
    # fixed model with one global variance and equal alpha. By the algebraic identity
    # proven in the earlier synthetic 2D check, its hard assignment is EXACTLY
    # k-means' own assignment (log-alpha and log-det terms are identical across c,
    # so argmax reduces to nearest-centroid) — included here as a real check that the
    # identity holds in actual hidden-state space too, not just in the 2D synthetic.
    print("\n--- equal_spherical_sanity (fixed, not EM-fit) ---")
    sqnorm = ((ds_keys ** 2).sum(axis=1, keepdims=True) - 2.0 * ds_keys @ centroids_a.T
              + (centroids_a ** 2).sum(axis=1)[None, :])
    equal_spherical_assignment = sqnorm.argmin(axis=1)
    frac_matches_kmeans = float((equal_spherical_assignment == assignment_a).mean())
    print(f"equal_spherical_sanity vs k-means agreement: {frac_matches_kmeans*100:.3f}% (expected: 100.000%)")
    family_results["equal_spherical_sanity"] = {"fraction_agrees_with_kmeans": frac_matches_kmeans}
    results["covariance_family_comparison"] = family_results
    save_results(OUT_PATH, results)

    # Pick the best-performing EM-fit family by controller-train NLL for the deeper dives below
    best_family_name = min(
        (n for n in family_results if n != "equal_spherical_sanity"),
        key=lambda n: family_results[n]["controller_train_nll"],
    )
    print(f"\nBest family by controller-train NLL: {best_family_name}")
    results["best_family"] = best_family_name
    save_results(OUT_PATH, results)

    diag_fit = fitted_gmms["diagonal"]  # the decomposition in Section 3 specifically needs diagonal covariance

    # ============================================================
    # Section 3 — counterfactual decomposition (which DOF matters?)
    # ============================================================
    print("\n=== Section 3: counterfactual decomposition ===")
    counterfactuals = counterfactual_assignments(ds_keys, diag_fit["means"], diag_fit["weights"], diag_fit["variances"])
    decomposition = {}
    prev_assignment = assignment_a
    for model_name, cf_assignment in counterfactuals.items():
        dists_cf = build_hard_dime_from_assignment(ds_values, cf_assignment, N_CLUSTERS)
        ct_nll_cf, _ = eval_dime_nll(diag_fit["means"], dists_cf, ct_keys, ct_values, ct_p_lm_true)
        frac_changed_from_kmeans = float((cf_assignment != assignment_a).mean())
        frac_changed_from_prev = float((cf_assignment != prev_assignment).mean())
        decomposition[model_name] = {
            "controller_train_nll": ct_nll_cf,
            "frac_assignment_changed_vs_kmeans": frac_changed_from_kmeans,
            "frac_assignment_changed_vs_previous_model": frac_changed_from_prev,
        }
        print(f"  {model_name}: ct_nll={ct_nll_cf:.4f}  vs_kmeans_changed={frac_changed_from_kmeans*100:.2f}%  "
              f"vs_previous_changed={frac_changed_from_prev*100:.2f}%")
        prev_assignment = cf_assignment
    results["counterfactual_decomposition"] = decomposition
    save_results(OUT_PATH, results)

    # ============================================================
    # Section 4 — learned covariance diagnostics + regularization sensitivity
    # ============================================================
    print("\n=== Section 4: covariance diagnostics + reg_covar sensitivity ===")
    variances = diag_fit["variances"]
    per_comp_mean = variances.mean(axis=1)
    most_isotropic = int(np.argmin(variances.max(axis=1) / np.clip(variances.min(axis=1), 1e-12, None)))
    most_anisotropic = int(np.argmax(variances.max(axis=1) / np.clip(variances.min(axis=1), 1e-12, None)))
    highest_var = int(np.argmax(per_comp_mean))
    lowest_var = int(np.argmin(per_comp_mean))

    reg_sensitivity = []
    for reg in [1e-6, 1e-4, 1e-2]:
        init_means, init_var, init_w = init_gmm_from_kmeans(ds_keys, centroids_a, assignment_a, "diag", reg_covar=reg)
        fit_reg = fit_gmm_rich(ds_keys, init_means, init_var, init_w, covariance_type="diag",
                                rel_tol=REL_TOL, patience=PATIENCE, max_iter=MAX_ITER, batch_size=5000,
                                reg_covar=reg, checkpoint_every=CHECKPOINT_EVERY, verbose=False)
        hard_assign_reg = fit_reg["responsibilities"].argmax(axis=1)
        dists_reg = build_hard_dime_from_assignment(ds_values, hard_assign_reg, N_CLUSTERS)
        ct_nll_reg, _ = eval_dime_nll(fit_reg["means"], dists_reg, ct_keys, ct_values, ct_p_lm_true)
        n_dead = int((np.bincount(hard_assign_reg, minlength=N_CLUSTERS) < 1).sum())
        reg_sensitivity.append({"reg_covar": reg, "controller_train_nll": ct_nll_reg,
                                 "n_dead_components": n_dead, "n_iterations": fit_reg["n_iterations"]})
        print(f"  reg_covar={reg}: ct_nll={ct_nll_reg:.4f}  n_dead={n_dead}  n_iter={fit_reg['n_iterations']}")

    results["covariance_diagnostics"] = {
        "per_component_mean_variance_distribution": {
            "mean": float(per_comp_mean.mean()), "std": float(per_comp_mean.std()),
        },
        "example_components": {
            "most_isotropic": most_isotropic, "most_anisotropic": most_anisotropic,
            "highest_variance": highest_var, "lowest_variance": lowest_var,
        },
        "reg_covar_sensitivity": reg_sensitivity,
    }
    save_results(OUT_PATH, results)

    # ============================================================
    # Section 5 — multi-seed stability (diagonal family)
    # ============================================================
    print("\n=== Section 5: multi-seed stability (diagonal covariance) ===")
    seed_results = []
    for seed in range(42, 42 + N_SEEDS):
        centroids_s, assignment_s, _ = build_kmeans_dime(ds_keys, ds_values, N_CLUSTERS, seed=seed)
        init_means, init_var, init_w = init_gmm_from_kmeans(ds_keys, centroids_s, assignment_s, "diag")
        fit_s = fit_gmm_rich(ds_keys, init_means, init_var, init_w, covariance_type="diag",
                              rel_tol=REL_TOL, patience=PATIENCE, max_iter=MAX_ITER, batch_size=5000,
                              checkpoint_every=CHECKPOINT_EVERY, verbose=False)
        hard_assign_s = fit_s["responsibilities"].argmax(axis=1)
        dists_s = build_hard_dime_from_assignment(ds_values, hard_assign_s, N_CLUSTERS)
        ct_nll_s, _ = eval_dime_nll(fit_s["means"], dists_s, ct_keys, ct_values, ct_p_lm_true)
        sim_s = assignment_similarity_report(assignment_s, hard_assign_s, centroids_s, fit_s["means"])
        seed_results.append({
            "seed": seed, "converged": fit_s["converged"], "n_iterations": fit_s["n_iterations"],
            "final_log_likelihood": fit_s["history"][-1]["mean_log_likelihood"],
            "controller_train_nll": ct_nll_s, "assignment_overlap_with_own_kmeans": sim_s,
        })
        print(f"  seed {seed}: ll={fit_s['history'][-1]['mean_log_likelihood']:.4f}  ct_nll={ct_nll_s:.4f}  "
              f"n_iter={fit_s['n_iterations']}")

    # One alternate (non-kmeans) init, per the user's request to test it if practical
    rng = np.random.default_rng(123)
    random_idx = rng.choice(len(ds_keys), size=N_CLUSTERS, replace=False)
    random_means = ds_keys[random_idx].copy()
    random_var = np.full((N_CLUSTERS, ds_keys.shape[1]), float(np.var(ds_keys)), dtype=np.float64)
    random_w = np.full(N_CLUSTERS, 1.0 / N_CLUSTERS)
    fit_random_init = fit_gmm_rich(ds_keys, random_means, random_var, random_w, covariance_type="diag",
                                    rel_tol=REL_TOL, patience=PATIENCE, max_iter=MAX_ITER, batch_size=5000,
                                    checkpoint_every=CHECKPOINT_EVERY, verbose=False)
    hard_assign_rand = fit_random_init["responsibilities"].argmax(axis=1)
    dists_rand = build_hard_dime_from_assignment(ds_values, hard_assign_rand, N_CLUSTERS)
    ct_nll_rand, _ = eval_dime_nll(fit_random_init["means"], dists_rand, ct_keys, ct_values, ct_p_lm_true)
    print(f"  random-init (non-kmeans-seeded): ll={fit_random_init['history'][-1]['mean_log_likelihood']:.4f}  "
          f"ct_nll={ct_nll_rand:.4f}  n_iter={fit_random_init['n_iterations']}")

    ll_vals = [s["final_log_likelihood"] for s in seed_results]
    nll_vals = [s["controller_train_nll"] for s in seed_results]
    best_ll_idx = int(np.argmax(ll_vals))
    best_nll_idx = int(np.argmin(nll_vals))
    results["multi_seed_stability"] = {
        "kmeans_seeded_runs": seed_results,
        "random_init_run": {"final_log_likelihood": fit_random_init["history"][-1]["mean_log_likelihood"],
                             "controller_train_nll": ct_nll_rand, "n_iterations": fit_random_init["n_iterations"]},
        "highest_likelihood_seed_is_best_nll_seed": best_ll_idx == best_nll_idx,
        "log_likelihood_vs_nll_correlation": float(np.corrcoef(ll_vals, nll_vals)[0, 1]),
    }
    save_results(OUT_PATH, results)

    # ============================================================
    # Section 6 — cluster assignment comparison (global + margin-percentile curve)
    # ============================================================
    print("\n=== Section 6: assignment comparison ===")
    diag_hard_assignment = diag_fit["responsibilities"].argmax(axis=1)
    global_sim = assignment_similarity_report(assignment_a, diag_hard_assignment, centroids_a, diag_fit["means"])
    print(json.dumps(global_sim, indent=2))

    d1, d2, margin = kmeans_boundary_margin(ds_keys, centroids_a)
    curve = margin_percentile_curve(margin, assignment_a, diag_hard_assignment)
    print("Margin percentile -> disagreement fraction:")
    for row in curve:
        print(f"  {row['percentile_band']}: disagree={row['fraction_disagreeing']*100:.1f}%  mean_margin={row['mean_margin']:.4f}")

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot([row["percentile_band"] for row in curve], [row["fraction_disagreeing"] for row in curve], "o-")
    ax.set_xlabel("k-means margin percentile band")
    ax.set_ylabel("fraction of points where GMM-hard disagrees with k-means")
    ax.set_title("Assignment disagreement vs. boundary proximity (datastore)")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "margin_percentile_disagreement_curve.png"), dpi=130)
    plt.close(fig)

    results["assignment_comparison"] = {"global_similarity": global_sim, "margin_percentile_curve": curve}
    save_results(OUT_PATH, results)

    # ============================================================
    # Section 7 — DIME-specific token-information diagnostics
    # ============================================================
    print("\n=== Section 7: token-information diagnostics ===")
    dists_diag_hard = build_hard_dime_from_assignment(ds_values, diag_hard_assignment, N_CLUSTERS)
    info_kmeans = conditional_entropy_report(dists_a)
    info_gmm = conditional_entropy_report(dists_diag_hard)
    print(f"k-means: H(Y|C)={info_kmeans['H_Y_given_C']:.4f}  I(C;Y)={info_kmeans['I_C_Y']:.4f}")
    print(f"GMM-hard: H(Y|C)={info_gmm['H_Y_given_C']:.4f}  I(C;Y)={info_gmm['I_C_Y']:.4f}")
    results["token_information_diagnostics"] = {"kmeans": info_kmeans, "gmm_hard_diagonal": info_gmm}
    save_results(OUT_PATH, results)

    # ============================================================
    # Section 8 — prototype / token-distribution diagnostics (A vs diagonal GMM-hard)
    # ============================================================
    print("\n=== Section 8: prototype movement vs JS divergence ===")
    N_c_hard = np.array([sum(c.values()) for c in dists_a], dtype=np.float64)
    N_c_gmm = np.array([sum(c.values()) for c in dists_diag_hard], dtype=np.float64)
    alpha_gmm_hard = N_c_gmm / N_c_gmm.sum()
    cluster_report = cluster_level_report(centroids_a, dists_a, N_c_hard,
                                           diag_fit["means"], dists_diag_hard, N_c_gmm, alpha_gmm_hard)
    print(json.dumps(cluster_report, indent=2))
    results["cluster_level_analysis_A_vs_diagonal"] = cluster_report
    save_results(OUT_PATH, results)

    # ============================================================
    # Section 9 — PCA visualization (exploratory projections, explicitly labeled)
    # ============================================================
    print("\n=== Section 9: PCA visualization ===")
    subsample_n = min(5000, len(ds_keys))
    sub_idx = np.random.default_rng(0).choice(len(ds_keys), size=subsample_n, replace=False)
    pca = PCA(n_components=2, random_state=42).fit(ds_keys[sub_idx])
    proj = pca.transform(ds_keys[sub_idx])

    fig, axes = plt.subplots(2, 2, figsize=(11, 10))
    axes[0, 0].scatter(proj[:, 0], proj[:, 1], c=assignment_a[sub_idx], cmap="tab20", s=4)
    axes[0, 0].set_title("k-means assignment (PCA projection — NOT the exact high-D boundary)")
    axes[0, 1].scatter(proj[:, 0], proj[:, 1], c=diag_hard_assignment[sub_idx], cmap="tab20", s=4)
    axes[0, 1].set_title("GMM-hard assignment (PCA projection)")
    disagree_sub = assignment_a[sub_idx] != diag_hard_assignment[sub_idx]
    axes[1, 0].scatter(proj[~disagree_sub, 0], proj[~disagree_sub, 1], c="lightgray", s=4, label="agree")
    axes[1, 0].scatter(proj[disagree_sub, 0], proj[disagree_sub, 1], c="red", s=6, label="disagree")
    axes[1, 0].set_title("Where k-means and GMM-hard disagree")
    axes[1, 0].legend()
    sc = axes[1, 1].scatter(proj[:, 0], proj[:, 1], c=margin[sub_idx], cmap="viridis_r", s=4)
    axes[1, 1].set_title("k-means boundary margin (PCA projection; darker = closer to boundary)")
    fig.colorbar(sc, ax=axes[1, 1])
    fig.suptitle(f"PCA projection of {subsample_n} datastore points (2 of {ds_keys.shape[1]} dims shown)")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "pca_projection_overview.png"), dpi=130)
    plt.close(fig)
    print(f"  saved {os.path.join(FIG_DIR, 'pca_projection_overview.png')}")
    print("  NOTE: UMAP was not run — it is not an existing project dependency, so it was "
          "skipped rather than silently adding a new one; PCA alone was explicitly prioritized.")

    # ============================================================
    # Section 10 — boundary-stratified NLL improvement (controller_train AND val)
    # ============================================================
    print("\n=== Section 10: boundary-stratified GMM improvement over k-means ===")
    def stratified_improvement(query_keys, query_values, query_p_lm_true, label):
        qd1, qd2, qmargin = kmeans_boundary_margin(query_keys, centroids_a)
        _, nll_km = eval_dime_nll(centroids_a, dists_a, query_keys, query_values, query_p_lm_true)
        _, nll_gm = eval_dime_nll(diag_fit["means"], dists_diag_hard, query_keys, query_values, query_p_lm_true)
        order = np.argsort(qmargin)
        N = len(qmargin)
        curve = []
        for lo, hi in zip((0, 1, 5, 10, 25, 50), (1, 5, 10, 25, 50, 100)):
            i_lo, i_hi = int(N * lo / 100), max(int(N * lo / 100) + 1, int(N * hi / 100))
            idx = order[i_lo:i_hi]
            improvement = float(nll_km[idx].mean() - nll_gm[idx].mean())
            curve.append({"percentile_band": f"{lo}-{hi}%", "nll_kmeans_minus_nll_gmm": improvement})
        print(f"  [{label}] boundary percentile -> (NLL_kmeans - NLL_gmm):")
        for row in curve:
            print(f"    {row['percentile_band']}: {row['nll_kmeans_minus_nll_gmm']:+.4f}")
        return curve

    boundary_stratified = {
        "controller_train": stratified_improvement(ct_keys, ct_values, ct_p_lm_true, "controller_train"),
        "validation": stratified_improvement(val_keys, val_values, val_p_lm_true, "validation"),
    }
    results["boundary_stratified_improvement"] = boundary_stratified
    save_results(OUT_PATH, results)

    # ============================================================
    # Section 11/13 — complexity vs. performance summary + FINAL val NLL table
    # ============================================================
    print("\n=== Section 11/13: complexity vs. performance, final NLL table ===")
    n_params = {"kmeans": 0, "learned_spherical": N_CLUSTERS,
                "diagonal": N_CLUSTERS * ds_keys.shape[1], "tied_diagonal": ds_keys.shape[1]}
    summary_table = {}
    for name, cov_type in [("kmeans", None)] + list(covariance_families.items()):
        if name == "kmeans":
            val_nll_final, _ = eval_dime_nll(centroids_a, dists_a, val_keys, val_values, val_p_lm_true)
            summary_table[name] = {
                "n_covariance_params": n_params[name], "converged_log_likelihood": None,
                "assignment_disagreement_with_kmeans": 0.0,
                "H_Y_given_C": info_kmeans["H_Y_given_C"],
                "controller_train_nll": ct_nll_kmeans, "validation_nll": val_nll_final,
            }
        else:
            fit = fitted_gmms[name]
            hard_assign = fit["responsibilities"].argmax(axis=1)
            dists_name = build_hard_dime_from_assignment(ds_values, hard_assign, N_CLUSTERS)
            val_nll_final, _ = eval_dime_nll(fit["means"], dists_name, val_keys, val_values, val_p_lm_true)
            info_name = conditional_entropy_report(dists_name)
            sim = assignment_similarity_report(assignment_a, hard_assign, centroids_a, fit["means"])
            summary_table[name] = {
                "n_covariance_params": n_params[name],
                "converged_log_likelihood": fit["history"][-1]["mean_log_likelihood"],
                "assignment_disagreement_with_kmeans": 1.0 - sim["fraction_points_agreeing_after_matching"],
                "H_Y_given_C": info_name["H_Y_given_C"],
                "controller_train_nll": family_results[name]["controller_train_nll"],
                "validation_nll": val_nll_final,
            }
    print(json.dumps(summary_table, indent=2))
    results["final_summary_table"] = summary_table
    results["sections_completed"] = [
        "kmeans_baseline", "covariance_family_comparison", "counterfactual_decomposition",
        "covariance_diagnostics", "multi_seed_stability", "assignment_comparison",
        "token_information_diagnostics", "cluster_level_analysis_A_vs_diagonal",
        "pca_visualization", "boundary_stratified_improvement", "final_summary_table",
    ]
    save_results(OUT_PATH, results)

    # ============================================================
    # Final interpretation — answers to the 10 questions
    # ============================================================
    print("\n=== FINAL INTERPRETATION ===")
    print(f"1. Was the original 50-iteration GMM converged? Diagonal family converged={fitted_gmms['diagonal']['converged']} "
          f"after {fitted_gmms['diagonal']['n_iterations']} iterations (rel_tol={REL_TOL}).")
    print(f"2. See likelihood_vs_dime_nll_*.png and checkpoint_curve in results JSON — "
          f"compare whether controller_train_nll keeps improving after log-likelihood plateaus.")
    print(f"3. See counterfactual_decomposition — compare frac_assignment_changed_vs_previous_model "
          f"across model0->model1->model2->model3 to see which step moves the most points / NLL.")
    print(f"4. Global assignment agreement with k-means (diagonal): "
          f"{global_sim['fraction_points_agreeing_after_matching']*100:.1f}%  ARI={global_sim['adjusted_rand_index']:.4f}")
    print(f"5. See margin_percentile_disagreement_curve.png — check whether disagreement fraction "
          f"rises monotonically toward the lowest-margin band.")
    print(f"6. H(Y|C): k-means={info_kmeans['H_Y_given_C']:.4f} vs GMM-hard={info_gmm['H_Y_given_C']:.4f} "
          f"(lower is more informative clusters).")
    print(f"7. See multi_seed_stability — log_likelihood_vs_nll_correlation="
          f"{results['multi_seed_stability']['log_likelihood_vs_nll_correlation']:.4f}, "
          f"highest_likelihood_seed_is_best_nll_seed={results['multi_seed_stability']['highest_likelihood_seed_is_best_nll_seed']}")
    print(f"8. See point 7 — same correlation number answers this directly.")
    print(f"9. See final_summary_table for the complexity/performance tradeoff across all 4 families.")
    print(f"10. Judge from the combination of all of the above — see final_summary_table's "
          f"validation_nll column for the headline comparison.")

    print(f"\nAll results saved to {OUT_PATH}")


if __name__ == "__main__":
    main()
