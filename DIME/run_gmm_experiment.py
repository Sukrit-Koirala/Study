"""Controlled experiment: does replacing DIME's hard k-means cluster boundary with
GMM/EM boundaries (Variant B) and soft responsibilities (Variant C) change DIME's
behavior, relative to the original hard k-means DIME (Variant A)? Spectral
clustering is deliberately out of scope here — isolating the GMM/EM effect first.

Uses the smallest DIME setting (TinyStories/gpt2) for a first real run, same
seed=42/seq_len=128/{350,75,75} chunk split as every other TinyStories/gpt2 result
in this project, so the numbers below are directly comparable to the established
baseline (GPT-only 2.798, raw kNN tuned 2.694, DIME tuned 2.745).

Retrieval hyperparameters are held FIXED and identical across all three variants
(k=20, tau=2.0, alpha=0.05 — DIME's own already-tuned config from the original grid
search) specifically so this run isolates the construction/clustering change, not a
retuned comparison. Per-method retuning is explicitly deferred to a later pass.
"""
import sys, os, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "GPT_Module"))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from gpt_class import FrozenGPT2
from dataset_helpers import collect_chunks_split
from extract import build_datastore_from_chunks, run_batch
from knn import build_datastore, query_knn
from mixing import mix_dime_and_lm
from results_io import save_results

from gmm_dime_construction import (
    build_kmeans_dime, build_gmm_hard_dime, build_gmm_soft_dime,
    kmeans_boundary_margin, gmm_responsibility_gap, boundary_point_report,
    cluster_level_report,
)

FIG_DIR = os.path.join(os.path.dirname(__file__), "figures")
os.makedirs(FIG_DIR, exist_ok=True)

N_CLUSTERS = 500
K, TAU, ALPHA = 20, 2.0, 0.05  # DIME's established tuned config at this scale — fixed across all 3 variants
COVARIANCE_TYPE = "diag"


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


def evaluate_variant(compressed_keys, compressed_dists, val_keys, val_values, val_p_lm_true):
    index, stored_values = build_datastore(compressed_keys, compressed_dists)
    k_eff = min(K, len(compressed_keys))
    distances, retrieved = query_knn(index, stored_values, val_keys, k=k_eff)
    _, nll = mix_dime_and_lm(distances, retrieved, val_values, val_p_lm_true, tau=TAU, alpha=ALPHA)
    return float(nll.mean()), nll


def main():
    gpt = FrozenGPT2()
    dataset = __import__("datasets").load_dataset(
        "roneneldan/TinyStories", split="train", streaming=True, trust_remote_code=True
    )
    chunks, _ = collect_chunks_split(
        dataset, gpt.tokenizer, seq_len=128,
        n_chunks_needed={"datastore": 350, "controller_train": 75, "val": 75},
        seed=42,
    )

    print("Encoding datastore / val...")
    ds_keys, ds_values = build_datastore_from_chunks(gpt, chunks["datastore"], batch_size=256)
    val_keys, val_values, val_p_lm_true = encode_split(gpt, chunks["val"])
    print(f"datastore: {len(ds_values)}  val: {len(val_values)}")

    results = {"phase": "gmm_experiment", "dataset": "tinystories", "model": "gpt2",
               "n_clusters": N_CLUSTERS, "fixed_config": {"k": K, "tau": TAU, "alpha": ALPHA},
               "covariance_type": COVARIANCE_TYPE}
    out_path = "results/gmm_experiment.json"

    # ============================================================
    # Variant A — original DIME (hard k-means)
    # ============================================================
    print("\n--- Variant A: k-means hard DIME ---")
    centroids_a, assignment_a, dists_a = build_kmeans_dime(ds_keys, ds_values, N_CLUSTERS, seed=42)
    nll_a_mean, nll_a = evaluate_variant(centroids_a, dists_a, val_keys, val_values, val_p_lm_true)
    print(f"Variant A val mean NLL: {nll_a_mean:.4f}")

    # ============================================================
    # Variant B — GMM/EM, hard final assignment (argmax responsibility)
    # ============================================================
    print("\n--- Variant B: GMM-hard DIME ---")
    gmm_b = build_gmm_hard_dime(ds_keys, ds_values, centroids_a, assignment_a,
                                 covariance_type=COVARIANCE_TYPE, tol=1e-4, max_iter=50,
                                 batch_size=5000, seed=42)
    # Sanity test: responsibilities must sum to 1 across components, for every point.
    resp_sums = gmm_b["responsibilities"].sum(axis=1)
    assert np.allclose(resp_sums, 1.0, atol=1e-3), \
        f"responsibilities don't sum to 1 (min={resp_sums.min()}, max={resp_sums.max()})"
    print(f"Sanity check passed: responsibilities sum to 1 (min={resp_sums.min():.6f}, max={resp_sums.max():.6f})")

    nll_b_mean, nll_b = evaluate_variant(gmm_b["means"], gmm_b["compressed_dists"], val_keys, val_values, val_p_lm_true)
    print(f"Variant B val mean NLL: {nll_b_mean:.4f}")

    # ============================================================
    # Variant C — GMM/EM, soft DIME construction (full responsibilities)
    # ============================================================
    print("\n--- Variant C: GMM-soft DIME ---")
    dists_c, N_c_eff, alpha_c = build_gmm_soft_dime(ds_values, gmm_b["means"], gmm_b["responsibilities"])
    nll_c_mean, nll_c = evaluate_variant(gmm_b["means"], dists_c, val_keys, val_values, val_p_lm_true)
    print(f"Variant C val mean NLL: {nll_c_mean:.4f}")

    results["nll"] = {"kmeans_hard_A": nll_a_mean, "gmm_hard_B": nll_b_mean, "gmm_soft_C": nll_c_mean}
    save_results(out_path, results)

    # ============================================================
    # Boundary analysis (on datastore points)
    # ============================================================
    print("\n--- Boundary analysis (datastore points) ---")
    d1, d2, margin = kmeans_boundary_margin(ds_keys, centroids_a)
    top1, top2, gap, entropy = gmm_responsibility_gap(gmm_b["responsibilities"])

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    axes[0].scatter(margin, gap, s=2, alpha=0.3)
    axes[0].set_xlabel("k-means boundary margin (d2 - d1)")
    axes[0].set_ylabel("GMM responsibility gap (gamma_1 - gamma_2)")
    axes[0].set_title("Margin vs. responsibility gap")
    axes[1].scatter(margin, entropy, s=2, alpha=0.3, color="darkorange")
    axes[1].set_xlabel("k-means boundary margin (d2 - d1)")
    axes[1].set_ylabel("GMM responsibility entropy")
    axes[1].set_title("Margin vs. responsibility entropy")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "boundary_margin_vs_gmm.png"), dpi=130)
    plt.close(fig)
    print(f"  saved {os.path.join(FIG_DIR, 'boundary_margin_vs_gmm.png')}")

    boundary_report = boundary_point_report(margin, gap, entropy, assignment_a, gmm_b["hard_assignment"],
                                             gmm_b["responsibilities"], percentiles=(1, 5, 10))
    results["boundary_analysis"] = boundary_report
    print(json.dumps(boundary_report, indent=2))
    save_results(out_path, results)

    # ============================================================
    # Cluster-level analysis
    # ============================================================
    print("\n--- Cluster-level analysis (k-means hard [A] vs. GMM-soft [C]) ---")
    N_c_hard = np.array([sum(c.values()) for c in dists_a], dtype=np.float64)
    cluster_report = cluster_level_report(
        centroids_a, dists_a, N_c_hard,
        gmm_b["means"], dists_c, N_c_eff, alpha_c,
    )
    results["cluster_level_analysis"] = cluster_report
    print(json.dumps(cluster_report, indent=2))
    save_results(out_path, results)

    # ============================================================
    # Boundary-stratified val evaluation — is any gain concentrated near boundaries?
    # ============================================================
    print("\n--- Boundary-stratified val evaluation ---")
    val_d1, val_d2, val_margin = kmeans_boundary_margin(val_keys, centroids_a)
    median_margin = np.median(val_margin)
    near_boundary = val_margin <= median_margin
    interior = ~near_boundary

    stratified = {}
    for name, nll_arr in [("kmeans_hard_A", nll_a), ("gmm_hard_B", nll_b), ("gmm_soft_C", nll_c)]:
        stratified[name] = {
            "near_boundary_mean_nll": float(nll_arr[near_boundary].mean()),
            "interior_mean_nll": float(nll_arr[interior].mean()),
            "n_near_boundary": int(near_boundary.sum()),
            "n_interior": int(interior.sum()),
        }
    results["boundary_stratified_val_nll"] = stratified
    print(json.dumps(stratified, indent=2))
    save_results(out_path, results)

    # ============================================================
    # Interpretation summary
    # ============================================================
    print("\n=== INTERPRETATION ===")
    print(f"A (k-means hard):  {nll_a_mean:.4f}")
    print(f"B (GMM hard):      {nll_b_mean:.4f}   (A vs B isolates: does the boundary alone matter?)")
    print(f"C (GMM soft):      {nll_c_mean:.4f}   (B vs C isolates: does soft membership add anything?)")
    print(f"Mean prototype movement B->C: {cluster_report['prototype_movement']['mean']:.6f}")
    print(f"Mean token-distribution JS divergence B->C: {cluster_report['token_distribution_js_divergence']['mean']:.6f}")
    for pct_key, pct_report in boundary_report.items():
        print(f"{pct_key}: GMM-hard disagrees with k-means on {pct_report['frac_gmm_hard_disagrees_with_kmeans']*100:.1f}% of points")
    print("Boundary-stratified gains (near-boundary vs interior val queries) — see "
          "results['boundary_stratified_val_nll'] for whether any A->B->C improvement "
          "is concentrated specifically near k-means' boundaries.")

    results["sections_completed"] = ["nll", "boundary_analysis", "cluster_level_analysis", "boundary_stratified_val_nll"]
    save_results(out_path, results)
    print(f"\nAll results saved to {out_path}")


if __name__ == "__main__":
    main()
