import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "GPT_Module"))

from cache_io import load_cache
from knn import build_datastore, query_knn, mix_knn_and_lm
from results_io import save_results
import numpy as np
from scipy import stats
from sklearn.cluster import MiniBatchKMeans

from state_object import minibatch_kmeans_partition
from mixing import mix_dime_and_lm
from grid_search import grid_search_hyperparams

CACHE_DIR = os.path.join(os.path.dirname(__file__), "..", "GPT_Module", "cache")
PREFIX = "wikitext103_gpt2-medium"

ds = load_cache(os.path.join(CACHE_DIR, f"{PREFIX}_datastore.npz"))
ct = load_cache(os.path.join(CACHE_DIR, f"{PREFIX}_controller_train.npz"))
val = load_cache(os.path.join(CACHE_DIR, f"{PREFIX}_val.npz"))

ds_keys, ds_values = ds["keys"], ds["values"]
ct_keys, ct_values, ct_p_lm_true = ct["keys"], ct["values"], ct["p_lm_true"]
val_keys, val_values, val_p_lm_true = val["keys"], val["values"], val["p_lm_true"]

print(f"datastore: {len(ds_values)}  controller_train: {len(ct_values)}  val: {len(val_values)}")

# --- 1. GPT-only baseline ---
nll_gpt_only_val = -np.log(val_p_lm_true + 1e-12)
print("mean NLL, GPT-only:", nll_gpt_only_val.mean())

# --- Build raw kNN datastore and DIME (minibatch_kmeans) datastore ---
raw_index, raw_stored_values = build_datastore(ds_keys, ds_values)

n_clusters = 4000  # ~95x compression, matching the ratio already validated on TinyStories
compressed_keys, compressed_dists = minibatch_kmeans_partition(ds_keys, ds_values, n_clusters=n_clusters, seed=42)
dime_index, dime_stored_values = build_datastore(compressed_keys, np.array(compressed_dists, dtype=object))

# --- 2. Grid search on controller_train (reusing the shape that worked for TinyStories, widened for the bigger datastore) ---
k_values = [10, 20, 50, 100]
tau_values = [0.5, 1.0, 2.0, 5.0]
alpha_values = [0.01, 0.05, 0.1, 0.25]
k_max = max(k_values)

raw_distances_ct, raw_retrieved_ct = query_knn(raw_index, raw_stored_values, ct_keys, k=k_max)
raw_grid_results, raw_best = grid_search_hyperparams(
    raw_distances_ct, raw_retrieved_ct, ct_values, ct_p_lm_true,
    mix_knn_and_lm, k_values, tau_values, alpha_values
)
print("raw kNN best on controller_train:", raw_best)

dime_distances_ct, dime_retrieved_ct = query_knn(dime_index, dime_stored_values, ct_keys, k=k_max)
dime_grid_results, dime_best = grid_search_hyperparams(
    dime_distances_ct, dime_retrieved_ct, ct_values, ct_p_lm_true,
    mix_dime_and_lm, k_values, tau_values, alpha_values
)
print("DIME (minibatch_kmeans) best on controller_train:", dime_best)

# --- 3. Evaluate winning configs on val — touched exactly once, here ---
raw_distances_val, raw_retrieved_val = query_knn(raw_index, raw_stored_values, val_keys, k=raw_best["k"])
_, nll_raw_val = mix_knn_and_lm(raw_distances_val, raw_retrieved_val, val_values, val_p_lm_true,
                                  tau=raw_best["tau"], alpha=raw_best["alpha"])

dime_distances_val, dime_retrieved_val = query_knn(dime_index, dime_stored_values, val_keys, k=dime_best["k"])
_, nll_dime_val = mix_dime_and_lm(dime_distances_val, dime_retrieved_val, val_values, val_p_lm_true,
                                    tau=dime_best["tau"], alpha=dime_best["alpha"])

print("mean NLL, raw kNN (tuned):", nll_raw_val.mean())
print("mean NLL, DIME minibatch_kmeans (tuned):", nll_dime_val.mean())

# --- 4. Best equal-budget raw variant from Phase E: raw_kmeans_representative ---
km = MiniBatchKMeans(n_clusters=n_clusters, random_state=42)
assignment = km.fit_predict(ds_keys)
centers = km.cluster_centers_
selected_idx = []
for c in range(n_clusters):
    member_idx = np.where(assignment == c)[0]
    if len(member_idx) == 0:
        continue
    dists = np.linalg.norm(ds_keys[member_idx] - centers[c], axis=1)
    selected_idx.append(member_idx[np.argmin(dists)])
selected_idx = np.array(selected_idx)

raw_rep_index, raw_rep_values = build_datastore(ds_keys[selected_idx], ds_values[selected_idx])
distances_rep, retrieved_rep = query_knn(raw_rep_index, raw_rep_values, val_keys, k=min(raw_best["k"], len(selected_idx)))
_, nll_raw_rep_val = mix_knn_and_lm(distances_rep, retrieved_rep, val_values, val_p_lm_true,
                                     tau=raw_best["tau"], alpha=raw_best["alpha"])

print(f"raw_kmeans_representative: n={len(selected_idx)}  mean NLL = {nll_raw_rep_val.mean()}")

# --- 5. Significance tests: GPT-only vs DIME, and best-raw-equal-budget vs DIME ---
def paired_tests(nll_a, nll_b, name_a, name_b):
    diff = nll_a - nll_b
    t_stat, t_p = stats.ttest_rel(nll_a, nll_b)
    w_stat, w_p = stats.wilcoxon(nll_a, nll_b)
    return {
        "comparison": f"{name_a} vs {name_b}",
        "mean_diff": float(diff.mean()),
        "paired_t_test": {"t_stat": float(t_stat), "p_value": float(t_p)},
        "wilcoxon_signed_rank": {"stat": float(w_stat), "p_value": float(w_p)},
    }


test_gpt_vs_dime = paired_tests(nll_gpt_only_val, nll_dime_val, "GPT-only", "minibatch_kmeans (tuned)")
test_rawrep_vs_dime = paired_tests(nll_raw_rep_val, nll_dime_val, "raw_kmeans_representative (equal budget)", "minibatch_kmeans (tuned)")

results = {
    "phase": "replication_wikitext103_gpt2-medium",
    "model": "gpt2-medium",
    "dataset": "wikitext103",
    "seq_len": 128,
    "n_clusters": n_clusters,
    "n_datastore": len(ds_values),
    "n_controller_train": len(ct_values),
    "n_val": len(val_values),
    "grid": {"k_values": k_values, "tau_values": tau_values, "alpha_values": alpha_values},
    "raw_knn": {"best_config": raw_best, "val_mean_nll": float(nll_raw_val.mean())},
    "dime_minibatch_kmeans": {"best_config": dime_best, "val_mean_nll": float(nll_dime_val.mean())},
    "raw_kmeans_representative": {"n_selected": int(len(selected_idx)), "val_mean_nll": float(nll_raw_rep_val.mean())},
    "gpt_only_mean_nll": float(nll_gpt_only_val.mean()),
    "significance_tests": [test_gpt_vs_dime, test_rawrep_vs_dime],
}

save_results("results/wikitext103_gpt2_medium_replication.json", results)

print()
print("=== SUMMARY ===")
print("GPT-only:                 ", nll_gpt_only_val.mean())
print("raw kNN (tuned):          ", nll_raw_val.mean())
print("DIME minibatch_kmeans:    ", nll_dime_val.mean())
print("raw_kmeans_representative:", nll_raw_rep_val.mean())
print()
print("Test 1 (GPT-only vs DIME):", test_gpt_vs_dime)
print()
print("Test 2 (best raw equal-budget vs DIME):", test_rawrep_vs_dime)
