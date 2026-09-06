import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "GPT_Module"))

from cache_io import load_cache
from knn import build_datastore, query_knn, mix_knn_and_lm
from results_io import save_results
import numpy as np

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

k_values = [50, 100, 200, 300]
tau_values = [0.5, 1.0, 2.0, 5.0]
alpha_values = [0.05, 0.1, 0.25, 0.5]
k_max = max(k_values)

# --- Raw kNN's own tuned ceiling: computed once, the reference line the whole sweep is checked against ---
raw_index, raw_stored_values = build_datastore(ds_keys, ds_values)
raw_distances_ct, raw_retrieved_ct = query_knn(raw_index, raw_stored_values, ct_keys, k=k_max)
raw_grid_results, raw_best = grid_search_hyperparams(
    raw_distances_ct, raw_retrieved_ct, ct_values, ct_p_lm_true,
    mix_knn_and_lm, k_values, tau_values, alpha_values
)
raw_distances_val, raw_retrieved_val = query_knn(raw_index, raw_stored_values, val_keys, k=raw_best["k"])
_, nll_raw_val = mix_knn_and_lm(raw_distances_val, raw_retrieved_val, val_values, val_p_lm_true,
                                  tau=raw_best["tau"], alpha=raw_best["alpha"])
raw_ceiling_nll = float(nll_raw_val.mean())
print("raw kNN tuned ceiling:", raw_best, "-> val mean NLL:", raw_ceiling_nll)

# --- Sweep DIME across several compression ratios, aggressive to mild ---
n_clusters_sweep = [4000, 15000, 40000, 100000]
sweep_results = []

for n_clusters in n_clusters_sweep:
    ratio = len(ds_values) / n_clusters
    print(f"\n--- n_clusters={n_clusters} ({ratio:.1f}x compression) ---")

    compressed_keys, compressed_dists = minibatch_kmeans_partition(ds_keys, ds_values, n_clusters=n_clusters, seed=42)
    dime_index, dime_stored_values = build_datastore(compressed_keys, np.array(compressed_dists, dtype=object))

    this_k_values = [k for k in k_values if k <= n_clusters]
    dime_distances_ct, dime_retrieved_ct = query_knn(dime_index, dime_stored_values, ct_keys, k=max(this_k_values))
    dime_grid_results, dime_best = grid_search_hyperparams(
        dime_distances_ct, dime_retrieved_ct, ct_values, ct_p_lm_true,
        mix_dime_and_lm, this_k_values, tau_values, alpha_values
    )

    dime_distances_val, dime_retrieved_val = query_knn(dime_index, dime_stored_values, val_keys, k=dime_best["k"])
    _, nll_dime_val = mix_dime_and_lm(dime_distances_val, dime_retrieved_val, val_values, val_p_lm_true,
                                        tau=dime_best["tau"], alpha=dime_best["alpha"])
    dime_val_nll = float(nll_dime_val.mean())
    beats_raw = dime_val_nll < raw_ceiling_nll

    print(f"n_clusters={n_clusters}  best_config={dime_best}  val NLL={dime_val_nll:.4f}  beats raw ceiling? {beats_raw}")

    sweep_results.append({
        "n_clusters": n_clusters,
        "compression_ratio": ratio,
        "best_config": dime_best,
        "val_mean_nll": dime_val_nll,
        "beats_raw_ceiling": beats_raw,
    })

results = {
    "phase": "compression_sweep_wikitext103_gpt2-medium",
    "model": "gpt2-medium",
    "dataset": "wikitext103",
    "seq_len": 128,
    "n_datastore": len(ds_values),
    "grid": {"k_values": k_values, "tau_values": tau_values, "alpha_values": alpha_values},
    "raw_knn_tuned_ceiling": {"best_config": raw_best, "val_mean_nll": raw_ceiling_nll},
    "sweep": sweep_results,
}

save_results("results/compression_sweep.json", results)

print("\n=== SWEEP SUMMARY ===")
print(f"raw kNN ceiling: {raw_ceiling_nll:.4f}")
for r in sweep_results:
    print(f"n_clusters={r['n_clusters']:>7}  ratio={r['compression_ratio']:6.1f}x  "
          f"val NLL={r['val_mean_nll']:.4f}  beats raw? {r['beats_raw_ceiling']}")
