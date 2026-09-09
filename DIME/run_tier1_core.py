import sys, os, argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "GPT_Module"))

from cache_io import load_cache
from knn import build_datastore, query_knn, mix_knn_and_lm
from results_io import save_results
from grid_search import grid_search_hyperparams
from state_object import minibatch_kmeans_partition
from mixing import mix_dime_and_lm
import numpy as np
from scipy import stats
from sklearn.cluster import MiniBatchKMeans

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", required=True)
parser.add_argument("--model", required=True)
parser.add_argument("--n_clusters", type=int, default=None, help="defaults to ~1/100 of the datastore size")
args = parser.parse_args()

CACHE_DIR = os.path.join(os.path.dirname(__file__), "..", "GPT_Module", "cache")
PREFIX = f"{args.dataset}_{args.model}"

ds = load_cache(os.path.join(CACHE_DIR, f"{PREFIX}_datastore.npz"))
ct = load_cache(os.path.join(CACHE_DIR, f"{PREFIX}_controller_train.npz"))
val = load_cache(os.path.join(CACHE_DIR, f"{PREFIX}_val.npz"))

ds_keys, ds_values = ds["keys"], ds["values"]
ct_keys, ct_values, ct_p_lm_true = ct["keys"], ct["values"], ct["p_lm_true"]
val_keys, val_values, val_p_lm_true = val["keys"], val["values"], val["p_lm_true"]

print(f"[{PREFIX}] datastore: {len(ds_values)}  controller_train: {len(ct_values)}  val: {len(val_values)}")

N_CLUSTERS = args.n_clusters or max(100, round(len(ds_values) / 100))
print(f"[{PREFIX}] n_clusters = {N_CLUSTERS} (ratio {len(ds_values)/N_CLUSTERS:.1f}x)")

# Wide grid covering the ranges that have worked across every setting so far (TinyStories preferred
# smaller k/alpha, WikiText-103 preferred larger k/alpha) — self-tuning per setting, no hardcoded configs.
k_values = [k for k in [10, 20, 50, 100, 200, 300] if k <= N_CLUSTERS] or [min(N_CLUSTERS, 10)]
tau_values = [0.5, 1.0, 2.0, 5.0, 10.0]
alpha_values = [0.01, 0.05, 0.1, 0.25, 0.5]
k_max = max(k_values)
k_max_raw = max([k for k in [10, 20, 50, 100, 200, 300] if k <= len(ds_values)])

results = {"phase": "tier1_core", "dataset": args.dataset, "model": args.model, "n_clusters": N_CLUSTERS}
out_path = f"results/{PREFIX}_tier1.json"

# --- GPT-only ---
nll_gpt_only = -np.log(val_p_lm_true + 1e-12)
results["gpt_only_mean_nll"] = float(nll_gpt_only.mean())
print(f"[{PREFIX}] GPT-only: {results['gpt_only_mean_nll']:.4f}")
save_results(out_path, results)

# --- raw kNN: grid search on controller_train ---
raw_index, raw_stored_values = build_datastore(ds_keys, ds_values)
raw_distances_ct, raw_retrieved_ct = query_knn(raw_index, raw_stored_values, ct_keys, k=k_max_raw)
raw_grid_results, raw_best = grid_search_hyperparams(
    raw_distances_ct, raw_retrieved_ct, ct_values, ct_p_lm_true,
    mix_knn_and_lm, [k for k in k_values if k <= len(ds_values)], tau_values, alpha_values
)
raw_distances_val, raw_retrieved_val = query_knn(raw_index, raw_stored_values, val_keys, k=raw_best["k"])
_, nll_raw_val = mix_knn_and_lm(raw_distances_val, raw_retrieved_val, val_values, val_p_lm_true, tau=raw_best["tau"], alpha=raw_best["alpha"])
results["raw_knn"] = {"best_config": raw_best, "val_mean_nll": float(nll_raw_val.mean())}
print(f"[{PREFIX}] raw kNN tuned: {raw_best} -> {float(nll_raw_val.mean()):.4f}")
save_results(out_path, results)

# --- DIME (minibatch_kmeans): grid search on controller_train ---
compressed_keys, compressed_dists = minibatch_kmeans_partition(ds_keys, ds_values, n_clusters=N_CLUSTERS, seed=42)
dime_index, dime_stored_values = build_datastore(compressed_keys, np.array(compressed_dists, dtype=object))
dime_distances_ct, dime_retrieved_ct = query_knn(dime_index, dime_stored_values, ct_keys, k=k_max)
dime_grid_results, dime_best = grid_search_hyperparams(
    dime_distances_ct, dime_retrieved_ct, ct_values, ct_p_lm_true,
    mix_dime_and_lm, k_values, tau_values, alpha_values
)
dime_distances_val, dime_retrieved_val = query_knn(dime_index, dime_stored_values, val_keys, k=dime_best["k"])
_, nll_dime_val = mix_dime_and_lm(dime_distances_val, dime_retrieved_val, val_values, val_p_lm_true, tau=dime_best["tau"], alpha=dime_best["alpha"])
results["dime_minibatch_kmeans"] = {"best_config": dime_best, "val_mean_nll": float(nll_dime_val.mean())}
print(f"[{PREFIX}] DIME tuned: {dime_best} -> {float(nll_dime_val.mean()):.4f}")
save_results(out_path, results)

# --- equal-budget raw: raw_kmeans_representative (strongest variant from earlier work) ---
km = MiniBatchKMeans(n_clusters=N_CLUSTERS, random_state=42)
assignment = km.fit_predict(ds_keys)
centers = km.cluster_centers_
selected_idx = []
for c in range(N_CLUSTERS):
    member_idx = np.where(assignment == c)[0]
    if len(member_idx) == 0:
        continue
    dists = np.linalg.norm(ds_keys[member_idx] - centers[c], axis=1)
    selected_idx.append(member_idx[np.argmin(dists)])
selected_idx = np.array(selected_idx)
raw_rep_index, raw_rep_values = build_datastore(ds_keys[selected_idx], ds_values[selected_idx])
distances_rep, retrieved_rep = query_knn(raw_rep_index, raw_rep_values, val_keys, k=min(raw_best["k"], len(selected_idx)))
_, nll_rep = mix_knn_and_lm(distances_rep, retrieved_rep, val_values, val_p_lm_true, tau=raw_best["tau"], alpha=raw_best["alpha"])
results["raw_kmeans_representative"] = {"n_selected": int(len(selected_idx)), "val_mean_nll": float(nll_rep.mean())}
print(f"[{PREFIX}] raw_kmeans_representative: n={len(selected_idx)} -> {float(nll_rep.mean()):.4f}")
save_results(out_path, results)

# --- significance testing (paired, per-position) ---
def paired_tests(nll_a, nll_b, name_a, name_b):
    diff = nll_a - nll_b
    t_stat, t_p = stats.ttest_rel(nll_a, nll_b)
    w_stat, w_p = stats.wilcoxon(nll_a, nll_b)
    return {"comparison": f"{name_a} vs {name_b}", "mean_diff": float(diff.mean()),
            "paired_t_test": {"t_stat": float(t_stat), "p_value": float(t_p)},
            "wilcoxon_signed_rank": {"stat": float(w_stat), "p_value": float(w_p)}}

test_gpt_vs_dime = paired_tests(nll_gpt_only, nll_dime_val, "GPT-only", "DIME (tuned)")
test_rawrep_vs_dime = paired_tests(nll_rep, nll_dime_val, "raw_kmeans_representative", "DIME (tuned)")
results["significance_tests"] = [test_gpt_vs_dime, test_rawrep_vs_dime]
save_results(out_path, results)

print(f"\n[{PREFIX}] === TIER 1 SUMMARY ===")
print(f"GPT-only: {results['gpt_only_mean_nll']:.4f}")
print(f"raw kNN tuned: {results['raw_knn']['val_mean_nll']:.4f}")
print(f"DIME tuned: {results['dime_minibatch_kmeans']['val_mean_nll']:.4f}")
print(f"raw_kmeans_representative: {results['raw_kmeans_representative']['val_mean_nll']:.4f}")
print("DIME vs GPT-only:", test_gpt_vs_dime)
print("DIME vs raw_kmeans_representative:", test_rawrep_vs_dime)
