import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "GPT_Module"))

from gpt_class import FrozenGPT2
from dataset_helpers import collect_chunks_split
from extract import build_datastore_from_chunks, run_batch
from knn import build_datastore, query_knn, query_knn_indices, mix_knn_and_lm
from results_io import save_results
from datasets import load_dataset
import numpy as np

from state_object import minibatch_kmeans_partition
from mixing import mix_dime_and_lm, build_global_freq
from grid_search import grid_search_hyperparams

VOCAB_SIZE = 50257

gpt = FrozenGPT2()
dataset = load_dataset("roneneldan/TinyStories", split="train", streaming=True, trust_remote_code=True)

chunks, _ = collect_chunks_split(
    dataset, gpt.tokenizer, seq_len=128,
    n_chunks_needed={"datastore": 350, "controller_train": 75, "val": 75},
    seed=42,
)


def encode_split(chunks_list, batch_size=256):
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


ds_keys, ds_values = build_datastore_from_chunks(gpt, chunks["datastore"], batch_size=256)
val_keys, val_true_targets, val_p_lm_true = encode_split(chunks["val"])
nll_gpt_only = -np.log(val_p_lm_true + 1e-12)

print(f"datastore: {len(ds_values)}  val: {len(val_true_targets)}")

# Already-tuned hyperparameters (Phase D11 round 2)
RAW_K, RAW_TAU, RAW_ALPHA = 50, 2.0, 0.1
DIME_K, DIME_TAU, DIME_ALPHA = 20, 2.0, 0.05
N_CLUSTERS = 500

raw_index, raw_stored_values = build_datastore(ds_keys, ds_values)

compressed_keys, compressed_dists = minibatch_kmeans_partition(ds_keys, ds_values, n_clusters=N_CLUSTERS, seed=42)
compressed_values = np.array(compressed_dists, dtype=object)
dime_index, dime_stored_values = build_datastore(compressed_keys, compressed_values)

results = {"phase": "tinystories_extras", "sections_completed": []}
out_path = "results/tinystories_extras.json"
save_results(out_path, results)


# ============================================================
# Section A — Point 5: mechanism diagnostics (on DIME memory)
# ============================================================
print("\n=== Section A: mechanism diagnostics ===")

K_DIAG = 8
distances_diag, retrieved_diag = query_knn(dime_index, compressed_values, val_keys, k=K_DIAG)
N = len(val_true_targets)

hit_at = {}
for K in [1, 4, 8]:
    hits = 0
    for i in range(N):
        true_tok = int(val_true_targets[i])
        if any(true_tok in retrieved_diag[i, j] for j in range(K)):
            hits += 1
    hit_at[K] = hits / N
print("hit@k:", hit_at)

p_state_true = np.zeros(N)
for i in range(N):
    true_tok = int(val_true_targets[i])
    counter = retrieved_diag[i, 0]
    total = sum(counter.values())
    if total > 0:
        p_state_true[i] = counter.get(true_tok, 0) / total
mean_p_state_true = float(p_state_true.mean())
print("mean p_state(true) at nearest state:", mean_p_state_true)

_, nearest_idx_arr = query_knn_indices(dime_index, val_keys, k=1)
nearest_idx_arr = nearest_idx_arr[:, 0]
active_fraction = len(set(nearest_idx_arr.tolist())) / N_CLUSTERS
print(f"active-state fraction: {active_fraction:.4f} ({len(set(nearest_idx_arr.tolist()))}/{N_CLUSTERS})")

distances_tuned, retrieved_tuned = query_knn(dime_index, compressed_values, val_keys, k=DIME_K)
_, nll_mixed_tuned = mix_dime_and_lm(distances_tuned, retrieved_tuned, val_true_targets, val_p_lm_true, tau=DIME_TAU, alpha=DIME_ALPHA)
nll_delta = nll_gpt_only - nll_mixed_tuned  # positive = retrieval helped this query

cluster_deltas = {}
for i in range(N):
    c = int(nearest_idx_arr[i])
    cluster_deltas.setdefault(c, []).append(float(nll_delta[i]))
cluster_mean_delta = {c: float(np.mean(v)) for c, v in cluster_deltas.items() if len(v) >= 3}  # ignore near-empty clusters
sorted_clusters = sorted(cluster_mean_delta.items(), key=lambda x: x[1])
top_harmful = sorted_clusters[:10]
top_helpful = sorted_clusters[-10:][::-1]
print("top 10 most helpful clusters (id, mean NLL delta):", top_helpful)
print("top 10 most harmful clusters (id, mean NLL delta):", top_harmful)

results["point5_diagnostics"] = {
    "k_diag": K_DIAG,
    "hit_at_k": hit_at,
    "mean_p_state_true": mean_p_state_true,
    "active_state_fraction": active_fraction,
    "n_active_states": len(set(nearest_idx_arr.tolist())),
    "n_total_states": N_CLUSTERS,
    "top_helpful_clusters": top_helpful,
    "top_harmful_clusters": top_harmful,
    "note_oracle_gap": "see DIME/results/q_read_multi_action_baseline.json — oracle 2.5342 vs learned Q-read 2.7308 vs GPT-only 2.7984",
}
results["sections_completed"].append("point5_diagnostics")
save_results(out_path, results)


# ============================================================
# Section B — Point 6: two more raw-baseline selection criteria
# ============================================================
print("\n=== Section B: raw_token_rarity, raw_coverage ===")

global_freq = build_global_freq(ds_values, VOCAB_SIZE)

# raw_token_rarity: the BUDGET positions whose target token is rarest corpus-wide
token_freq_per_position = global_freq[ds_values]
rarity_idx = np.argsort(token_freq_per_position)[:N_CLUSTERS]

raw_rarity_index, raw_rarity_values = build_datastore(ds_keys[rarity_idx], ds_values[rarity_idx])
distances_rarity, retrieved_rarity = query_knn(raw_rarity_index, raw_rarity_values, val_keys, k=min(RAW_K, len(rarity_idx)))
_, nll_rarity = mix_knn_and_lm(distances_rarity, retrieved_rarity, val_true_targets, val_p_lm_true, tau=RAW_TAU, alpha=RAW_ALPHA)
nll_rarity_mean = float(nll_rarity.mean())
print(f"raw_token_rarity: n={len(rarity_idx)}  mean NLL = {nll_rarity_mean:.4f}")

# raw_coverage: greedy farthest-first traversal — maximize minimum pairwise distance
rng = np.random.default_rng(42)
N_ds = len(ds_keys)
start = int(rng.integers(N_ds))
selected = [start]
min_dist = np.linalg.norm(ds_keys - ds_keys[start], axis=1)
min_dist[start] = -1
for _ in range(N_CLUSTERS - 1):
    next_idx = int(np.argmax(min_dist))
    selected.append(next_idx)
    new_dist = np.linalg.norm(ds_keys - ds_keys[next_idx], axis=1)
    min_dist = np.minimum(min_dist, new_dist)
    min_dist[next_idx] = -1
coverage_idx = np.array(selected)

raw_coverage_index, raw_coverage_values = build_datastore(ds_keys[coverage_idx], ds_values[coverage_idx])
distances_coverage, retrieved_coverage = query_knn(raw_coverage_index, raw_coverage_values, val_keys, k=min(RAW_K, len(coverage_idx)))
_, nll_coverage = mix_knn_and_lm(distances_coverage, retrieved_coverage, val_true_targets, val_p_lm_true, tau=RAW_TAU, alpha=RAW_ALPHA)
nll_coverage_mean = float(nll_coverage.mean())
print(f"raw_coverage: n={len(coverage_idx)}  mean NLL = {nll_coverage_mean:.4f}")

results["point6_raw_baselines"] = {
    "raw_token_rarity": {"n_selected": int(len(rarity_idx)), "mean_nll": nll_rarity_mean},
    "raw_coverage": {"n_selected": int(len(coverage_idx)), "mean_nll": nll_coverage_mean},
    "reference_gpt_only": float(nll_gpt_only.mean()),
    "reference_dime_tuned": None,  # filled in below once computed
}
results["sections_completed"].append("point6_raw_baselines")
save_results(out_path, results)


# ============================================================
# Section C — compression-ratio-vs-quality sweep (TinyStories scale)
# ============================================================
print("\n=== Section C: compression sweep ===")

k_values = [10, 20, 30, 50]
tau_values = [1.0, 2.0, 5.0, 10.0]
alpha_values = [0.01, 0.05, 0.1, 0.25]
k_max = max(k_values)

ct_keys, ct_true_targets, ct_p_lm_true = encode_split(chunks["controller_train"])

raw_distances_ct, raw_retrieved_ct = query_knn(raw_index, raw_stored_values, ct_keys, k=k_max)
raw_grid_results, raw_best = grid_search_hyperparams(
    raw_distances_ct, raw_retrieved_ct, ct_true_targets, ct_p_lm_true,
    mix_knn_and_lm, k_values, tau_values, alpha_values
)
raw_distances_val, raw_retrieved_val = query_knn(raw_index, raw_stored_values, val_keys, k=raw_best["k"])
_, nll_raw_val = mix_knn_and_lm(raw_distances_val, raw_retrieved_val, val_true_targets, val_p_lm_true, tau=raw_best["tau"], alpha=raw_best["alpha"])
raw_ceiling_nll = float(nll_raw_val.mean())
print("raw kNN tuned ceiling:", raw_best, "-> val mean NLL:", raw_ceiling_nll)

n_clusters_sweep = [500, 2000, 5000, 13000]
sweep_results = []

for n_clusters in n_clusters_sweep:
    ratio = len(ds_values) / n_clusters
    print(f"\n--- n_clusters={n_clusters} ({ratio:.1f}x compression) ---")

    c_keys, c_dists = minibatch_kmeans_partition(ds_keys, ds_values, n_clusters=n_clusters, seed=42)
    c_index, c_values = build_datastore(c_keys, np.array(c_dists, dtype=object))

    this_k_values = [k for k in k_values if k <= n_clusters]
    dime_distances_ct, dime_retrieved_ct = query_knn(c_index, c_values, ct_keys, k=max(this_k_values))
    dime_grid_results, dime_best = grid_search_hyperparams(
        dime_distances_ct, dime_retrieved_ct, ct_true_targets, ct_p_lm_true,
        mix_dime_and_lm, this_k_values, tau_values, alpha_values
    )

    dime_distances_val, dime_retrieved_val = query_knn(c_index, c_values, val_keys, k=dime_best["k"])
    _, nll_dime_val = mix_dime_and_lm(dime_distances_val, dime_retrieved_val, val_true_targets, val_p_lm_true, tau=dime_best["tau"], alpha=dime_best["alpha"])
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

    if n_clusters == N_CLUSTERS:
        results["point6_raw_baselines"]["reference_dime_tuned"] = dime_val_nll

    results["compression_sweep"] = {
        "raw_knn_tuned_ceiling": {"best_config": raw_best, "val_mean_nll": raw_ceiling_nll},
        "sweep": sweep_results,
    }
    save_results(out_path, results)

results["sections_completed"].append("compression_sweep")
save_results(out_path, results)

print("\n=== ALL SECTIONS DONE ===")
print(f"raw kNN ceiling: {raw_ceiling_nll:.4f}")
for r in sweep_results:
    print(f"n_clusters={r['n_clusters']:>6}  ratio={r['compression_ratio']:6.1f}x  val NLL={r['val_mean_nll']:.4f}  beats raw? {r['beats_raw_ceiling']}")
