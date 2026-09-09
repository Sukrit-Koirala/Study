import sys, os, argparse, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "GPT_Module"))

from cache_io import load_cache
from knn import build_datastore, query_knn, query_knn_indices, mix_knn_and_lm
from results_io import save_results
from state_object import minibatch_kmeans_partition, random_partition, utility_weighted_partition, query_kmeans_partition
from mixing import mix_dime_and_lm, build_global_freq
from q_read_dense import counters_to_dense, compute_reward_matrix_dense, build_obs_features, build_qsa_dataset
from action_grid import build_action_grid, build_action_features
from q_read import train_q_read_controller
import numpy as np
import torch
from collections import Counter

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", required=True)
parser.add_argument("--model", required=True)
args = parser.parse_args()

CACHE_DIR = os.path.join(os.path.dirname(__file__), "..", "GPT_Module", "cache")
PREFIX = f"{args.dataset}_{args.model}"

ds = load_cache(os.path.join(CACHE_DIR, f"{PREFIX}_datastore.npz"))
ct = load_cache(os.path.join(CACHE_DIR, f"{PREFIX}_controller_train.npz"))
val = load_cache(os.path.join(CACHE_DIR, f"{PREFIX}_val.npz"))

ds_keys, ds_values, ds_p_lm_true, ds_entropy = ds["keys"], ds["values"], ds["p_lm_true"], ds["entropy"]
ct_keys, ct_values, ct_p_lm_true, ct_entropy = ct["keys"], ct["values"], ct["p_lm_true"], ct["entropy"]
val_keys, val_values, val_p_lm_true, val_entropy = val["keys"], val["values"], val["p_lm_true"], val["entropy"]

with open(f"results/{PREFIX}_tier1.json") as f:
    tier1 = json.load(f)

N_CLUSTERS = tier1["n_clusters"]
RAW_K, RAW_TAU, RAW_ALPHA = tier1["raw_knn"]["best_config"]["k"], tier1["raw_knn"]["best_config"]["tau"], tier1["raw_knn"]["best_config"]["alpha"]
DIME_K, DIME_TAU, DIME_ALPHA = tier1["dime_minibatch_kmeans"]["best_config"]["k"], tier1["dime_minibatch_kmeans"]["best_config"]["tau"], tier1["dime_minibatch_kmeans"]["best_config"]["alpha"]

print(f"[{PREFIX}] n_clusters={N_CLUSTERS}  raw tuned k={RAW_K},tau={RAW_TAU},alpha={RAW_ALPHA}  dime tuned k={DIME_K},tau={DIME_TAU},alpha={DIME_ALPHA}")

results = {"phase": "tier3_extras", "dataset": args.dataset, "model": args.model, "sections_completed": []}
out_path = f"results/{PREFIX}_tier3.json"

compressed_keys, compressed_dists = minibatch_kmeans_partition(ds_keys, ds_values, n_clusters=N_CLUSTERS, seed=42)
dime_index, dime_stored_values = build_datastore(compressed_keys, np.array(compressed_dists, dtype=object))


def eval_dime_construction(keys, dists, name):
    index, values_arr = build_datastore(keys, np.array(dists, dtype=object))
    k = min(DIME_K, len(keys))
    distances, retrieved = query_knn(index, values_arr, val_keys, k=k)
    _, nll = mix_dime_and_lm(distances, retrieved, val_values, val_p_lm_true, tau=DIME_TAU, alpha=DIME_ALPHA)
    print(f"[{PREFIX}] construction={name}: {float(nll.mean()):.4f}")
    return float(nll.mean())


# ============================================================
# Section 1 — remaining DIME construction methods
# ============================================================
results["construction_methods"] = {"minibatch_kmeans": eval_dime_construction(compressed_keys, compressed_dists, "minibatch_kmeans")}

rand_keys, rand_dists = random_partition(ds_keys, ds_values, n_clusters=N_CLUSTERS, seed=42)
results["construction_methods"]["random_partition"] = eval_dime_construction(rand_keys, rand_dists, "random_partition")

util_keys, util_dists = utility_weighted_partition(ds_keys, ds_values, -np.log(ds_p_lm_true + 1e-12), n_clusters=N_CLUSTERS, seed=42)
results["construction_methods"]["utility_weighted"] = eval_dime_construction(util_keys, util_dists, "utility_weighted")

qk_keys, qk_dists = query_kmeans_partition(ds_keys, ds_values, ct_keys, n_clusters=N_CLUSTERS, seed=42)
results["construction_methods"]["query_kmeans"] = eval_dime_construction(qk_keys, qk_dists, "query_kmeans")

results["sections_completed"].append("construction_methods")
save_results(out_path, results)


# ============================================================
# Section 2 — remaining raw-baseline equal-budget variants (Point 6)
# ============================================================
def eval_raw_subset(idx, name):
    idx = np.asarray(idx)
    index, values_arr = build_datastore(ds_keys[idx], ds_values[idx])
    k = min(RAW_K, len(idx))
    distances, retrieved = query_knn(index, values_arr, val_keys, k=k)
    _, nll = mix_knn_and_lm(distances, retrieved, val_values, val_p_lm_true, tau=RAW_TAU, alpha=RAW_ALPHA)
    print(f"[{PREFIX}] raw variant={name}: n={len(idx)} -> {float(nll.mean()):.4f}")
    return float(nll.mean())


rng = np.random.default_rng(42)
N_total = len(ds_values)

idx_random = rng.choice(N_total, size=min(N_CLUSTERS, N_total), replace=False)
results_raw = {"raw_random": eval_raw_subset(idx_random, "raw_random")}

order_by_loss = np.argsort(-np.log(ds_p_lm_true + 1e-12))
results_raw["raw_high_gpt_loss"] = eval_raw_subset(order_by_loss[-N_CLUSTERS:], "raw_high_gpt_loss")
results_raw["raw_low_gpt_loss"] = eval_raw_subset(order_by_loss[:N_CLUSTERS], "raw_low_gpt_loss")

order_by_entropy = np.argsort(ds_entropy)
results_raw["raw_high_entropy"] = eval_raw_subset(order_by_entropy[-N_CLUSTERS:], "raw_high_entropy")
results_raw["raw_low_entropy"] = eval_raw_subset(order_by_entropy[:N_CLUSTERS], "raw_low_entropy")

global_freq = build_global_freq(ds_values)
rarity = -np.log(global_freq[ds_values] + 1e-12)
order_by_rarity = np.argsort(-rarity)
results_raw["raw_token_rarity"] = eval_raw_subset(order_by_rarity[:N_CLUSTERS], "raw_token_rarity")

print(f"[{PREFIX}] starting greedy farthest-first coverage selection (n={N_CLUSTERS})...")
selected = [rng.integers(0, N_total)]
min_dist = np.linalg.norm(ds_keys - ds_keys[selected[0]], axis=1)
for step in range(1, N_CLUSTERS):
    next_idx = int(np.argmax(min_dist))
    selected.append(next_idx)
    new_dist = np.linalg.norm(ds_keys - ds_keys[next_idx], axis=1)
    min_dist = np.minimum(min_dist, new_dist)
    if step % 500 == 0:
        print(f"[{PREFIX}] coverage selection: {step}/{N_CLUSTERS}")
results_raw["raw_coverage"] = eval_raw_subset(np.array(selected), "raw_coverage")

results["raw_baseline_variants"] = results_raw
results["sections_completed"].append("raw_baseline_variants")
save_results(out_path, results)


# ============================================================
# Section 3 — rich (10-variant) ablation table
# ============================================================
def evaluate_ablation(values_array, name):
    distances, retrieved = query_knn(dime_index, values_array, val_keys, k=DIME_K)
    _, nll = mix_dime_and_lm(distances, retrieved, val_values, val_p_lm_true, tau=DIME_TAU, alpha=DIME_ALPHA)
    print(f"[{PREFIX}] ablation={name}: {float(nll.mean()):.4f}")
    return float(nll.mean())


ablations = {"original": evaluate_ablation(dime_stored_values, "original")}

for top_k in [64, 32, 16, 5]:
    trunc = np.array([Counter(dict(d.most_common(top_k))) for d in compressed_dists], dtype=object)
    ablations[f"top{top_k}"] = evaluate_ablation(trunc, f"top{top_k}")

majority = []
for d in compressed_dists:
    if len(d) > 0:
        top_tok, _ = d.most_common(1)[0]
        majority.append(Counter({top_tok: sum(d.values())}))
    else:
        majority.append(Counter())
ablations["majority_token"] = evaluate_ablation(np.array(majority, dtype=object), "majority_token")

global_counts = Counter(ds_values.tolist())
global_unigram = np.array([Counter(global_counts) for _ in compressed_dists], dtype=object)
ablations["global_unigram"] = evaluate_ablation(global_unigram, "global_unigram")

rp_keys, rp_dists = random_partition(ds_keys, ds_values, n_clusters=N_CLUSTERS, seed=43)
rp_index, rp_values = build_datastore(rp_keys, np.array(rp_dists, dtype=object))
rp_distances, rp_retrieved = query_knn(rp_index, rp_values, val_keys, k=DIME_K)
_, rp_nll = mix_dime_and_lm(rp_distances, rp_retrieved, val_values, val_p_lm_true, tau=DIME_TAU, alpha=DIME_ALPHA)
ablations["random_partition"] = float(rp_nll.mean())
print(f"[{PREFIX}] ablation=random_partition: {ablations['random_partition']:.4f}")

perm = rng.permutation(len(compressed_dists))
shuffled_distribution = np.array(compressed_dists, dtype=object)[perm]
ablations["shuffled_distribution"] = evaluate_ablation(shuffled_distribution, "shuffled_distribution")

shuffled_keys, _ = build_datastore(compressed_keys[perm], np.array(compressed_dists, dtype=object))
distances_sp, retrieved_sp = query_knn(shuffled_keys, np.array(compressed_dists, dtype=object), val_keys, k=DIME_K)
_, nll_sp = mix_dime_and_lm(distances_sp, retrieved_sp, val_values, val_p_lm_true, tau=DIME_TAU, alpha=DIME_ALPHA)
ablations["shuffled_prototype"] = float(nll_sp.mean())
print(f"[{PREFIX}] ablation=shuffled_prototype: {ablations['shuffled_prototype']:.4f}")

results["rich_ablations"] = ablations
results["sections_completed"].append("rich_ablations")
save_results(out_path, results)


# ============================================================
# Section 4 — multi-action Q-read controller (dense/GPU)
# ============================================================
device = "cuda" if torch.cuda.is_available() else "cpu"
actions_full = build_action_grid(fast=(N_CLUSTERS < 300))
actions = [a for a in actions_full if a["k"] <= N_CLUSTERS]
k_max = max(a["k"] for a in actions)
print(f"[{PREFIX}] multi-action Q-read: {len(actions)} actions, k_max={k_max}, device={device}")

token_ids, token_counts, total_counts = counters_to_dense(compressed_dists)
global_freq_full = build_global_freq(ds_values)

dist_ct_max, idx_ct_max = query_knn_indices(dime_index, ct_keys, k=k_max)
reward_ct = compute_reward_matrix_dense(idx_ct_max, dist_ct_max, ct_values, ct_p_lm_true, token_ids, token_counts, total_counts, actions, global_freq=global_freq_full, device=device)
obs_ct = build_obs_features(ct_entropy, dist_ct_max, idx_ct_max, token_counts, total_counts, device=device)
action_features = build_action_features(actions, k_max=k_max)

X_train, y_train = build_qsa_dataset(obs_ct, action_features, reward_ct)
q_model = train_q_read_controller(X_train, y_train)

dist_val_max, idx_val_max = query_knn_indices(dime_index, val_keys, k=k_max)
reward_val = compute_reward_matrix_dense(idx_val_max, dist_val_max, val_values, val_p_lm_true, token_ids, token_counts, total_counts, actions, global_freq=global_freq_full, device=device)
obs_val = build_obs_features(val_entropy, dist_val_max, idx_val_max, token_counts, total_counts, device=device)

N_val, A = reward_val.shape
X_val_qsa, _ = build_qsa_dataset(obs_val, action_features, reward_val)
pred_q = q_model.predict(X_val_qsa).reshape(N_val, A)
chosen_action = pred_q.argmax(axis=1)

nll_gpt_val = -np.log(val_p_lm_true + 1e-12)
nll_matrix_val = nll_gpt_val[:, None] - reward_val
final_nll_multi = nll_matrix_val[np.arange(N_val), chosen_action]

action_names = np.array([a["name"] for a in actions])
chosen_names, chosen_counts = np.unique(action_names[chosen_action], return_counts=True)
top_chosen = sorted(zip(chosen_names.tolist(), chosen_counts.tolist()), key=lambda x: -x[1])[:10]

results["multi_action_q_read"] = {
    "n_actions": len(actions),
    "mean_nll": float(final_nll_multi.mean()),
    "frac_gpt_only": float((action_names[chosen_action] == "gpt_only").mean()),
    "top_chosen_actions": top_chosen,
}
print(f"[{PREFIX}] multi-action Q-read: {float(final_nll_multi.mean()):.4f}  frac_gpt_only={results['multi_action_q_read']['frac_gpt_only']:.3f}")
results["sections_completed"].append("multi_action_q_read")
save_results(out_path, results)


# ============================================================
# Section 5 — Point 5 diagnostics (hit@k, p_state(true), active-state, helpful/harmful clusters)
# ============================================================
_, nearest_idx_val = query_knn_indices(dime_index, val_keys, k=DIME_K)

hit_at_k = np.zeros(len(val_values), dtype=bool)
p_state_true = np.zeros(len(val_values), dtype=np.float64)
for i in range(len(val_values)):
    true_tok = int(val_values[i])
    top1_cluster = compressed_dists[nearest_idx_val[i, 0]]
    total = sum(top1_cluster.values())
    p_state_true[i] = (top1_cluster.get(true_tok, 0) / total) if total > 0 else 0.0
    for j in range(DIME_K):
        if compressed_dists[nearest_idx_val[i, j]].get(true_tok, 0) > 0:
            hit_at_k[i] = True
            break

active_clusters = np.unique(nearest_idx_val[:, 0])
active_fraction = len(active_clusters) / N_CLUSTERS

distances_diag, retrieved_diag = query_knn(dime_index, dime_stored_values, val_keys, k=DIME_K)
_, nll_dime_val_diag = mix_dime_and_lm(distances_diag, retrieved_diag, val_values, val_p_lm_true, tau=DIME_TAU, alpha=DIME_ALPHA)
reward_diag = nll_gpt_val - nll_dime_val_diag

cluster_avg_reward = {}
for c in active_clusters:
    mask = nearest_idx_val[:, 0] == c
    cluster_avg_reward[int(c)] = float(reward_diag[mask].mean())

sorted_clusters = sorted(cluster_avg_reward.items(), key=lambda x: -x[1])
helpful_top10 = sorted_clusters[:10]
harmful_top10 = sorted_clusters[-10:]

results["point5_diagnostics"] = {
    "hit_at_k": float(hit_at_k.mean()),
    "p_state_true_mean": float(p_state_true.mean()),
    "active_state_fraction": {"active": int(len(active_clusters)), "total": int(N_CLUSTERS), "fraction": float(active_fraction)},
    "helpful_clusters_top10": helpful_top10,
    "harmful_clusters_top10": harmful_top10,
}
print(f"[{PREFIX}] hit@{DIME_K}={hit_at_k.mean():.4f}  p_state(true)={p_state_true.mean():.4f}  active={len(active_clusters)}/{N_CLUSTERS}")
results["sections_completed"].append("point5_diagnostics")
save_results(out_path, results)

print(f"\n[{PREFIX}] === TIER 3 DONE ===")
