import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "GPT_Module"))

import numpy as np
import torch

from gpt_class import FrozenGPT2
from dataset_helpers import collect_chunks_split
from extract import build_datastore_from_chunks, run_batch_with_entropy
from knn import build_datastore, query_knn_indices
from results_io import save_results
from datasets import load_dataset

from state_object import minibatch_kmeans_partition
from mixing import build_global_freq
from q_read import train_q_read_controller
from q_read_dense import counters_to_dense, compute_reward_matrix_dense, build_obs_features, build_qsa_dataset
from action_grid import build_action_grid, build_action_features

device = "cuda" if torch.cuda.is_available() else "cpu"

dataset = load_dataset("roneneldan/TinyStories", split="train", streaming=True, trust_remote_code=True)
gpt = FrozenGPT2()

chunks, chunk_story_ids = collect_chunks_split(
    dataset, gpt.tokenizer, seq_len=128,
    n_chunks_needed={"datastore": 350, "controller_train": 75, "val": 75}
)


def encode_split_with_entropy(chunks_list, batch_size=256):
    all_keys, all_targets, all_p_lm, all_entropy = [], [], [], []
    for i in range(0, len(chunks_list), batch_size):
        batch = chunks_list[i:i + batch_size]
        h_pred, y_target, p_true, nll, entropy = run_batch_with_entropy(gpt, batch)
        B, L, D = h_pred.shape
        all_keys.append(h_pred.reshape(B * L, D).cpu().numpy())
        all_targets.append(y_target.reshape(B * L).cpu().numpy())
        all_p_lm.append(p_true.reshape(B * L).cpu().numpy())
        all_entropy.append(entropy.reshape(B * L).cpu().numpy())
    return (np.concatenate(all_keys, axis=0),
            np.concatenate(all_targets, axis=0),
            np.concatenate(all_p_lm, axis=0),
            np.concatenate(all_entropy, axis=0))


# Build the raw datastore, then compress it via minibatch_kmeans (same 500-cluster setup as Phase D)
ds_keys, ds_values = build_datastore_from_chunks(gpt, chunks["datastore"], batch_size=256)
n_clusters = 500
compressed_keys, compressed_dists = minibatch_kmeans_partition(ds_keys, ds_values, n_clusters=n_clusters, seed=42)
index, _ = build_datastore(compressed_keys, compressed_dists)

# Dense per-cluster arrays, built once and reused for every action/query
token_ids, token_counts, total_counts = counters_to_dense(compressed_dists, top_k=64)
global_freq = build_global_freq(ds_values)

# fast=True (17 actions) verified the pipeline end-to-end; now the full grid
actions = build_action_grid(fast=False)
action_features = build_action_features(actions, k_max=300, beta_max=20.0)
k_max = max(a["k"] for a in actions)


def build_qsa_and_reward(keys, true_targets, p_lm_true, entropy):
    distances_max, cluster_idx_max = query_knn_indices(index, keys, k=k_max)
    reward = compute_reward_matrix_dense(
        cluster_idx_max, distances_max, true_targets, p_lm_true,
        token_ids, token_counts, total_counts, actions,
        global_freq=global_freq, device=device,
    )
    obs = build_obs_features(entropy, distances_max, cluster_idx_max, token_counts, total_counts, device=device)
    X, y = build_qsa_dataset(obs, action_features, reward)
    return X, y, reward


# --- controller_train: build (state, action) -> reward dataset and train ---
ct_keys, ct_true_targets, ct_p_lm_true, ct_entropy = encode_split_with_entropy(chunks["controller_train"])
X_ct, y_ct, reward_ct = build_qsa_and_reward(ct_keys, ct_true_targets, ct_p_lm_true, ct_entropy)

model = train_q_read_controller(X_ct, y_ct)
print(f"controller_train: {len(ct_true_targets)} queries x {len(actions)} actions "
      f"= {len(y_ct)} (state,action) rows, mean reward = {y_ct.mean():.4f}")

# --- val: touched exactly once. Predict reward for every action, pick argmax per query ---
val_keys, val_true_targets, val_p_lm_true, val_entropy = encode_split_with_entropy(chunks["val"])
X_val, _, reward_val = build_qsa_and_reward(val_keys, val_true_targets, val_p_lm_true, val_entropy)

N_val, A = reward_val.shape
predicted_reward = model.predict(X_val).reshape(N_val, A)
best_action_idx = predicted_reward.argmax(axis=1)
oracle_action_idx = reward_val.argmax(axis=1)

nll_gpt_val = -np.log(val_p_lm_true + 1e-12)
nll_matrix_val = nll_gpt_val[:, None] - reward_val  # [N_val, A] — reuses the real per-action mixing already computed

final_nll_q_read = nll_matrix_val[np.arange(N_val), best_action_idx]
final_nll_oracle = nll_matrix_val[np.arange(N_val), oracle_action_idx]

# Best single fixed action, selected on controller_train (never val) — same rule as
# Phase D11's grid search, so this stays a fair baseline instead of a val-peeking one.
best_fixed_idx = int(reward_ct.mean(axis=0).argmax())

action_names = [a["name"] for a in actions]
chosen_action_counts = {name: int((best_action_idx == i).sum()) for i, name in enumerate(action_names)}

results = {
    "phase": "post_F_multi_action_q_read",
    "model": "gpt2",
    "seq_len": 128,
    "dime_method": "minibatch_kmeans",
    "n_clusters": n_clusters,
    "n_actions": len(actions),
    "k_max": k_max,
    "n_controller_train_queries": len(ct_true_targets),
    "n_val_queries": N_val,
    "mean_nll_gpt_only": float(nll_gpt_val.mean()),
    "mean_nll_q_read_multi_action": float(final_nll_q_read.mean()),
    "mean_nll_oracle_multi_action": float(final_nll_oracle.mean()),
    "best_fixed_action": action_names[best_fixed_idx],
    "mean_nll_best_fixed_action": float(nll_matrix_val[:, best_fixed_idx].mean()),
    "chosen_action_counts": chosen_action_counts,
}

save_results("results/q_read_multi_action_baseline.json", results)
print(f"val: {N_val} queries (collect_chunks_split stops once ALL splits hit >= their target, "
      f"so this can be well above the nominal 75-chunk count)")
assert sum(chosen_action_counts.values()) == N_val
print("mean NLL, GPT-only:                    ", results["mean_nll_gpt_only"])
print("mean NLL, best single fixed action:    ", results["mean_nll_best_fixed_action"], f"({results['best_fixed_action']})")
print("mean NLL, Q-read (learned, per-query): ", results["mean_nll_q_read_multi_action"])
print("mean NLL, oracle (perfect per-query):  ", results["mean_nll_oracle_multi_action"])
print("action usage (top 10):", sorted(chosen_action_counts.items(), key=lambda kv: -kv[1])[:10])
