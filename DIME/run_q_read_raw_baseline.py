import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "GPT_Module"))

from gpt_class import FrozenGPT2
from dataset_helpers import collect_chunks_split
from extract import build_datastore_from_chunks, run_batch_with_entropy
from knn import build_datastore, query_knn, mix_knn_and_lm
from results_io import save_results
from datasets import load_dataset
import numpy as np

from q_read import retrieval_purity_entropy, train_q_read_controller

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


# Raw (uncompressed) datastore — no clustering this time
ds_keys, ds_values = build_datastore_from_chunks(gpt, chunks["datastore"], batch_size=256)
index, stored_values = build_datastore(ds_keys, ds_values)

# Tuned hyperparameters from Phase D step 11's grid search (raw kNN's winner)
k, tau, alpha = 50, 2.0, 0.1


def build_features_and_reward(keys, true_targets, p_lm_true, entropy):
    distances, retrieved_values = query_knn(index, stored_values, keys, k=k)
    _, nll_mixed = mix_knn_and_lm(distances, retrieved_values, true_targets, p_lm_true, tau=tau, alpha=alpha)
    nll_gpt = -np.log(p_lm_true + 1e-12)
    reward = nll_gpt - nll_mixed

    nearest_distance = distances[:, 0]
    retrieved_top1 = retrieved_values[:, 0]  # plain token ids for raw kNN — retrieval_purity_entropy
    retrieval_entropy, retrieval_purity = retrieval_purity_entropy(retrieved_top1)  # handles this (entropy=0, purity=1)

    X = np.stack([entropy, nearest_distance, retrieval_entropy, retrieval_purity], axis=1)
    return X, reward, nll_gpt, nll_mixed


# --- Build controller_train's labeled dataset and train the controller ---
ct_keys, ct_true_targets, ct_p_lm_true, ct_entropy = encode_split_with_entropy(chunks["controller_train"])
X_ct, reward_ct, ct_nll_gpt, ct_nll_mixed = build_features_and_reward(ct_keys, ct_true_targets, ct_p_lm_true, ct_entropy)

model = train_q_read_controller(X_ct, reward_ct)
print(f"controller_train: {len(reward_ct)} examples, mean reward = {reward_ct.mean():.4f}")

# --- Apply the trained controller to val — touched exactly once ---
val_keys, val_true_targets, val_p_lm_true, val_entropy = encode_split_with_entropy(chunks["val"])
X_val, reward_val_oracle, val_nll_gpt, val_nll_mixed = build_features_and_reward(val_keys, val_true_targets, val_p_lm_true, val_entropy)

predicted_reward = model.predict(X_val)
use_retrieval = predicted_reward > 0

final_nll = np.where(use_retrieval, val_nll_mixed, val_nll_gpt)

mean_nll_q_read = float(final_nll.mean())
mean_nll_gpt_only = float(val_nll_gpt.mean())
mean_nll_always_mixed = float(val_nll_mixed.mean())
frac_retrieved = float(use_retrieval.mean())

results = {
    "phase": "D13_q_read_raw_knn",
    "model": "gpt2",
    "seq_len": 128,
    "memory": "raw_knn",
    "n_raw_datastore_entries": len(ds_values),
    "tuned_config": {"k": k, "tau": tau, "alpha": alpha},
    "n_controller_train": len(reward_ct),
    "n_val": len(final_nll),
    "frac_val_retrieved": frac_retrieved,
    "mean_nll_gpt_only": mean_nll_gpt_only,
    "mean_nll_always_mixed": mean_nll_always_mixed,
    "mean_nll_q_read": mean_nll_q_read,
}

save_results("results/q_read_raw_baseline.json", results)
print("mean NLL, GPT-only:          ", mean_nll_gpt_only)
print("mean NLL, always mixed:      ", mean_nll_always_mixed)
print("mean NLL, Q-read (adaptive): ", mean_nll_q_read)
print("fraction of val queries where Q-read chose to retrieve:", frac_retrieved)
