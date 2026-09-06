from gpt_class import FrozenGPT2
from dataset_helpers import collect_chunks_split
from extract import build_datastore_from_chunks, run_batch
from knn import build_datastore, query_knn, mix_knn_and_lm
from results_io import save_results
from datasets import load_dataset
import numpy as np

dataset = load_dataset("roneneldan/TinyStories", split="train", streaming=True, trust_remote_code=True)
gpt = FrozenGPT2()

chunks, chunk_story_ids = collect_chunks_split(
    dataset, gpt.tokenizer, seq_len=128,
    n_chunks_needed={"datastore": 350, "controller_train": 75, "val": 75}
)

# Step 5: build the raw datastore
ds_keys, ds_values = build_datastore_from_chunks(gpt, chunks["datastore"], batch_size=256)
index, stored_values = build_datastore(ds_keys, ds_values)

# Encode val split, batched — need p_true this time, so used run_batch directly (not build_datastore_from_chunks)
val_chunks = chunks["val"]
batch_size = 256
all_val_keys, all_true_targets, all_p_lm_true = [], [], []
for i in range(0, len(val_chunks), batch_size):
    batch = val_chunks[i:i + batch_size]
    h_pred, y_target, p_true, nll = run_batch(gpt, batch)
    B, L, D = h_pred.shape
    all_val_keys.append(h_pred.reshape(B * L, D).cpu().numpy())
    all_true_targets.append(y_target.reshape(B * L).cpu().numpy())
    all_p_lm_true.append(p_true.reshape(B * L).cpu().numpy())

val_keys = np.concatenate(all_val_keys, axis=0)
val_true_targets = np.concatenate(all_true_targets, axis=0)
val_p_lm_true = np.concatenate(all_p_lm_true, axis=0)

# Step 6: retrieve + mix
k, tau, alpha = 5, 1.0, 0.25
distances, retrieved = query_knn(index, stored_values, val_keys, k=k)
p_mixed, nll_mixed = mix_knn_and_lm(distances, retrieved, val_true_targets, val_p_lm_true, tau=tau, alpha=alpha)

mean_nll_mixed = float(nll_mixed.mean())
mean_nll_pure_lm = float((-np.log(val_p_lm_true + 1e-12)).mean())

results = {
    "phase": "B5_B6_raw_knn_baseline",
    "model": "gpt2",
    "seq_len": 128,
    "n_datastore_entries": len(ds_values),
    "n_query_positions": len(val_true_targets),
    "k": k, "tau": tau, "alpha": alpha,
    "mean_nll_mixed": mean_nll_mixed,
    "mean_nll_pure_lm_same_positions": mean_nll_pure_lm,
    "per_position_nll_mixed": nll_mixed.tolist(),
}

save_results("results/raw_knn_baseline.json", results)
print("datastore size:", ds_keys.shape)
print("mean pure GPT NLL (same val positions):", mean_nll_pure_lm)
print("mean raw kNN-mixed NLL:", mean_nll_mixed)
