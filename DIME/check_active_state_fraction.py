import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "GPT_Module"))

from gpt_class import FrozenGPT2
from dataset_helpers import collect_chunks_split
from extract import build_datastore_from_chunks, run_batch
from knn import build_datastore, query_knn_indices
from datasets import load_dataset
import numpy as np

from state_object import minibatch_kmeans_partition

gpt = FrozenGPT2()
dataset = load_dataset("roneneldan/TinyStories", split="train", streaming=True, trust_remote_code=True)

chunks, _ = collect_chunks_split(
    dataset, gpt.tokenizer, seq_len=128,
    n_chunks_needed={"datastore": 350, "controller_train": 75, "val": 75},
    seed=42,
)


def encode_split(chunks_list, batch_size=256):
    all_keys = []
    for i in range(0, len(chunks_list), batch_size):
        batch = chunks_list[i:i + batch_size]
        h_pred, y_target, p_true, nll = run_batch(gpt, batch)
        B, L, D = h_pred.shape
        all_keys.append(h_pred.reshape(B * L, D).cpu().numpy())
    return np.concatenate(all_keys, axis=0)


ds_keys, ds_values = build_datastore_from_chunks(gpt, chunks["datastore"], batch_size=256)
val_keys = encode_split(chunks["val"])

N_CLUSTERS = 500
compressed_keys, compressed_dists = minibatch_kmeans_partition(ds_keys, ds_values, n_clusters=N_CLUSTERS, seed=42)
dime_index, _ = build_datastore(compressed_keys, np.array(compressed_dists, dtype=object))

_, nearest_idx_arr = query_knn_indices(dime_index, val_keys, k=1)
nearest_idx_arr = nearest_idx_arr[:, 0]

print("min index:", nearest_idx_arr.min(), " max index:", nearest_idx_arr.max())  # sanity: must be in [0, 499]
n_active = len(set(nearest_idx_arr.tolist()))
active_fraction = n_active / N_CLUSTERS
print(f"active-state fraction: {active_fraction:.4f} ({n_active}/{N_CLUSTERS})")
