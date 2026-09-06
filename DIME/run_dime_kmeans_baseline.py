import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "GPT_Module"))

from gpt_class import FrozenGPT2
from dataset_helpers import collect_chunks_split
from extract import build_datastore_from_chunks, run_batch
from knn import build_datastore, query_knn
from results_io import save_results
from datasets import load_dataset
import numpy as np

from state_object import minibatch_kmeans_partition
from mixing import mix_dime_and_lm

dataset = load_dataset("roneneldan/TinyStories", split="train", streaming=True, trust_remote_code=True)
gpt = FrozenGPT2()

chunks, chunk_story_ids = collect_chunks_split(
    dataset, gpt.tokenizer, seq_len=128,
    n_chunks_needed={"datastore": 350, "controller_train": 75, "val": 75}
)

ds_keys, ds_values = build_datastore_from_chunks(gpt, chunks["datastore"], batch_size=256)

n_clusters = 500  # same as random_partition, for a fair comparison
compressed_keys, compressed_dists = minibatch_kmeans_partition(ds_keys, ds_values, n_clusters=n_clusters, seed=42)
compressed_values = np.array(compressed_dists, dtype=object)

index, stored_values = build_datastore(compressed_keys, compressed_values)

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

k, tau, alpha = 5, 1.0, 0.25
distances, retrieved_dists = query_knn(index, stored_values, val_keys, k=k)

print("sanity check — first query's retrieved cluster distributions:")
for j in range(k):
    print(f"  neighbor {j}: distance={distances[0, j]:.3f}  dist={dict(retrieved_dists[0, j])}")

p_mixed, nll_mixed = mix_dime_and_lm(distances, retrieved_dists, val_true_targets, val_p_lm_true, tau=tau, alpha=alpha)

mean_nll_mixed = float(nll_mixed.mean())
mean_nll_pure_lm = float((-np.log(val_p_lm_true + 1e-12)).mean())

results = {
    "phase": "C8_minibatch_kmeans_baseline",
    "model": "gpt2",
    "seq_len": 128,
    "n_raw_datastore_entries": len(ds_values),
    "n_clusters": n_clusters,
    "n_query_positions": len(val_true_targets),
    "k": k, "tau": tau, "alpha": alpha,
    "mean_nll_mixed": mean_nll_mixed,
    "mean_nll_pure_lm_same_positions": mean_nll_pure_lm,
    "per_position_nll_mixed": nll_mixed.tolist(),
}

save_results("results/minibatch_kmeans_baseline.json", results)
print("raw datastore size:", ds_keys.shape)
print("compressed to n_clusters:", n_clusters)
print("mean pure GPT NLL (same val positions):", mean_nll_pure_lm)
print("mean minibatch_kmeans DIME NLL:", mean_nll_mixed)
