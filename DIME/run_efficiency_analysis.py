import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "GPT_Module"))

from gpt_class import FrozenGPT2
from dataset_helpers import collect_chunks_split
from extract import build_datastore_from_chunks, run_batch
from knn import build_datastore, query_knn
from results_io import save_results
from datasets import load_dataset
import numpy as np
import pickle
import time
from collections import Counter

from state_object import minibatch_kmeans_partition

dataset = load_dataset("roneneldan/TinyStories", split="train", streaming=True, trust_remote_code=True)
gpt = FrozenGPT2()

chunks, chunk_story_ids = collect_chunks_split(
    dataset, gpt.tokenizer, seq_len=128,
    n_chunks_needed={"datastore": 350, "controller_train": 75, "val": 75}
)


def encode_keys(chunks_list, batch_size=256):
    all_keys = []
    for i in range(0, len(chunks_list), batch_size):
        batch = chunks_list[i:i + batch_size]
        h_pred, y_target, p_true, nll = run_batch(gpt, batch)
        B, L, D = h_pred.shape
        all_keys.append(h_pred.reshape(B * L, D).cpu().numpy())
    return np.concatenate(all_keys, axis=0)


ds_keys, ds_values = build_datastore_from_chunks(gpt, chunks["datastore"], batch_size=256)
n_clusters = 500
compressed_keys, compressed_dists = minibatch_kmeans_partition(ds_keys, ds_values, n_clusters=n_clusters, seed=42)

val_keys = encode_keys(chunks["val"])

# --- Entry-count comparison ---
n_raw_entries = len(ds_values)
n_dime_entries = n_clusters
entry_compression_ratio = n_raw_entries / n_dime_entries

# --- Real measured bytes (not formulas) ---
raw_keys_bytes = ds_keys.nbytes
raw_values_bytes = ds_values.nbytes
raw_total_bytes = raw_keys_bytes + raw_values_bytes

dime_keys_bytes = compressed_keys.nbytes
dime_values_bytes_full = len(pickle.dumps(compressed_dists))

topk_dists = [Counter(dict(d.most_common(5))) for d in compressed_dists]
dime_values_bytes_top5 = len(pickle.dumps(topk_dists))

majority_dists = []
for d in compressed_dists:
    if len(d) > 0:
        top_tok, _ = d.most_common(1)[0]
        majority_dists.append(Counter({top_tok: sum(d.values())}))
    else:
        majority_dists.append(Counter())
dime_values_bytes_majority = len(pickle.dumps(majority_dists))

dime_total_bytes_full = dime_keys_bytes + dime_values_bytes_full
dime_total_bytes_top5 = dime_keys_bytes + dime_values_bytes_top5
dime_total_bytes_majority = dime_keys_bytes + dime_values_bytes_majority

# --- Real wall-clock retrieval latency ---
raw_index, raw_stored_values = build_datastore(ds_keys, ds_values)
dime_index, dime_stored_values = build_datastore(compressed_keys, np.array(compressed_dists, dtype=object))


def time_retrieval(index, values, query_keys, k, n_warmup=5, n_timed=20):
    for _ in range(n_warmup):
        query_knn(index, values, query_keys[:100], k=k)
    start = time.perf_counter()
    for _ in range(n_timed):
        query_knn(index, values, query_keys, k=k)
    elapsed = time.perf_counter() - start
    return (elapsed / n_timed) / len(query_keys) * 1000  # ms per single query


raw_latency_ms = time_retrieval(raw_index, raw_stored_values, val_keys, k=50)
dime_latency_ms = time_retrieval(dime_index, dime_stored_values, val_keys, k=20)

results = {
    "phase": "F16_efficiency_analysis",
    "model": "gpt2",
    "seq_len": 128,
    "entry_count": {
        "raw_kNN": n_raw_entries,
        "dime_minibatch_kmeans": n_dime_entries,
        "compression_ratio": entry_compression_ratio,
    },
    "bytes_measured": {
        "raw_kNN": {"keys_bytes": int(raw_keys_bytes), "values_bytes": int(raw_values_bytes), "total_bytes": int(raw_total_bytes)},
        "dime_full": {"keys_bytes": int(dime_keys_bytes), "values_bytes": int(dime_values_bytes_full), "total_bytes": int(dime_total_bytes_full)},
        "dime_top5_truncation": {"keys_bytes": int(dime_keys_bytes), "values_bytes": int(dime_values_bytes_top5), "total_bytes": int(dime_total_bytes_top5)},
        "dime_majority_token": {"keys_bytes": int(dime_keys_bytes), "values_bytes": int(dime_values_bytes_majority), "total_bytes": int(dime_total_bytes_majority)},
    },
    "byte_compression_ratio_vs_raw": {
        "dime_full": raw_total_bytes / dime_total_bytes_full,
        "dime_top5_truncation": raw_total_bytes / dime_total_bytes_top5,
        "dime_majority_token": raw_total_bytes / dime_total_bytes_majority,
    },
    "latency_ms_per_query": {
        "raw_kNN_k50": raw_latency_ms,
        "dime_minibatch_kmeans_k20": dime_latency_ms,
    },
}

save_results("results/efficiency_analysis.json", results)
print("=== Entry count ===")
print(f"raw kNN: {n_raw_entries}  DIME: {n_dime_entries}  compression: {entry_compression_ratio:.1f}x")
print("=== Real measured bytes ===")
print(f"raw kNN total:         {raw_total_bytes/1e6:.2f} MB")
print(f"DIME full:             {dime_total_bytes_full/1e6:.2f} MB  ({raw_total_bytes/dime_total_bytes_full:.1f}x smaller than raw)")
print(f"DIME top-5 truncation: {dime_total_bytes_top5/1e6:.2f} MB  ({raw_total_bytes/dime_total_bytes_top5:.1f}x smaller than raw)")
print(f"DIME majority-token:   {dime_total_bytes_majority/1e6:.2f} MB  ({raw_total_bytes/dime_total_bytes_majority:.1f}x smaller than raw)")
print("=== Retrieval latency (ms per query) ===")
print(f"raw kNN (k=50):               {raw_latency_ms:.4f} ms/query")
print(f"DIME minibatch_kmeans (k=20): {dime_latency_ms:.4f} ms/query")
