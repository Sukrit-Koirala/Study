import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "GPT_Module"))

from gpt_class import FrozenGPT2
from dataset_helpers import collect_chunks_split
from extract import build_datastore_from_chunks, run_batch
from knn import build_datastore, query_knn, mix_knn_and_lm
from results_io import save_results
from datasets import load_dataset
import numpy as np

from state_object import minibatch_kmeans_partition
from mixing import mix_dime_and_lm
from grid_search import grid_search_hyperparams

dataset = load_dataset("roneneldan/TinyStories", split="train", streaming=True, trust_remote_code=True)
gpt = FrozenGPT2()

chunks, chunk_story_ids = collect_chunks_split(
    dataset, gpt.tokenizer, seq_len=128,
    n_chunks_needed={"datastore": 350, "controller_train": 75, "val": 75}
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


# Build raw datastore + minibatch_kmeans compressed datastore (both from the same datastore split)
ds_keys, ds_values = build_datastore_from_chunks(gpt, chunks["datastore"], batch_size=256)

n_clusters = 500
compressed_keys, compressed_dists = minibatch_kmeans_partition(ds_keys, ds_values, n_clusters=n_clusters, seed=42)
compressed_values = np.array(compressed_dists, dtype=object)

raw_index, raw_stored_values = build_datastore(ds_keys, ds_values)
dime_index, dime_stored_values = build_datastore(compressed_keys, compressed_values)

# Encode controller_train (for tuning) and val (touched once, at the very end)
ct_keys, ct_true_targets, ct_p_lm_true = encode_split(chunks["controller_train"])
val_keys, val_true_targets, val_p_lm_true = encode_split(chunks["val"])

k_values = [10, 20, 30, 50]        # dropped k=3 (never competitive), extended upward for raw kNN's sake
tau_values = [1.0, 2.0, 5.0, 10.0]  # dropped 0.5 (never competitive), extended slightly upward
alpha_values = [0.01, 0.05, 0.1, 0.25]  # dropped 0.5/0.75 (clearly much worse), extended downward — this is the important change

k_max = max(k_values)

# --- Raw kNN grid search on controller_train ---
raw_distances_ct, raw_retrieved_ct = query_knn(raw_index, raw_stored_values, ct_keys, k=k_max)
raw_grid_results, raw_best = grid_search_hyperparams(
    raw_distances_ct, raw_retrieved_ct, ct_true_targets, ct_p_lm_true,
    mix_knn_and_lm, k_values, tau_values, alpha_values
)
print("raw kNN best on controller_train:", raw_best)

# --- DIME (minibatch_kmeans) grid search on controller_train ---
dime_distances_ct, dime_retrieved_ct = query_knn(dime_index, dime_stored_values, ct_keys, k=k_max)
dime_grid_results, dime_best = grid_search_hyperparams(
    dime_distances_ct, dime_retrieved_ct, ct_true_targets, ct_p_lm_true,
    mix_dime_and_lm, k_values, tau_values, alpha_values
)
print("DIME (minibatch_kmeans) best on controller_train:", dime_best)

# --- Evaluate the winning configs on val — touched exactly once, here ---
raw_distances_val, raw_retrieved_val = query_knn(raw_index, raw_stored_values, val_keys, k=raw_best["k"])
_, raw_val_nll = mix_knn_and_lm(raw_distances_val, raw_retrieved_val, val_true_targets, val_p_lm_true,
                                  tau=raw_best["tau"], alpha=raw_best["alpha"])

dime_distances_val, dime_retrieved_val = query_knn(dime_index, dime_stored_values, val_keys, k=dime_best["k"])
_, dime_val_nll = mix_dime_and_lm(dime_distances_val, dime_retrieved_val, val_true_targets, val_p_lm_true,
                                    tau=dime_best["tau"], alpha=dime_best["alpha"])

results = {
    "phase": "D11_grid_search",
    "model": "gpt2",
    "seq_len": 128,
    "grid": {"k_values": k_values, "tau_values": tau_values, "alpha_values": alpha_values},
    "raw_knn": {
        "best_config": raw_best,
        "val_mean_nll": float(raw_val_nll.mean()),
        "ct_grid_top5": raw_grid_results[:5],
    },
    "dime_minibatch_kmeans": {
        "best_config": dime_best,
        "val_mean_nll": float(dime_val_nll.mean()),
        "ct_grid_top5": dime_grid_results[:5],
    },
}

save_results("results/grid_search_results.json", results)
print("raw kNN: best config", raw_best, "-> val mean NLL:", float(raw_val_nll.mean()))
print("DIME minibatch_kmeans: best config", dime_best, "-> val mean NLL:", float(dime_val_nll.mean()))
