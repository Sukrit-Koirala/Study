import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "GPT_Module"))

from gpt_class import FrozenGPT2
from dataset_helpers import collect_chunks_split
from extract import build_datastore_from_chunks, run_batch
from knn import build_datastore, query_knn
from results_io import save_results
from datasets import load_dataset
import numpy as np
from collections import Counter

from state_object import minibatch_kmeans_partition
from mixing import mix_dime_and_lm

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


ds_keys, ds_values = build_datastore_from_chunks(gpt, chunks["datastore"], batch_size=256)
n_clusters = 500
compressed_keys, compressed_dists = minibatch_kmeans_partition(ds_keys, ds_values, n_clusters=n_clusters, seed=42)

val_keys, val_true_targets, val_p_lm_true = encode_split(chunks["val"])

# Tuned config from Phase D step 11 (minibatch_kmeans winner)
k, tau, alpha = 20, 2.0, 0.05

index, _ = build_datastore(compressed_keys, np.array(compressed_dists, dtype=object))


def evaluate(values_array):
    distances, retrieved_dists = query_knn(index, values_array, val_keys, k=k)
    _, nll_mixed = mix_dime_and_lm(distances, retrieved_dists, val_true_targets, val_p_lm_true, tau=tau, alpha=alpha)
    return float(nll_mixed.mean())


# --- Full (unablated) state object ---
full_values = np.array(compressed_dists, dtype=object)
nll_full = evaluate(full_values)

# --- Ablation 1: majority-token — collapse each cluster to just its top token ---
majority_dists = []
for d in compressed_dists:
    if len(d) > 0:
        top_tok, _ = d.most_common(1)[0]
        majority_dists.append(Counter({top_tok: sum(d.values())}))
    else:
        majority_dists.append(Counter())
nll_majority = evaluate(np.array(majority_dists, dtype=object))

# --- Ablation 2: top-5 truncation — keep only the 5 most frequent tokens per cluster ---
TOPK = 5
topk_dists = [Counter(dict(d.most_common(TOPK))) for d in compressed_dists]
nll_topk = evaluate(np.array(topk_dists, dtype=object))

# --- Ablation 3: shuffled — keep centroids where they are, scramble which distribution each reports ---
rng = np.random.default_rng(42)
perm = rng.permutation(len(compressed_dists))
shuffled_dists = [compressed_dists[i] for i in perm]
nll_shuffled = evaluate(np.array(shuffled_dists, dtype=object))

results = {
    "phase": "F15_state_object_ablations",
    "model": "gpt2",
    "seq_len": 128,
    "n_clusters": n_clusters,
    "tuned_config": {"k": k, "tau": tau, "alpha": alpha},
    "top_k_truncation": TOPK,
    "mean_nll_full": nll_full,
    "mean_nll_majority_token": nll_majority,
    "mean_nll_topk_truncation": nll_topk,
    "mean_nll_shuffled": nll_shuffled,
}

save_results("results/state_object_ablations.json", results)
print("full (unablated):     ", nll_full)
print("majority-token:       ", nll_majority)
print(f"top-{TOPK} truncation:     ", nll_topk)
print("shuffled (illusion check):", nll_shuffled)
