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

from state_object import minibatch_kmeans_partition, random_partition
from mixing import mix_dime_and_lm

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

N_CLUSTERS = 500
K, TAU, ALPHA = 20, 2.0, 0.05  # tuned DIME config (Phase D11), used for every variant so the comparison is apples-to-apples


def evaluate(keys, values_array, index=None):
    """index (geometry, fit only on keys) can be reused across calls with different
    content — but the lookup must always use the freshly-passed values_array, never
    a stale cached one, or ablations that only change content silently no-op."""
    if index is None:
        index, _ = build_datastore(keys, values_array)
    distances, retrieved = query_knn(index, values_array, val_keys, k=K)
    _, nll = mix_dime_and_lm(distances, retrieved, val_true_targets, val_p_lm_true, tau=TAU, alpha=ALPHA)
    return float(nll.mean())


results = {"phase": "tinystories_rich_ablations", "k": K, "tau": TAU, "alpha": ALPHA, "n_clusters": N_CLUSTERS}
out_path = "results/tinystories_rich_ablations.json"

# --- base minibatch_kmeans DIME ("original") ---
compressed_keys, compressed_dists = minibatch_kmeans_partition(ds_keys, ds_values, n_clusters=N_CLUSTERS, seed=42)
base_index, _ = build_datastore(compressed_keys, np.array(compressed_dists, dtype=object))

nll_original = evaluate(compressed_keys, np.array(compressed_dists, dtype=object), base_index)
print("original:", nll_original)
results["original"] = nll_original
save_results(out_path, results)

# --- top-K truncation levels (finer than the earlier top-5-only ablation) ---
for K_trunc in [64, 32, 16, 5]:
    trunc_dists = [Counter(dict(d.most_common(K_trunc))) for d in compressed_dists]
    nll = evaluate(compressed_keys, np.array(trunc_dists, dtype=object), base_index)
    print(f"top{K_trunc}:", nll)
    results[f"top{K_trunc}"] = nll
    save_results(out_path, results)

# --- majority_token ---
majority_dists = []
for d in compressed_dists:
    if len(d) > 0:
        top_tok, _ = d.most_common(1)[0]
        majority_dists.append(Counter({top_tok: sum(d.values())}))
    else:
        majority_dists.append(Counter())
nll_majority = evaluate(compressed_keys, np.array(majority_dists, dtype=object), base_index)
print("majority_token:", nll_majority)
results["majority_token"] = nll_majority
save_results(out_path, results)

# --- global_unigram: every cluster reports the SAME corpus-wide token distribution ---
# (a sharper illusion-check than shuffling — removes ALL cluster-specific content, not just scrambles it)
global_counts_arr = np.bincount(ds_values, minlength=VOCAB_SIZE)
global_counter = Counter({int(tok): int(cnt) for tok, cnt in enumerate(global_counts_arr) if cnt > 0})
global_dists = [global_counter for _ in compressed_dists]  # same shared Counter reused for every cluster
nll_global_unigram = evaluate(compressed_keys, np.array(global_dists, dtype=object), base_index)
print("global_unigram:", nll_global_unigram)
results["global_unigram"] = nll_global_unigram
save_results(out_path, results)

# --- random_partition (recomputed under this table's shared tuned config, not the old Phase C untuned number) ---
rp_keys, rp_dists = random_partition(ds_keys, ds_values, n_clusters=N_CLUSTERS, seed=42)
nll_random_partition = evaluate(rp_keys, np.array(rp_dists, dtype=object))
print("random_partition:", nll_random_partition)
results["random_partition"] = nll_random_partition
save_results(out_path, results)

# --- shuffled_distribution: correct prototypes, scrambled content (the existing "shuffled" ablation) ---
rng = np.random.default_rng(42)
perm_values = rng.permutation(len(compressed_dists))
shuffled_dist_values = [compressed_dists[i] for i in perm_values]
nll_shuffled_distribution = evaluate(compressed_keys, np.array(shuffled_dist_values, dtype=object), base_index)
print("shuffled_distribution:", nll_shuffled_distribution)
results["shuffled_distribution"] = nll_shuffled_distribution
save_results(out_path, results)

# --- shuffled_prototype: correct distributions, scrambled keys (breaks routing, not content) ---
perm_keys = rng.permutation(len(compressed_keys))
shuffled_keys = compressed_keys[perm_keys]
nll_shuffled_prototype = evaluate(shuffled_keys, np.array(compressed_dists, dtype=object))
print("shuffled_prototype:", nll_shuffled_prototype)
results["shuffled_prototype"] = nll_shuffled_prototype
save_results(out_path, results)

print("\n=== SUMMARY ===")
for k, v in results.items():
    if k not in ("phase", "k", "tau", "alpha", "n_clusters"):
        print(f"{k}: {v:.4f}")
