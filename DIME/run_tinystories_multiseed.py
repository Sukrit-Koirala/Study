import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "GPT_Module"))

from gpt_class import FrozenGPT2
from dataset_helpers import collect_chunks_split
from extract import build_datastore_from_chunks, run_batch
from knn import build_datastore, query_knn, mix_knn_and_lm
from results_io import save_results
from datasets import load_dataset
import numpy as np
from sklearn.cluster import MiniBatchKMeans

from state_object import minibatch_kmeans_partition
from mixing import mix_dime_and_lm

gpt = FrozenGPT2()

# Already-tuned hyperparameters from the seed=42 grid search (Phase D11) — reused
# across every seed here. Re-tuning per seed would be far more expensive and isn't
# the point of this test: we're measuring how stable the ALREADY-established result
# is across different random data splits, not re-discovering the best config.
RAW_K, RAW_TAU, RAW_ALPHA = 50, 2.0, 0.1
DIME_K, DIME_TAU, DIME_ALPHA = 20, 2.0, 0.05
N_CLUSTERS = 500


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


def run_one_seed(seed):
    dataset = load_dataset("roneneldan/TinyStories", split="train", streaming=True, trust_remote_code=True)
    chunks, _ = collect_chunks_split(
        dataset, gpt.tokenizer, seq_len=128,
        n_chunks_needed={"datastore": 350, "controller_train": 75, "val": 75},
        seed=seed,
    )

    ds_keys, ds_values = build_datastore_from_chunks(gpt, chunks["datastore"], batch_size=256)
    val_keys, val_true_targets, val_p_lm_true = encode_split(chunks["val"])

    nll_gpt_only = float((-np.log(val_p_lm_true + 1e-12)).mean())

    # Raw kNN, tuned config
    raw_index, raw_stored_values = build_datastore(ds_keys, ds_values)
    distances_raw, retrieved_raw = query_knn(raw_index, raw_stored_values, val_keys, k=RAW_K)
    _, nll_raw = mix_knn_and_lm(distances_raw, retrieved_raw, val_true_targets, val_p_lm_true, tau=RAW_TAU, alpha=RAW_ALPHA)
    nll_raw_mean = float(nll_raw.mean())

    # DIME minibatch_kmeans, tuned config
    compressed_keys, compressed_dists = minibatch_kmeans_partition(ds_keys, ds_values, n_clusters=N_CLUSTERS, seed=seed)
    dime_index, dime_stored_values = build_datastore(compressed_keys, np.array(compressed_dists, dtype=object))
    distances_dime, retrieved_dime = query_knn(dime_index, dime_stored_values, val_keys, k=DIME_K)
    _, nll_dime = mix_dime_and_lm(distances_dime, retrieved_dime, val_true_targets, val_p_lm_true, tau=DIME_TAU, alpha=DIME_ALPHA)
    nll_dime_mean = float(nll_dime.mean())

    # Best equal-budget raw variant (raw_kmeans_representative)
    km = MiniBatchKMeans(n_clusters=N_CLUSTERS, random_state=seed)
    assignment = km.fit_predict(ds_keys)
    centers = km.cluster_centers_
    selected_idx = []
    for c in range(N_CLUSTERS):
        member_idx = np.where(assignment == c)[0]
        if len(member_idx) == 0:
            continue
        dists = np.linalg.norm(ds_keys[member_idx] - centers[c], axis=1)
        selected_idx.append(member_idx[np.argmin(dists)])
    selected_idx = np.array(selected_idx)
    raw_rep_index, raw_rep_values = build_datastore(ds_keys[selected_idx], ds_values[selected_idx])
    distances_rep, retrieved_rep = query_knn(raw_rep_index, raw_rep_values, val_keys, k=min(RAW_K, len(selected_idx)))
    _, nll_rep = mix_knn_and_lm(distances_rep, retrieved_rep, val_true_targets, val_p_lm_true, tau=RAW_TAU, alpha=RAW_ALPHA)
    nll_rep_mean = float(nll_rep.mean())

    return {
        "seed": seed,
        "gpt_only": nll_gpt_only,
        "raw_knn_tuned": nll_raw_mean,
        "dime_tuned": nll_dime_mean,
        "raw_kmeans_representative": nll_rep_mean,
    }


SEEDS = list(range(42, 142))  # seed=42 (already known, sanity re-check) + 99 new ones
results = []
out_path = "results/tinystories_multiseed.json"

for seed in SEEDS:
    print(f"--- seed {seed} ---", flush=True)
    r = run_one_seed(seed)
    print(r, flush=True)
    results.append(r)
    save_results(out_path, {"seeds_completed": len(results), "total_planned": len(SEEDS), "results": results})

print("\n=== SUMMARY ===")
for key in ["gpt_only", "raw_knn_tuned", "dime_tuned", "raw_kmeans_representative"]:
    vals = [r[key] for r in results]
    print(f"{key}: mean={np.mean(vals):.4f}  std={np.std(vals):.4f}  n={len(vals)}")
