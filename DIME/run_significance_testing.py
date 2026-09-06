import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "GPT_Module"))

from gpt_class import FrozenGPT2
from dataset_helpers import collect_chunks_split
from extract import build_datastore_from_chunks, run_batch
from knn import build_datastore, query_knn, mix_knn_and_lm
from results_io import save_results
from datasets import load_dataset
import numpy as np
from scipy import stats
from sklearn.cluster import MiniBatchKMeans

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
val_keys, val_true_targets, val_p_lm_true = encode_split(chunks["val"])

nll_gpt_only = -np.log(val_p_lm_true + 1e-12)

# --- minibatch_kmeans (DIME, tuned config from Phase D step 11) ---
n_clusters = 500
compressed_keys, compressed_dists = minibatch_kmeans_partition(ds_keys, ds_values, n_clusters=n_clusters, seed=42)
dime_index, dime_stored_values = build_datastore(compressed_keys, np.array(compressed_dists, dtype=object))
k_dime, tau_dime, alpha_dime = 20, 2.0, 0.05
distances_dime, retrieved_dime = query_knn(dime_index, dime_stored_values, val_keys, k=k_dime)
_, nll_dime = mix_dime_and_lm(distances_dime, retrieved_dime, val_true_targets, val_p_lm_true, tau=tau_dime, alpha=alpha_dime)

# --- raw_kmeans_representative (Phase E's best equal-budget raw variant) ---
km = MiniBatchKMeans(n_clusters=n_clusters, random_state=42)
assignment = km.fit_predict(ds_keys)
centers = km.cluster_centers_
selected_idx = []
for c in range(n_clusters):
    member_idx = np.where(assignment == c)[0]
    if len(member_idx) == 0:
        continue
    dists = np.linalg.norm(ds_keys[member_idx] - centers[c], axis=1)
    selected_idx.append(member_idx[np.argmin(dists)])
selected_idx = np.array(selected_idx)

raw_rep_index, raw_rep_values = build_datastore(ds_keys[selected_idx], ds_values[selected_idx])
k_raw, tau_raw, alpha_raw = 50, 2.0, 0.1
distances_raw, retrieved_raw = query_knn(raw_rep_index, raw_rep_values, val_keys, k=min(k_raw, len(selected_idx)))
_, nll_raw_rep = mix_knn_and_lm(distances_raw, retrieved_raw, val_true_targets, val_p_lm_true, tau=tau_raw, alpha=alpha_raw)


def paired_tests(nll_a, nll_b, name_a, name_b):
    """Positive mean_diff means a has higher (worse) NLL than b."""
    diff = nll_a - nll_b
    t_stat, t_p = stats.ttest_rel(nll_a, nll_b)
    w_stat, w_p = stats.wilcoxon(nll_a, nll_b)
    return {
        "comparison": f"{name_a} vs {name_b}",
        "mean_diff": float(diff.mean()),
        "paired_t_test": {"t_stat": float(t_stat), "p_value": float(t_p)},
        "wilcoxon_signed_rank": {"stat": float(w_stat), "p_value": float(w_p)},
    }


test_gpt_vs_dime = paired_tests(nll_gpt_only, nll_dime, "GPT-only", "minibatch_kmeans (tuned)")
test_rawrep_vs_dime = paired_tests(nll_raw_rep, nll_dime, "raw_kmeans_representative (equal budget)", "minibatch_kmeans (tuned)")

results = {
    "phase": "F17_significance_testing",
    "model": "gpt2",
    "seq_len": 128,
    "n_val_positions": len(nll_gpt_only),
    "mean_nll": {
        "gpt_only": float(nll_gpt_only.mean()),
        "minibatch_kmeans_tuned": float(nll_dime.mean()),
        "raw_kmeans_representative": float(nll_raw_rep.mean()),
    },
    "significance_tests": [test_gpt_vs_dime, test_rawrep_vs_dime],
}

save_results("results/significance_testing.json", results)
print("mean NLL — GPT-only:", nll_gpt_only.mean())
print("mean NLL — minibatch_kmeans (tuned):", nll_dime.mean())
print("mean NLL — raw_kmeans_representative:", nll_raw_rep.mean())
print()
print("Test 1 (GPT-only vs DIME):", test_gpt_vs_dime)
print()
print("Test 2 (best raw equal-budget vs DIME):", test_rawrep_vs_dime)
