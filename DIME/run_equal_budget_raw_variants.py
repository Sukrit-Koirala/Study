import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "GPT_Module"))

from gpt_class import FrozenGPT2
from dataset_helpers import collect_chunks_split
from extract import run_batch_with_entropy
from knn import build_datastore, query_knn, mix_knn_and_lm
from results_io import save_results
from datasets import load_dataset
import numpy as np
from sklearn.cluster import MiniBatchKMeans

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


ds_keys, ds_values, ds_p_lm, ds_entropy = encode_split_with_entropy(chunks["datastore"])
ds_nll = -np.log(ds_p_lm + 1e-12)

val_keys, val_true_targets, val_p_lm_true, val_entropy = encode_split_with_entropy(chunks["val"])

N = len(ds_values)
BUDGET = 500  # same as DIME's n_clusters — a genuinely equal-budget comparison
k, tau, alpha = 50, 2.0, 0.1  # raw kNN's tuned config from Phase D step 11 (reused as-is, not re-tuned per variant)

rng = np.random.default_rng(42)


def select_kmeans_representative(keys, n_select, seed=42):
    """Cluster like DIME does, but keep the single real raw entry closest to each
    centroid instead of merging into a synthetic prototype — tests diversity-driven
    selection without actually compressing anything."""
    km = MiniBatchKMeans(n_clusters=n_select, random_state=seed)
    assignment = km.fit_predict(keys)
    centers = km.cluster_centers_
    selected_idx = []
    for c in range(n_select):
        member_idx = np.where(assignment == c)[0]
        if len(member_idx) == 0:
            continue
        dists = np.linalg.norm(keys[member_idx] - centers[c], axis=1)
        selected_idx.append(member_idx[np.argmin(dists)])
    return np.array(selected_idx)


variants = {
    "raw_random": rng.choice(N, size=BUDGET, replace=False),
    "raw_high_gpt_loss": np.argsort(-ds_nll)[:BUDGET],
    "raw_low_gpt_loss": np.argsort(ds_nll)[:BUDGET],
    "raw_high_entropy": np.argsort(-ds_entropy)[:BUDGET],
    "raw_low_entropy": np.argsort(ds_entropy)[:BUDGET],
    "raw_kmeans_representative": select_kmeans_representative(ds_keys, BUDGET, seed=42),
}

results = {
    "phase": "E14_equal_budget_raw_variants",
    "model": "gpt2",
    "seq_len": 128,
    "budget": BUDGET,
    "tuned_config": {"k": k, "tau": tau, "alpha": alpha},
    "variants": {},
}

print(f"equal-budget raw variants, budget={BUDGET}")
for name, idx in variants.items():
    sub_keys = ds_keys[idx]
    sub_values = ds_values[idx]
    index, stored_values = build_datastore(sub_keys, sub_values)
    distances, retrieved_values = query_knn(index, stored_values, val_keys, k=min(k, len(idx)))
    _, nll_mixed = mix_knn_and_lm(distances, retrieved_values, val_true_targets, val_p_lm_true, tau=tau, alpha=alpha)
    mean_nll = float(nll_mixed.mean())
    results["variants"][name] = {"n_selected": int(len(idx)), "mean_nll": mean_nll}
    print(f"  {name:28s} n={len(idx):4d}  mean NLL = {mean_nll:.4f}")

save_results("results/equal_budget_raw_variants.json", results)
print("\nreference lines:")
print("  GPT-only:                     2.7984323501586914")
print("  minibatch_kmeans (500, tuned): 2.7451901708278066")
print("  raw kNN (49657, tuned):        2.6940908318605463")
