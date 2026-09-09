import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "GPT_Module"))

from cache_io import load_cache
from knn import build_datastore, query_knn, mix_knn_and_lm
from results_io import save_results
import numpy as np
import pickle, time
from collections import Counter
from sklearn.cluster import MiniBatchKMeans

from state_object import minibatch_kmeans_partition, random_partition, utility_weighted_partition, query_kmeans_partition
from mixing import mix_dime_and_lm, build_global_freq
from q_read import retrieval_purity_entropy, train_q_read_controller

VOCAB_SIZE = 50257
CACHE_DIR = os.path.join(os.path.dirname(__file__), "..", "GPT_Module", "cache")
PREFIX = "wikitext103_gpt2-medium"

ds = load_cache(os.path.join(CACHE_DIR, f"{PREFIX}_datastore.npz"))
ct = load_cache(os.path.join(CACHE_DIR, f"{PREFIX}_controller_train.npz"))
val = load_cache(os.path.join(CACHE_DIR, f"{PREFIX}_val.npz"))

ds_keys, ds_values, ds_p_lm, ds_entropy = ds["keys"], ds["values"], ds["p_lm_true"], ds["entropy"]
ct_keys, ct_values, ct_p_lm, ct_entropy = ct["keys"], ct["values"], ct["p_lm_true"], ct["entropy"]
val_keys, val_values, val_p_lm, val_entropy = val["keys"], val["values"], val["p_lm_true"], val["entropy"]

ds_nll = -np.log(ds_p_lm + 1e-12)
nll_gpt_only_val = -np.log(val_p_lm + 1e-12)

print(f"datastore: {len(ds_values)}  controller_train: {len(ct_values)}  val: {len(val_values)}")

# Established tuned configs from the WikiText-103 replication (round 2)
RAW_K, RAW_TAU, RAW_ALPHA = 300, 1.0, 0.25
DIME_K, DIME_TAU, DIME_ALPHA = 200, 2.0, 0.1
N_CLUSTERS = 4000  # DIME's established WikiText-103 budget — also used for all equal-budget raw variants

raw_index, raw_stored_values = build_datastore(ds_keys, ds_values)

results = {"phase": "wikitext_extras"}
out_path = "results/wikitext_extras.json"
save_results(out_path, results)


# ============================================================
# Section A — binary Q-read on raw kNN and on tuned DIME
# ============================================================
print("\n=== Section A: binary Q-read (raw + DIME) ===")

compressed_keys, compressed_dists = minibatch_kmeans_partition(ds_keys, ds_values, n_clusters=N_CLUSTERS, seed=42)
dime_index, dime_stored_values = build_datastore(compressed_keys, np.array(compressed_dists, dtype=object))


def build_binary_features_and_reward(index, stored_values, mix_fn, k, tau, alpha, keys, true_targets, p_lm_true, entropy):
    distances, retrieved = query_knn(index, stored_values, keys, k=k)
    _, nll_mixed = mix_fn(distances, retrieved, true_targets, p_lm_true, tau=tau, alpha=alpha)
    nll_gpt = -np.log(p_lm_true + 1e-12)
    reward = nll_gpt - nll_mixed
    nearest_distance = distances[:, 0]
    retrieved_top1 = retrieved[:, 0]
    retrieval_entropy, retrieval_purity = retrieval_purity_entropy(retrieved_top1)
    X = np.stack([entropy, nearest_distance, retrieval_entropy, retrieval_purity], axis=1)
    return X, reward, nll_gpt, nll_mixed


# Raw kNN Q-read
X_ct_raw, reward_ct_raw, _, _ = build_binary_features_and_reward(
    raw_index, raw_stored_values, mix_knn_and_lm, RAW_K, RAW_TAU, RAW_ALPHA, ct_keys, ct_values, ct_p_lm, ct_entropy)
model_raw = train_q_read_controller(X_ct_raw, reward_ct_raw)
X_val_raw, _, val_nll_gpt_raw, val_nll_mixed_raw = build_binary_features_and_reward(
    raw_index, raw_stored_values, mix_knn_and_lm, RAW_K, RAW_TAU, RAW_ALPHA, val_keys, val_values, val_p_lm, val_entropy)
pred_reward_raw = model_raw.predict(X_val_raw)
use_retrieval_raw = pred_reward_raw > 0
final_nll_raw = np.where(use_retrieval_raw, val_nll_mixed_raw, val_nll_gpt_raw)
print("raw kNN Q-read: mean NLL =", float(final_nll_raw.mean()), " frac_retrieved =", float(use_retrieval_raw.mean()))

# DIME Q-read
X_ct_dime, reward_ct_dime, _, _ = build_binary_features_and_reward(
    dime_index, dime_stored_values, mix_dime_and_lm, DIME_K, DIME_TAU, DIME_ALPHA, ct_keys, ct_values, ct_p_lm, ct_entropy)
model_dime = train_q_read_controller(X_ct_dime, reward_ct_dime)
X_val_dime, _, val_nll_gpt_dime, val_nll_mixed_dime = build_binary_features_and_reward(
    dime_index, dime_stored_values, mix_dime_and_lm, DIME_K, DIME_TAU, DIME_ALPHA, val_keys, val_values, val_p_lm, val_entropy)
pred_reward_dime = model_dime.predict(X_val_dime)
use_retrieval_dime = pred_reward_dime > 0
final_nll_dime = np.where(use_retrieval_dime, val_nll_mixed_dime, val_nll_gpt_dime)
print("DIME Q-read: mean NLL =", float(final_nll_dime.mean()), " frac_retrieved =", float(use_retrieval_dime.mean()))

results["binary_q_read"] = {
    "raw_knn": {"mean_nll": float(final_nll_raw.mean()), "frac_retrieved": float(use_retrieval_raw.mean())},
    "dime": {"mean_nll": float(final_nll_dime.mean()), "frac_retrieved": float(use_retrieval_dime.mean())},
    "gpt_only_reference": float(nll_gpt_only_val.mean()),
}
results["sections_completed"] = ["binary_q_read"]
save_results(out_path, results)


# ============================================================
# Section B — remaining 3 DIME construction methods (at DIME's tuned config)
# ============================================================
print("\n=== Section B: remaining construction methods ===")

def eval_dime_variant(keys, dists):
    index, stored = build_datastore(keys, np.array(dists, dtype=object))
    distances, retrieved = query_knn(index, stored, val_keys, k=DIME_K)
    _, nll = mix_dime_and_lm(distances, retrieved, val_values, val_p_lm, tau=DIME_TAU, alpha=DIME_ALPHA)
    return float(nll.mean())


rp_keys, rp_dists = random_partition(ds_keys, ds_values, n_clusters=N_CLUSTERS, seed=42)
nll_random_partition = eval_dime_variant(rp_keys, rp_dists)
print("random_partition:", nll_random_partition)

uw_keys, uw_dists = utility_weighted_partition(ds_keys, ds_values, ds_nll, n_clusters=N_CLUSTERS, seed=42)
nll_utility_weighted = eval_dime_variant(uw_keys, uw_dists)
print("utility_weighted:", nll_utility_weighted)

qk_keys, qk_dists = query_kmeans_partition(ds_keys, ds_values, ct_keys, n_clusters=N_CLUSTERS, seed=42)
nll_query_kmeans = eval_dime_variant(qk_keys, qk_dists)
print("query_kmeans:", nll_query_kmeans)

results["construction_methods"] = {
    "minibatch_kmeans_reference": float(mix_dime_and_lm(*query_knn(dime_index, dime_stored_values, val_keys, k=DIME_K), val_values, val_p_lm, tau=DIME_TAU, alpha=DIME_ALPHA)[1].mean()),
    "random_partition": nll_random_partition,
    "utility_weighted": nll_utility_weighted,
    "query_kmeans": nll_query_kmeans,
}
results["sections_completed"].append("construction_methods")
save_results(out_path, results)


# ============================================================
# Section C — remaining equal-budget raw variants + Point 6
# ============================================================
print("\n=== Section C: remaining equal-budget raw variants + Point 6 ===")

rng = np.random.default_rng(42)


def eval_raw_subset(idx):
    sub_keys, sub_values = ds_keys[idx], ds_values[idx]
    index, stored = build_datastore(sub_keys, sub_values)
    distances, retrieved = query_knn(index, stored, val_keys, k=min(RAW_K, len(idx)))
    _, nll = mix_knn_and_lm(distances, retrieved, val_values, val_p_lm, tau=RAW_TAU, alpha=RAW_ALPHA)
    return float(nll.mean())


variant_results = {}

variant_results["raw_random"] = eval_raw_subset(rng.choice(len(ds_values), size=N_CLUSTERS, replace=False))
print("raw_random:", variant_results["raw_random"])

variant_results["raw_high_gpt_loss"] = eval_raw_subset(np.argsort(-ds_nll)[:N_CLUSTERS])
print("raw_high_gpt_loss:", variant_results["raw_high_gpt_loss"])

variant_results["raw_low_gpt_loss"] = eval_raw_subset(np.argsort(ds_nll)[:N_CLUSTERS])
print("raw_low_gpt_loss:", variant_results["raw_low_gpt_loss"])

variant_results["raw_high_entropy"] = eval_raw_subset(np.argsort(-ds_entropy)[:N_CLUSTERS])
print("raw_high_entropy:", variant_results["raw_high_entropy"])

variant_results["raw_low_entropy"] = eval_raw_subset(np.argsort(ds_entropy)[:N_CLUSTERS])
print("raw_low_entropy:", variant_results["raw_low_entropy"])

global_freq = build_global_freq(ds_values, VOCAB_SIZE)
token_freq_per_position = global_freq[ds_values]
variant_results["raw_token_rarity"] = eval_raw_subset(np.argsort(token_freq_per_position)[:N_CLUSTERS])
print("raw_token_rarity:", variant_results["raw_token_rarity"])

results["equal_budget_variants_partial"] = variant_results
results["sections_completed"].append("equal_budget_variants_partial")
save_results(out_path, results)

# raw_coverage: greedy farthest-first — the expensive one, checkpointed on its own
print("computing raw_coverage (greedy farthest-first, this is the slow one)...")
t0 = time.time()
N_ds = len(ds_keys)
start = int(rng.integers(N_ds))
selected = [start]
min_dist = np.linalg.norm(ds_keys - ds_keys[start], axis=1)
min_dist[start] = -1
for step in range(N_CLUSTERS - 1):
    next_idx = int(np.argmax(min_dist))
    selected.append(next_idx)
    new_dist = np.linalg.norm(ds_keys - ds_keys[next_idx], axis=1)
    min_dist = np.minimum(min_dist, new_dist)
    min_dist[next_idx] = -1
    if step % 500 == 0:
        print(f"  coverage selection: {step}/{N_CLUSTERS-1}  elapsed={time.time()-t0:.1f}s", flush=True)
coverage_idx = np.array(selected)
print(f"coverage selection done in {time.time()-t0:.1f}s")

variant_results["raw_coverage"] = eval_raw_subset(coverage_idx)
print("raw_coverage:", variant_results["raw_coverage"])

results["equal_budget_variants_partial"] = variant_results
results["sections_completed"].append("raw_coverage")
save_results(out_path, results)


# ============================================================
# Section D — efficiency analysis (real measured bytes/latency)
# ============================================================
print("\n=== Section D: efficiency analysis ===")

raw_keys_bytes = ds_keys.nbytes
raw_values_bytes = ds_values.nbytes
raw_total_bytes = raw_keys_bytes + raw_values_bytes

dime_keys_bytes = compressed_keys.nbytes
dime_values_bytes = len(pickle.dumps(compressed_dists))
dime_total_bytes = dime_keys_bytes + dime_values_bytes


def time_retrieval(index, values, query_keys, k, n_warmup=5, n_timed=20):
    for _ in range(n_warmup):
        query_knn(index, values, query_keys[:100], k=k)
    start_t = time.perf_counter()
    for _ in range(n_timed):
        query_knn(index, values, query_keys, k=k)
    elapsed = time.perf_counter() - start_t
    return (elapsed / n_timed) / len(query_keys) * 1000


raw_latency_ms = time_retrieval(raw_index, raw_stored_values, val_keys, k=RAW_K)
dime_latency_ms = time_retrieval(dime_index, dime_stored_values, val_keys, k=DIME_K)

results["efficiency"] = {
    "entry_count": {"raw": len(ds_values), "dime": N_CLUSTERS, "ratio": len(ds_values) / N_CLUSTERS},
    "bytes": {"raw_total_mb": raw_total_bytes / 1e6, "dime_total_mb": dime_total_bytes / 1e6,
              "ratio": raw_total_bytes / dime_total_bytes},
    "latency_ms_per_query": {"raw": raw_latency_ms, "dime": dime_latency_ms, "ratio": raw_latency_ms / dime_latency_ms},
}
results["sections_completed"].append("efficiency")
save_results(out_path, results)

print("\n=== ALL SECTIONS DONE ===")
print(results["efficiency"])
