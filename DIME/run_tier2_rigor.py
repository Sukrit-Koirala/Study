import sys, os, argparse, json, pickle, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "GPT_Module"))

from cache_io import load_cache
from knn import build_datastore, query_knn, mix_knn_and_lm
from results_io import save_results
from state_object import minibatch_kmeans_partition
from mixing import mix_dime_and_lm
from q_read import retrieval_purity_entropy, train_q_read_controller
import numpy as np
from collections import Counter

#Safer

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", required=True)
parser.add_argument("--model", required=True)
args = parser.parse_args()

CACHE_DIR = os.path.join(os.path.dirname(__file__), "..", "GPT_Module", "cache")
PREFIX = f"{args.dataset}_{args.model}"

ds = load_cache(os.path.join(CACHE_DIR, f"{PREFIX}_datastore.npz"))
ct = load_cache(os.path.join(CACHE_DIR, f"{PREFIX}_controller_train.npz"))
val = load_cache(os.path.join(CACHE_DIR, f"{PREFIX}_val.npz"))

ds_keys, ds_values = ds["keys"], ds["values"]
ct_keys, ct_values, ct_p_lm_true, ct_entropy = ct["keys"], ct["values"], ct["p_lm_true"], ct["entropy"]
val_keys, val_values, val_p_lm_true, val_entropy = val["keys"], val["values"], val["p_lm_true"], val["entropy"]

with open(f"results/{PREFIX}_tier1.json") as f:
    tier1 = json.load(f)

N_CLUSTERS = tier1["n_clusters"]
RAW_K, RAW_TAU, RAW_ALPHA = tier1["raw_knn"]["best_config"]["k"], tier1["raw_knn"]["best_config"]["tau"], tier1["raw_knn"]["best_config"]["alpha"]
DIME_K, DIME_TAU, DIME_ALPHA = tier1["dime_minibatch_kmeans"]["best_config"]["k"], tier1["dime_minibatch_kmeans"]["best_config"]["tau"], tier1["dime_minibatch_kmeans"]["best_config"]["alpha"]

print(f"[{PREFIX}] using tuned configs: raw k={RAW_K},tau={RAW_TAU},alpha={RAW_ALPHA}  dime k={DIME_K},tau={DIME_TAU},alpha={DIME_ALPHA}")

raw_index, raw_stored_values = build_datastore(ds_keys, ds_values)
compressed_keys, compressed_dists = minibatch_kmeans_partition(ds_keys, ds_values, n_clusters=N_CLUSTERS, seed=42)
dime_index, dime_stored_values = build_datastore(compressed_keys, np.array(compressed_dists, dtype=object))

results = {"phase": "tier2_rigor", "dataset": args.dataset, "model": args.model}
out_path = f"results/{PREFIX}_tier2.json"


# ============================================================
# Section A — binary Q-read (raw + DIME)
# ============================================================
def build_binary_features_and_reward(index, stored_values, mix_fn, k, tau, alpha, keys, true_targets, p_lm_true, entropy):
    distances, retrieved = query_knn(index, stored_values, keys, k=k)
    _, nll_mixed = mix_fn(distances, retrieved, true_targets, p_lm_true, tau=tau, alpha=alpha)
    nll_gpt = -np.log(p_lm_true + 1e-12)
    reward = nll_gpt - nll_mixed
    nearest_distance = distances[:, 0]
    retrieval_entropy, retrieval_purity = retrieval_purity_entropy(retrieved[:, 0])
    X = np.stack([entropy, nearest_distance, retrieval_entropy, retrieval_purity], axis=1)
    return X, reward, nll_gpt, nll_mixed


X_ct_raw, reward_ct_raw, _, _ = build_binary_features_and_reward(raw_index, raw_stored_values, mix_knn_and_lm, RAW_K, RAW_TAU, RAW_ALPHA, ct_keys, ct_values, ct_p_lm_true, ct_entropy)
model_raw = train_q_read_controller(X_ct_raw, reward_ct_raw)
X_val_raw, _, val_nll_gpt_raw, val_nll_mixed_raw = build_binary_features_and_reward(raw_index, raw_stored_values, mix_knn_and_lm, RAW_K, RAW_TAU, RAW_ALPHA, val_keys, val_values, val_p_lm_true, val_entropy)
use_retrieval_raw = model_raw.predict(X_val_raw) > 0
final_nll_raw = np.where(use_retrieval_raw, val_nll_mixed_raw, val_nll_gpt_raw)
print(f"[{PREFIX}] raw Q-read: {float(final_nll_raw.mean()):.4f}  frac_retrieved={float(use_retrieval_raw.mean()):.3f}")

X_ct_dime, reward_ct_dime, _, _ = build_binary_features_and_reward(dime_index, dime_stored_values, mix_dime_and_lm, DIME_K, DIME_TAU, DIME_ALPHA, ct_keys, ct_values, ct_p_lm_true, ct_entropy)
model_dime = train_q_read_controller(X_ct_dime, reward_ct_dime)
X_val_dime, _, val_nll_gpt_dime, val_nll_mixed_dime = build_binary_features_and_reward(dime_index, dime_stored_values, mix_dime_and_lm, DIME_K, DIME_TAU, DIME_ALPHA, val_keys, val_values, val_p_lm_true, val_entropy)
use_retrieval_dime = model_dime.predict(X_val_dime) > 0
final_nll_dime = np.where(use_retrieval_dime, val_nll_mixed_dime, val_nll_gpt_dime)
print(f"[{PREFIX}] DIME Q-read: {float(final_nll_dime.mean()):.4f}  frac_retrieved={float(use_retrieval_dime.mean()):.3f}")

results["binary_q_read"] = {
    "raw_knn": {"mean_nll": float(final_nll_raw.mean()), "frac_retrieved": float(use_retrieval_raw.mean())},
    "dime": {"mean_nll": float(final_nll_dime.mean()), "frac_retrieved": float(use_retrieval_dime.mean())},
}
results["sections_completed"] = ["binary_q_read"]
save_results(out_path, results)


# ============================================================
# Section B — efficiency (real measured bytes/latency)
# ============================================================
raw_total_bytes = ds_keys.nbytes + ds_values.nbytes
dime_total_bytes = compressed_keys.nbytes + len(pickle.dumps(compressed_dists))


def time_retrieval(index, values, query_keys, k, n_warmup=5, n_timed=20, max_queries=500):
    # Latency-per-query only needs a representative sample, not the full val set —
    # timing 20 full passes over hundreds of thousands of queries (WikiText-2's true
    # scale) is what blew through the 4-hour SLURM limit here.
    sample = query_keys[:min(max_queries, len(query_keys))]
    for _ in range(n_warmup):
        query_knn(index, values, sample[:min(100, len(sample))], k=k)
    t0 = time.perf_counter()
    for _ in range(n_timed):
        query_knn(index, values, sample, k=k)
    return ((time.perf_counter() - t0) / n_timed) / len(sample) * 1000


raw_latency_ms = time_retrieval(raw_index, raw_stored_values, val_keys, k=RAW_K)
dime_latency_ms = time_retrieval(dime_index, dime_stored_values, val_keys, k=DIME_K)

results["efficiency"] = {
    "entry_count": {"raw": len(ds_values), "dime": N_CLUSTERS, "ratio": len(ds_values) / N_CLUSTERS},
    "bytes": {"raw_total_mb": raw_total_bytes / 1e6, "dime_total_mb": dime_total_bytes / 1e6, "ratio": raw_total_bytes / dime_total_bytes},
    "latency_ms_per_query": {"raw": raw_latency_ms, "dime": dime_latency_ms, "ratio": raw_latency_ms / dime_latency_ms},
}
print(f"[{PREFIX}] efficiency:", results["efficiency"])
results["sections_completed"].append("efficiency")
save_results(out_path, results)


# ============================================================
# Section C — original 3-variant ablation (illusion-check)
# ============================================================
def evaluate(keys, values_array, index=None):
    if index is None:
        index, _ = build_datastore(keys, values_array)
    distances, retrieved = query_knn(index, values_array, val_keys, k=DIME_K)
    _, nll = mix_dime_and_lm(distances, retrieved, val_values, val_p_lm_true, tau=DIME_TAU, alpha=DIME_ALPHA)
    return float(nll.mean())


majority_dists = []
for d in compressed_dists:
    if len(d) > 0:
        top_tok, _ = d.most_common(1)[0]
        majority_dists.append(Counter({top_tok: sum(d.values())}))
    else:
        majority_dists.append(Counter())
nll_majority = evaluate(compressed_keys, np.array(majority_dists, dtype=object), dime_index)

trunc5_dists = [Counter(dict(d.most_common(5))) for d in compressed_dists]
nll_top5 = evaluate(compressed_keys, np.array(trunc5_dists, dtype=object), dime_index)

rng = np.random.default_rng(42)
perm = rng.permutation(len(compressed_dists))
shuffled_dists = [compressed_dists[i] for i in perm]
nll_shuffled = evaluate(compressed_keys, np.array(shuffled_dists, dtype=object), dime_index)

results["ablation_3variant"] = {"majority_token": nll_majority, "top5": nll_top5, "shuffled": nll_shuffled}
print(f"[{PREFIX}] ablations: majority_token={nll_majority:.4f}  top5={nll_top5:.4f}  shuffled={nll_shuffled:.4f}")
results["sections_completed"].append("ablation_3variant")
save_results(out_path, results)

print(f"\n[{PREFIX}] === TIER 2 DONE ===")
