"""Re-measure retrieval latency and storage under ONE protocol for every setting.

Why: Tier 2's latency column mixed protocols (older runs timed full validation passes; after
commit 321dbe1 a batch of 500 was timed, where fixed per-call overhead dominates DIME's small
index). This script times three stores with identical code:
  raw_full : all N raw entries              (uncompressed kNN-LM datastore)
  raw_B    : B randomly chosen raw entries  (same entry count as DIME)
  dime     : B prototypes + token Counters  (DIME)
under two query protocols:
  single   : one query per call (online serving), per-query wall time
  batch    : 256 queries per call, per-query amortised time
Retrieval = sklearn NearestNeighbors.kneighbors + value lookup (the same query_knn used everywhere).
This is exact CPU brute-force search; ANN indexes (FAISS, HNSW) would change absolute numbers.

Also reports bytes for the same three stores, including a compact sparse layout for DIME's
distributions (the pickle used in Tier 2 carries Python object overhead).

Usage: python run_latency.py --dataset wikitext103 --model gpt2
Output: results/{dataset}_{model}_latency.json
"""
import argparse, os, pickle, time
import numpy as np

from exp_common import (load_setting, load_json, kmeans_fit, build_dists, build_datastore, query_knn)
from results_io import save_results

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", required=True)
parser.add_argument("--model", required=True)
parser.add_argument("--n_single", type=int, default=200)
parser.add_argument("--batch_size", type=int, default=256)
parser.add_argument("--n_batch_repeats", type=int, default=5)
args = parser.parse_args()

S = load_setting(args.dataset, args.model)
P = S["prefix"]
tier1 = load_json(f"results/{P}_tier1.json")
B = tier1["n_clusters"]
RAW_K = tier1["raw_knn"]["best_config"]["k"]
DIME_K = tier1["dime_minibatch_kmeans"]["best_config"]["k"]

ds_keys, ds_values = S["ds"]["keys"], S["ds"]["values"]
N, D = ds_keys.shape
queries = S["val"]["keys"][: max(args.n_single, args.batch_size)]
print(f"[{P}] N={N} B={B} D={D} raw_k={RAW_K} dime_k={DIME_K}", flush=True)

# --- three stores ---
raw_full = build_datastore(ds_keys, ds_values)

rng = np.random.default_rng(42)
idx_b = rng.choice(N, size=min(B, N), replace=False)
raw_b_keys, raw_b_values = ds_keys[idx_b], ds_values[idx_b]
raw_b = build_datastore(raw_b_keys, raw_b_values)

assignment, centers = kmeans_fit(ds_keys, B, seed=42)
dists = build_dists(assignment, ds_values, B)
dime_keys = centers.astype(np.float32)
dime = build_datastore(dime_keys, np.array(dists, dtype=object))


def time_single(store, k, n_warmup=10):
    index, values = store
    k = min(k, len(values))
    for i in range(n_warmup):
        query_knn(index, values, queries[i:i + 1], k)
    ts = []
    for i in range(args.n_single):
        t0 = time.perf_counter()
        query_knn(index, values, queries[i:i + 1], k)
        ts.append(time.perf_counter() - t0)
    return {"mean_ms": float(np.mean(ts) * 1e3), "median_ms": float(np.median(ts) * 1e3)}


def time_batch(store, k, n_warmup=2):
    index, values = store
    k = min(k, len(values))
    q = queries[: args.batch_size]
    for _ in range(n_warmup):
        query_knn(index, values, q, k)
    t0 = time.perf_counter()
    for _ in range(args.n_batch_repeats):
        query_knn(index, values, q, k)
    return {"per_query_ms": float((time.perf_counter() - t0) / args.n_batch_repeats / len(q) * 1e3)}


lat = {}
for name, store, k in (("raw_full", raw_full, RAW_K), ("raw_B", raw_b, RAW_K), ("dime", dime, DIME_K)):
    lat[name] = {"k": min(k, len(store[1])), "n_entries": int(len(store[1])),
                 "single": time_single(store, k), "batch": time_batch(store, k)}
    print(f"[{P}] {name}: {lat[name]}", flush=True)

ratios = {
    "single_median_raw_full_over_dime": lat["raw_full"]["single"]["median_ms"] / lat["dime"]["single"]["median_ms"],
    "batch_raw_full_over_dime": lat["raw_full"]["batch"]["per_query_ms"] / lat["dime"]["batch"]["per_query_ms"],
    "single_median_raw_B_over_dime": lat["raw_B"]["single"]["median_ms"] / lat["dime"]["single"]["median_ms"],
    "batch_raw_B_over_dime": lat["raw_B"]["batch"]["per_query_ms"] / lat["dime"]["batch"]["per_query_ms"],
}

# --- bytes ---
raw_full_bytes = ds_keys.nbytes + ds_values.nbytes
raw_b_bytes = raw_b_keys.nbytes + raw_b_values.nbytes
nnz = int(sum(len(c) for c in dists))
dime_pickle_bytes = dime_keys.nbytes + len(pickle.dumps(dists))
dime_sparse_bytes = dime_keys.nbytes + nnz * (4 + 4) + (B + 1) * 4  # int32 token id + int32 count per nonzero, int32 offsets
bytes_ = {
    "raw_full": raw_full_bytes, "raw_B": raw_b_bytes,
    "dime_pickle": dime_pickle_bytes, "dime_sparse": dime_sparse_bytes,
    "dime_nonzeros": nnz,
    "ratio_raw_full_over_dime_pickle": raw_full_bytes / dime_pickle_bytes,
    "ratio_raw_full_over_dime_sparse": raw_full_bytes / dime_sparse_bytes,
    "dime_sparse_over_raw_B": dime_sparse_bytes / raw_b_bytes,
    "dime_pickle_over_raw_B": dime_pickle_bytes / raw_b_bytes,
}
print(f"[{P}] bytes: {bytes_}", flush=True)

save_results(f"results/{P}_latency.json", {
    "phase": "latency_bytes_single_protocol", "dataset": args.dataset, "model": args.model,
    "N": int(N), "B": int(B), "D": int(D),
    "protocol": {"single": f"{args.n_single} queries, one per call, median/mean per-query ms",
                 "batch": f"{args.batch_size} queries per call x {args.n_batch_repeats} repeats, per-query ms",
                 "search": "sklearn NearestNeighbors brute force, CPU"},
    "latency": lat, "ratios": ratios, "bytes": bytes_,
})
print(f"[{P}] done", flush=True)
