"""Retrieval latency and storage for raw_full / raw_B / DIME under ONE protocol, v2.

v1 timed sklearn's NearestNeighbors. Results were dominated by fixed per-call overhead that changed
with the node's thread setup (shared node: DIME ~5-10 ms/query; --exclusive node: ~55-85 ms/query;
speed-ups of 22-46x vs 1-4x for the same code), so they were not a reliable measurement.

v2 uses an explicit exact flat search in PyTorch (CPU): squared distances via ||x||^2 - 2 q.x
(the ||q||^2 term does not change the ranking), torch.topk for the k nearest, then the value lookup
(np array for raw stores, object array of Counters for DIME -- the same fetch as query_knn).
Thread count is pinned per measurement (default 1 and 8) so the result does not depend on how many
cores the node exposes. Run it INSIDE the normal 8-CPU allocation, without --exclusive.

Stores: raw_full (all N entries), raw_B (B random entries), dime (B prototypes + Counters).
Protocols: single = one query per call (median/mean per-query ms over --n_single queries);
           batch  = --batch_size queries per call, amortised per-query ms.
Also reports bytes (raw_full, raw_B, DIME pickle, DIME compact sparse layout).

Usage: python run_latency.py --dataset wikitext103 --model gpt2 [--threads 1,8]
Output: results/{dataset}_{model}_latency.json
"""
import argparse, pickle, time
import numpy as np
import torch

from exp_common import load_setting, load_json, kmeans_fit, build_dists
from results_io import save_results

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", required=True)
parser.add_argument("--model", required=True)
parser.add_argument("--threads", default="1,8")
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
queries = torch.from_numpy(np.ascontiguousarray(S["val"]["keys"][: max(args.n_single, args.batch_size)], dtype=np.float32))
print(f"[{P}] N={N} B={B} D={D} raw_k={RAW_K} dime_k={DIME_K} cores_visible={torch.get_num_threads()}", flush=True)

# --- three stores ---
rng = np.random.default_rng(42)
idx_b = rng.choice(N, size=min(B, N), replace=False)
raw_b_keys, raw_b_values = ds_keys[idx_b], ds_values[idx_b]

assignment, centers = kmeans_fit(ds_keys, B, seed=42)
dists = build_dists(assignment, ds_values, B)
dime_keys = centers.astype(np.float32)
dime_values = np.array(dists, dtype=object)


class Store:
    def __init__(self, keys, values):
        self.X = torch.from_numpy(np.ascontiguousarray(keys, dtype=np.float32))
        self.sq = torch.einsum("ij,ij->i", self.X, self.X)
        self.values = values

    def search(self, q, k):
        d = self.sq[None, :] - 2.0 * (q @ self.X.T)
        idx = torch.topk(d, min(k, self.X.shape[0]), dim=1, largest=False).indices.numpy()
        return self.values[idx]  # value fetch included, as in query_knn


stores = {
    "raw_full": (Store(ds_keys, ds_values), RAW_K),
    "raw_B": (Store(raw_b_keys, raw_b_values), RAW_K),
    "dime": (Store(dime_keys, dime_values), DIME_K),
}


def time_single(store, k, n_warmup=10):
    for i in range(n_warmup):
        store.search(queries[i:i + 1], k)
    ts = []
    for i in range(args.n_single):
        t0 = time.perf_counter()
        store.search(queries[i:i + 1], k)
        ts.append(time.perf_counter() - t0)
    return {"mean_ms": float(np.mean(ts) * 1e3), "median_ms": float(np.median(ts) * 1e3)}


def time_batch(store, k, n_warmup=2):
    q = queries[: args.batch_size]
    for _ in range(n_warmup):
        store.search(q, k)
    t0 = time.perf_counter()
    for _ in range(args.n_batch_repeats):
        store.search(q, k)
    return {"per_query_ms": float((time.perf_counter() - t0) / args.n_batch_repeats / len(q) * 1e3)}


latency, ratios = {}, {}
for t in [int(x) for x in args.threads.split(",")]:
    torch.set_num_threads(t)
    lat = {}
    for name, (store, k) in stores.items():
        lat[name] = {"k": min(k, store.X.shape[0]), "n_entries": int(store.X.shape[0]),
                     "single": time_single(store, k), "batch": time_batch(store, k)}
        print(f"[{P}] threads={t} {name}: {lat[name]}", flush=True)
    latency[f"threads_{t}"] = lat
    ratios[f"threads_{t}"] = {
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
    "dime_pickle": dime_pickle_bytes, "dime_sparse": dime_sparse_bytes, "dime_nonzeros": nnz,
    "ratio_raw_full_over_dime_pickle": raw_full_bytes / dime_pickle_bytes,
    "ratio_raw_full_over_dime_sparse": raw_full_bytes / dime_sparse_bytes,
    "dime_sparse_over_raw_B": dime_sparse_bytes / raw_b_bytes,
    "dime_pickle_over_raw_B": dime_pickle_bytes / raw_b_bytes,
}
print(f"[{P}] bytes: {bytes_}", flush=True)

save_results(f"results/{P}_latency.json", {
    "phase": "latency_bytes_v2_torch_flat", "dataset": args.dataset, "model": args.model,
    "N": int(N), "B": int(B), "D": int(D),
    "protocol": {"search": "exact flat search, torch CPU (squared distance via ||x||^2 - 2q.x, topk), value fetch included",
                 "single": f"{args.n_single} queries, one per call, median/mean per-query ms",
                 "batch": f"{args.batch_size} queries per call x {args.n_batch_repeats} repeats, per-query ms",
                 "threads": args.threads},
    "latency": latency, "ratios": ratios, "bytes": bytes_,
})
print(f"[{P}] done", flush=True)
