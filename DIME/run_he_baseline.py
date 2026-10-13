"""Baseline: k-means datastore pruning in the style of He et al. (2021), "Efficient Nearest
Neighbor Language Models" -- the closest published cluster-based compression.

Store: for every cluster c and every distinct next-token v seen in it, one entry
(centroid_c, count s_{c,v}, token v). So storage is T = #(cluster, token) pairs, not #clusters.
Retrieval: the k nearest ENTRIES to the query (entries of one cluster share a centroid), scored
    P(y | q) proportional to  sum over retrieved entries with token y of  s * exp(-d / tau)
and mixed with the LM as in kNN-LM. NOTE: implemented from a summary of their method; check the
weighting against the paper before describing it as a faithful reproduction.

Two operating points, each tuned on controller_train (Tier 1 grid, alpha=0 allowed):
  matched : number of clusters C chosen so that T ~= B   (the same ENTRY budget as DIME)
  their   : C chosen so that T ~= N / 5                  (the ~5x compression regime they report)
Plus the same-cluster-count point C = B (uses T >= B entries) for reference.
DIME's and the best re-tuned raw baseline's val NLL are copied in for side-by-side comparison.

Usage: python run_he_baseline.py --dataset tinystories --model gpt2
Output: results/{dataset}_{model}_he_baseline.json
"""
import argparse, os
import numpy as np

from exp_common import (load_setting, load_json, kmeans_fit, k_grid_for, TAU_GRID, RAW_ALPHA_GRID,
                        build_datastore, query_knn_indices, paired)
from results_io import save_results

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", required=True)
parser.add_argument("--model", required=True)
parser.add_argument("--tol", type=float, default=0.08, help="relative tolerance when matching an entry budget")
parser.add_argument("--max_fits", type=int, default=10)
args = parser.parse_args()

S = load_setting(args.dataset, args.model)
P = S["prefix"]
tier1 = load_json(f"results/{P}_tier1.json")
B = tier1["n_clusters"]
ds_keys, ds_values = S["ds"]["keys"], S["ds"]["values"]
ct, val = S["ct"], S["val"]
N, D = ds_keys.shape
V = int(ds_values.max()) + 1

_fit_cache = {}


def fit(C):
    if C not in _fit_cache:
        _fit_cache[C] = kmeans_fit(ds_keys, C, seed=42)
    return _fit_cache[C]


def build_triples(assignment, centers):
    """Distinct (cluster, token) pairs -> keys [T,D], token [T], count [T]."""
    code = assignment.astype(np.int64) * V + ds_values.astype(np.int64)
    uniq, counts = np.unique(code, return_counts=True)
    cl, tok = uniq // V, uniq % V
    return centers[cl].astype(np.float32), tok.astype(ds_values.dtype), counts.astype(np.float64)


def n_triples(C):
    a, _ = fit(C)
    return len(np.unique(a.astype(np.int64) * V + ds_values.astype(np.int64)))


def match_budget(target, lo=2, hi=None):
    """Geometric bisection on C so that T(C) ~= target (T grows with C)."""
    hi = hi or N
    best = None
    for _ in range(args.max_fits):
        C = int(round(np.sqrt(lo * hi)))
        C = max(2, min(C, N))
        T = n_triples(C)
        if best is None or abs(T - target) < abs(best[1] - target):
            best = (C, T)
        if abs(T - target) <= args.tol * target or hi - lo <= 1:
            break
        if T < target:
            lo = C
        else:
            hi = C
    return best


def he_nll(dist, vals, cnts, true, p_lm, k, tau, alpha):
    d = dist[:, :k]
    w = cnts[:, :k] * np.exp(-(d - d.min(axis=1, keepdims=True)) / tau)
    p_knn = (w * (vals[:, :k] == true[:, None])).sum(axis=1) / w.sum(axis=1)
    return -np.log(alpha * p_knn + (1 - alpha) * p_lm + 1e-12)


def run_point(C, label):
    a, centers = fit(C)
    keys, toks, cnts = build_triples(a, centers)
    T = len(toks)
    index, _ = build_datastore(keys, toks)
    kg = k_grid_for(T)
    d_ct, i_ct = query_knn_indices(index, ct["keys"], k=max(kg))
    v_ct, c_ct = toks[i_ct], cnts[i_ct]
    best = None
    for k in kg:
        for tau in TAU_GRID:
            for alpha in RAW_ALPHA_GRID:
                m = float(he_nll(d_ct, v_ct, c_ct, ct["values"], ct["p_lm_true"], k, tau, alpha).mean())
                if best is None or m < best["mean_nll"]:
                    best = {"k": k, "tau": tau, "alpha": alpha, "mean_nll": m}
    d_v, i_v = query_knn_indices(index, val["keys"], k=best["k"])
    nll = he_nll(d_v, toks[i_v], cnts[i_v], val["values"], val["p_lm_true"], best["k"], best["tau"], best["alpha"])
    out = {"label": label, "n_clusters": int(C), "n_entries": int(T), "entry_ratio_N_over_T": N / T,
           "bytes": int(keys.nbytes + toks.nbytes + cnts.astype(np.float32).nbytes),
           "best_config": best, "val_mean_nll": float(nll.mean())}
    print(f"[{P}] He et al. {label}: C={C} T={T} -> {out['val_mean_nll']:.4f}  {best}", flush=True)
    return out, nll


out_path = f"results/{P}_he_baseline.json"
results = {"phase": "he_baseline", "dataset": args.dataset, "model": args.model, "N": int(N), "B": int(B),
           "gpt_only_mean_nll": tier1["gpt_only_mean_nll"],
           "dime_val_nll": tier1["dime_minibatch_kmeans"]["val_mean_nll"], "points": []}
retune_path = f"results/{P}_raw_retune.json"
if os.path.exists(retune_path):
    rt = load_json(retune_path)
    best_name = rt.get("best_strategy_by_controller_train")
    if best_name:
        results["best_retuned_raw"] = {"name": best_name, "val_mean_nll": rt["strategies"][best_name]["val_mean_nll_retuned"]}

# same cluster count as DIME (uses >= B entries)
p, _ = run_point(B, "same_clusters_as_DIME")
results["points"].append(p)
save_results(out_path, results)

# entry-matched to DIME. With ONE cluster the triple store already holds one entry per distinct
# next-token, so T can never go below the number of distinct tokens in the datastore. If that floor
# is above B, an entry-matched He-style store does not exist -- record that instead of running it.
T_floor = int(len(np.unique(ds_values)))
results["min_entries_any_C"] = T_floor
if T_floor > B * (1 + args.tol):
    results["entry_matched"] = {"feasible": False, "min_entries": T_floor, "B": int(B),
                                "floor_over_B": T_floor / B}
    print(f"[{P}] entry-matched He store infeasible: floor T={T_floor} (distinct tokens) > B={B} "
          f"({T_floor / B:.1f}x)", flush=True)
else:
    C_m, T_m = match_budget(B, lo=1)
    p, _ = run_point(C_m, f"entry_matched_T~{B}")
    results["points"].append(p)
    results["entry_matched"] = {"feasible": True}
save_results(out_path, results)

# their ~5x regime
C_t, T_t = match_budget(int(N / 5))
p, _ = run_point(C_t, "their_regime_T~N/5")
results["points"].append(p)
save_results(out_path, results)

print(f"\n[{P}] === HE BASELINE SUMMARY ===  DIME {results['dime_val_nll']:.4f}  GPT-only {results['gpt_only_mean_nll']:.4f}")
for q in results["points"]:
    print(f"  {q['label']:28s} C={q['n_clusters']:>6d} T={q['n_entries']:>7d}  val {q['val_mean_nll']:.4f}")
