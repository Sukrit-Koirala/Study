"""Quality vs. storage budget: sweep the compression ratio N/B and, at every B, compare
DIME with equal-budget raw baselines that are each re-tuned at that B.

For every ratio r: B = round(N / r). Everything is tuned on controller_train with the Tier 1 grid
(raw baselines additionally get alpha=0) and evaluated once on val.
  dime                       : minibatch k-means (seed 42) prototypes + token distributions
  raw_kmeans_representative  : same clusters, one real entry per cluster
  raw_random                 : B uniformly random entries
The full-datastore raw kNN ceiling and GPT-only are copied from the Tier 1 JSON.

Use ratios other than the Tier 1 default (100); pass --ratios 1000,300,100,30,10 to include it
as a consistency check (DIME at r=100 must reproduce Tier 1).

Usage: python run_sweep_B.py --dataset tinystories --model gpt2 --ratios 1000,300,30,10
Output: results/{dataset}_{model}_sweep_B.json (saved after every ratio)
"""
import argparse
import numpy as np

from exp_common import (load_setting, load_json, kmeans_fit, build_dists, kmeans_representative,
                        tune_raw_store, tune_dime_store, paired)
from results_io import save_results

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", required=True)
parser.add_argument("--model", required=True)
parser.add_argument("--ratios", default="1000,300,30,10")
args = parser.parse_args()

S = load_setting(args.dataset, args.model)
P = S["prefix"]
tier1 = load_json(f"results/{P}_tier1.json")
ds_keys, ds_values = S["ds"]["keys"], S["ds"]["values"]
N = len(ds_values)

out_path = f"results/{P}_sweep_B.json"
results = {"phase": "sweep_B", "dataset": args.dataset, "model": args.model, "N": int(N),
           "gpt_only_mean_nll": tier1["gpt_only_mean_nll"],
           "raw_knn_full_val_nll": tier1["raw_knn"]["val_mean_nll"],
           "tier1_B": tier1["n_clusters"], "tier1_dime_val_nll": tier1["dime_minibatch_kmeans"]["val_mean_nll"],
           "points": []}

for r in [float(x) for x in args.ratios.split(",")]:
    B = int(round(N / r))
    if B < 10:
        print(f"[{P}] skip ratio {r}: B={B} < 10", flush=True)
        continue
    print(f"[{P}] === ratio {r:g}x  B={B} ===", flush=True)

    assignment, centers = kmeans_fit(ds_keys, B, seed=42)
    dists = build_dists(assignment, ds_values, B)

    dime_best, nll_dime = tune_dime_store(centers, dists, S)
    print(f"[{P}] B={B} DIME {dime_best} -> {float(nll_dime.mean()):.4f}", flush=True)

    sel = kmeans_representative(ds_keys, assignment, centers, B)
    rep_best, nll_rep = tune_raw_store(ds_keys[sel], ds_values[sel], S)
    print(f"[{P}] B={B} raw_kmeans_representative n={len(sel)} {rep_best} -> {float(nll_rep.mean()):.4f}", flush=True)

    rng = np.random.default_rng(42)
    idx_rand = rng.choice(N, size=min(B, N), replace=False)
    rand_best, nll_rand = tune_raw_store(ds_keys[idx_rand], ds_values[idx_rand], S)
    print(f"[{P}] B={B} raw_random {rand_best} -> {float(nll_rand.mean()):.4f}", flush=True)

    best_raw_name, best_raw_nll = min([("raw_kmeans_representative", nll_rep), ("raw_random", nll_rand)],
                                      key=lambda x: float(x[1].mean()))
    results["points"].append({
        "ratio": r, "B": B,
        "dime": {"best_config": dime_best, "val_mean_nll": float(nll_dime.mean())},
        "raw_kmeans_representative": {"n_selected": int(len(sel)), "best_config": rep_best, "val_mean_nll": float(nll_rep.mean())},
        "raw_random": {"n_selected": int(len(idx_rand)), "best_config": rand_best, "val_mean_nll": float(nll_rand.mean())},
        "best_raw_name": best_raw_name,
        "dime_vs_best_raw": paired(best_raw_nll, nll_dime, f"{best_raw_name} vs DIME"),
    })
    save_results(out_path, results)

print(f"\n[{P}] === SWEEP SUMMARY ===  GPT-only {results['gpt_only_mean_nll']:.4f}  raw kNN {results['raw_knn_full_val_nll']:.4f}")
for p in results["points"]:
    print(f"  {p['ratio']:>6g}x B={p['B']:>6d}  DIME {p['dime']['val_mean_nll']:.4f}  "
          f"kmeans_rep {p['raw_kmeans_representative']['val_mean_nll']:.4f}  random {p['raw_random']['val_mean_nll']:.4f}")
