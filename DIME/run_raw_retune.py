"""Re-tune the equal-budget raw baselines at their own size B.

Tier 1 / Tier 3 evaluated every equal-budget raw baseline with the (k, tau, alpha)
tuned for the FULL uncompressed datastore, while DIME got a configuration tuned at
B entries. Here each raw selection strategy gets the same treatment as DIME: grid
search on controller_train (same grid as Tier 1), then a single evaluation on val.
The selected entries are identical to the earlier runs (same seeds, same rng call
order); only the hyperparameters change.

Also recomputes DIME's val NLL with the Tier 1 config as a sanity check, and runs
the paired tests of DIME against the best re-tuned strategy. "Best" is chosen on
controller_train NLL, never on val.

Requires results/{dataset}_{model}_tier1.json. Optional: {..}_tier3.json (only used
to report the old, un-retuned numbers side by side).

Usage: python run_raw_retune.py --dataset wikitext103 --model gpt2 [--skip_coverage]
Output: results/{dataset}_{model}_raw_retune.json
"""
import sys, os, argparse, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "GPT_Module"))

from cache_io import load_cache
from knn import build_datastore, query_knn, mix_knn_and_lm
from results_io import save_results
from grid_search import grid_search_hyperparams
from mixing import mix_dime_and_lm, build_global_freq
import numpy as np
import torch
from collections import Counter
from scipy import stats
from sklearn.cluster import MiniBatchKMeans

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", required=True)
parser.add_argument("--model", required=True)
parser.add_argument("--skip_coverage", action="store_true", help="skip raw_coverage (greedy farthest-first)")
args = parser.parse_args()

CACHE_DIR = os.path.join(os.path.dirname(__file__), "..", "GPT_Module", "cache")
PREFIX = f"{args.dataset}_{args.model}"

ds = load_cache(os.path.join(CACHE_DIR, f"{PREFIX}_datastore.npz"))
ct = load_cache(os.path.join(CACHE_DIR, f"{PREFIX}_controller_train.npz"))
val = load_cache(os.path.join(CACHE_DIR, f"{PREFIX}_val.npz"))

ds_keys, ds_values, ds_p_lm_true, ds_entropy = ds["keys"], ds["values"], ds["p_lm_true"], ds["entropy"]
ct_keys, ct_values, ct_p_lm_true = ct["keys"], ct["values"], ct["p_lm_true"]
val_keys, val_values, val_p_lm_true = val["keys"], val["values"], val["p_lm_true"]

with open(f"results/{PREFIX}_tier1.json") as f:
    tier1 = json.load(f)
tier3 = {}
if os.path.exists(f"results/{PREFIX}_tier3.json"):
    with open(f"results/{PREFIX}_tier3.json") as f:
        tier3 = json.load(f)

B = tier1["n_clusters"]
N_total = len(ds_values)
DIME_CFG = tier1["dime_minibatch_kmeans"]["best_config"]
print(f"[{PREFIX}] N={N_total}  B={B}  ct={len(ct_values)}  val={len(val_values)}")

# Same grid as run_tier1_core.py, so raw and DIME are tuned over identical ranges.
K_GRID = [10, 20, 50, 100, 200, 300]
TAU_GRID = [0.5, 1.0, 2.0, 5.0, 10.0]
ALPHA_GRID = [0.01, 0.05, 0.1, 0.25, 0.5]
# A raw baseline can always switch retrieval off (alpha=0 == GPT-only). Without it, a selection that
# carries no signal lands on the alpha=0.01 grid floor and scores ~GPT-only + 0.01 (-log 0.99), which
# understates that baseline. DIME's tuned alpha is interior, so adding 0 does not change DIME's result.
RAW_ALPHA_GRID = [0.0] + ALPHA_GRID

out_path = f"results/{PREFIX}_raw_retune.json"
results = {"phase": "raw_retune", "dataset": args.dataset, "model": args.model, "n_clusters": B,
           "grid": {"k": K_GRID, "tau": TAU_GRID, "alpha": RAW_ALPHA_GRID},
           "gpt_only_mean_nll": float((-np.log(val_p_lm_true + 1e-12)).mean()),
           "strategies": {}}

# ---- one clustering fit, shared by DIME and raw_kmeans_representative ----
# Same call as minibatch_kmeans_partition(seed=42) and run_tier1_core.py.
km = MiniBatchKMeans(n_clusters=B, random_state=42)
assignment = km.fit_predict(ds_keys)
centers = km.cluster_centers_

# ---- DIME with the Tier 1 config: sanity check + per-position NLL for paired tests ----
dime_keys = centers.astype(np.float32)
dime_dists = [Counter(ds_values[assignment == c].tolist()) for c in range(B)]
dime_index, dime_stored = build_datastore(dime_keys, np.array(dime_dists, dtype=object))
d_dime, r_dime = query_knn(dime_index, dime_stored, val_keys, k=min(DIME_CFG["k"], B))
_, nll_dime = mix_dime_and_lm(d_dime, r_dime, val_values, val_p_lm_true, tau=DIME_CFG["tau"], alpha=DIME_CFG["alpha"])
dime_val = float(nll_dime.mean())
print(f"[{PREFIX}] DIME (tier1 config) val NLL recomputed: {dime_val:.4f}  "
      f"(tier1 stored: {tier1['dime_minibatch_kmeans']['val_mean_nll']:.4f})")
results["dime"] = {"config": DIME_CFG, "val_mean_nll_recomputed": dime_val,
                   "val_mean_nll_tier1": tier1["dime_minibatch_kmeans"]["val_mean_nll"]}
save_results(out_path, results)


# ---- selection strategies (identical to run_tier1_core.py / run_tier3_extras.py) ----
def select_kmeans_representative():
    selected = []
    for c in range(B):
        members = np.where(assignment == c)[0]
        if len(members) == 0:
            continue
        d = np.linalg.norm(ds_keys[members] - centers[c], axis=1)
        selected.append(members[np.argmin(d)])
    return np.array(selected)


def farthest_first(keys, n_select, start_idx):
    """Greedy farthest-first traversal, on GPU when available (the numpy version took
    ~15h extrapolated at WikiText-2 scale). Uses squared distances via ||x||^2 - 2x.y
    + ||y||^2; selection can differ from the numpy version only on float ties."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    X = torch.from_numpy(np.ascontiguousarray(keys, dtype=np.float32)).to(device)
    sq = torch.einsum("ij,ij->i", X, X)
    selected = [int(start_idx)]
    min_d = (sq - 2 * (X @ X[selected[0]]) + sq[selected[0]]).clamp_(min=0)
    for step in range(1, n_select):
        nxt = int(torch.argmax(min_d))
        selected.append(nxt)
        d = (sq - 2 * (X @ X[nxt]) + sq[nxt]).clamp_(min=0)
        min_d = torch.minimum(min_d, d)
        if step % 2000 == 0:
            print(f"[{PREFIX}] coverage selection: {step}/{n_select}", flush=True)
    return np.array(selected)


n_pick = min(B, N_total)
rng = np.random.default_rng(42)
idx_random = rng.choice(N_total, size=n_pick, replace=False)  # rng call order matters: matches run_tier3_extras.py

order_by_loss = np.argsort(-np.log(ds_p_lm_true + 1e-12))
order_by_entropy = np.argsort(ds_entropy)
global_freq = build_global_freq(ds_values)
order_by_rarity = np.argsort(np.log(global_freq[ds_values] + 1e-12))  # rarest next-token first

selectors = {
    "raw_kmeans_representative": select_kmeans_representative,
    "raw_random": lambda: idx_random,
    "raw_high_gpt_loss": lambda: order_by_loss[-n_pick:],
    "raw_low_gpt_loss": lambda: order_by_loss[:n_pick],
    "raw_high_entropy": lambda: order_by_entropy[-n_pick:],
    "raw_low_entropy": lambda: order_by_entropy[:n_pick],
    "raw_token_rarity": lambda: order_by_rarity[:n_pick],
}
if not args.skip_coverage:
    selectors["raw_coverage"] = lambda: farthest_first(ds_keys, n_pick, rng.integers(0, N_total))


def old_val_nll(name):
    if name == "raw_kmeans_representative":
        return tier1["raw_kmeans_representative"]["val_mean_nll"]
    return tier3.get("raw_baseline_variants", {}).get(name)


val_nll_arrays = {}
for name, select in selectors.items():
    idx = np.asarray(select())
    size = len(idx)
    index, stored = build_datastore(ds_keys[idx], ds_values[idx])
    k_grid = [k for k in K_GRID if k <= size] or [min(size, 10)]

    d_ct, r_ct = query_knn(index, stored, ct_keys, k=max(k_grid))
    _, best = grid_search_hyperparams(d_ct, r_ct, ct_values, ct_p_lm_true, mix_knn_and_lm,
                                      k_grid, TAU_GRID, RAW_ALPHA_GRID)
    d_val, r_val = query_knn(index, stored, val_keys, k=best["k"])
    _, nll_val = mix_knn_and_lm(d_val, r_val, val_values, val_p_lm_true, tau=best["tau"], alpha=best["alpha"])
    val_nll_arrays[name] = nll_val

    results["strategies"][name] = {
        "n_selected": int(size),
        "best_config": best,                       # includes controller_train mean_nll
        "val_mean_nll_retuned": float(nll_val.mean()),
        "val_mean_nll_original": old_val_nll(name),  # evaluated with full-datastore config
    }
    print(f"[{PREFIX}] {name}: n={size}  best={best}  val retuned={float(nll_val.mean()):.4f}  "
          f"original={old_val_nll(name)}", flush=True)
    save_results(out_path, results)

# ---- best strategy chosen on controller_train, then paired tests against DIME on val ----
best_name = min(results["strategies"], key=lambda n: results["strategies"][n]["best_config"]["mean_nll"])
nll_best = val_nll_arrays[best_name]
gpt_only = -np.log(val_p_lm_true + 1e-12)


def paired(nll_a, nll_b, label):
    t_stat, t_p = stats.ttest_rel(nll_a, nll_b)
    w_stat, w_p = stats.wilcoxon(nll_a, nll_b)
    return {"comparison": label, "mean_diff": float((nll_a - nll_b).mean()),
            "paired_t_test": {"t_stat": float(t_stat), "p_value": float(t_p)},
            "wilcoxon_signed_rank": {"stat": float(w_stat), "p_value": float(w_p)}}


results["best_strategy_by_controller_train"] = best_name
results["significance_tests"] = [
    paired(nll_best, nll_dime, f"{best_name} (retuned) vs DIME (tuned)"),
    paired(gpt_only, nll_best, f"GPT-only vs {best_name} (retuned)"),
]
save_results(out_path, results)

print(f"\n[{PREFIX}] === RAW RETUNE SUMMARY ===")
print(f"GPT-only {results['gpt_only_mean_nll']:.4f} | DIME {dime_val:.4f}")
for name, r in results["strategies"].items():
    print(f"  {name:28s} retuned {r['val_mean_nll_retuned']:.4f}   original {r['val_mean_nll_original']}")
print(f"best on controller_train: {best_name}")
print(results["significance_tests"][0])
