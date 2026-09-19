"""100-seed test on TinyStories/gpt2 with the equal-budget raw baselines RE-TUNED at B.

The original run (run_tinystories_multiseed.py) evaluated raw_kmeans_representative with the
full-datastore config (k=50, tau=2, alpha=0.1). After re-tuning at B=500 that baseline is
stronger (val 2.781 -> 2.764 at seed 42), so the seed-level claim must be redone against it.

Same design as the original: a fresh random story split per seed (seed -> collect_chunks_split
seed and k-means seed); hyperparameters are FIXED across seeds (tuned once at seed 42 -- the DIME
config from Tier 1, the raw configs from results/tinystories_gpt2_raw_retune.json), because
re-tuning per seed would answer a different question.

Reports, per seed: GPT-only, raw kNN (full datastore, Tier 1 config), DIME, and the re-tuned
raw_kmeans_representative and raw_random. The summary gives win counts and paired tests over seeds.

Usage: python run_tinystories_multiseed_retuned.py [--n_seeds 100]
Output: results/tinystories_multiseed_retuned.json (saved after every seed)
"""
import sys, os, argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "GPT_Module"))

from gpt_class import FrozenGPT2
from dataset_helpers import collect_chunks_split
from extract import build_datastore_from_chunks, run_batch
from knn import build_datastore, query_knn, mix_knn_and_lm
from results_io import save_results
from datasets import load_dataset
from mixing import mix_dime_and_lm
from exp_common import load_json, kmeans_fit, build_dists, kmeans_representative
import numpy as np
from scipy import stats

parser = argparse.ArgumentParser()
parser.add_argument("--n_seeds", type=int, default=100)
parser.add_argument("--first_seed", type=int, default=42)
args = parser.parse_args()

tier1 = load_json("results/tinystories_gpt2_tier1.json")
retune = load_json("results/tinystories_gpt2_raw_retune.json")
RAW = tier1["raw_knn"]["best_config"]
DIME = tier1["dime_minibatch_kmeans"]["best_config"]
REP = retune["strategies"]["raw_kmeans_representative"]["best_config"]
RND = retune["strategies"]["raw_random"]["best_config"]
N_CLUSTERS = tier1["n_clusters"]
print(f"configs: raw={RAW}\n dime={DIME}\n rep(retuned)={REP}\n random(retuned)={RND}\n B={N_CLUSTERS}", flush=True)

gpt = FrozenGPT2()


def encode_split(chunks_list, batch_size=256):
    keys, targets, p_lm = [], [], []
    for i in range(0, len(chunks_list), batch_size):
        h_pred, y_target, p_true, _ = run_batch(gpt, chunks_list[i:i + batch_size])
        B, L, D = h_pred.shape
        keys.append(h_pred.reshape(B * L, D).cpu().numpy())
        targets.append(y_target.reshape(B * L).cpu().numpy())
        p_lm.append(p_true.reshape(B * L).cpu().numpy())
    return np.concatenate(keys), np.concatenate(targets), np.concatenate(p_lm)


def raw_eval(keys, values, cfg, vk, vt, vp):
    index, stored = build_datastore(keys, values)
    d, r = query_knn(index, stored, vk, k=min(cfg["k"], len(values)))
    _, nll = mix_knn_and_lm(d, r, vt, vp, tau=cfg["tau"], alpha=cfg["alpha"])
    return float(nll.mean())


def run_one_seed(seed):
    dataset = load_dataset("roneneldan/TinyStories", split="train", streaming=True, trust_remote_code=True)
    chunks, _ = collect_chunks_split(dataset, gpt.tokenizer, seq_len=128,
                                     n_chunks_needed={"datastore": 350, "controller_train": 75, "val": 75}, seed=seed)
    ds_keys, ds_values = build_datastore_from_chunks(gpt, chunks["datastore"], batch_size=256)
    vk, vt, vp = encode_split(chunks["val"])

    out = {"seed": seed, "n_datastore": int(len(ds_values)), "n_val": int(len(vt)),
           "gpt_only": float((-np.log(vp + 1e-12)).mean())}
    out["raw_knn_tuned"] = raw_eval(ds_keys, ds_values, RAW, vk, vt, vp)

    assignment, centers = kmeans_fit(ds_keys, N_CLUSTERS, seed=seed)
    dists = build_dists(assignment, ds_values, N_CLUSTERS)
    dime_index, dime_vals = build_datastore(centers.astype(np.float32), np.array(dists, dtype=object))
    d, r = query_knn(dime_index, dime_vals, vk, k=min(DIME["k"], N_CLUSTERS))
    _, nll = mix_dime_and_lm(d, r, vt, vp, tau=DIME["tau"], alpha=DIME["alpha"])
    out["dime_tuned"] = float(nll.mean())

    sel = kmeans_representative(ds_keys, assignment, centers, N_CLUSTERS)
    out["raw_kmeans_representative_retuned"] = raw_eval(ds_keys[sel], ds_values[sel], REP, vk, vt, vp)

    rng = np.random.default_rng(seed)
    idx = rng.choice(len(ds_values), size=min(N_CLUSTERS, len(ds_values)), replace=False)
    out["raw_random_retuned"] = raw_eval(ds_keys[idx], ds_values[idx], RND, vk, vt, vp)
    return out


seeds = list(range(args.first_seed, args.first_seed + args.n_seeds))
results, out_path = [], "results/tinystories_multiseed_retuned.json"
meta = {"note": "raw baselines re-tuned at B on seed 42; hyperparameters fixed across seeds",
        "configs": {"raw_knn": RAW, "dime": DIME, "raw_kmeans_representative": REP, "raw_random": RND}}
for seed in seeds:
    print(f"--- seed {seed} ---", flush=True)
    r = run_one_seed(seed)
    print(r, flush=True)
    results.append(r)
    save_results(out_path, {**meta, "seeds_completed": len(results), "total_planned": len(seeds), "results": results})

print("\n=== SUMMARY ===")
arr = {k: np.array([r[k] for r in results]) for k in ["gpt_only", "raw_knn_tuned", "dime_tuned",
                                                     "raw_kmeans_representative_retuned", "raw_random_retuned"]}
for k, v in arr.items():
    print(f"{k}: mean={v.mean():.4f} std={v.std():.4f} n={len(v)}")
for base in ["gpt_only", "raw_kmeans_representative_retuned", "raw_random_retuned", "raw_knn_tuned"]:
    wins = int((arr["dime_tuned"] < arr[base]).sum())
    t = stats.ttest_rel(arr[base], arr["dime_tuned"])
    print(f"DIME beats {base}: {wins}/{len(results)}  mean diff {float((arr[base] - arr['dime_tuned']).mean()):+.4f}  paired t p={t.pvalue:.3e}")
