import sys, os, argparse, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "GPT_Module"))

from cache_io import load_cache
from knn import build_datastore, query_knn, mix_knn_and_lm
from results_io import save_results
from state_object import minibatch_kmeans_partition
from mixing import mix_dime_and_lm
import numpy as np
from scipy import stats
from sklearn.cluster import MiniBatchKMeans

# NOTE on scope: unlike the original TinyStories multi-seed test (which re-extracted
# a fresh random story split from raw text per seed — cheap there since TinyStories
# streams and gpt2-small forward passes are fast), this generic version reuses ONE
# already-extracted cache (datastore/controller_train/val, fixed split) and varies
# only the DIME/raw_kmeans_representative CONSTRUCTION seed (the k-means random_state).
# Re-extracting per seed at WikiText-103 + gpt2-medium scale would mean N_SEEDS full
# GPU forward passes over the corpus, which is not worth the cost here. This still
# answers a real question — "is DIME's advantage an artifact of one clustering draw,
# or does it hold across many?" — just not the data-split-variance question the
# TinyStories run answered. Say so plainly when reporting these numbers.

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", required=True)
parser.add_argument("--model", required=True)
parser.add_argument("--n_seeds", type=int, default=30)
args = parser.parse_args()

CACHE_DIR = os.path.join(os.path.dirname(__file__), "..", "GPT_Module", "cache")
PREFIX = f"{args.dataset}_{args.model}"

ds = load_cache(os.path.join(CACHE_DIR, f"{PREFIX}_datastore.npz"))
val = load_cache(os.path.join(CACHE_DIR, f"{PREFIX}_val.npz"))
ds_keys, ds_values = ds["keys"], ds["values"]
val_keys, val_values, val_p_lm_true = val["keys"], val["values"], val["p_lm_true"]

with open(f"results/{PREFIX}_tier1.json") as f:
    tier1 = json.load(f)

N_CLUSTERS = tier1["n_clusters"]
RAW_K, RAW_TAU, RAW_ALPHA = tier1["raw_knn"]["best_config"]["k"], tier1["raw_knn"]["best_config"]["tau"], tier1["raw_knn"]["best_config"]["alpha"]
DIME_K, DIME_TAU, DIME_ALPHA = tier1["dime_minibatch_kmeans"]["best_config"]["k"], tier1["dime_minibatch_kmeans"]["best_config"]["tau"], tier1["dime_minibatch_kmeans"]["best_config"]["alpha"]

nll_gpt_only = -np.log(val_p_lm_true + 1e-12)
gpt_only_mean = float(nll_gpt_only.mean())

raw_index, raw_stored_values = build_datastore(ds_keys, ds_values)
distances_raw, retrieved_raw = query_knn(raw_index, raw_stored_values, val_keys, k=RAW_K)
_, nll_raw = mix_knn_and_lm(distances_raw, retrieved_raw, val_values, val_p_lm_true, tau=RAW_TAU, alpha=RAW_ALPHA)
raw_knn_mean = float(nll_raw.mean())

print(f"[{PREFIX}] fixed baselines: gpt_only={gpt_only_mean:.4f}  raw_knn_tuned={raw_knn_mean:.4f}")
print(f"[{PREFIX}] running {args.n_seeds} construction seeds for DIME + raw_kmeans_representative...")

SEEDS = list(range(42, 42 + args.n_seeds))
results_list = []
out_path = f"results/{PREFIX}_multiseed.json"

for seed in SEEDS:
    compressed_keys, compressed_dists = minibatch_kmeans_partition(ds_keys, ds_values, n_clusters=N_CLUSTERS, seed=seed)
    dime_index, dime_stored_values = build_datastore(compressed_keys, np.array(compressed_dists, dtype=object))
    distances_dime, retrieved_dime = query_knn(dime_index, dime_stored_values, val_keys, k=DIME_K)
    _, nll_dime = mix_dime_and_lm(distances_dime, retrieved_dime, val_values, val_p_lm_true, tau=DIME_TAU, alpha=DIME_ALPHA)
    dime_mean = float(nll_dime.mean())

    km = MiniBatchKMeans(n_clusters=N_CLUSTERS, random_state=seed)
    assignment = km.fit_predict(ds_keys)
    centers = km.cluster_centers_
    selected_idx = []
    for c in range(N_CLUSTERS):
        member_idx = np.where(assignment == c)[0]
        if len(member_idx) == 0:
            continue
        dists = np.linalg.norm(ds_keys[member_idx] - centers[c], axis=1)
        selected_idx.append(member_idx[np.argmin(dists)])
    selected_idx = np.array(selected_idx)
    raw_rep_index, raw_rep_values = build_datastore(ds_keys[selected_idx], ds_values[selected_idx])
    distances_rep, retrieved_rep = query_knn(raw_rep_index, raw_rep_values, val_keys, k=min(RAW_K, len(selected_idx)))
    _, nll_rep = mix_knn_and_lm(distances_rep, retrieved_rep, val_values, val_p_lm_true, tau=RAW_TAU, alpha=RAW_ALPHA)
    rep_mean = float(nll_rep.mean())

    r = {"seed": seed, "gpt_only": gpt_only_mean, "raw_knn_tuned": raw_knn_mean,
         "dime_tuned": dime_mean, "raw_kmeans_representative": rep_mean}
    print(f"[{PREFIX}] seed {seed}: dime={dime_mean:.4f}  raw_rep={rep_mean:.4f}", flush=True)
    results_list.append(r)
    save_results(out_path, {"dataset": args.dataset, "model": args.model, "note": "construction-seed variance only, fixed data split (see script header)",
                             "seeds_completed": len(results_list), "total_planned": len(SEEDS), "results": results_list})

dime_vals = np.array([r["dime_tuned"] for r in results_list])
rep_vals = np.array([r["raw_kmeans_representative"] for r in results_list])

n_beats_gpt = int((dime_vals < gpt_only_mean).sum())
n_beats_rep = int((dime_vals < rep_vals).sum())
t_gpt = stats.ttest_1samp(dime_vals, gpt_only_mean)
t_rep = stats.ttest_rel(dime_vals, rep_vals)

print(f"\n[{PREFIX}] === MULTI-SEED SUMMARY ({len(results_list)} construction seeds) ===")
print(f"DIME beats GPT-only: {n_beats_gpt}/{len(results_list)}  (one-sample t p={t_gpt.pvalue:.3e})")
print(f"DIME beats raw_kmeans_representative: {n_beats_rep}/{len(results_list)}  (paired t p={t_rep.pvalue:.3e})")
print(f"DIME mean±std: {dime_vals.mean():.4f} ± {dime_vals.std():.4f}")
