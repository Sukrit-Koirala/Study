"""Shared helpers for the follow-up experiments (latency, B-sweep, He et al. baseline).

Same grids, seeds and clustering call as run_tier1_core.py / run_raw_retune.py, so results
from these scripts are directly comparable to the Tier 1-3 tables.
"""
import sys, os, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "GPT_Module"))

import numpy as np
from collections import Counter
from scipy import stats
from sklearn.cluster import MiniBatchKMeans

from cache_io import load_cache
from knn import build_datastore, query_knn, query_knn_indices, mix_knn_and_lm
from mixing import mix_dime_and_lm
from grid_search import grid_search_hyperparams

K_GRID = [10, 20, 50, 100, 200, 300]
TAU_GRID = [0.5, 1.0, 2.0, 5.0, 10.0]
ALPHA_GRID = [0.01, 0.05, 0.1, 0.25, 0.5]
# A raw/He baseline can always switch retrieval off (alpha=0 == GPT-only).
RAW_ALPHA_GRID = [0.0] + ALPHA_GRID

CACHE_DIR = os.path.join(os.path.dirname(__file__), "..", "GPT_Module", "cache")


def load_setting(dataset, model):
    prefix = f"{dataset}_{model}"
    out = {"prefix": prefix}
    for split, short in (("datastore", "ds"), ("controller_train", "ct"), ("val", "val")):
        d = load_cache(os.path.join(CACHE_DIR, f"{prefix}_{split}.npz"))
        out[short] = d
    return out


def load_json(path):
    with open(path) as f:
        return json.load(f)


def k_grid_for(size):
    return [k for k in K_GRID if k <= size] or [max(1, min(size, 10))]


def kmeans_fit(keys, n_clusters, seed=42):
    """Identical call to minibatch_kmeans_partition / run_tier1_core.py."""
    km = MiniBatchKMeans(n_clusters=n_clusters, random_state=seed)
    assignment = km.fit_predict(keys)
    return assignment, km.cluster_centers_


def build_dists(assignment, values, n_clusters):
    """Per-cluster Counter of next tokens (sort-based, O(N log N) instead of B boolean masks)."""
    order = np.argsort(assignment, kind="stable")
    sorted_assign = assignment[order]
    sorted_vals = values[order]
    bounds = np.searchsorted(sorted_assign, np.arange(n_clusters + 1))
    return [Counter(sorted_vals[bounds[c]:bounds[c + 1]].tolist()) for c in range(n_clusters)]


def kmeans_representative(keys, assignment, centers, n_clusters):
    """raw_kmeans_representative selection: real entry nearest each centroid."""
    order = np.argsort(assignment, kind="stable")
    bounds = np.searchsorted(assignment[order], np.arange(n_clusters + 1))
    selected = []
    for c in range(n_clusters):
        members = order[bounds[c]:bounds[c + 1]]
        if len(members) == 0:
            continue
        d = np.linalg.norm(keys[members] - centers[c], axis=1)
        selected.append(members[np.argmin(d)])
    return np.array(selected)


def tune_raw_store(keys, values, S, alpha_grid=RAW_ALPHA_GRID):
    """Grid-search (k, tau, alpha) on controller_train for a raw (single-token) store,
    then evaluate once on val. Returns (best_config, per-position val NLL)."""
    index, stored = build_datastore(keys, values)
    kg = k_grid_for(len(values))
    d_ct, r_ct = query_knn(index, stored, S["ct"]["keys"], k=max(kg))
    _, best = grid_search_hyperparams(d_ct, r_ct, S["ct"]["values"], S["ct"]["p_lm_true"],
                                      mix_knn_and_lm, kg, TAU_GRID, alpha_grid)
    d_v, r_v = query_knn(index, stored, S["val"]["keys"], k=best["k"])
    _, nll = mix_knn_and_lm(d_v, r_v, S["val"]["values"], S["val"]["p_lm_true"], tau=best["tau"], alpha=best["alpha"])
    return best, nll


def tune_dime_store(centers, dists, S):
    """Same tuning as run_tier1_core.py for the DIME store."""
    keys = centers.astype(np.float32)
    index, stored = build_datastore(keys, np.array(dists, dtype=object))
    kg = k_grid_for(len(dists))
    d_ct, r_ct = query_knn(index, stored, S["ct"]["keys"], k=max(kg))
    _, best = grid_search_hyperparams(d_ct, r_ct, S["ct"]["values"], S["ct"]["p_lm_true"],
                                      mix_dime_and_lm, kg, TAU_GRID, ALPHA_GRID)
    d_v, r_v = query_knn(index, stored, S["val"]["keys"], k=best["k"])
    _, nll = mix_dime_and_lm(d_v, r_v, S["val"]["values"], S["val"]["p_lm_true"], tau=best["tau"], alpha=best["alpha"])
    return best, nll


def paired(nll_a, nll_b, label):
    """Same two-sided paired tests as run_tier1_core.py. mean_diff > 0 means a is worse (higher NLL)."""
    t_stat, t_p = stats.ttest_rel(nll_a, nll_b)
    w_stat, w_p = stats.wilcoxon(nll_a, nll_b)
    return {"comparison": label, "mean_diff": float((nll_a - nll_b).mean()),
            "paired_t_test": {"t_stat": float(t_stat), "p_value": float(t_p)},
            "wilcoxon_signed_rank": {"stat": float(w_stat), "p_value": float(w_p)}}
