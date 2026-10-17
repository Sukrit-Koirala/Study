"""E1 diagnostic pipeline: where does DIME lose predictive performance relative to
uncompressed kNN-LM, and what characterizes affected queries? WikiText-103/gpt2,
reusing the already-validated extraction cache and tuned hyperparameters (confirmed
to match this experiment's required config exactly: n_clusters=3811, seq_len=128,
same seed=42 split) rather than re-extracting or re-tuning from scratch.

Scope note, stated plainly rather than silently handled: the cached .npz files
(cache_io.py) store only final hidden vectors, target token ids, and GPT's own
p(true token) — NOT the raw context token sequences. This is sufficient for E1A
(geometry), E1B (distribution shift), and E1C (predictive degradation) in full, and
for E1D's worst-case *numeric* diagnostics (distances, probabilities, JS divergence,
retrieved token ids). It is NOT sufficient for E1D's "decoded context snippets"
requirement — reconstructing those needs a fresh, cheap (CPU-only) re-tokenization
pass whose chunk order must then be validated against this cache's p_lm_true before
trusting the alignment. That validation needs a GPU forward pass this script does
run (see validate_cache_alignment), but decoded-context recovery itself is left as
an explicitly separate, smaller follow-up script so this one isn't gated on it.

Sequence/document-level bootstrap: WikiText rows are mostly short paragraphs
concatenated across many rows per 128-token chunk (confirmed earlier this session:
WikiText-2's entire train split is 36,718 *rows* for a few hundred actual articles)
-- so "one chunk" is already a mix of several rows' content, not a clean row or
article unit. Chunks are bootstrapped as the sequence unit instead (each chunk
contributes exactly seq_len-1=127 positions, in order, so chunk_id = position_index
// 127 recovers this with no extra metadata needed) -- matching the spec's own
"source-row OR sequence level" allowance.
"""
import sys, os, json, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "GPT_Module"))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

from cache_io import load_cache
from knn import build_datastore, query_knn, query_knn_indices, mix_knn_and_lm, softmax_np
from mixing import mix_dime_and_lm
from state_object import minibatch_kmeans_partition
from results_io import save_results
from e1_metrics import (
    raw_standalone_prob, dime_standalone_prob, raw_full_distribution,
    dime_full_distribution, js_divergence_dict, retrieval_coverage,
)

CACHE_DIR = os.path.join(os.path.dirname(__file__), "..", "GPT_Module", "cache")
OUT_DIR = os.path.join(os.path.dirname(__file__), "results", "stage3_e1_wikitext103")
PLOT_DIR = os.path.join(OUT_DIR, "plots")
METRICS_DIR = os.path.join(OUT_DIR, "metrics")
DIAG_DIR = os.path.join(OUT_DIR, "diagnostics")
for d in (OUT_DIR, PLOT_DIR, METRICS_DIR, DIAG_DIR):
    os.makedirs(d, exist_ok=True)

N_CLUSTERS = 3811
SEQ_LEN = 128
POSITIONS_PER_CHUNK = SEQ_LEN - 1
MATCHED_CONFIG = {"k": 20, "tau": 2.0, "alpha": 0.25}  # Comparison A, a predeclared diagnostic choice, not from the paper


def load_tuned_config():
    with open("results/wikitext103_gpt2_tier1.json") as f:
        tier1 = json.load(f)
    assert tier1["n_clusters"] == N_CLUSTERS, "cached tier1 config n_clusters mismatch -- do not silently proceed"
    return tier1["raw_knn"]["best_config"], tier1["dime_minibatch_kmeans"]["best_config"], tier1["gpt_only_mean_nll"]


def manifest(ds, ct, val, raw_tuned, dime_tuned):
    import platform
    return {
        "dataset": "wikitext103", "model": "gpt2", "seq_len": SEQ_LEN, "n_clusters": N_CLUSTERS,
        "seed": 42, "distance_metric": "euclidean", "representation": "hidden_states[-1] (final-layer output)",
        "data_sizes": {"datastore": len(ds["values"]), "controller_train": len(ct["values"]), "val": len(val["values"])},
        "matched_config": MATCHED_CONFIG, "tuned_raw_config": raw_tuned, "tuned_dime_config": dime_tuned,
        "clustering": "sklearn.cluster.MiniBatchKMeans, seed=42",
        "retrieval_search": "sklearn.neighbors.NearestNeighbors (exact, not approximate)",
        "numpy_version": np.__version__, "python_version": platform.python_version(),
        "cache_files": ["wikitext103_gpt2_datastore.npz", "wikitext103_gpt2_controller_train.npz", "wikitext103_gpt2_val.npz"],
        "note_on_sequence_bootstrap_unit": "one 128-token chunk (127 positions), NOT a WikiText row or article -- "
                                           "rows are concatenated across many rows per chunk, so chunk is the "
                                           "cleanest available sequence-level unit without extra metadata.",
    }


def quantile_summary(x):
    return {"median": float(np.median(x)), "p90": float(np.percentile(x, 90)),
            "p95": float(np.percentile(x, 95)), "p99": float(np.percentile(x, 99)),
            "mean": float(np.mean(x)), "std": float(np.std(x))}


def document_bootstrap_ci(chunk_ids, x, y, n_boot=2000, seed=42):
    rng = np.random.default_rng(seed)
    unique_chunks = np.unique(chunk_ids)
    chunk_to_pos = {c: np.where(chunk_ids == c)[0] for c in unique_chunks}
    point_estimate = float((x - y).mean())
    boot = np.zeros(n_boot)
    for b in range(n_boot):
        sampled = rng.choice(unique_chunks, size=len(unique_chunks), replace=True)
        idx = np.concatenate([chunk_to_pos[c] for c in sampled])
        boot[b] = (x[idx] - y[idx]).mean()
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return {"point_estimate": point_estimate, "ci_95_low": float(lo), "ci_95_high": float(hi),
            "n_sequences": int(len(unique_chunks)), "n_bootstrap": n_boot}


def main():
    print("Loading cache (datastore / controller_train / val)...")
    ds = load_cache(os.path.join(CACHE_DIR, "wikitext103_gpt2_datastore.npz"))
    ct = load_cache(os.path.join(CACHE_DIR, "wikitext103_gpt2_controller_train.npz"))
    val = load_cache(os.path.join(CACHE_DIR, "wikitext103_gpt2_val.npz"))
    raw_tuned, dime_tuned, gpt_only_nll = load_tuned_config()
    print(f"datastore={len(ds['values'])} controller_train={len(ct['values'])} val={len(val['values'])}")
    print(f"reusing tuned configs: raw={raw_tuned}  dime={dime_tuned}  gpt_only_nll={gpt_only_nll:.4f}")

    chunk_ids_val = np.arange(len(val["values"])) // POSITIONS_PER_CHUNK

    run_manifest = manifest(ds, ct, val, raw_tuned, dime_tuned)
    with open(os.path.join(OUT_DIR, "run_manifest.json"), "w") as f:
        json.dump(run_manifest, f, indent=2)

    print("Building raw datastore index and DIME clusters...")
    raw_index, raw_values = build_datastore(ds["keys"], ds["values"])
    t0 = time.perf_counter()
    dime_centroids, dime_dists = minibatch_kmeans_partition(ds["keys"], ds["values"], n_clusters=N_CLUSTERS, seed=42)
    dime_dists = np.array(dime_dists, dtype=object)
    dime_index, dime_values = build_datastore(dime_centroids, dime_dists)
    clustering_time = time.perf_counter() - t0
    print(f"DIME clustering done in {clustering_time:.1f}s")

    k_max_raw = min(max(MATCHED_CONFIG["k"], raw_tuned["k"]), len(ds["values"]))
    k_max_dime = min(max(MATCHED_CONFIG["k"], dime_tuned["k"]), N_CLUSTERS)
    assert k_max_raw >= MATCHED_CONFIG["k"] and k_max_dime >= min(MATCHED_CONFIG["k"], N_CLUSTERS), \
        "MATCHED_CONFIG's k exceeds the available datastore/cluster pool -- not expected at real WikiText-103 scale"
    print(f"Querying val set against both indices (k_max_raw={k_max_raw}, k_max_dime={k_max_dime})...")
    d_raw_full, idx_raw_full = query_knn_indices(raw_index, val["keys"], k=k_max_raw)
    d_dime_full, idx_dime_full = query_knn_indices(dime_index, val["keys"], k=k_max_dime)
    retrieved_raw_full = raw_values[idx_raw_full]
    retrieved_dime_full = dime_values[idx_dime_full]

    val_values, val_p_lm = val["values"], val["p_lm_true"]
    nll_lm = -np.log(np.clip(val_p_lm, 1e-12, None))

    # ============================================================
    # E1A — retrieval geometry
    # ============================================================
    print("\n=== E1A: retrieval geometry ===")
    d_raw_nearest = d_raw_full[:, 0]
    d_dime_nearest = d_dime_full[:, 0]
    delta_d = d_dime_nearest - d_raw_nearest
    abs_delta_d = np.abs(delta_d)

    e1a_summary = {
        "d_raw_nearest": quantile_summary(d_raw_nearest),
        "d_dime_nearest": quantile_summary(d_dime_nearest),
        "delta_d_signed": quantile_summary(delta_d),
        "delta_d_abs": quantile_summary(abs_delta_d),
        "frac_centroid_closer_than_nearest_raw_point": float((delta_d < 0).mean()),
    }
    print(json.dumps(e1a_summary, indent=2))

    fig, ax = plt.subplots(figsize=(6, 6))
    sample_idx = np.random.default_rng(0).choice(len(d_raw_nearest), size=min(20000, len(d_raw_nearest)), replace=False)
    ax.scatter(d_raw_nearest[sample_idx], d_dime_nearest[sample_idx], s=2, alpha=0.2)
    lims = [0, max(d_raw_nearest.max(), d_dime_nearest.max())]
    ax.plot(lims, lims, "r--", linewidth=1, label="y = x")
    ax.set_xlabel("raw nearest-neighbor distance (Euclidean)")
    ax.set_ylabel("DIME nearest-centroid distance (Euclidean)")
    ax.set_title("E1A: raw vs. compressed nearest-representative distance")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, "raw_vs_centroid_distance.png"), dpi=130)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(delta_d, bins=100)
    ax.set_xlabel("signed delta d = d_DIME - d_raw (Euclidean distance units)")
    ax.set_ylabel("count")
    ax.set_title("E1A: signed distance-change histogram")
    fig.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, "distance_delta_histogram.png"), dpi=130)
    plt.close(fig)

    # ============================================================
    # E1B — retrieval-distribution shift (standalone, pre-interpolation)
    # ============================================================
    print("\n=== E1B: retrieval-distribution shift (tuned config) ===")
    k_raw_t, tau_raw_t = raw_tuned["k"], raw_tuned["tau"]
    k_dime_t, tau_dime_t = dime_tuned["k"], dime_tuned["tau"]
    d_raw_t = d_raw_full[:, :k_raw_t]
    r_raw_t = retrieved_raw_full[:, :k_raw_t]
    d_dime_t = d_dime_full[:, :k_dime_t]
    r_dime_t = retrieved_dime_full[:, :k_dime_t]

    p_raw_standalone = raw_standalone_prob(d_raw_t, r_raw_t, val_values, tau=tau_raw_t)
    p_dime_standalone = dime_standalone_prob(d_dime_t, r_dime_t, val_values, tau=tau_dime_t)

    print("Computing per-query JS divergence (this loops over all val queries, dict-based)...")
    t0 = time.perf_counter()
    raw_dists_list = raw_full_distribution(d_raw_t, r_raw_t, tau=tau_raw_t)
    dime_dists_list = dime_full_distribution(d_dime_t, r_dime_t, tau=tau_dime_t)
    js_div = np.array([js_divergence_dict(raw_dists_list[i], dime_dists_list[i]) for i in range(len(val_values))])
    print(f"JS divergence computed in {time.perf_counter()-t0:.1f}s")

    e1b_summary = {
        "js_divergence": quantile_summary(js_div),
        "raw_coverage": retrieval_coverage(p_raw_standalone),
        "dime_coverage": retrieval_coverage(p_dime_standalone),
        "p_raw_true_token": quantile_summary(p_raw_standalone),
        "p_dime_true_token": quantile_summary(p_dime_standalone),
    }
    print(json.dumps(e1b_summary, indent=2))

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(js_div, bins=100)
    ax.set_xlabel("Jensen-Shannon divergence (nats)")
    ax.set_ylabel("count")
    ax.set_title("E1B: JS divergence between raw and DIME standalone retrieval distributions")
    fig.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, "js_histogram.png"), dpi=130)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(abs_delta_d[sample_idx], js_div[sample_idx], s=2, alpha=0.2)
    ax.set_xlabel("|delta d| (absolute nearest-distance change)")
    ax.set_ylabel("JS divergence (nats)")
    ax.set_title("E1B: distance change vs. distribution-shift magnitude")
    fig.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, "distance_vs_js.png"), dpi=130)
    plt.close(fig)

    distance_quartiles = np.quantile(abs_delta_d, [0.25, 0.5, 0.75])
    distance_bin = np.digitize(abs_delta_d, distance_quartiles)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.boxplot([js_div[distance_bin == b] for b in range(4)],
               tick_labels=["Q1 (smallest |delta d|)", "Q2", "Q3", "Q4 (largest |delta d|)"])
    ax.set_ylabel("JS divergence (nats)")
    ax.set_title("E1B: JS divergence grouped by |delta d| quantile")
    fig.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, "js_by_distance_quantile_boxplot.png"), dpi=130)
    plt.close(fig)

    # High-JS, low-distance-change cases: predeclared criterion (bottom quartile of
    # |delta d|, top quartile of JS), not chosen after looking at results.
    low_delta_threshold = np.percentile(abs_delta_d, 25)
    high_js_threshold = np.percentile(js_div, 75)
    interesting_mask = (abs_delta_d <= low_delta_threshold) & (js_div >= high_js_threshold)
    e1b_summary["n_small_distance_change_high_js"] = int(interesting_mask.sum())
    print(f"queries with small |delta d| (<=p25) AND high JS (>=p75): {interesting_mask.sum()}")

    # ============================================================
    # E1C — predictive degradation
    # ============================================================
    print("\n=== E1C: predictive degradation ===")
    results_by_config = {}
    for config_name, (kr, taur, alphar, kd, taud, alphad) in {
        "matched": (MATCHED_CONFIG["k"], MATCHED_CONFIG["tau"], MATCHED_CONFIG["alpha"],
                    MATCHED_CONFIG["k"], MATCHED_CONFIG["tau"], MATCHED_CONFIG["alpha"]),
        "tuned": (raw_tuned["k"], raw_tuned["tau"], raw_tuned["alpha"],
                  dime_tuned["k"], dime_tuned["tau"], dime_tuned["alpha"]),
    }.items():
        print(f"--- Comparison: {config_name} ---")
        dr, rr = d_raw_full[:, :kr], retrieved_raw_full[:, :kr]
        dd, rd = d_dime_full[:, :kd], retrieved_dime_full[:, :kd]
        _, nll_raw_mixed = mix_knn_and_lm(dr, rr, val_values, val_p_lm, tau=taur, alpha=alphar)
        _, nll_dime_mixed = mix_dime_and_lm(dd, rd, val_values, val_p_lm, tau=taud, alpha=alphad)

        delta_L = nll_dime_mixed - nll_raw_mixed
        G_raw = nll_lm - nll_raw_mixed
        G_dime = nll_lm - nll_dime_mixed

        raw_benefits = G_raw > 0
        denom = G_raw[raw_benefits].sum()
        R = float(G_dime[raw_benefits].sum() / denom) if denom > 1e-9 else None

        preserved = raw_benefits & (G_dime >= G_raw * 0.9)  # within 10% of raw's own gain, a declared threshold
        dime_worse_than_raw = raw_benefits & (G_dime < G_raw) & (G_dime >= 0)
        dime_worse_than_lm = raw_benefits & (G_dime < 0)
        dime_beats_raw = G_dime > G_raw
        neither_helps = (G_raw <= 0) & (G_dime <= 0)

        config_result = {
            "mean_nll_lm": float(nll_lm.mean()), "mean_nll_raw_mixed": float(nll_raw_mixed.mean()),
            "mean_nll_dime_mixed": float(nll_dime_mixed.mean()),
            "delta_L": quantile_summary(delta_L),
            "gain_retention_R": R, "sum_G_raw_positive": float(denom), "sum_G_dime_on_raw_positive": float(G_dime[raw_benefits].sum()),
            "category_fractions": {
                "raw_improves_dime_preserves": float(preserved.mean()),
                "raw_improves_dime_worse_than_raw": float(dime_worse_than_raw.mean()),
                "raw_improves_dime_worse_than_lm": float(dime_worse_than_lm.mean()),
                "dime_beats_raw": float(dime_beats_raw.mean()),
                "neither_helps": float(neither_helps.mean()),
            },
        }
        results_by_config[config_name] = config_result
        print(json.dumps(config_result, indent=2))

        if config_name == "tuned":
            boot_dime_vs_raw = document_bootstrap_ci(chunk_ids_val, nll_dime_mixed, nll_raw_mixed)
            boot_dime_vs_lm = document_bootstrap_ci(chunk_ids_val, nll_dime_mixed, nll_lm)
            t_dime_raw = stats.ttest_rel(nll_dime_mixed, nll_raw_mixed)
            w_dime_raw = stats.wilcoxon(nll_dime_mixed, nll_raw_mixed)
            config_result["bootstrap_dime_vs_raw"] = boot_dime_vs_raw
            config_result["bootstrap_dime_vs_lm"] = boot_dime_vs_lm
            config_result["paired_t_dime_vs_raw"] = {"t": float(t_dime_raw.statistic), "p": float(t_dime_raw.pvalue)}
            config_result["wilcoxon_dime_vs_raw_p"] = float(w_dime_raw.pvalue)
            print(f"document bootstrap (DIME-raw): {boot_dime_vs_raw}")

            fig, ax = plt.subplots(figsize=(6, 6))
            ax.scatter(G_raw[sample_idx], G_dime[sample_idx], s=2, alpha=0.2)
            lims = [min(G_raw.min(), G_dime.min()), max(G_raw.max(), G_dime.max())]
            ax.plot(lims, lims, "r--", linewidth=1, label="y = x")
            ax.axhline(0, color="gray", linewidth=0.5)
            ax.axvline(0, color="gray", linewidth=0.5)
            ax.set_xlabel("raw kNN gain over LM (G_raw = L_LM - L_raw)")
            ax.set_ylabel("DIME gain over LM (G_DIME = L_LM - L_DIME)")
            ax.set_title("E1C: raw gain vs. DIME gain (tuned config)")
            ax.legend()
            fig.tight_layout()
            fig.savefig(os.path.join(PLOT_DIR, "raw_gain_vs_dime_gain.png"), dpi=130)
            plt.close(fig)

            fig, ax = plt.subplots(figsize=(7, 5))
            ax.hist(delta_L, bins=100)
            ax.set_xlabel("delta L = L_DIME - L_raw (nats)")
            ax.set_ylabel("count")
            ax.set_title("E1C: per-query NLL difference, DIME minus raw kNN (tuned config)")
            fig.tight_layout()
            fig.savefig(os.path.join(PLOT_DIR, "delta_nll_histogram.png"), dpi=130)
            plt.close(fig)

            fig, ax = plt.subplots(figsize=(6, 6))
            ax.scatter(js_div[sample_idx], delta_L[sample_idx], s=2, alpha=0.2)
            ax.set_xlabel("JS divergence (nats)")
            ax.set_ylabel("delta L (nats)")
            ax.set_title("E1C: distribution shift vs. NLL degradation")
            fig.tight_layout()
            fig.savefig(os.path.join(PLOT_DIR, "js_vs_delta_nll.png"), dpi=130)
            plt.close(fig)

            js_quartiles = np.quantile(js_div, [0.25, 0.5, 0.75])
            js_bin = np.digitize(js_div, js_quartiles)
            fig, ax = plt.subplots(figsize=(7, 5))
            ax.boxplot([delta_L[js_bin == b] for b in range(4)],
                       tick_labels=["Q1 (smallest JS)", "Q2", "Q3", "Q4 (largest JS)"])
            ax.set_ylabel("delta L = L_DIME - L_raw (nats)")
            ax.set_title("E1C: delta L grouped by JS-divergence quantile (tuned config)")
            fig.tight_layout()
            fig.savefig(os.path.join(PLOT_DIR, "delta_nll_by_js_quantile_boxplot.png"), dpi=130)
            plt.close(fig)

            fig, ax = plt.subplots(figsize=(6, 6))
            ax.scatter(abs_delta_d[sample_idx], delta_L[sample_idx], s=2, alpha=0.2)
            ax.set_xlabel("|delta d| (absolute nearest-distance change)")
            ax.set_ylabel("delta L (nats)")
            ax.set_title("E1C: absolute distance change vs. NLL degradation (tuned config)")
            fig.tight_layout()
            fig.savefig(os.path.join(PLOT_DIR, "distance_vs_delta_nll.png"), dpi=130)
            plt.close(fig)

            fig, ax = plt.subplots(figsize=(9, 5))
            cat_labels = list(config_result["category_fractions"].keys())
            cat_values = list(config_result["category_fractions"].values())
            ax.barh(cat_labels, cat_values)
            ax.set_xlabel("fraction of evaluation queries")
            ax.set_title("E1C: gain-retention category breakdown (tuned config)")
            for i, v in enumerate(cat_values):
                ax.text(v, i, f" {v*100:.1f}%", va="center")
            fig.tight_layout()
            fig.savefig(os.path.join(PLOT_DIR, "gain_retention_summary.png"), dpi=130)
            plt.close(fig)

            fig, ax = plt.subplots(figsize=(8, 5))
            labels = ["GPT-only", "raw kNN (tuned)", "DIME (tuned)"]
            means = [nll_lm.mean(), nll_raw_mixed.mean(), nll_dime_mixed.mean()]
            ax.bar(labels, means)
            ax.set_ylabel("mean NLL (nats)")
            ax.set_title("E1C: mean NLL comparison")
            for i, v in enumerate(means):
                ax.text(i, v + 0.01, f"{v:.3f}", ha="center")
            fig.tight_layout()
            fig.savefig(os.path.join(PLOT_DIR, "nll_comparison.png"), dpi=130)
            plt.close(fig)

            # keep the tuned config's arrays for E1D
            tuned_delta_L, tuned_js_div = delta_L, js_div
            tuned_nll_lm, tuned_nll_raw, tuned_nll_dime = nll_lm, nll_raw_mixed, nll_dime_mixed
            tuned_p_raw_true, tuned_p_dime_true = p_raw_standalone, p_dime_standalone
            tuned_G_raw, tuned_G_dime = G_raw, G_dime

    # ============================================================
    # E1D — failure-case discovery (numeric; decoded context deferred, see module docstring)
    # ============================================================
    print("\n=== E1D: failure-case discovery (tuned config) ===")
    order_worst = np.argsort(-tuned_delta_L)  # highest DIME-worse-than-raw first
    rng = np.random.default_rng(42)

    def record(i):
        return {
            "position_index": int(i), "chunk_id": int(chunk_ids_val[i]),
            "true_token_id": int(val_values[i]),
            "p_lm_true": float(val_p_lm[i]), "p_raw_standalone_true": float(tuned_p_raw_true[i]),
            "p_dime_standalone_true": float(tuned_p_dime_true[i]),
            "nll_lm": float(tuned_nll_lm[i]), "nll_raw_mixed": float(tuned_nll_raw[i]), "nll_dime_mixed": float(tuned_nll_dime[i]),
            "delta_L": float(tuned_delta_L[i]), "js_divergence": float(tuned_js_div[i]),
            "G_raw": float(tuned_G_raw[i]), "G_dime": float(tuned_G_dime[i]),
            "d_raw_nearest": float(d_raw_nearest[i]), "d_dime_nearest": float(d_dime_nearest[i]),
            "top_raw_neighbor_ids": idx_raw_full[i, :5].tolist(), "top_raw_neighbor_distances": d_raw_full[i, :5].tolist(),
            "top_raw_neighbor_tokens": retrieved_raw_full[i, :5].tolist(),
            "top_dime_cluster_ids": idx_dime_full[i, :5].tolist(), "top_dime_cluster_distances": d_dime_full[i, :5].tolist(),
            "note": "decoded context snippet not available from cache -- see run script docstring",
        }

    worst_cases = [record(i) for i in order_worst[:50]]
    small_d_high_js_idx = np.where(interesting_mask)[0]
    small_d_high_js_cases = [record(i) for i in small_d_high_js_idx[:50]]
    raw_helps_dime_hurts_idx = np.where((tuned_G_raw > 0) & (tuned_G_dime < tuned_G_raw))[0]
    raw_helps_dime_hurts_cases = [record(i) for i in raw_helps_dime_hurts_idx[:50]]
    dime_beats_raw_idx = np.where(tuned_G_dime > tuned_G_raw)[0]
    dime_beats_raw_cases = [record(i) for i in rng.choice(dime_beats_raw_idx, size=min(50, len(dime_beats_raw_idx)), replace=False)]
    representative_sample = [record(i) for i in rng.choice(len(val_values), size=50, replace=False)]

    with open(os.path.join(DIAG_DIR, "worst_cases.jsonl"), "w") as f:
        for c in worst_cases:
            f.write(json.dumps(c) + "\n")
    with open(os.path.join(DIAG_DIR, "small_distance_change_high_js_cases.jsonl"), "w") as f:
        for c in small_d_high_js_cases:
            f.write(json.dumps(c) + "\n")
    with open(os.path.join(DIAG_DIR, "raw_helps_dime_hurts_cases.jsonl"), "w") as f:
        for c in raw_helps_dime_hurts_cases:
            f.write(json.dumps(c) + "\n")
    with open(os.path.join(DIAG_DIR, "dime_beats_raw_cases.jsonl"), "w") as f:
        for c in dime_beats_raw_cases:
            f.write(json.dumps(c) + "\n")
    with open(os.path.join(DIAG_DIR, "representative_cases.jsonl"), "w") as f:
        for c in representative_sample:
            f.write(json.dumps(c) + "\n")
    print(f"saved worst/interesting/representative case files to {DIAG_DIR}")

    # ============================================================
    # Per-query metrics table (CSV -- pyarrow not in requirements.txt, flagged earlier)
    # ============================================================
    print("\nSaving per-query metrics table (CSV)...")
    import csv
    with open(os.path.join(METRICS_DIR, "per_query.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["position_index", "chunk_id", "true_token_id", "p_lm_true",
                          "d_raw_nearest", "d_dime_nearest", "delta_d", "js_divergence",
                          "p_raw_standalone_true", "p_dime_standalone_true",
                          "nll_lm", "nll_raw_mixed_tuned", "nll_dime_mixed_tuned", "delta_L_tuned"])
        for i in range(len(val_values)):
            writer.writerow([i, int(chunk_ids_val[i]), int(val_values[i]), float(val_p_lm[i]),
                              float(d_raw_nearest[i]), float(d_dime_nearest[i]), float(delta_d[i]), float(tuned_js_div[i]),
                              float(tuned_p_raw_true[i]), float(tuned_p_dime_true[i]),
                              float(tuned_nll_lm[i]), float(tuned_nll_raw[i]), float(tuned_nll_dime[i]), float(tuned_delta_L[i])])

    full_results = {
        "e1a_geometry": e1a_summary, "e1b_distribution_shift": e1b_summary,
        "e1c_predictive_degradation": results_by_config,
        "run_manifest": run_manifest, "clustering_time_sec": clustering_time,
    }
    save_results(os.path.join(OUT_DIR, "summary.json"), full_results)
    print(f"\nAll results saved under {OUT_DIR}")


if __name__ == "__main__":
    main()
