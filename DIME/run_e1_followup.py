"""E1 follow-up analysis — reuses the existing WikiText-103/gpt2 cache (no GPT-2
extraction rerun) and re-derives DIME's clusters once (seed=42, deterministic,
identical result to the original run — centroids were never persisted to disk, so
this is the one "necessary" recompute, not a full re-extraction). Everything in
items 1-4 below only needs the cache + one re-clustering + batched kNN queries, all
CPU/cheap. Items 5-6 additionally need a SMALL, targeted re-tokenization + decode
pass (just the specific ~45-60 queries requested, not the full val set), validated
against the cache's own p_lm_true before trusting any decoded text.
"""
import sys, os, json, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "GPT_Module"))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

from cache_io import load_cache
from knn import build_datastore, query_knn, query_knn_indices, mix_knn_and_lm
from mixing import mix_dime_and_lm
from state_object import minibatch_kmeans_partition
from gpt_class import FrozenGPT2
from dataset_helpers import collect_chunks_split
from results_io import save_results
from e1_metrics import raw_standalone_prob, dime_standalone_prob, js_divergence_dict, raw_full_distribution, dime_full_distribution

CACHE_DIR = os.path.join(os.path.dirname(__file__), "..", "GPT_Module", "cache")
E1_DIR = os.path.join(os.path.dirname(__file__), "results", "stage3_e1_wikitext103")
FOLLOWUP_DIR = os.path.join(os.path.dirname(__file__), "results", "stage3_e1_followup")
PLOT_DIR = os.path.join(FOLLOWUP_DIR, "plots")
os.makedirs(PLOT_DIR, exist_ok=True)

N_CLUSTERS = 3811
SEQ_LEN = 128
POSITIONS_PER_CHUNK = SEQ_LEN - 1
MATCHED = {"k": 20, "tau": 2.0, "alpha": 0.25}


def spearman_with_ci(x, y, n_boot=2000, seed=42):
    rng = np.random.default_rng(seed)
    rho, p = stats.spearmanr(x, y)
    n = len(x)
    boot = np.zeros(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boot[b] = stats.spearmanr(x[idx], y[idx])[0]
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return {"rho": float(rho), "p_value": float(p), "ci_95_low": float(lo), "ci_95_high": float(hi)}


def conditional_means(x, y, n_bins=10):
    edges = np.quantile(x, np.linspace(0, 1, n_bins + 1))
    bin_idx = np.clip(np.digitize(x, edges[1:-1]), 0, n_bins - 1)
    out = []
    for b in range(n_bins):
        mask = bin_idx == b
        if mask.sum() == 0:
            continue
        out.append({"bin": b, "n": int(mask.sum()), "x_mean": float(x[mask].mean()),
                     "y_mean": float(y[mask].mean()), "y_median": float(np.median(y[mask]))})
    return out


def main():
    print("Loading cache (no extraction)...")
    ds = load_cache(os.path.join(CACHE_DIR, "wikitext103_gpt2_datastore.npz"))
    val = load_cache(os.path.join(CACHE_DIR, "wikitext103_gpt2_val.npz"))
    with open("results/wikitext103_gpt2_tier1.json") as f:
        tier1 = json.load(f)
    raw_tuned, dime_tuned = tier1["raw_knn"]["best_config"], tier1["dime_minibatch_kmeans"]["best_config"]
    assert tier1["n_clusters"] == N_CLUSTERS
    print(f"val={len(val['values'])}  raw_tuned={raw_tuned}  dime_tuned={dime_tuned}")

    val_values, val_p_lm = val["values"], val["p_lm_true"]
    nll_lm = -np.log(np.clip(val_p_lm, 1e-12, None))
    chunk_ids_val = np.arange(len(val_values)) // POSITIONS_PER_CHUNK

    print("Building raw index, re-deriving DIME clusters (seed=42, deterministic)...")
    raw_index, raw_values = build_datastore(ds["keys"], ds["values"])
    t0 = time.perf_counter()
    dime_centroids, dime_dists = minibatch_kmeans_partition(ds["keys"], ds["values"], n_clusters=N_CLUSTERS, seed=42)
    dime_dists = np.array(dime_dists, dtype=object)
    dime_index, dime_values = build_datastore(dime_centroids, dime_dists)
    print(f"clustering done in {time.perf_counter()-t0:.1f}s")

    k_max_raw = max(MATCHED["k"], raw_tuned["k"])
    k_max_dime = max(MATCHED["k"], dime_tuned["k"])
    d_raw_full, idx_raw_full = query_knn_indices(raw_index, val["keys"], k=k_max_raw)
    d_dime_full, idx_dime_full = query_knn_indices(dime_index, val["keys"], k=k_max_dime)
    retrieved_raw_full = raw_values[idx_raw_full]
    retrieved_dime_full = dime_values[idx_dime_full]
    d_raw_nearest, d_dime_nearest = d_raw_full[:, 0], d_dime_full[:, 0]
    delta_d, abs_delta_d = d_dime_nearest - d_raw_nearest, np.abs(d_dime_nearest - d_raw_nearest)

    configs = {
        "matched": (MATCHED["k"], MATCHED["tau"], MATCHED["alpha"], MATCHED["k"], MATCHED["tau"], MATCHED["alpha"]),
        "tuned": (raw_tuned["k"], raw_tuned["tau"], raw_tuned["alpha"], dime_tuned["k"], dime_tuned["tau"], dime_tuned["alpha"]),
    }
    per_config = {}
    for name, (kr, taur, alphar, kd, taud, alphad) in configs.items():
        dr, rr = d_raw_full[:, :kr], retrieved_raw_full[:, :kr]
        dd, rd = d_dime_full[:, :kd], retrieved_dime_full[:, :kd]
        p_raw_mixed, nll_raw_mixed = mix_knn_and_lm(dr, rr, val_values, val_p_lm, tau=taur, alpha=alphar)
        p_dime_mixed, nll_dime_mixed = mix_dime_and_lm(dd, rd, val_values, val_p_lm, tau=taud, alpha=alphad)
        p_raw_standalone = raw_standalone_prob(dr, rr, val_values, tau=taur)
        p_dime_standalone = dime_standalone_prob(dd, rd, val_values, tau=taud)
        raw_dists = raw_full_distribution(dr, rr, tau=taur)
        dime_dists_full = dime_full_distribution(dd, rd, tau=taud)
        js = np.array([js_divergence_dict(raw_dists[i], dime_dists_full[i]) for i in range(len(val_values))])
        delta_L = nll_dime_mixed - nll_raw_mixed
        G_raw, G_dime = nll_lm - nll_raw_mixed, nll_lm - nll_dime_mixed
        per_config[name] = dict(nll_raw_mixed=nll_raw_mixed, nll_dime_mixed=nll_dime_mixed, p_raw_standalone=p_raw_standalone,
                                 p_dime_standalone=p_dime_standalone, js=js, delta_L=delta_L, G_raw=G_raw, G_dime=G_dime,
                                 k_raw=kr, tau_raw=taur, alpha_raw=alphar, k_dime=kd, tau_dime=taud, alpha_dime=alphad)
        print(f"[{name}] mean nll_raw={nll_raw_mixed.mean():.4f}  mean nll_dime={nll_dime_mixed.mean():.4f}")

    report = {"configs": {}}

    # ============================================================
    # Item 1 — NLL loss concentration
    # ============================================================
    print("\n=== Item 1: loss concentration ===")
    for name, c in per_config.items():
        dL = c["delta_L"]
        pos, neg, zero = dL > 0, dL < 0, dL == 0
        pos_sum, neg_sum = float(dL[pos].sum()), float(dL[neg].sum())
        order = np.argsort(-dL)
        tail_stats = {}
        for pct in (1, 5, 10):
            n = max(1, int(len(dL) * pct / 100))
            idx = order[:n]
            tail_stats[f"worst_{pct}pct"] = {
                "n": int(n), "frac_of_total_positive_loss": float(dL[idx].clip(min=0).sum() / pos_sum) if pos_sum > 0 else None,
                "mean_p_raw_standalone": float(c["p_raw_standalone"][idx].mean()),
                "mean_p_dime_standalone": float(c["p_dime_standalone"][idx].mean()),
                "mean_G_raw": float(c["G_raw"][idx].mean()), "mean_js": float(c["js"][idx].mean()),
            }
        report["configs"].setdefault(name, {})["loss_concentration"] = {
            "frac_positive": float(pos.mean()), "frac_negative": float(neg.mean()), "frac_zero": float(zero.mean()),
            "total_positive_sum": pos_sum, "total_negative_sum": neg_sum, "tails": tail_stats,
        }
        print(f"[{name}] +:{pos.mean()*100:.1f}% -:{neg.mean()*100:.1f}% 0:{zero.mean()*100:.1f}%  "
              f"pos_sum={pos_sum:.2f} neg_sum={neg_sum:.2f}")
        for k, v in tail_stats.items():
            print(f"  {k}: {v}")

    # ============================================================
    # Item 2 — the spike
    # ============================================================
    print("\n=== Item 2: spike investigation (tuned config) ===")
    tuned = per_config["tuned"]
    alpha_raw_t, alpha_dime_t = tuned["alpha_raw"], tuned["alpha_dime"]
    predicted_spike = np.log((1 - alpha_raw_t) / (1 - alpha_dime_t))
    both_zero = (tuned["p_raw_standalone"] == 0) & (tuned["p_dime_standalone"] == 0)
    near_spike = np.abs(tuned["delta_L"] - predicted_spike) < 1e-4
    near_zero_raw_contrib = tuned["p_raw_standalone"] * alpha_raw_t < 1e-6
    near_zero_dime_contrib = tuned["p_dime_standalone"] * alpha_dime_t < 1e-6
    spike_report = {
        "predicted_delta_L_when_both_zero": float(predicted_spike),
        "n_both_standalone_exactly_zero": int(both_zero.sum()),
        "frac_both_standalone_exactly_zero": float(both_zero.mean()),
        "n_within_1e-4_of_predicted_spike": int(near_spike.sum()),
        "frac_within_1e-4_of_predicted_spike": float(near_spike.mean()),
        "n_both_contributions_negligible": int((near_zero_raw_contrib & near_zero_dime_contrib).sum()),
        "fraction_of_both_zero_cases_that_land_at_predicted_spike": float(near_spike[both_zero].mean()) if both_zero.sum() > 0 else None,
    }
    report["spike_investigation"] = spike_report
    print(json.dumps(spike_report, indent=2))

    fig, ax = plt.subplots(figsize=(6, 6))
    sample_idx = np.random.default_rng(0).choice(len(val_values), size=min(20000, len(val_values)), replace=False)
    ax.scatter(tuned["G_raw"][sample_idx], tuned["G_dime"][sample_idx], s=2, alpha=0.15)
    ax.axhline(0, color="gray", linewidth=0.5)
    ax.axvline(0, color="gray", linewidth=0.5)
    lims = [min(tuned["G_raw"].min(), tuned["G_dime"].min()), max(tuned["G_raw"].max(), tuned["G_dime"].max())]
    ax.plot(lims, lims, "r--", linewidth=1)
    ax.set_xlabel("G_raw"); ax.set_ylabel("G_DIME")
    ax.set_title("Item 2: raw gain vs DIME gain (annotated concentrations)")
    fig.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, "gain_scatter_annotated.png"), dpi=130)
    plt.close(fig)

    # ============================================================
    # Item 3 — matched-config relationships
    # ============================================================
    print("\n=== Item 3: matched-config relationships ===")
    matched = per_config["matched"]
    for pair_name, (x, y) in {
        "distance_vs_js": (abs_delta_d, matched["js"]),
        "js_vs_deltaL": (matched["js"], matched["delta_L"]),
        "distance_vs_deltaL": (abs_delta_d, matched["delta_L"]),
    }.items():
        sp = spearman_with_ci(x, y)
        cm = conditional_means(x, y)
        report.setdefault("matched_relationships", {})[pair_name] = {"spearman": sp, "conditional_means": cm}
        print(f"[matched] {pair_name}: spearman={sp}")

        fig, ax = plt.subplots(figsize=(6, 6))
        ax.scatter(x[sample_idx], y[sample_idx], s=2, alpha=0.15)
        ax.set_xlabel(pair_name.split("_vs_")[0])
        ax.set_ylabel(pair_name.split("_vs_")[1])
        ax.set_title(f"Item 3 (matched config): {pair_name}")
        fig.tight_layout()
        fig.savefig(os.path.join(PLOT_DIR, f"matched_{pair_name}.png"), dpi=130)
        plt.close(fig)

    denom = matched["G_raw"][matched["G_raw"] > 0].sum()
    R_matched = float(matched["G_dime"][matched["G_raw"] > 0].sum() / denom) if denom > 1e-9 else None
    report["matched_relationships"]["gain_retention_R"] = R_matched
    print(f"[matched] gain retention R = {R_matched}")

    # ============================================================
    # Item 4 — 2x2 coverage table (tuned config)
    # ============================================================
    print("\n=== Item 4: 2x2 coverage table (tuned config) ===")
    raw_covers = tuned["p_raw_standalone"] > 0
    dime_covers = tuned["p_dime_standalone"] > 0
    coverage_table = {}
    for name, mask in {
        "both_cover": raw_covers & dime_covers, "raw_only": raw_covers & ~dime_covers,
        "dime_only": ~raw_covers & dime_covers, "neither_covers": ~raw_covers & ~dime_covers,
    }.items():
        n = int(mask.sum())
        coverage_table[name] = {
            "n": n, "mean_delta_L": float(tuned["delta_L"][mask].mean()) if n else None,
            "median_delta_L": float(np.median(tuned["delta_L"][mask])) if n else None,
            "mean_p_lm_true": float(val_p_lm[mask].mean()) if n else None,
            "mean_G_raw": float(tuned["G_raw"][mask].mean()) if n else None,
            "mean_G_dime": float(tuned["G_dime"][mask].mean()) if n else None,
        }
    report["coverage_2x2"] = coverage_table
    print(json.dumps(coverage_table, indent=2))

    # ============================================================
    # Items 5/6 — targeted decode (small, specific query set only)
    # ============================================================
    print("\n=== Items 5/6: targeted context recovery ===")
    order_pos = np.argsort(-tuned["delta_L"])[:15]
    order_neg = np.argsort(tuned["delta_L"])[:15]
    small_d_thresh, high_js_thresh = np.percentile(abs_delta_d, 25), np.percentile(matched["js"], 75)
    small_d_high_js = np.where((abs_delta_d <= small_d_thresh) & (matched["js"] >= high_js_thresh))[0][:15]
    near_zero_raw = np.where(d_raw_nearest < np.percentile(d_raw_nearest, 0.5))[0]  # bottom 0.5% as 'near-zero' audit pool

    target_positions = np.unique(np.concatenate([order_pos, order_neg, small_d_high_js, near_zero_raw[:50]]))
    target_chunks = np.unique(target_positions // POSITIONS_PER_CHUNK)
    print(f"re-tokenizing {len(target_chunks)} val chunks to recover context for {len(target_positions)} target positions...")

    gpt = FrozenGPT2()
    dataset = __import__("datasets").load_dataset("wikitext", "wikitext-103-raw-v1", split="train", streaming=True)
    chunks, _ = collect_chunks_split(dataset, gpt.tokenizer, seq_len=SEQ_LEN,
                                       n_chunks_needed={"datastore": 3000, "controller_train": 500, "val": 500}, seed=42)
    val_chunks = chunks["val"]

    from extract import run_batch
    import torch
    validated_chunks = {}
    for cid in target_chunks:
        h_pred, y_target, p_true, nll = run_batch(gpt, [val_chunks[cid]])
        cached_p = val_p_lm[cid * POSITIONS_PER_CHUNK:(cid + 1) * POSITIONS_PER_CHUNK]
        fresh_p = p_true[0].cpu().numpy()
        max_diff = float(np.abs(cached_p - fresh_p).max())
        validated_chunks[int(cid)] = {"max_p_lm_diff_vs_cache": max_diff, "aligned": max_diff < 1e-3}

    n_misaligned = sum(1 for v in validated_chunks.values() if not v["aligned"])
    print(f"alignment check: {len(validated_chunks)-n_misaligned}/{len(validated_chunks)} chunks aligned "
          f"(max p_lm diff < 1e-3); {n_misaligned} MISALIGNED" + (" -- DO NOT TRUST DECODED CONTEXT BELOW" if n_misaligned else ""))

    def decode_case(i):
        cid = int(i // POSITIONS_PER_CHUNK)
        pos_in_chunk = int(i % POSITIONS_PER_CHUNK)
        aligned = validated_chunks.get(cid, {}).get("aligned", False)
        context_ids = val_chunks[cid][:pos_in_chunk + 1] if aligned else None
        target_id = val_chunks[cid][pos_in_chunk + 1] if aligned else None
        return {
            "position_index": int(i), "chunk_id": cid, "alignment_validated": aligned,
            "decoded_context": gpt.tokenizer.decode(context_ids[-40:]) if aligned else "UNAVAILABLE (alignment check failed)",
            "true_next_token": gpt.tokenizer.decode([target_id]) if aligned else None,
            "true_token_id": int(val_values[i]),
            "p_lm_true": float(val_p_lm[i]),
            "p_raw_standalone": float(tuned["p_raw_standalone"][i]), "p_dime_standalone": float(tuned["p_dime_standalone"][i]),
            "delta_L_tuned": float(tuned["delta_L"][i]), "js_tuned": float(tuned["js"][i]),
            "d_raw_nearest": float(d_raw_nearest[i]), "d_dime_nearest": float(d_dime_nearest[i]),
            "top_raw_neighbor_tokens": [gpt.tokenizer.decode([t]) for t in retrieved_raw_full[i, :5].tolist()],
            "top_raw_neighbor_distances": d_raw_full[i, :5].tolist(),
            "top_dime_cluster_distances": d_dime_full[i, :5].tolist(),
        }

    cases = {
        "largest_positive_deltaL": [decode_case(i) for i in order_pos],
        "largest_negative_deltaL": [decode_case(i) for i in order_neg],
        "small_distance_high_js": [decode_case(i) for i in small_d_high_js],
    }
    with open(os.path.join(FOLLOWUP_DIR, "extreme_cases_decoded.json"), "w") as f:
        json.dump(cases, f, indent=2)

    # Item 6 — near-zero raw-nearest-distance audit
    near_zero_thresh = np.percentile(d_raw_nearest, 0.5)
    near_zero_mask = d_raw_nearest < near_zero_thresh
    near_zero_analysis = {
        "threshold_used_p0.5": float(near_zero_thresh), "n_near_zero": int(near_zero_mask.sum()),
        "frac_near_zero": float(near_zero_mask.mean()),
        "mean_delta_L_near_zero": float(tuned["delta_L"][near_zero_mask].mean()),
        "note": "flagged as a descriptive audit, not automatically labeled leakage per instruction -- "
                "see extreme_cases_decoded.json's small_distance_high_js entries and the chunk-level "
                "alignment check above for whether repeated short contexts/near-duplicate text is visible "
                "in the decoded samples actually inspected.",
    }
    report["near_zero_raw_distance_audit"] = near_zero_analysis
    print(json.dumps(near_zero_analysis, indent=2))

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(d_raw_nearest, bins=200)
    ax.axvline(near_zero_thresh, color="r", linestyle="--", label=f"p0.5 threshold = {near_zero_thresh:.4f}")
    ax.set_xlabel("raw nearest-neighbor distance")
    ax.set_ylabel("count")
    ax.set_title("Item 6: raw nearest-distance distribution (near-zero concentration)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, "near_zero_distance_audit.png"), dpi=130)
    plt.close(fig)

    report["alignment_validation"] = validated_chunks
    save_results(os.path.join(FOLLOWUP_DIR, "followup_summary.json"), report)
    print(f"\nAll follow-up results saved under {FOLLOWUP_DIR}")


if __name__ == "__main__":
    main()
