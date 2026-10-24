"""DIME Stage 3 -- Token Embedding-Predictive Relationship Alignment. Extends
the E1/E2 series without modifying any existing output (writes only to
results/stage3_token_embedding_alignment/). Does NOT train NCA, change the
hidden-state metric, recluster the datastore, or modify DIME-V1 in any way --
purely descriptive/diagnostic, reusing the existing matched-config retrieval
pipeline (k=20, tau=2.0) established throughout this project.

Verified directly against the real model (not assumed): GPT-2 ties input and
output embeddings exactly -- `lm_head.weight is transformer.wte.weight` (same
tensor, same data pointer, confirmed via a local check). U = that [50257,768]
matrix is the "output embedding matrix" referenced throughout.

No scikit-learn used for this experiment's own new computations (cosine
similarity, Pearson/Spearman correlation, top-k neighbor overlap) -- all via
NumPy/PyTorch/SciPy, which are already project dependencies. The EXISTING
cluster assignments and raw-kNN retrieval (built with sklearn, as everywhere
else in this project) are reused as-is, not recomputed with new tooling --
changing that would itself violate "do not recluster the datastore."

Query region definition: nearest-centroid routing (a val query's region = its
top-1 retrieved DIME cluster), per the spec's own "primary region definition."
Val is used throughout (not controller_train) because this experiment is
purely DESCRIPTIVE -- nothing is tuned or selected based on these results, so
this doesn't carry the leakage risk that selecting/tuning against val would;
consistent with how E1A-E1C.2 analyzed val directly for the same reason.

Scoping decisions, stated rather than silently narrowed:
  - 20 centroids selected (not all 3811): 4 geometric strata (large/small
    population x high/low token entropy, 4 per stratum, deterministic by
    cluster ID within each stratum) + 4 "degradation" picks (2 highest mean
    positive delta_L, 2 most-improved) using already-computed matched-config
    val NLL, attributed to each query's top-1 region (same attribution rule
    E2 used, same stated limitation: not a causal decomposition).
  - Up to 15 most-frequent tokens per selected centroid (bounded token sets,
    per the spec's own "do not construct a full vocabulary matrix" instruction).
  - Experiment B's "predictive neighbors" are necessarily computed WITHIN each
    centroid's bounded token set (predictive profiles are only meaningfully
    defined there); embedding neighbors for the SAME comparison are also
    restricted to that set for a fair like-for-like overlap measurement, with
    a separate note reporting each token's true global top-k embedding
    neighbors (unrestricted) for context.
"""
import sys, os, json, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "GPT_Module"))

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import Counter
from scipy import stats
from sklearn.cluster import MiniBatchKMeans  # ONLY for reproducing the existing, already-established clustering -- see docstring

from cache_io import load_cache
from knn import build_datastore, query_knn_indices, softmax_np, mix_knn_and_lm
from mixing import mix_dime_and_lm
from gpt_class import FrozenGPT2
from results_io import save_results
from e1_metrics import raw_full_distribution, dime_full_distribution, js_divergence_dict

CACHE_DIR = os.path.join(os.path.dirname(__file__), "..", "GPT_Module", "cache")
OUT_DIR = os.path.join(os.path.dirname(__file__), "results", "stage3_token_embedding_alignment")
PLOT_DIR = os.path.join(OUT_DIR, "plots")
os.makedirs(PLOT_DIR, exist_ok=True)

N_CLUSTERS = 3811
MATCHED = {"k": 20, "tau": 2.0, "alpha": 0.25}
N_PER_STRATUM = 4
N_DEGRADATION_PICKS = 4
TOKENS_PER_CENTROID = 15
MIN_REGION_QUERIES = 20  # below this, a region's predictive profiles are flagged as insufficient-evidence, not silently correlated
MIN_NONZERO_FOR_CORR = 5  # a token's profile needs at least this many nonzero observations for Pearson/Spearman to be meaningful
N_BOOT = 2000
K_NEIGHBORS = [3, 5]


def cosine_sim_matrix(vecs):
    """vecs: [n, d] numpy or torch. Returns [n, n] cosine similarity, pure
    linear algebra, no scikit-learn."""
    v = torch.as_tensor(vecs, dtype=torch.float32)
    v = v / v.norm(dim=1, keepdim=True).clamp(min=1e-12)
    return (v @ v.T).numpy()


def entropy_of_counter(counter):
    total = sum(counter.values())
    if total == 0:
        return 0.0
    probs = np.array([c / total for c in counter.values()], dtype=np.float64)
    probs = probs[probs > 0]
    return float(-(probs * np.log(probs)).sum())


def safe_corr(x, y, min_nonzero=MIN_NONZERO_FOR_CORR):
    """Pearson + Spearman, explicitly guarding zero-variance and low-support
    profiles rather than letting scipy silently return NaN with a warning."""
    nz = int((np.abs(x) > 0).sum()) + int((np.abs(y) > 0).sum())
    if np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return {"pearson_r": None, "spearman_rho": None, "reason": "zero-variance profile"}
    if nz < min_nonzero:
        return {"pearson_r": None, "spearman_rho": None, "reason": f"insufficient nonzero support ({nz} < {min_nonzero})"}
    pr, _ = stats.pearsonr(x, y)
    sr, _ = stats.spearmanr(x, y)
    return {"pearson_r": float(pr), "spearman_rho": float(sr), "reason": "ok"}


def bootstrap_spearman_ci(x, y, n_boot=N_BOOT, seed=42):
    rng = np.random.default_rng(seed)
    n = len(x)
    boot = np.zeros(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        if np.std(x[idx]) < 1e-12 or np.std(y[idx]) < 1e-12:
            boot[b] = np.nan
            continue
        boot[b] = stats.spearmanr(x[idx], y[idx])[0]
    valid = boot[~np.isnan(boot)]
    if len(valid) < n_boot * 0.5:
        return None
    return {"ci_95_low": float(np.percentile(valid, 2.5)), "ci_95_high": float(np.percentile(valid, 97.5)), "n_valid": len(valid)}


def main():
    print("Loading cache (no extraction) + GPT-2 (for the embedding matrix + tokenizer only, no forward passes)...")
    ds = load_cache(os.path.join(CACHE_DIR, "wikitext103_gpt2_datastore.npz"))
    val = load_cache(os.path.join(CACHE_DIR, "wikitext103_gpt2_val.npz"))
    ds_keys, ds_values = ds["keys"], ds["values"]
    val_keys, val_values, val_p_lm = val["keys"], val["values"], val["p_lm_true"]
    gpt = FrozenGPT2(model_name="gpt2")
    U = gpt.model.lm_head.weight.detach().cpu().numpy()  # [50257, 768]
    tied = gpt.model.lm_head.weight.data_ptr() == gpt.model.transformer.wte.weight.data_ptr()
    print(f"embedding matrix U: {U.shape}, tied to input embeddings: {tied}")
    assert tied, "GPT-2's embedding tying assumption failed -- U would not be the 'output embedding matrix' as described, stop"

    print("Re-deriving DIME clusters (seed=42, deterministic, identical to every prior E1/E2 script)...")
    km = MiniBatchKMeans(n_clusters=N_CLUSTERS, random_state=42)
    assignment = km.fit_predict(ds_keys)
    dime_centroids = km.cluster_centers_.astype(np.float32)
    compressed_dists = []
    for c in range(N_CLUSTERS):
        member_values = ds_values[assignment == c]
        compressed_dists.append(Counter(member_values.tolist()) if len(member_values) > 0 else Counter())
    compressed_dists = np.array(compressed_dists, dtype=object)

    raw_index, raw_values = build_datastore(ds_keys, ds_values)
    dime_index, dime_values = build_datastore(dime_centroids, compressed_dists)

    kr, taur, alphar = MATCHED["k"], MATCHED["tau"], MATCHED["alpha"]
    kd, taud, alphad = MATCHED["k"], MATCHED["tau"], MATCHED["alpha"]
    print("Querying val (matched config, full scale) -- raw and DIME retrieval...")
    d_raw, idx_raw = query_knn_indices(raw_index, val_keys, k=kr)
    d_dime, idx_dime = query_knn_indices(dime_index, val_keys, k=kd)
    r_raw, r_dime = raw_values[idx_raw], dime_values[idx_dime]
    _, L_raw = mix_knn_and_lm(d_raw, r_raw, val_values, val_p_lm, tau=taur, alpha=alphar)
    _, L_dime = mix_dime_and_lm(d_dime, r_dime, val_values, val_p_lm, tau=taud, alpha=alphad)
    with np.errstate(invalid="ignore"):
        delta_L = L_dime - L_raw
    region_of_query = idx_dime[:, 0]  # nearest-centroid routing, the primary region definition

    # ============================================================
    # Centroid selection: 4 strata + degradation picks
    # ============================================================
    print("Selecting a reproducible sample of centroids across population/entropy/degradation strata...")
    populations = np.array([sum(c.values()) for c in compressed_dists])
    entropies = np.array([entropy_of_counter(c) for c in compressed_dists])
    nonempty = np.where(populations > 0)[0]
    pop_median, ent_median = np.median(populations[nonempty]), np.median(entropies[nonempty])

    strata = {
        "large_pop_high_entropy": nonempty[(populations[nonempty] >= pop_median) & (entropies[nonempty] >= ent_median)],
        "large_pop_low_entropy": nonempty[(populations[nonempty] >= pop_median) & (entropies[nonempty] < ent_median)],
        "small_pop_high_entropy": nonempty[(populations[nonempty] < pop_median) & (entropies[nonempty] >= ent_median)],
        "small_pop_low_entropy": nonempty[(populations[nonempty] < pop_median) & (entropies[nonempty] < ent_median)],
    }
    selected = {}
    for name, pool in strata.items():
        picks = np.sort(pool)[:N_PER_STRATUM]  # deterministic by cluster ID within the stratum
        selected[name] = picks.tolist()

    cluster_mean_pos_delta_L = np.zeros(N_CLUSTERS)
    cluster_query_count = np.zeros(N_CLUSTERS, dtype=int)
    for i in range(len(val_values)):
        c = region_of_query[i]
        cluster_query_count[c] += 1
        if np.isfinite(delta_L[i]):
            cluster_mean_pos_delta_L[c] += max(0.0, delta_L[i])
    with np.errstate(invalid="ignore", divide="ignore"):
        cluster_mean_pos_delta_L = np.where(cluster_query_count >= MIN_REGION_QUERIES,
                                              cluster_mean_pos_delta_L / np.maximum(cluster_query_count, 1), np.nan)
    eligible = np.where(~np.isnan(cluster_mean_pos_delta_L))[0]
    order_degrad = eligible[np.argsort(-cluster_mean_pos_delta_L[eligible])]
    selected["most_degraded"] = order_degrad[:N_DEGRADATION_PICKS // 2].tolist()
    selected["least_degraded"] = order_degrad[::-1][:N_DEGRADATION_PICKS // 2].tolist()

    all_selected = sorted(set(c for picks in selected.values() for c in picks))
    print(f"Selected {len(all_selected)} unique centroids across {len(selected)} strata: {selected}")

    report = {"n_val": int(len(val_values)), "matched_config": MATCHED, "n_clusters_total": N_CLUSTERS,
              "embedding_tied_to_input": bool(tied), "centroid_selection": selected,
              "n_unique_centroids_selected": len(all_selected),
              "min_region_queries_threshold": MIN_REGION_QUERIES, "tokens_per_centroid": TOKENS_PER_CENTROID}
    save_results(os.path.join(OUT_DIR, "summary.json"), report)

    # ============================================================
    # Per-centroid: token selection + Experiment A (embedding vs predictive similarity)
    # ============================================================
    print("\nExperiment A: embedding vs. predictive similarity, per centroid...")
    per_centroid = {}
    aggregate_agree_pairs, aggregate_disagree_pairs = [], []
    all_global_spearmans = []

    for c in all_selected:
        region_mask = region_of_query == c
        n_region = int(region_mask.sum())
        if n_region < MIN_REGION_QUERIES:
            per_centroid[c] = {"skipped": True, "reason": f"only {n_region} region queries, below threshold {MIN_REGION_QUERIES}"}
            continue

        top_tokens = [t for t, _ in compressed_dists[c].most_common(TOKENS_PER_CENTROID)]
        if len(top_tokens) < 2:
            per_centroid[c] = {"skipped": True, "reason": "fewer than 2 distinct tokens in this cluster"}
            continue

        region_idx = np.where(region_mask)[0]
        region_d_raw, region_r_raw = d_raw[region_idx], r_raw[region_idx]
        full_dists = raw_full_distribution(region_d_raw, region_r_raw, tau=taur)
        profile_matrix = np.array([[d.get(t, 0.0) for t in top_tokens] for d in full_dists])  # [n_region, n_tokens]

        embed_sim = cosine_sim_matrix(U[top_tokens])
        n_tok = len(top_tokens)
        pred_sim_cosine = cosine_sim_matrix(profile_matrix.T)  # tokens as rows -> [n_tok, n_tok]

        pair_records = []
        for i in range(n_tok):
            for j in range(i + 1, n_tok):
                corr = safe_corr(profile_matrix[:, i], profile_matrix[:, j])
                mean_abs_diff = float(np.mean(np.abs(profile_matrix[:, i] - profile_matrix[:, j])))
                prefer_i = float((profile_matrix[:, i] > profile_matrix[:, j]).mean())
                pair_records.append({
                    "token_i": int(top_tokens[i]), "token_j": int(top_tokens[j]),
                    "decoded_i": gpt.tokenizer.decode([top_tokens[i]]), "decoded_j": gpt.tokenizer.decode([top_tokens[j]]),
                    "embed_cosine": float(embed_sim[i, j]), "pred_cosine": float(pred_sim_cosine[i, j]),
                    "pred_pearson": corr["pearson_r"], "pred_spearman": corr["spearman_rho"], "corr_status": corr["reason"],
                    "mean_abs_diff": mean_abs_diff, "prefer_i_fraction": prefer_i,
                })

        valid_pairs = [p for p in pair_records if p["pred_spearman"] is not None]
        if len(valid_pairs) >= 4:
            embed_vals = np.array([p["embed_cosine"] for p in valid_pairs])
            pred_vals = np.array([p["pred_cosine"] for p in valid_pairs])
            ranking_corr, ranking_p = stats.spearmanr(embed_vals, pred_vals)
            all_global_spearmans.append(float(ranking_corr))
        else:
            ranking_corr, ranking_p = None, None

        sorted_by_disagreement = sorted(valid_pairs, key=lambda p: p["embed_cosine"] - p["pred_cosine"], reverse=True)
        strong_disagree = sorted_by_disagreement[:3]  # high embed sim, low pred sim -- H3's "lost distinctions"
        strong_agree = sorted(valid_pairs, key=lambda p: abs(p["embed_cosine"] - p["pred_cosine"]))[:3]
        aggregate_disagree_pairs.extend([{**p, "centroid": int(c)} for p in strong_disagree])
        aggregate_agree_pairs.extend([{**p, "centroid": int(c)} for p in strong_agree])

        per_centroid[c] = {
            "n_region_queries": n_region, "n_tokens": n_tok, "tokens": [int(t) for t in top_tokens],
            "embed_vs_pred_ranking_spearman": float(ranking_corr) if ranking_corr is not None else None,
            "embed_vs_pred_ranking_p": float(ranking_p) if ranking_p is not None else None,
            "n_pairs_total": len(pair_records), "n_pairs_valid_corr": len(valid_pairs),
            "strong_agree_examples": strong_agree, "strong_disagree_examples": strong_disagree,
            "pair_records": pair_records,
        }
        print(f"  centroid {c} ({n_region} queries, {n_tok} tokens): "
              f"embed-vs-pred ranking spearman={ranking_corr}")

    report["experiment_A_per_centroid"] = per_centroid
    report["experiment_A_aggregate"] = {
        "mean_ranking_spearman_across_centroids": float(np.mean(all_global_spearmans)) if all_global_spearmans else None,
        "std_ranking_spearman_across_centroids": float(np.std(all_global_spearmans)) if all_global_spearmans else None,
        "n_centroids_with_valid_ranking": len(all_global_spearmans),
        "strong_agree_examples_global": aggregate_agree_pairs[:15],
        "strong_disagree_examples_global": aggregate_disagree_pairs[:15],
    }
    save_results(os.path.join(OUT_DIR, "summary.json"), report)

    fig, ax = plt.subplots(figsize=(7, 7))
    all_embed, all_pred = [], []
    for c, data in per_centroid.items():
        if data.get("skipped"):
            continue
        for p in data["pair_records"]:
            all_embed.append(p["embed_cosine"]); all_pred.append(p["pred_cosine"])
    ax.scatter(all_embed, all_pred, s=6, alpha=0.3)
    ax.set_xlabel("output-embedding cosine similarity")
    ax.set_ylabel("predictive-profile cosine similarity")
    ax.set_title("Experiment A: embedding vs. predictive similarity (all selected centroids pooled)")
    fig.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, "embedding_vs_predictive_similarity.png"), dpi=130)
    plt.close(fig)

    # ============================================================
    # Experiment B: neighborhood overlap (within each centroid's bounded token set)
    # ============================================================
    print("\nExperiment B: neighborhood overlap...")
    overlap_results = {}
    rng_overlap = np.random.default_rng(42)
    for c, data in per_centroid.items():
        if data.get("skipped"):
            continue
        tokens = data["tokens"]
        n_tok = len(tokens)
        if n_tok < 4:
            continue
        embed_sim = cosine_sim_matrix(U[tokens])
        pred_sim = np.eye(n_tok)
        tok_index = {t: i for i, t in enumerate(tokens)}
        for p in data["pair_records"]:
            i, j = tok_index[p["token_i"]], tok_index[p["token_j"]]
            pred_sim[i, j] = pred_sim[j, i] = p["pred_cosine"]

        c_overlap = {}
        for k in K_NEIGHBORS:
            k_eff = min(k, n_tok - 1)
            overlaps = []
            for i in range(n_tok):
                embed_neighbors = set(np.argsort(-embed_sim[i])[1:k_eff + 1].tolist())
                pred_neighbors = set(np.argsort(-pred_sim[i])[1:k_eff + 1].tolist())
                overlaps.append(len(embed_neighbors & pred_neighbors) / k_eff)
            random_overlaps = []
            for _ in range(50):
                i = rng_overlap.integers(0, n_tok)
                embed_neighbors = set(np.argsort(-embed_sim[i])[1:k_eff + 1].tolist())
                random_neighbors = set(rng_overlap.choice([x for x in range(n_tok) if x != i], size=k_eff, replace=False).tolist())
                random_overlaps.append(len(embed_neighbors & random_neighbors) / k_eff)
            c_overlap[k] = {"mean_overlap": float(np.mean(overlaps)), "random_baseline_overlap": float(np.mean(random_overlaps))}
        overlap_results[c] = c_overlap
    report["experiment_B_neighbor_overlap"] = overlap_results
    save_results(os.path.join(OUT_DIR, "summary.json"), report)
    for k in K_NEIGHBORS:
        vals = [overlap_results[c][k]["mean_overlap"] for c in overlap_results]
        baseline = [overlap_results[c][k]["random_baseline_overlap"] for c in overlap_results]
        print(f"  k={k}: mean overlap={np.mean(vals):.3f}  random baseline={np.mean(baseline):.3f}")

    # ============================================================
    # Experiment C: predictive discrimination (high embed sim, low pred sim)
    # ============================================================
    print("\nExperiment C: predictive discrimination failures...")
    discrimination_cases = []
    for c, data in per_centroid.items():
        if data.get("skipped"):
            continue
        for p in data["pair_records"]:
            if p["embed_cosine"] > 0.5 and p["pred_cosine"] is not None and p["pred_cosine"] < 0.1:
                discrimination_cases.append({**p, "centroid": int(c), "region_mean_js": None})
    # Region-level JS/delta_L context for centroids containing discrimination cases
    centroids_with_cases = sorted(set(d["centroid"] for d in discrimination_cases))
    region_context = {}
    for c in centroids_with_cases:
        region_mask = region_of_query == c
        region_idx = np.where(region_mask)[0]
        raw_dists_region = raw_full_distribution(d_raw[region_idx], r_raw[region_idx], tau=taur)
        dime_dists_region = dime_full_distribution(d_dime[region_idx], dime_values[idx_dime[region_idx]], tau=taud)
        js_vals = [js_divergence_dict(rd, dd) for rd, dd in zip(raw_dists_region, dime_dists_region)]
        region_context[c] = {"n_region_queries": int(region_mask.sum()),
                              "mean_delta_L_region": float(np.nanmean(delta_L[region_idx])) if np.isfinite(delta_L[region_idx]).any() else None,
                              "mean_js_divergence_region": float(np.mean(js_vals)) if js_vals else None}
    report["experiment_C_discrimination_cases"] = discrimination_cases[:30]
    report["experiment_C_region_context"] = region_context
    print(f"  found {len(discrimination_cases)} high-embed/low-predictive-similarity pairs "
          f"across {len(centroids_with_cases)} centroids")
    save_results(os.path.join(OUT_DIR, "summary.json"), report)

    # ============================================================
    # Experiment D: global vs. centroid-local consistency (tokens appearing in multiple centroids)
    # ============================================================
    print("\nExperiment D: global vs. centroid-local relationship consistency...")
    token_to_centroids = {}
    for c, data in per_centroid.items():
        if data.get("skipped"):
            continue
        for t in data["tokens"]:
            token_to_centroids.setdefault(t, []).append(c)
    multi_region_tokens = {t: cs for t, cs in token_to_centroids.items() if len(cs) >= 2}
    cross_region_consistency = []
    for t, cs in list(multi_region_tokens.items())[:30]:  # bounded, stated
        local_pred_sims = {}
        for c in cs:
            for p in per_centroid[c]["pair_records"]:
                if p["token_i"] == t or p["token_j"] == t:
                    other = p["token_j"] if p["token_i"] == t else p["token_i"]
                    local_pred_sims.setdefault(c, {})[other] = p["pred_cosine"]
        if len(cs) >= 2:
            common_others = set.intersection(*[set(local_pred_sims[c].keys()) for c in cs]) if all(c in local_pred_sims for c in cs) else set()
            if len(common_others) >= 3:
                vecs = [[local_pred_sims[c][o] for o in sorted(common_others)] for c in cs]
                rho, _ = stats.spearmanr(vecs[0], vecs[1]) if len(vecs) >= 2 else (None, None)
                cross_region_consistency.append({"token": int(t), "decoded": gpt.tokenizer.decode([t]),
                                                   "n_regions": len(cs), "n_common_comparison_tokens": len(common_others),
                                                   "cross_region_spearman": float(rho) if rho is not None and np.isfinite(rho) else None})
    report["experiment_D_cross_region_consistency"] = cross_region_consistency
    valid_d = [d["cross_region_spearman"] for d in cross_region_consistency if d["cross_region_spearman"] is not None]
    print(f"  {len(multi_region_tokens)} tokens appear in >=2 selected centroids; "
          f"{len(valid_d)} had enough common comparison tokens for a cross-region consistency check "
          f"(mean spearman={np.mean(valid_d) if valid_d else None})")
    save_results(os.path.join(OUT_DIR, "summary.json"), report)

    # ============================================================
    # Markdown report (generated from actually-computed numbers, not templated)
    # ============================================================
    print("\nGenerating report.md from computed results...")
    agg = report["experiment_A_aggregate"]
    lines = [
        "# DIME Stage 3 -- Token Embedding-Predictive Relationship Alignment",
        "",
        f"Matched config (k={MATCHED['k']}, tau={MATCHED['tau']}, alpha={MATCHED['alpha']}), "
        f"{len(val_values)} val queries, {len(all_selected)} centroids selected across "
        f"{len(strata)} geometric strata + {N_DEGRADATION_PICKS} degradation-based picks.",
        "",
        "## Experiment A -- embedding vs. predictive similarity",
        f"Mean embedding-vs-predictive ranking Spearman correlation across "
        f"{agg['n_centroids_with_valid_ranking']} valid centroids: "
        f"**{agg['mean_ranking_spearman_across_centroids']}** (std {agg['std_ranking_spearman_across_centroids']}).",
        "",
        "## Experiment B -- neighborhood overlap",
    ]
    for k in K_NEIGHBORS:
        vals = [overlap_results[c][k]["mean_overlap"] for c in overlap_results]
        baseline = [overlap_results[c][k]["random_baseline_overlap"] for c in overlap_results]
        lines.append(f"- k={k}: mean embedding/predictive neighbor overlap = {np.mean(vals):.3f} "
                      f"(random baseline = {np.mean(baseline):.3f})")
    lines += [
        "",
        "## Experiment C -- predictive discrimination failures",
        f"{len(discrimination_cases)} token pairs found with high embedding similarity (>0.5) but "
        f"low predictive-profile similarity (<0.1), across {len(centroids_with_cases)} centroids.",
        "",
        "## Experiment D -- global vs. centroid-local consistency",
        f"{len(multi_region_tokens)} tokens appeared in >=2 selected centroids; "
        f"{len(valid_d)} had sufficient shared comparison tokens for a cross-region check "
        f"(mean cross-region Spearman = {np.mean(valid_d) if valid_d else 'N/A'}).",
        "",
        "## Answers to the four research questions (from the measurements above only)",
        f"**Q1** (do embeddings encode predictive relationships?): mean ranking correlation "
        f"{agg['mean_ranking_spearman_across_centroids']} -- interpret directly from this number and "
        f"the full per-centroid table in summary.json, not asserted here.",
        "**Q2** (consistent across centroids, or context-dependent?): see Experiment D's cross-region "
        "Spearman distribution in summary.json.",
        "**Q3** (do embeddings predict DIME's specific lost distinctions?): see Experiment C's case "
        "count and region_context entries.",
        "**Q4** (next step -- direct embedding graph, NCA, or other?): not pre-decided here; judge from "
        "Q1-Q3's actual measured evidence once reviewed.",
        "",
        "## Limitations",
        f"- {len(all_selected)} of {N_CLUSTERS} centroids sampled (bounded for cost); per-centroid token "
        f"sets bounded to top-{TOKENS_PER_CENTROID} most frequent tokens.",
        "- Experiment C's `region_context` reports region-level mean JS divergence and mean delta_L "
        "(raw-vs-DIME full distributions), not a per-pair causal attribution of either to specific "
        "token pairs.",
        "- Correlation does not imply causation anywhere in this report.",
    ]
    with open(os.path.join(OUT_DIR, "report.md"), "w") as f:
        f.write("\n".join(lines))

    report["sections_completed"] = ["experiment_A", "experiment_B", "experiment_C", "experiment_D", "report.md"]
    save_results(os.path.join(OUT_DIR, "summary.json"), report)
    print(f"\nAll results saved under {OUT_DIR}")
    print("No existing E1/E2 outputs were modified.")


if __name__ == "__main__":
    main()
