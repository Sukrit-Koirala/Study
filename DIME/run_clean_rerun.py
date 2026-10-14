"""Clean rerun: DIME(h_out) vs DIME(h_att), plus Xu et al. Section 4.1's learned-head
comparison, plus two domain-adaptation controls added after the audit of the first
run flagged the FFN head's gain as plausibly explained by corpus-frequency fitting
rather than genuine representation value.

Nothing is loaded from any earlier run — no cached hidden states, datastores,
trained heads, or tuned hyperparameters. Every artifact here is rebuilt from the
frozen pretrained model from scratch (confirmed by the earlier audit that nothing
from the first run was ever persisted to disk in the first place, so there is
nothing to accidentally reuse). Results go to a new directory
(`results_clean_rerun/`), entirely separate from `results/representation_ensembling_experiment.json`
and its audit — neither is touched.

Configs:
  A. Base LM
  B. Base + DIME(h_out)              C. Base + DIME(h_att)
  D. Base + learned FFN head (h_out) E. Base + learned ATT head (h_att)
  F. Base + full raw kNN (h_out)     G. Base + full raw kNN (h_att)
  H. Base + unigram control (domain-adaptation control: marginal frequency only)
  I. Base + FFN head trained on SHUFFLED labels (domain-adaptation control:
     isolates marginal-frequency fitting from genuine context-dependent learning)

Both standalone (alpha=1.0, no base-LM blending) and validation-tuned-interpolated
NLL are logged for every applicable config — the audit flagged this as missing
before. Per-position NLL arrays are saved to a companion .npz, not just scalars, so
future re-analysis doesn't require rerunning.
"""
import sys, os, argparse, json, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "GPT_Module"))

import numpy as np
from scipy import stats

from gpt_class import FrozenGPT2
from dataset_helpers import collect_chunks_split
from extract_dual import build_dual_datastore_from_chunks, encode_dual_split
from knn import build_datastore, query_knn, mix_knn_and_lm
from mixing import mix_dime_and_lm, build_global_freq
from state_object import minibatch_kmeans_partition
from grid_search import grid_search_hyperparams
from results_io import save_results
from sklearn.cluster import MiniBatchKMeans

from learned_head import train_head, head_log_probs_for_targets

DATASET_REGISTRY = {
    "tinystories": lambda: __import__("datasets").load_dataset(
        "roneneldan/TinyStories", split="train", streaming=True, trust_remote_code=True),
    "wikitext103": lambda: __import__("datasets").load_dataset(
        "wikitext", "wikitext-103-raw-v1", split="train", streaming=True),
}
CHUNK_BUDGET = {
    "tinystories": {"datastore": 350, "controller_train": 75, "val": 75},
    "wikitext103": {"datastore": 3000, "controller_train": 500, "val": 500},
}

OUT_DIR = "results_clean_rerun"
K_VALUES = [10, 20, 50, 100, 200, 300]
TAU_VALUES = [0.5, 1.0, 2.0, 5.0, 10.0]
ALPHA_VALUES = [0.01, 0.05, 0.1, 0.25, 0.5]


def eps_log(p):
    return -np.log(np.clip(p, 1e-12, None))


def tune_alpha(p_base_val, p_other_val, p_base_test, p_other_test, grid=np.round(np.arange(0.0, 1.0001, 0.05), 2)):
    best_alpha, best_nll = None, float("inf")
    for alpha in grid:
        nll = eps_log(alpha * p_other_val + (1 - alpha) * p_base_val).mean()
        if nll < best_nll:
            best_nll, best_alpha = nll, float(alpha)
    nll_test = eps_log(best_alpha * p_other_test + (1 - best_alpha) * p_base_test)
    return best_alpha, float(nll_test.mean()), nll_test


def retrieval_config(name, keys, values, query_ct, query_val,
                      ct_values, ct_p_lm, val_values, val_p_lm, k_values, mix_fn):
    index, stored = build_datastore(keys, values)
    k_max = max([k for k in k_values if k <= len(keys)] or [min(len(keys), max(k_values))])
    dist_ct, retr_ct = query_knn(index, stored, query_ct, k=k_max)
    _, best = grid_search_hyperparams(dist_ct, retr_ct, ct_values, ct_p_lm, mix_fn,
                                       [k for k in k_values if k <= len(keys)], TAU_VALUES, ALPHA_VALUES)
    dist_val, retr_val = query_knn(index, stored, query_val, k=best["k"])
    p_mixed_test, nll_test = mix_fn(dist_val, retr_val, val_values, val_p_lm, tau=best["tau"], alpha=best["alpha"])
    _, nll_standalone_test = mix_fn(dist_val, retr_val, val_values, val_p_lm, tau=best["tau"], alpha=1.0)
    print(f"  {name}: {best} -> interpolated test NLL={nll_test.mean():.4f}  standalone test NLL={nll_standalone_test.mean():.4f}")
    return {"best_config": best, "interpolated_test_nll": float(nll_test.mean()),
            "standalone_test_nll": float(nll_standalone_test.mean())}, nll_test


def document_bootstrap_ci(story_ids, nll_a, nll_b, n_boot=2000, seed=42):
    rng = np.random.default_rng(seed)
    unique_stories = np.unique(story_ids)
    story_to_positions = {s: np.where(story_ids == s)[0] for s in unique_stories}
    point_estimate = float((nll_a - nll_b).mean())
    boot_means = np.zeros(n_boot)
    for b in range(n_boot):
        sampled = rng.choice(unique_stories, size=len(unique_stories), replace=True)
        idx = np.concatenate([story_to_positions[s] for s in sampled])
        boot_means[b] = (nll_a[idx] - nll_b[idx]).mean()
    ci_lo, ci_hi = np.percentile(boot_means, [2.5, 97.5])
    return {"point_estimate": point_estimate, "ci_95_low": float(ci_lo), "ci_95_high": float(ci_hi),
            "n_documents": int(len(unique_stories)), "n_bootstrap": n_boot}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=list(DATASET_REGISTRY.keys()))
    parser.add_argument("--model", required=True)
    parser.add_argument("--n_clusters", type=int, default=None)
    parser.add_argument("--seq_len", type=int, default=128)
    args = parser.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    prefix = f"{args.dataset}_{args.model}"
    out_json = os.path.join(OUT_DIR, f"{prefix}_summary.json")
    out_npz = os.path.join(OUT_DIR, f"{prefix}_positions.npz")

    results = {"phase": "clean_rerun", "dataset": args.dataset, "model": args.model,
               "split_mapping": {"train": "datastore", "validation": "controller_train", "test": "val"}}

    gpt = FrozenGPT2(model_name=args.model)
    dataset = DATASET_REGISTRY[args.dataset]()
    chunks, chunk_story_ids = collect_chunks_split(
        dataset, gpt.tokenizer, seq_len=args.seq_len,
        n_chunks_needed=CHUNK_BUDGET[args.dataset], seed=42,
    )

    print(f"[{prefix}] dual-extracting datastore / controller_train(VALIDATION) / val(TEST)...")
    t0 = time.perf_counter()
    ds_ffn, ds_att, ds_values = build_dual_datastore_from_chunks(gpt, chunks["datastore"], batch_size=256)
    ct_ffn, ct_att, ct_values, ct_p_lm = encode_dual_split(gpt, chunks["controller_train"], batch_size=256)
    val_ffn, val_att, val_values, val_p_lm = encode_dual_split(gpt, chunks["val"], batch_size=256)
    extraction_time = time.perf_counter() - t0
    print(f"[{prefix}] datastore={len(ds_values)} controller_train={len(ct_values)} val={len(val_values)}  "
          f"[{extraction_time:.1f}s]")

    positions_per_chunk = args.seq_len - 1
    val_story_ids = np.concatenate([np.full(positions_per_chunk, sid) for sid in chunk_story_ids["val"]])
    assert len(val_story_ids) == len(val_values), "story-id expansion length mismatch — check chunk length assumption"

    n_clusters = args.n_clusters or max(100, round(len(ds_values) / 100))
    print(f"[{prefix}] n_clusters={n_clusters}")

    results["data_sizes"] = {"datastore": len(ds_values), "controller_train": len(ct_values), "val": len(val_values)}
    results["n_clusters"] = n_clusters
    results["extraction_time_sec"] = extraction_time
    save_results(out_json, results)

    nll_arrays = {}

    # ============================================================
    # Config A — base LM
    # ============================================================
    nll_A_test = eps_log(val_p_lm)
    nll_arrays["A_base"] = nll_A_test
    print(f"[{prefix}] A (base LM): test NLL={nll_A_test.mean():.4f}")
    results["configs"] = {"A_base_lm": {"test_nll": float(nll_A_test.mean())}}
    save_results(out_json, results)

    # ============================================================
    # Configs B/C — DIME(h_out) / DIME(h_att), separate clustering fits
    # ============================================================
    print(f"[{prefix}] --- Config B: DIME(h_out) ---")
    t0 = time.perf_counter()
    ck_out, cd_out = minibatch_kmeans_partition(ds_ffn, ds_values, n_clusters=n_clusters, seed=42)
    (b_result, nll_B_test) = retrieval_config(
        "B (DIME/h_out)", ck_out, np.array(cd_out, dtype=object),
        ct_ffn, val_ffn, ct_values, ct_p_lm, val_values, val_p_lm, K_VALUES, mix_dime_and_lm,
    )
    b_result["train_time_sec"] = time.perf_counter() - t0
    nll_arrays["B_dime_hout"] = nll_B_test

    print(f"[{prefix}] --- Config C: DIME(h_att) ---")
    t0 = time.perf_counter()
    ck_att, cd_att = minibatch_kmeans_partition(ds_att, ds_values, n_clusters=n_clusters, seed=42)
    (c_result, nll_C_test) = retrieval_config(
        "C (DIME/h_att)", ck_att, np.array(cd_att, dtype=object),
        ct_att, val_att, ct_values, ct_p_lm, val_values, val_p_lm, K_VALUES, mix_dime_and_lm,
    )
    c_result["train_time_sec"] = time.perf_counter() - t0
    nll_arrays["C_dime_hatt"] = nll_C_test

    results["configs"]["B_dime_hout"] = b_result
    results["configs"]["C_dime_hatt"] = c_result
    save_results(out_json, results)

    # ============================================================
    # Configs F/G — full raw kNN(h_out) / full raw kNN(h_att)
    # ============================================================
    print(f"[{prefix}] --- Config F: full raw kNN(h_out) ---")
    t0 = time.perf_counter()
    (f_result, nll_F_test) = retrieval_config(
        "F (raw kNN/h_out)", ds_ffn, ds_values, ct_ffn, val_ffn,
        ct_values, ct_p_lm, val_values, val_p_lm, K_VALUES, mix_knn_and_lm,
    )
    f_result["train_time_sec"] = time.perf_counter() - t0
    nll_arrays["F_rawknn_hout"] = nll_F_test

    print(f"[{prefix}] --- Config G: full raw kNN(h_att) ---")
    t0 = time.perf_counter()
    (g_result, nll_G_test) = retrieval_config(
        "G (raw kNN/h_att)", ds_att, ds_values, ct_att, val_att,
        ct_values, ct_p_lm, val_values, val_p_lm, K_VALUES, mix_knn_and_lm,
    )
    g_result["train_time_sec"] = time.perf_counter() - t0
    nll_arrays["G_rawknn_hatt"] = nll_G_test

    results["configs"]["F_rawknn_hout"] = f_result
    results["configs"]["G_rawknn_hatt"] = g_result
    save_results(out_json, results)

    # ============================================================
    # Configs D/E — learned heads (h_out / h_att)
    # ============================================================
    print(f"[{prefix}] --- Config D: learned FFN head (h_out) ---")
    t0 = time.perf_counter()
    head_ffn, hist_ffn, best_val_ffn = train_head(ds_ffn, ds_values, ct_ffn, ct_values, verbose=False)
    train_time_d = time.perf_counter() - t0
    p_head_ffn_ct = np.exp(head_log_probs_for_targets(head_ffn, ct_ffn, ct_values))
    p_head_ffn_test = np.exp(head_log_probs_for_targets(head_ffn, val_ffn, val_values))
    alpha_D, nll_D_interp, nll_D_test = tune_alpha(ct_p_lm, p_head_ffn_ct, val_p_lm, p_head_ffn_test)
    nll_D_standalone = eps_log(p_head_ffn_test)
    print(f"  D: alpha={alpha_D}  interpolated test NLL={nll_D_interp:.4f}  standalone test NLL={nll_D_standalone.mean():.4f}")
    nll_arrays["D_ffn_head_interp"] = nll_D_test
    nll_arrays["D_ffn_head_standalone"] = nll_D_standalone

    print(f"[{prefix}] --- Config E: learned ATT head (h_att) ---")
    t0 = time.perf_counter()
    head_att, hist_att, best_val_att = train_head(ds_att, ds_values, ct_att, ct_values, verbose=False)
    train_time_e = time.perf_counter() - t0
    p_head_att_ct = np.exp(head_log_probs_for_targets(head_att, ct_att, ct_values))
    p_head_att_test = np.exp(head_log_probs_for_targets(head_att, val_att, val_values))
    alpha_E, nll_E_interp, nll_E_test = tune_alpha(ct_p_lm, p_head_att_ct, val_p_lm, p_head_att_test)
    nll_E_standalone = eps_log(p_head_att_test)
    print(f"  E: alpha={alpha_E}  interpolated test NLL={nll_E_interp:.4f}  standalone test NLL={nll_E_standalone.mean():.4f}")
    nll_arrays["E_att_head_interp"] = nll_E_test
    nll_arrays["E_att_head_standalone"] = nll_E_standalone

    results["configs"]["D_ffn_head"] = {
        "alpha": alpha_D, "interpolated_test_nll": nll_D_interp, "standalone_test_nll": float(nll_D_standalone.mean()),
        "train_time_sec": train_time_d, "memory_bytes": head_ffn.memory_bytes(),
    }
    results["configs"]["E_att_head"] = {
        "alpha": alpha_E, "interpolated_test_nll": nll_E_interp, "standalone_test_nll": float(nll_E_standalone.mean()),
        "train_time_sec": train_time_e, "memory_bytes": head_att.memory_bytes(),
    }
    save_results(out_json, results)

    # ============================================================
    # Config H — unigram control (domain-adaptation: marginal frequency only)
    # ============================================================
    print(f"[{prefix}] --- Config H: unigram control ---")
    global_freq = build_global_freq(ds_values)
    p_unigram_ct = global_freq[ct_values]
    p_unigram_test = global_freq[val_values]
    alpha_H, nll_H_interp, nll_H_test = tune_alpha(ct_p_lm, p_unigram_ct, val_p_lm, p_unigram_test)
    nll_H_standalone = eps_log(p_unigram_test)
    print(f"  H: alpha={alpha_H}  interpolated test NLL={nll_H_interp:.4f}  standalone test NLL={nll_H_standalone.mean():.4f}")
    nll_arrays["H_unigram_interp"] = nll_H_test
    nll_arrays["H_unigram_standalone"] = nll_H_standalone
    results["configs"]["H_unigram_control"] = {
        "alpha": alpha_H, "interpolated_test_nll": nll_H_interp, "standalone_test_nll": float(nll_H_standalone.mean()),
    }
    save_results(out_json, results)

    # ============================================================
    # Config I — FFN head trained on SHUFFLED labels (isolates marginal-frequency
    # fitting from genuine context-dependent learning: same h_out contexts, same
    # marginal token-frequency distribution, but context<->token pairing destroyed)
    # ============================================================
    print(f"[{prefix}] --- Config I: FFN head trained on shuffled labels ---")
    rng = np.random.default_rng(42)
    shuffled_ds_values = ds_values[rng.permutation(len(ds_values))]
    t0 = time.perf_counter()
    head_shuf, hist_shuf, best_val_shuf = train_head(ds_ffn, shuffled_ds_values, ct_ffn, ct_values, verbose=False)
    train_time_i = time.perf_counter() - t0
    p_shuf_ct = np.exp(head_log_probs_for_targets(head_shuf, ct_ffn, ct_values))
    p_shuf_test = np.exp(head_log_probs_for_targets(head_shuf, val_ffn, val_values))
    alpha_I, nll_I_interp, nll_I_test = tune_alpha(ct_p_lm, p_shuf_ct, val_p_lm, p_shuf_test)
    nll_I_standalone = eps_log(p_shuf_test)
    print(f"  I: alpha={alpha_I}  interpolated test NLL={nll_I_interp:.4f}  standalone test NLL={nll_I_standalone.mean():.4f}")
    nll_arrays["I_shuffled_head_interp"] = nll_I_test
    nll_arrays["I_shuffled_head_standalone"] = nll_I_standalone
    results["configs"]["I_shuffled_label_head"] = {
        "alpha": alpha_I, "interpolated_test_nll": nll_I_interp, "standalone_test_nll": float(nll_I_standalone.mean()),
        "train_time_sec": train_time_i,
        "note": "same h_out contexts and same marginal token-frequency distribution as D, "
                "but context<->token pairing destroyed by shuffling labels before training.",
    }
    save_results(out_json, results)

    # ============================================================
    # Primary comparisons — paired tests + document-level bootstrap
    # ============================================================
    print(f"[{prefix}] --- Primary comparisons ---")
    comparisons = {
        "DIME_hout_vs_DIME_hatt": (nll_B_test, nll_C_test),
        "rawknn_hout_vs_rawknn_hatt": (nll_F_test, nll_G_test),
        "FFNhead_vs_ATThead": (nll_D_test, nll_E_test),
        "DIME_hatt_vs_rawknn_hatt": (nll_C_test, nll_G_test),
        "learnedheads_vs_base": (nll_D_test, nll_A_test),
    }
    primary_results = {}
    for name, (nll_x, nll_y) in comparisons.items():
        ttest = stats.ttest_rel(nll_x, nll_y)
        wilcoxon = stats.wilcoxon(nll_x, nll_y)
        boot = document_bootstrap_ci(val_story_ids, nll_x, nll_y)
        primary_results[name] = {
            "mean_diff_x_minus_y": float((nll_x - nll_y).mean()),
            "paired_t_p": float(ttest.pvalue), "wilcoxon_p": float(wilcoxon.pvalue),
            "document_bootstrap": boot,
        }
        print(f"  {name}: mean_diff={primary_results[name]['mean_diff_x_minus_y']:.4f}  "
              f"boot_CI=[{boot['ci_95_low']:.4f}, {boot['ci_95_high']:.4f}]")

    results["primary_comparisons"] = primary_results
    results["sections_completed"] = ["configs", "primary_comparisons"]
    save_results(out_json, results)

    np.savez(out_npz, val_story_ids=val_story_ids, **nll_arrays)
    print(f"[{prefix}] per-position NLL arrays saved to {out_npz}")
    print(f"[{prefix}] summary saved to {out_json}")


if __name__ == "__main__":
    main()
