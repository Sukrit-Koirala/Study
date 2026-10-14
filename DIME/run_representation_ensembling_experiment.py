"""Representation-ensembling control experiment (Xu et al. 2023, Section 4.1,
adapted to DIME). TinyStories/gpt2, same seed=42/seq_len=128/{350,75,75} split as
every other TinyStories/gpt2 result in this project.

Split mapping onto DIME's existing 3-way split (an explicit interpretive choice,
documented here rather than silently assumed): `datastore` = TRAIN (for the learned
heads' gradient updates — the same text DIME's own datastore is built from, so no
new data-access asymmetry is introduced), `controller_train` = VALIDATION (head
checkpoint selection AND every interpolation-weight tuning decision, matching DIME's
own established controller_train-for-tuning convention), `val` = TEST (touched once,
for every final NLL/PPL number and the per-token overlap analysis).

h_ATT = the FFN-input representation (output of ln_2, i.e. post-attention-and-residual,
post-layer-norm, immediately feeding the MLP) of the LAST transformer block, captured
via attn_hook.py's forward hook. This is NOT a guess or an arbitrary "earlier layer" —
verified directly against Khandelwal et al. (2020)'s own Section 5 ablation
("Key Function"), where this exact representation ("FFN input, after layer norm")
was their best/adopted choice (16.06 dev PPL on WikiText-103, vs. 17.07 for "model
output" — what hidden_states[-1]/h_FFN actually is, confirmed worse in their own
experiments). Confirmed locally that h_ATT differs substantially from h_FFN (mean
abs diff ~0.72 on a real GPT-2 forward pass) and that h_FFN really is the exact
tensor producing the LM's own logits (lm_head(h_ffn) == logits exactly). Per the
chosen experimental design, DIME's datastore is REBUILT on h_ATT for this experiment
(configs D/E/F), not reused from any earlier h_FFN-based result — a deliberate
deviation from every earlier result in this project, required to make Control B/C a
meaningful test rather than a duplicate of Control A.
"""
import sys, os, json, pickle, time
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

OUT_PATH = "results/representation_ensembling_experiment.json"
N_CLUSTERS = 500
LAYER_IDX = -1
ALPHA_GRID = np.round(np.arange(0.0, 1.0001, 0.05), 2)
SIMPLEX_STEP = 0.05


def eps_log(p):
    return -np.log(np.clip(p, 1e-12, None))


def tune_alpha_two_way(p_true_base_val, p_true_other_val, p_true_base_test, p_true_other_test):
    """Tunes a single interpolation weight alpha for p = alpha*other + (1-alpha)*base,
    on VALIDATION (ct) only; applies the winning alpha once to TEST (val)."""
    best_alpha, best_nll = None, float("inf")
    for alpha in ALPHA_GRID:
        p_mixed = alpha * p_true_other_val + (1 - alpha) * p_true_base_val
        nll = eps_log(p_mixed).mean()
        if nll < best_nll:
            best_nll, best_alpha = nll, float(alpha)
    p_mixed_test = best_alpha * p_true_other_test + (1 - best_alpha) * p_true_base_test
    return best_alpha, float(eps_log(p_mixed_test).mean()), eps_log(p_mixed_test)


def tune_simplex_three_way(p_a_val, p_b_val, p_c_val, p_a_test, p_b_test, p_c_test, step=SIMPLEX_STEP):
    """Grid search over the 2-simplex (w_a,w_b,w_c >= 0, sum=1) on VALIDATION only."""
    grid = np.round(np.arange(0.0, 1.0001, step), 3)
    best_w, best_nll = None, float("inf")
    for wa in grid:
        for wb in grid:
            wc = round(1.0 - wa - wb, 3)
            if wc < 0 or wc > 1:
                continue
            p_mixed = wa * p_a_val + wb * p_b_val + wc * p_c_val
            nll = eps_log(p_mixed).mean()
            if nll < best_nll:
                best_nll, best_w = nll, (float(wa), float(wb), float(wc))
    wa, wb, wc = best_w
    p_mixed_test = wa * p_a_test + wb * p_b_test + wc * p_c_test
    return best_w, float(eps_log(p_mixed_test).mean()), eps_log(p_mixed_test)


def build_equal_budget_raw(ds_keys, ds_values, n_select, seed=42):
    km = MiniBatchKMeans(n_clusters=n_select, random_state=seed)
    assignment = km.fit_predict(ds_keys)
    centers = km.cluster_centers_
    selected_idx = []
    for c in range(n_select):
        member_idx = np.where(assignment == c)[0]
        if len(member_idx) == 0:
            continue
        dists = np.linalg.norm(ds_keys[member_idx] - centers[c], axis=1)
        selected_idx.append(member_idx[np.argmin(dists)])
    selected_idx = np.array(selected_idx)
    return ds_keys[selected_idx], ds_values[selected_idx], selected_idx


def document_bootstrap_ci(story_ids, nll_a, nll_b, n_boot=2000, seed=42):
    """Resamples at the STORY level (not per-token), per Step 5's explicit instruction
    not to treat adjacent tokens as independent observations. Returns the mean
    (nll_a - nll_b) improvement and its bootstrap 95% CI."""
    rng = np.random.default_rng(seed)
    unique_stories = np.unique(story_ids)
    story_to_positions = {s: np.where(story_ids == s)[0] for s in unique_stories}
    point_estimate = float((nll_a - nll_b).mean())

    boot_means = np.zeros(n_boot)
    n_stories = len(unique_stories)
    for b in range(n_boot):
        sampled_stories = rng.choice(unique_stories, size=n_stories, replace=True)
        idx = np.concatenate([story_to_positions[s] for s in sampled_stories])
        boot_means[b] = (nll_a[idx] - nll_b[idx]).mean()
    ci_lo, ci_hi = np.percentile(boot_means, [2.5, 97.5])
    return {"point_estimate": point_estimate, "ci_95_low": float(ci_lo), "ci_95_high": float(ci_hi),
            "n_documents": int(n_stories), "n_bootstrap": n_boot}


def main():
    results = {"phase": "representation_ensembling_experiment", "dataset": "tinystories", "model": "gpt2",
               "n_clusters": N_CLUSTERS, "layer_idx": LAYER_IDX,
               "split_mapping": {"train": "datastore", "validation": "controller_train", "test": "val"}}

    gpt = FrozenGPT2()
    dataset = __import__("datasets").load_dataset(
        "roneneldan/TinyStories", split="train", streaming=True, trust_remote_code=True
    )
    chunks, chunk_story_ids = collect_chunks_split(
        dataset, gpt.tokenizer, seq_len=128,
        n_chunks_needed={"datastore": 350, "controller_train": 75, "val": 75}, seed=42,
    )

    print("Dual-extracting datastore / controller_train(VALIDATION) / val(TEST)...")
    t0 = time.perf_counter()
    ds_ffn, ds_att, ds_values = build_dual_datastore_from_chunks(gpt, chunks["datastore"], layer_idx=LAYER_IDX, batch_size=256)
    ct_ffn, ct_att, ct_values, ct_p_lm = encode_dual_split(gpt, chunks["controller_train"], layer_idx=LAYER_IDX, batch_size=256)
    val_ffn, val_att, val_values, val_p_lm = encode_dual_split(gpt, chunks["val"], layer_idx=LAYER_IDX, batch_size=256)
    extraction_time = time.perf_counter() - t0
    print(f"datastore: {len(ds_values)}  controller_train(VAL): {len(ct_values)}  val(TEST): {len(val_values)}  "
          f"[{extraction_time:.1f}s]")

    # Expand chunk-level story ids to position-level (seq_len=128 -> 127 positions/chunk after shift)
    val_story_ids = np.concatenate([np.full(127, sid) for sid in chunk_story_ids["val"]])
    assert len(val_story_ids) == len(val_values), "story-id expansion length mismatch"

    results["data_sizes"] = {"datastore": len(ds_values), "controller_train": len(ct_values), "val": len(val_values)}
    results["extraction_time_sec"] = extraction_time
    save_results(OUT_PATH, results)

    # ============================================================
    # Config A — base LM
    # ============================================================
    nll_A_test = eps_log(val_p_lm)
    nll_A_val_set = eps_log(ct_p_lm)
    print(f"Config A (base LM) test NLL: {nll_A_test.mean():.4f}")

    # ============================================================
    # Configs B/C — learned heads (FFN / ATT representation)
    # ============================================================
    print("\n--- Training FFN head (Control A: trained on the SAME representation as the output layer) ---")
    t0 = time.perf_counter()
    head_ffn, hist_ffn, best_val_ffn = train_head(ds_ffn, ds_values, ct_ffn, ct_values, verbose=False)
    train_time_ffn = time.perf_counter() - t0
    p_head_ffn_val_set = np.exp(head_log_probs_for_targets(head_ffn, ct_ffn, ct_values))
    p_head_ffn_test = np.exp(head_log_probs_for_targets(head_ffn, val_ffn, val_values))
    alpha_B, nll_B_mean, nll_B_test = tune_alpha_two_way(ct_p_lm, p_head_ffn_val_set, val_p_lm, p_head_ffn_test)
    print(f"Config B (base + FFN head): alpha={alpha_B}  test NLL={nll_B_mean:.4f}")

    print("\n--- Training ATT head (Control B: trained on the FFN-input/post-ln_2 representation — "
          "Khandelwal et al.'s own best-ablated choice, genuinely different from h_FFN) ---")
    t0 = time.perf_counter()
    head_att, hist_att, best_val_att = train_head(ds_att, ds_values, ct_att, ct_values, verbose=False)
    train_time_att = time.perf_counter() - t0
    p_head_att_val_set = np.exp(head_log_probs_for_targets(head_att, ct_att, ct_values))
    p_head_att_test = np.exp(head_log_probs_for_targets(head_att, val_att, val_values))
    alpha_C, nll_C_mean, nll_C_test = tune_alpha_two_way(ct_p_lm, p_head_att_val_set, val_p_lm, p_head_att_test)
    print(f"Config C (base + ATT head): alpha={alpha_C}  test NLL={nll_C_mean:.4f}")

    results["configs"] = {
        "A_base_lm": {"test_nll": float(nll_A_test.mean())},
        "B_base_plus_ffn_head": {"alpha": alpha_B, "test_nll": nll_B_mean, "train_time_sec": train_time_ffn,
                                  "memory_bytes": head_ffn.memory_bytes()},
        "C_base_plus_att_head": {"alpha": alpha_C, "test_nll": nll_C_mean, "train_time_sec": train_time_att,
                                  "memory_bytes": head_att.memory_bytes()},
    }
    save_results(OUT_PATH, results)

    # ============================================================
    # Configs D/E/F — DIME / full raw kNN / equal-budget raw, all on h_ATT
    # ============================================================
    print("\n--- Config E: full raw kNN on h_ATT ---")
    raw_index, raw_values = build_datastore(ds_att, ds_values)
    k_vals = [10, 20, 50, 100, 200]
    tau_vals = [0.5, 1.0, 2.0, 5.0]
    alpha_vals = [0.01, 0.05, 0.1, 0.25, 0.5]
    k_max_raw = max([k for k in k_vals if k <= len(ds_values)])
    dist_ct_raw, retr_ct_raw = query_knn(raw_index, raw_values, ct_att, k=k_max_raw)
    _, best_raw = grid_search_hyperparams(dist_ct_raw, retr_ct_raw, ct_values, ct_p_lm, mix_knn_and_lm, k_vals, tau_vals, alpha_vals)
    dist_val_raw, retr_val_raw = query_knn(raw_index, raw_values, val_att, k=best_raw["k"])
    _, nll_E_test = mix_knn_and_lm(dist_val_raw, retr_val_raw, val_values, val_p_lm, tau=best_raw["tau"], alpha=best_raw["alpha"])
    print(f"Config E (full raw kNN): {best_raw} -> test NLL={nll_E_test.mean():.4f}")

    print("\n--- Config D: DIME (minibatch_kmeans) on h_ATT ---")
    compressed_keys, compressed_dists = minibatch_kmeans_partition(ds_att, ds_values, n_clusters=N_CLUSTERS, seed=42)
    dime_index, dime_values = build_datastore(compressed_keys, np.array(compressed_dists, dtype=object))
    k_vals_dime = [k for k in [10, 20, 50, 100, 200] if k <= N_CLUSTERS]
    dist_ct_dime, retr_ct_dime = query_knn(dime_index, dime_values, ct_att, k=max(k_vals_dime))
    _, best_dime = grid_search_hyperparams(dist_ct_dime, retr_ct_dime, ct_values, ct_p_lm, mix_dime_and_lm, k_vals_dime, tau_vals, alpha_vals)
    dist_val_dime, retr_val_dime = query_knn(dime_index, dime_values, val_att, k=best_dime["k"])
    p_dime_mixed_test, nll_D_test = mix_dime_and_lm(dist_val_dime, retr_val_dime, val_values, val_p_lm, tau=best_dime["tau"], alpha=best_dime["alpha"])
    print(f"Config D (DIME): {best_dime} -> test NLL={nll_D_test.mean():.4f}")

    # DIME's RAW retrieval-only distribution (alpha=1.0 reduces mix_dime_and_lm exactly
    # to the softmax-weighted cluster vote, no LM blending) — this is what feeds Config G,
    # so base LM's own signal isn't silently double-counted inside the 3-way ensemble.
    p_dime_raw_val_set, _ = mix_dime_and_lm(dist_ct_dime, retr_ct_dime, ct_values, ct_p_lm, tau=best_dime["tau"], alpha=1.0)
    p_dime_raw_test, _ = mix_dime_and_lm(dist_val_dime, retr_val_dime, val_values, val_p_lm, tau=best_dime["tau"], alpha=1.0)
    frac_dime_zero_test = float((p_dime_raw_test < 1e-11).mean())
    print(f"  fraction of test positions where DIME's raw retrieval prob is ~0 (true token never in top-k clusters): "
          f"{frac_dime_zero_test*100:.2f}%")

    print("\n--- Config F: equal-budget raw retrieval on h_ATT ---")
    eq_keys, eq_values, _ = build_equal_budget_raw(ds_att, ds_values, N_CLUSTERS, seed=42)
    eq_index, eq_stored = build_datastore(eq_keys, eq_values)
    k_vals_eq = [k for k in k_vals_dime if k <= len(eq_keys)]
    dist_ct_eq, retr_ct_eq = query_knn(eq_index, eq_stored, ct_att, k=max(k_vals_eq))
    _, best_eq = grid_search_hyperparams(dist_ct_eq, retr_ct_eq, ct_values, ct_p_lm, mix_knn_and_lm, k_vals_eq, tau_vals, alpha_vals)
    dist_val_eq, retr_val_eq = query_knn(eq_index, eq_stored, val_att, k=best_eq["k"])
    _, nll_F_test = mix_knn_and_lm(dist_val_eq, retr_val_eq, val_values, val_p_lm, tau=best_eq["tau"], alpha=best_eq["alpha"])
    print(f"Config F (equal-budget raw, n={len(eq_keys)}): {best_eq} -> test NLL={nll_F_test.mean():.4f}")

    results["configs"]["D_base_plus_dime"] = {"best_config": best_dime, "test_nll": float(nll_D_test.mean())}
    results["configs"]["E_full_raw_knn"] = {"best_config": best_raw, "test_nll": float(nll_E_test.mean())}
    results["configs"]["F_equal_budget_raw"] = {"best_config": best_eq, "n_selected": int(len(eq_keys)),
                                                  "test_nll": float(nll_F_test.mean())}
    save_results(OUT_PATH, results)

    # ============================================================
    # Config G — 3-way ensemble (base LM + ATT head + DIME's raw retrieval)
    # ============================================================
    print("\n--- Config G: base LM + ATT head + DIME (3-way, validation-tuned simplex) ---")
    w_G, nll_G_mean, nll_G_test = tune_simplex_three_way(
        ct_p_lm, p_head_att_val_set, p_dime_raw_val_set,
        val_p_lm, p_head_att_test, p_dime_raw_test,
    )
    print(f"Config G weights (lm, head_att, dime)={w_G}  test NLL={nll_G_mean:.4f}")

    # Validation control: same 3-way simplex tuning, but with DIME's component replaced by
    # a SHUFFLED version (correct cluster geometry, content randomly reassigned — the same
    # illusion-check methodology already established elsewhere in this project). If G beats
    # this control by no more than the shuffled version beats C, the gain is just extra
    # tuning flexibility, not real DIME signal.
    rng = np.random.default_rng(42)
    perm = rng.permutation(len(compressed_dists))
    shuffled_dists = np.array(compressed_dists, dtype=object)[perm]
    shuffled_index, shuffled_values = build_datastore(compressed_keys, shuffled_dists)
    dist_ct_shuf, retr_ct_shuf = query_knn(shuffled_index, shuffled_values, ct_att, k=best_dime["k"])
    p_shuf_val_set, _ = mix_dime_and_lm(dist_ct_shuf, retr_ct_shuf, ct_values, ct_p_lm, tau=best_dime["tau"], alpha=1.0)
    dist_val_shuf, retr_val_shuf = query_knn(shuffled_index, shuffled_values, val_att, k=best_dime["k"])
    p_shuf_test, _ = mix_dime_and_lm(dist_val_shuf, retr_val_shuf, val_values, val_p_lm, tau=best_dime["tau"], alpha=1.0)

    w_Gctrl, nll_Gctrl_mean, nll_Gctrl_test = tune_simplex_three_way(
        ct_p_lm, p_head_att_val_set, p_shuf_val_set,
        val_p_lm, p_head_att_test, p_shuf_test,
    )
    print(f"Config G control (shuffled DIME component) weights={w_Gctrl}  test NLL={nll_Gctrl_mean:.4f}")

    results["configs"]["G_base_plus_att_head_plus_dime"] = {
        "weights_lm_head_dime": w_G, "test_nll": float(nll_G_mean),
        "validation_control_shuffled_dime": {"weights_lm_head_dime": w_Gctrl, "test_nll": float(nll_Gctrl_mean)},
    }
    save_results(OUT_PATH, results)

    # ============================================================
    # Delta metrics
    # ============================================================
    nll_A, nll_C, nll_D, nll_G = nll_A_test.mean(), nll_C_test, nll_D_test.mean(), nll_G_test
    delta_dime = float(nll_A - nll_D)
    delta_head = float(nll_A - nll_C.mean())
    r_head = delta_head / delta_dime if delta_dime != 0 else float("nan")
    delta_incremental = float(nll_C.mean() - nll_G.mean())
    delta_incremental_control = float(nll_C.mean() - nll_Gctrl_test.mean())

    results["delta_metrics"] = {
        "delta_DIME": delta_dime, "delta_head": delta_head, "R_head": r_head,
        "delta_incremental": delta_incremental,
        "delta_incremental_shuffled_control": delta_incremental_control,
        "note": "R_head measures how much of DIME's NLL gain the learned intermediate "
                "predictor reproduces under THESE conditions — not a causal attribution. "
                "delta_incremental should be compared against delta_incremental_shuffled_control "
                "to judge whether G's gain over C is real DIME signal or just extra tuning freedom.",
    }
    save_results(OUT_PATH, results)

    # ============================================================
    # Per-token overlap analysis + document-level bootstrap
    # ============================================================
    print("\n--- Per-token overlap analysis ---")
    dime_helps = nll_D_test < nll_A_test  # per-position: did DIME's config beat base LM here?
    head_helps = nll_C_test < nll_A_test
    both_help = (dime_helps & head_helps).mean()
    only_dime = (dime_helps & ~head_helps).mean()
    only_head = (~dime_helps & head_helps).mean()
    both_hurt = (~dime_helps & ~head_helps).mean()

    overlap = {"both_help": float(both_help), "only_dime_helps": float(only_dime),
               "only_head_helps": float(only_head), "both_hurt": float(both_hurt)}
    print(json.dumps(overlap, indent=2))

    ttest = stats.ttest_rel(nll_A_test, nll_D_test)
    wilcoxon = stats.wilcoxon(nll_A_test, nll_D_test)
    boot_dime = document_bootstrap_ci(val_story_ids, nll_A_test, nll_D_test)
    boot_head = document_bootstrap_ci(val_story_ids, nll_A_test, nll_C_test)
    print(f"Document-level bootstrap, A-D improvement: {boot_dime}")
    print(f"Document-level bootstrap, A-C improvement: {boot_head}")

    # Token-frequency stratification
    global_freq = build_global_freq(ds_values)
    freq_of_true_token = global_freq[val_values]
    quartiles = np.quantile(freq_of_true_token, [0.25, 0.5, 0.75])
    freq_bucket = np.digitize(freq_of_true_token, quartiles)  # 0=rarest quartile .. 3=most common
    freq_stratified = {}
    for b in range(4):
        mask = freq_bucket == b
        label = ["rarest_quartile", "q2", "q3", "most_common_quartile"][b]
        freq_stratified[label] = {
            "n": int(mask.sum()),
            "delta_DIME": float((nll_A_test[mask] - nll_D_test[mask]).mean()),
            "delta_head": float((nll_A_test[mask] - nll_C_test[mask]).mean()),
        }
    print(json.dumps(freq_stratified, indent=2))

    results["overlap_analysis"] = overlap
    results["paired_significance_A_vs_D"] = {"t_stat": float(ttest.statistic), "p_value": float(ttest.pvalue),
                                              "wilcoxon_p": float(wilcoxon.pvalue)}
    results["document_bootstrap"] = {"A_minus_D": boot_dime, "A_minus_C": boot_head}
    results["token_frequency_stratified"] = freq_stratified
    results["frac_dime_raw_retrieval_zero_on_test"] = frac_dime_zero_test
    results["sections_completed"] = ["configs", "delta_metrics", "overlap_analysis",
                                      "document_bootstrap", "token_frequency_stratified"]
    save_results(OUT_PATH, results)

    print("\n=== SUMMARY ===")
    print(f"A (base):              {nll_A:.4f}")
    print(f"B (base+FFN head):      {nll_B_mean:.4f}")
    print(f"C (base+ATT head):      {nll_C.mean():.4f}")
    print(f"D (base+DIME):          {nll_D:.4f}")
    print(f"E (full raw kNN):       {nll_E_test.mean():.4f}")
    print(f"F (equal-budget raw):   {nll_F_test.mean():.4f}")
    print(f"G (base+ATT head+DIME): {nll_G.mean():.4f}")
    print(f"Delta_DIME={delta_dime:.4f}  Delta_head={delta_head:.4f}  R_head={r_head:.4f}  "
          f"Delta_incremental={delta_incremental:.4f}  (shuffled control: {delta_incremental_control:.4f})")
    print(f"\nAll results saved to {OUT_PATH}")


if __name__ == "__main__":
    main()
