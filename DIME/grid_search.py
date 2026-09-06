import numpy as np

def grid_search_hyperparams(distances, retrieved, true_targets, p_lm_true, mix_fn,
                              k_values, tau_values, alpha_values):
    """distances, retrieved: [M, k_max] — retrieved once with the largest k needed.
    mix_fn: mix_knn_and_lm (raw) or mix_dime_and_lm (compressed) — same signature.
    Returns (all results sorted best-first, the single best entry)."""
    results = []
    for k in k_values:
        d_k = distances[:, :k]
        r_k = retrieved[:, :k]
        for tau in tau_values:
            for alpha in alpha_values:
                _, nll_mixed = mix_fn(d_k, r_k, true_targets, p_lm_true, tau=tau, alpha=alpha)
                results.append({"k": k, "tau": tau, "alpha": alpha, "mean_nll": float(nll_mixed.mean())})

    results.sort(key=lambda r: r["mean_nll"])
    return results, results[0]
