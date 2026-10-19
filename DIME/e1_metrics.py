"""E1-specific metrics — standalone (pre-interpolation) retrieval probabilities and
full retrieval distributions, computed WITHOUT the epsilon-smoothing
mix_knn_and_lm/mix_dime_and_lm apply (correct for THEIR purpose — the interpolated
NLL, where p_lm_true>0 always guards against log(0) — but hides whether raw
retrieval assigns exact zero probability to the true token, which Section 7 of the
E1 spec explicitly wants preserved and reported, not concealed).

Reuses the exact same softmax-weighted-vote math as knn.py/mixing.py (same
formula, just exposed standalone, not folded into an interpolation) — not a
reimplementation, a read-out of an intermediate quantity those functions already
compute internally but don't return.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "GPT_Module"))

import numpy as np
from scipy import stats as _stats
from knn import softmax_np


def raw_standalone_prob(distances, retrieved_values, true_targets, tau=1.0):
    """P_raw(y_true | q), exact — zero preserved when the true token never
    appears among the retrieved neighbors, no epsilon added."""
    weights = softmax_np(-distances / tau, axis=1)
    match_mask = (retrieved_values == true_targets[:, None])
    return (weights * match_mask).sum(axis=1)


def dime_standalone_prob(distances, retrieved_dists, true_targets, tau=1.0):
    """P_DIME(y_true | q), exact — same zero-preservation as above."""
    weights = softmax_np(-distances / tau, axis=1)
    M, k = distances.shape
    p = np.zeros(M, dtype=np.float64)
    for i in range(M):
        true_tok = int(true_targets[i])
        for j in range(k):
            counter = retrieved_dists[i, j]
            total = sum(counter.values())
            if total > 0:
                p[i] += weights[i, j] * (counter.get(true_tok, 0) / total)
    return p


def raw_full_distribution(distances, retrieved_values, tau=1.0):
    """Per query: dict {token_id: prob}, summing to exactly 1 (softmax weights
    partitioned across whichever distinct tokens appear among the k neighbors)."""
    weights = softmax_np(-distances / tau, axis=1)
    out = []
    for i in range(distances.shape[0]):
        d = {}
        for j in range(distances.shape[1]):
            tok = int(retrieved_values[i, j])
            d[tok] = d.get(tok, 0.0) + float(weights[i, j])
        out.append(d)
    return out


def dime_full_distribution(distances, retrieved_dists, tau=1.0):
    """Per query: dict {token_id: prob}, summing to exactly 1 (each neighbor's
    softmax weight split proportionally across its cluster's token counts)."""
    weights = softmax_np(-distances / tau, axis=1)
    out = []
    for i in range(distances.shape[0]):
        d = {}
        for j in range(distances.shape[1]):
            counter = retrieved_dists[i, j]
            total = sum(counter.values())
            if total > 0:
                w = float(weights[i, j])
                for tok, cnt in counter.items():
                    d[tok] = d.get(tok, 0.0) + w * (cnt / total)
        out.append(d)
    return out


def js_divergence_dict(p, q):
    """JS divergence in nats between two sparse {token_id: prob} dicts, each
    already summing to 1 over the full vocab (implicitly 0 elsewhere) — computed
    only over the union of their nonzero support, exact (not approximate), since
    every token outside the union contributes exactly 0 to both KL terms."""
    support = set(p.keys()) | set(q.keys())
    if not support:
        return 0.0
    m = {t: 0.5 * (p.get(t, 0.0) + q.get(t, 0.0)) for t in support}

    def kl(a, b):
        s = 0.0
        for t in support:
            at = a.get(t, 0.0)
            if at > 0:
                s += at * np.log(at / b[t])
        return s

    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def retrieval_coverage(standalone_probs):
    """Fraction of queries where standalone retrieval assigns nonzero probability
    to the true token ('correct-target retrieval coverage')."""
    return float((standalone_probs > 0).mean())


# ============================================================
# Shared statistics helpers (moved here from run_e1a2_distance_prediction.py so
# E1B.2 and any later extension can reuse them instead of redefining).
# ============================================================

def correlations_with_ci(x, y, n_boot=2000, seed=42):
    """Pearson + Spearman, each with a bootstrap 95% CI, computed over the SAME
    resamples (one bootstrap loop, not two) for efficiency."""
    rng = np.random.default_rng(seed)
    n = len(x)
    pear_r, pear_p = _stats.pearsonr(x, y)
    spear_r, spear_p = _stats.spearmanr(x, y)
    pear_boot, spear_boot = np.zeros(n_boot), np.zeros(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        pear_boot[b] = _stats.pearsonr(x[idx], y[idx])[0]
        spear_boot[b] = _stats.spearmanr(x[idx], y[idx])[0]
    pl, ph = np.percentile(pear_boot, [2.5, 97.5])
    sl, sh = np.percentile(spear_boot, [2.5, 97.5])
    return {
        "n": int(n),
        "pearson": {"r": float(pear_r), "p_value": float(pear_p), "ci_95_low": float(pl), "ci_95_high": float(ph)},
        "spearman": {"rho": float(spear_r), "p_value": float(spear_p), "ci_95_low": float(sl), "ci_95_high": float(sh)},
    }


def binned_stats(x, y, n_bins=10):
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


def plot_binned(bins, xlabel, ylabel, title, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 5))
    xs = [b["x_mean"] for b in bins]
    means = [b["y_mean"] for b in bins]
    medians = [b["y_median"] for b in bins]
    ns = [b["n"] for b in bins]
    ax.plot(xs, means, "o-", label="mean")
    ax.plot(xs, medians, "s--", label="median")
    for x, m, n in zip(xs, means, ns):
        ax.annotate(f"n={n}", (x, m), fontsize=7, textcoords="offset points", xytext=(0, 6))
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


# ============================================================
# Distribution-shape diagnostics (RQ4 — entropy / concentration), computed from
# the same {token_id: prob} dicts raw_full_distribution/dime_full_distribution
# already produce, each summing to exactly 1.
# ============================================================

def distribution_shape_stats(dist_list):
    """Per query, from a list of {token_id: prob} dicts: entropy (nats),
    effective support (exp(entropy)), max probability (purity), and the number
    of tokens with nonzero probability (raw support size)."""
    n = len(dist_list)
    entropy = np.zeros(n, dtype=np.float64)
    max_prob = np.zeros(n, dtype=np.float64)
    support_size = np.zeros(n, dtype=np.int64)
    top1_token = np.zeros(n, dtype=np.int64)
    for i, d in enumerate(dist_list):
        if not d:
            top1_token[i] = -1
            continue
        probs = np.array(list(d.values()), dtype=np.float64)
        probs = probs[probs > 0]
        entropy[i] = float(-(probs * np.log(probs)).sum())
        max_prob[i] = float(probs.max())
        support_size[i] = int(len(probs))
        top1_token[i] = max(d, key=d.get)
    effective_support = np.exp(entropy)
    return {"entropy": entropy, "effective_support": effective_support,
            "max_prob": max_prob, "support_size": support_size, "top1_token": top1_token}
