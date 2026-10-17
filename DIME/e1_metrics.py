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
