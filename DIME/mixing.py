import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "GPT_Module"))
from knn import softmax_np  # reused from GPT_Module, not duplicated


# Only job is mixing
def mix_dime_and_lm(distances, retrieved_dists, true_targets, p_lm_true, tau=1.0, alpha=0.25):
    """distances, retrieved_dists: [M, k] (retrieved_dists holds Counter objects).
    true_targets, p_lm_true: [M]."""
    M, k = distances.shape
    weights = softmax_np(-distances / tau, axis=1)  # [M, k]

    p_knn_true = np.zeros(M, dtype=np.float64)
    for i in range(M):
        true_tok = int(true_targets[i])
        for j in range(k):
            dist_counter = retrieved_dists[i, j]
            total = sum(dist_counter.values())
            if total > 0:
                p_knn_true[i] += weights[i, j] * (dist_counter.get(true_tok, 0) / total)

    p_mixed = alpha * p_knn_true + (1 - alpha) * p_lm_true
    nll_mixed = -np.log(p_mixed + 1e-12)
    return p_mixed, nll_mixed
