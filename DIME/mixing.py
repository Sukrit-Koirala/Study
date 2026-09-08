import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "GPT_Module"))
from knn import softmax_np  # reused from GPT_Module, not duplicated


# Only job is mixing
def mix_dime_and_lm(distances, retrieved_dists, true_targets, p_lm_true, tau=1.0, alpha=0.25, beta = 0.0, global_freq=None):
    """distances, retrieved_dists: [M, k] (retrieved_dists holds Counter objects).
    true_targets, p_lm_true: [M]."""
    M, k = distances.shape
    weights = softmax_np(-distances / tau, axis=1)  # [M, k] #Very important: The closest centroid matters the most, so this does that, tau is the parameter that controls how much weight the closest centroid gets, higher the tau smoother favouring 

    p_knn_true = np.zeros(M, dtype=np.float64)
    for i in range(M):
        true_tok = int(true_targets[i])
        global_p = global_freq[true_tok] if (beta > 0 and global_freq is not None) else 0.0
        for j in range(k):
            dist_counter = retrieved_dists[i, j]
            total = sum(dist_counter.values())
            denom = total + beta
            if denom > 0:
                count = dist_counter.get(true_tok, 0)
                p_state = (count + beta * global_p) / denom
                p_knn_true[i] += weights[i, j] * p_state

    p_mixed = alpha * p_knn_true + (1 - alpha) * p_lm_true
    nll_mixed = -np.log(p_mixed + 1e-12)
    return p_mixed, nll_mixed

# A function to give the probabilty of that token appearing in the vocab, used to know rare tokens 
def build_global_freq(values, vocab_size=50257):
    """values: [N] datastore target token ids. Returns [vocab_size] array of
    P_global(token) = corpus-wide frequency, indexable directly by token id.""" 
    counts = np.bincount(values, minlength=vocab_size)
    return counts / counts.sum()




