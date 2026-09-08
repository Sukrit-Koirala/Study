from sklearn.neighbors import NearestNeighbors
import numpy as np
def build_datastore(keys, values, metric="euclidean"):
    """keys: [N, D] array of hidden states. values: [N] array of target token ids."""
    index = NearestNeighbors(metric=metric)
    index.fit(keys) # This is the part that builds the datastore that can be queries over
    return index, values

def query_knn(index, values, query_keys, k):
    """query_keys: [M, D]. Returns distances [M, k] and retrieved token ids [M, k]."""
    distances, neighbor_idx = index.kneighbors(query_keys, n_neighbors=k)
    retrieved_values = values[neighbor_idx]
    return distances, retrieved_values

def query_knn_indices(index, query_keys, k):
    """Same search as query_knn, but returns raw neighbor indices instead of
    gathering into a values array — needed for dense/GPU-vectorized lookups."""
    distances, neighbor_idx = index.kneighbors(query_keys, n_neighbors=k)
    return distances, neighbor_idx

def softmax_np(x, axis=-1):
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)  

def mix_knn_and_lm(distances, retrieved_values, true_targets, p_lm_true, tau=1.0, alpha=0.25):
    weights = softmax_np(-distances / tau, axis=1) # Turing the predictions into probabilites
    match_mask = (retrieved_values == true_targets[:, None])  # [M, k] — which neighbors "voted" for the true token
    p_knn_true = (weights * match_mask).sum(axis=1) # Getting the probability for the predicted next token
    p_mixed = alpha * p_knn_true + (1 - alpha) * p_lm_true     

    # This voting is weighted based on distance so its not exactly a vote only
    #Later on the mixing is what we bring in reading and Q-Learning for

    nll_mixed = -np.log(p_mixed + 1e-12)
    return p_mixed, nll_mixed
