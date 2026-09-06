import numpy as np
from sklearn.neural_network import MLPRegressor

#This function handles both a single entry/state_object
def retrieval_purity_entropy(retrieved_top1):
    """retrieved_top1: the nearest retrieved item per query — either a plain token id
    (raw kNN, degenerate point-mass distribution) or a Counter (compressed DIME).
    Returns (entropy, purity) arrays."""
    entropies, purities = [], []
    for item in retrieved_top1:
        if isinstance(item, dict):
            total = sum(item.values())
            if total > 0:
                probs = np.array([v / total for v in item.values()])
                entropies.append(float(-(probs * np.log(probs + 1e-12)).sum()))
                purities.append(float(probs.max()))
            else:
                entropies.append(0.0)
                purities.append(1.0)
        else:  # raw kNN: single token id, no real "spread" to measure
            entropies.append(0.0)
            purities.append(1.0)
    return np.array(entropies), np.array(purities)


#Trainning Wrapper
def train_q_read_controller(X, y, seed=42):
    """X: [N, n_features] state features. y: [N] observed reward (NLL_GPT - NLL_action).
    Returns a fitted regressor predicting expected reward from state features."""
    model = MLPRegressor(hidden_layer_sizes=(16,), random_state=seed, max_iter=2000)
    model.fit(X, y)
    return model


