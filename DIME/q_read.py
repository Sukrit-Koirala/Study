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


class NormalizedMLPWrapper:
    """Wraps an MLPRegressor fitted on normalized (X, y) so callers can keep calling
    .predict(X) with raw, unnormalized inputs and get back raw-scale predictions —
    no changes needed anywhere the model is already used."""
    def __init__(self, model, x_mean, x_std, y_mean, y_std):
        self.model = model
        self.x_mean = x_mean
        self.x_std = x_std
        self.y_mean = y_mean
        self.y_std = y_std

    def predict(self, X):
        X_norm = (X - self.x_mean) / self.x_std
        y_norm = self.model.predict(X_norm)
        return y_norm * self.y_std + self.y_mean


#Trainning Wrapper
def train_q_read_controller(X, y, seed=42):
    """X: [N, n_features] state features. y: [N] observed reward (NLL_GPT - NLL_action).
    Returns a fitted regressor predicting expected reward from state features.
    Normalizes X and y before fitting (features/rewards have very different scales,
    which was causing the MLP to collapse to a near-constant output) — wrapped so
    the returned object's .predict() still takes/returns raw-scale values."""
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    x_mean = X.mean(axis=0)
    x_std = X.std(axis=0)
    x_std[x_std < 1e-8] = 1.0   # guard divide-by-zero for constant columns
    y_mean = y.mean()
    y_std = y.std() if y.std() > 1e-8 else 1.0

    X_norm = (X - x_mean) / x_std
    y_norm = (y - y_mean) / y_std

    model = MLPRegressor(hidden_layer_sizes=(128, 64), random_state=seed, max_iter=2000,
                          early_stopping=True, validation_fraction=0.1, n_iter_no_change=15)
    model.fit(X_norm, y_norm)

    return NormalizedMLPWrapper(model, x_mean, x_std, y_mean, y_std)


