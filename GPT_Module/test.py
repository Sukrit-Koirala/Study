import numpy as np
from knn import mix_knn_and_lm

distances = np.array([
    [0.1, 5.0, 6.0],
    [0.1, 5.0, 6.0],
])
retrieved_values = np.array([
    [5, 2, 9],   # nearest neighbor's value MATCHES true target (5)
    [2, 9, 3],   # no neighbor matches true target (7)
])
true_targets = np.array([5, 7])
p_lm_true = np.array([0.3, 0.3])

p_mixed, nll_mixed = mix_knn_and_lm(distances, retrieved_values, true_targets, p_lm_true, tau=1.0, alpha=0.25)

print("p_mixed:", p_mixed)
print("nll_mixed:", nll_mixed)
print("pure LM NLL (no retrieval):", -np.log(p_lm_true))
