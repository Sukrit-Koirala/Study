import numpy as np
import sys,os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "GPT_Module"))
from knn import mix_knn_and_lm
from grid_search import grid_search_hyperparams

distances = np.array([[0.1, 10.0]])          # one query, k_max=2 — dominant near neighbor, far noisy one
retrieved = np.array([[5, 99]])              # nearest neighbor's value EXACTLY matches true target (5)
true_targets = np.array([5])
p_lm_true = np.array([0.5])                  # GPT itself is mediocre here

results, best = grid_search_hyperparams(
    distances, retrieved, true_targets, p_lm_true, mix_knn_and_lm,
    k_values=[2], tau_values=[0.5, 1.0, 2.0], alpha_values=[0.1, 0.5, 0.9]
)

print("best config:", best)
