import numpy as np

#This function handles both a single entry/state_object
import numpy as np

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


