import numpy as np
from collections import Counter
from state_object import utility_weighted_partition

keys = np.array([[0.0, 0.0], [0.1, 0.1]], dtype=np.float32)  # 2 points, same tiny cluster
values = np.array([10, 20])       # position 0 -> token 10, position 1 -> token 20
nll    = np.array([0.1, 5.0])     # position 0: GPT was confident. position 1: GPT struggled a lot

compressed_keys, compressed_dists = utility_weighted_partition(keys, values, nll, n_clusters=1, seed=42)

print("cluster 0 weighted distribution:", dict(compressed_dists[0]))
# unweighted count would be {10: 1, 20: 1} — equal.
# weighted should show token 20 with much more mass than token 10 (5.1 vs 0.2),
# reflecting that GPT needed help there far more than at position 0.
