import numpy as np
from collections import Counter
from q_read_dense import counters_to_dense, build_obs_features

dists = [Counter({5: 8, 2: 2}), Counter({9: 5}), Counter({1: 3}), Counter({7: 1})]
token_ids, token_counts, total_counts = counters_to_dense(dists, top_k=4)

cluster_idx = np.array([[0, 1, 2, 3, 0, 1, 2, 3]])  # 1 query, 8 "neighbors" (toy — reusing 4 clusters twice)
distances = np.array([[0.1, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]])
gpt_entropy = np.array([1.2])

obs = build_obs_features(gpt_entropy, distances, cluster_idx, token_counts, total_counts, device="cpu")
print("obs shape:", obs.shape)   # expect (1, 11)
print("obs:", obs)
print("purity_top1 (cluster 0, {5:8,2:2}) should be 0.8:", obs[0, 7])
