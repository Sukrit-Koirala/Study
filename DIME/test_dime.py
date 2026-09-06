import numpy as np
from collections import Counter
from state_object import minibatch_kmeans_partition

rng = np.random.default_rng(7)
blob_a = rng.normal(loc=0.0,   scale=0.1, size=(5, 4)).astype(np.float32)
blob_b = rng.normal(loc=20.0,  scale=0.1, size=(5, 4)).astype(np.float32)
blob_c = rng.normal(loc=-20.0, scale=0.1, size=(5, 4)).astype(np.float32)
keys = np.concatenate([blob_a, blob_b, blob_c], axis=0)
values = np.array([10]*5 + [20]*5 + [30]*5)  # blob A's positions are all tagged 10, etc.

compressed_keys, compressed_dists = minibatch_kmeans_partition(keys, values, n_clusters=3, seed=42)

print("compressed_keys shape:", compressed_keys.shape)
for i, d in enumerate(compressed_dists):
    print(f"cluster {i}: {dict(d)}")
