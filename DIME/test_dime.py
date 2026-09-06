import numpy as np
from state_object import random_partition

rng_check = np.random.default_rng(123)
keys = rng_check.normal(size=(12, 4)).astype(np.float32)
values = np.array([10, 10, 20, 20, 20, 30, 10, 20, 30, 30, 10, 20])

compressed_keys, compressed_dists = random_partition(keys, values, n_clusters=3, seed=1)

print("compressed_keys shape:", compressed_keys.shape)
total_counted = sum(sum(d.values()) for d in compressed_dists)
print("total counted (should be 12):", total_counted)
for i, d in enumerate(compressed_dists):
    print(f"cluster {i}: {dict(d)}")
