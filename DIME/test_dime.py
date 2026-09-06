import numpy as np
from collections import Counter
from state_object import query_kmeans_partition

rng = np.random.default_rng(11)

ct_blob_a = rng.normal(loc=0.0,   scale=0.1, size=(5, 4)).astype(np.float32)
ct_blob_b = rng.normal(loc=20.0,  scale=0.1, size=(5, 4)).astype(np.float32)
ct_blob_c = rng.normal(loc=-20.0, scale=0.1, size=(5, 4)).astype(np.float32)
ct_keys = np.concatenate([ct_blob_a, ct_blob_b, ct_blob_c], axis=0)

ds_blob_a = rng.normal(loc=0.5,   scale=0.1, size=(5, 4)).astype(np.float32)   # offset from ct_blob_a's center
ds_blob_b = rng.normal(loc=20.5,  scale=0.1, size=(5, 4)).astype(np.float32)   # offset from ct_blob_b's center
ds_blob_c = rng.normal(loc=-19.5, scale=0.1, size=(5, 4)).astype(np.float32)   # offset from ct_blob_c's center
ds_keys = np.concatenate([ds_blob_a, ds_blob_b, ds_blob_c], axis=0)
ds_values = np.array([10]*5 + [20]*5 + [30]*5)

compressed_keys, compressed_dists = query_kmeans_partition(ds_keys, ds_values, ct_keys, n_clusters=3, seed=42)

print("compressed_keys shape:", compressed_keys.shape)
for i, d in enumerate(compressed_dists):
    print(f"cluster {i}: {dict(d)}")
