import numpy as np
from knn import build_datastore, query_knn

rng = np.random.default_rng(42)
keys = rng.normal(size=(20, 768)).astype(np.float32)
values = np.arange(20)  # value i is just "i", so we know exactly what should come back

index, stored_values = build_datastore(keys, values)

# query = key #5 plus tiny noise — should retrieve value 5 as the closest match
query = keys[5:6] + rng.normal(scale=0.01, size=(1, 768)).astype(np.float32)
distances, retrieved = query_knn(index, stored_values, query, k=3)

print("distances:", distances)
print("retrieved values (should have 5 first):", retrieved)
