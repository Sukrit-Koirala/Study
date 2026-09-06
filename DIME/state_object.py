import numpy as np
from collections import Counter

# We will be adding different methods of clustering here



# Since I have a hard time trusting distance based methods, random partition is a good start
def random_partition(keys, values, n_clusters, seed=42):
    """keys: [N, D] raw datastore keys. values: [N] raw datastore target token ids.
    Returns compressed_keys [n_clusters, D] (centroids) and compressed_dists
    (list of Counter, one per cluster: {token_id: count})."""
    rng = np.random.default_rng(seed)
    N = keys.shape[0]

    assignment = rng.integers(0,n_clusters,size=N) #Randomly assign buckets to data, 1D, entry 0 -> assign[0]

    #init
    compressed_keys = np.zeros((n_clusters,keys.shape[1]), dtype=np.float32)
    compressed_dists = []

    for c in range(n_clusters): #The previous steps have already assigned clusters to entries
        member_mask = assignment == c # Working at one cluster level
        member_keys = keys[member_mask] # Taking keys from the cluster
        member_values = values[member_mask] # Same

        if member_keys.shape[0] > 0:
            compressed_keys[c] = member_keys.mean(axis=0)
            compressed_dists.append(Counter(member_values.tolist()))
        else:
            compressed_dists.append(Counter())  # empty cluster rare, but handle it

    return compressed_keys, compressed_dists

#Basically we created a function that returns the avg keys from a random cluster and its compressed distribution





