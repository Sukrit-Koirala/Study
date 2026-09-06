import numpy as np
from collections import Counter
from sklearn.cluster import MiniBatchKMeans

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


def minibatch_kmeans_partition(keys, values, n_clusters, seed=42):
    """keys: [N, D] raw datastore keys. values: [N] raw datastore target token ids.
    Returns compressed_keys [n_clusters, D] (k-means centroids) and compressed_dists
    (list of Counter, one per cluster: {token_id: count})."""

    #Clustering via K nearest neighbours ig
    km = MiniBatchKMeans(n_clusters=n_clusters,random_state=seed)
    assignment = km.fit_predict(keys)

    compressed_keys = km.cluster_centers_.astype(np.float32)  # k-means already gives centroids directly

    compressed_dists = []
    for c in range(n_clusters):
        member_mask = assignment == c
        member_values = values[member_mask]
        compressed_dists.append(Counter(member_values.tolist()) if member_values.shape[0] > 0 else Counter())

    return compressed_keys, compressed_dists


def utility_weighted_partition(keys, values, nll, n_clusters, seed=42):
    """Same clustering as minibatch_kmeans_partition, but each raw entry's
    contribution to its cluster's token distribution is weighted by GPT's own
    NLL at that position (+ a small floor), not a flat count of 1, positions
    where GPT struggled count for more than ones it already predicted well."""
    km = MiniBatchKMeans(n_clusters=n_clusters, random_state=seed)
    assignment = km.fit_predict(keys)
    compressed_keys = km.cluster_centers_.astype(np.float32)

    floor = 0.1
    weights = nll + floor #To keep small values relevant

    compressed_dists = []
    for c in range(n_clusters):
        member_mask = assignment == c
        member_values = values[member_mask]
        member_weights = weights[member_mask]
        dist = Counter()
        for tok, w in zip(member_values.tolist(), member_weights.tolist()):
            dist[tok] += w
        compressed_dists.append(dist)

    return compressed_keys, compressed_dists








