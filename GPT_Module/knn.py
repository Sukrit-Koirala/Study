from sklearn.neighbors import NearestNeighbors

def build_datastore(keys, values, metric="euclidean"):
    """keys: [N, D] array of hidden states. values: [N] array of target token ids."""
    index = NearestNeighbors(metric=metric)
    index.fit(keys) # This is the part that builds the datastore that can be queries over
    return index, values

def query_knn(index, values, query_keys, k):
    """query_keys: [M, D]. Returns distances [M, k] and retrieved token ids [M, k]."""
    distances, neighbor_idx = index.kneighbors(query_keys, n_neighbors=k)
    retrieved_values = values[neighbor_idx]
    return distances, retrieved_values
