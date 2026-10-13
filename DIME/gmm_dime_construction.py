"""Three DIME construction variants for the k-means-vs-GMM experiment, plus the
boundary and cluster-level analyses the experiment needs. Retrieval is untouched —
every variant still hands back (compressed_keys, compressed_dists) in exactly the
shape knn.py's build_datastore/query_knn and mixing.py's mix_dime_and_lm already
expect (compressed_dists: object-array of collections.Counter), so the existing
retrieval pipeline needs zero changes to run any of the three.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "GPT_Module"))

import numpy as np
from collections import Counter
from sklearn.cluster import MiniBatchKMeans
from sklearn.neighbors import NearestNeighbors
from scipy.optimize import linear_sum_assignment
from scipy.sparse import csr_matrix
from scipy.spatial.distance import cdist

from gmm import init_gmm_from_kmeans, fit_gmm

VOCAB_SIZE = 50257


def build_kmeans_dime(ds_keys, ds_values, n_clusters, seed=42):
    """VARIANT A — original DIME. Hard k-means assignment z_ic in {0,1}.
    Returns centroids [B,D], hard assignment [N] int, and compressed_dists
    (Counter per cluster, built from members' raw token counts)."""
    km = MiniBatchKMeans(n_clusters=n_clusters, random_state=seed)
    assignment = km.fit_predict(ds_keys)
    centroids = km.cluster_centers_.astype(np.float64)

    compressed_dists = []
    for c in range(n_clusters):
        member_values = ds_values[assignment == c]
        compressed_dists.append(Counter(member_values.tolist()) if len(member_values) > 0 else Counter())

    return centroids, assignment, np.array(compressed_dists, dtype=object)


def build_gmm_hard_dime(ds_keys, ds_values, kmeans_centroids, kmeans_assignment,
                         covariance_type="diag", tol=1e-4, max_iter=50, batch_size=5000, seed=42):
    """VARIANT B — GMM/EM fit (seeded from k-means), then collapsed back to a hard
    assignment via argmax responsibility, c_i = argmax_c gamma_ic. Isolates whether
    the GMM's DIFFERENT DECISION BOUNDARY alone (vs k-means' boundary) matters,
    without yet using soft membership in the construction.
    Returns means [B,D], variances, weights, responsibilities [N,B] (float32),
    hard assignment [N] int (argmax gamma), compressed_dists (Counter per cluster)."""
    init_means, init_variances, init_weights = init_gmm_from_kmeans(
        ds_keys, kmeans_centroids, kmeans_assignment, covariance_type
    )
    means, variances, weights, responsibilities, loglik_history = fit_gmm(
        ds_keys, init_means, init_variances, init_weights,
        covariance_type=covariance_type, tol=tol, max_iter=max_iter,
        batch_size=batch_size,
    )

    hard_assignment = responsibilities.argmax(axis=1)
    n_clusters = means.shape[0]
    compressed_dists = []
    for c in range(n_clusters):
        member_values = ds_values[hard_assignment == c]
        compressed_dists.append(Counter(member_values.tolist()) if len(member_values) > 0 else Counter())

    return {
        "means": means, "variances": variances, "weights": weights,
        "responsibilities": responsibilities, "hard_assignment": hard_assignment,
        "compressed_dists": np.array(compressed_dists, dtype=object),
        "loglik_history": loglik_history,
    }


def build_gmm_soft_dime(ds_values, gmm_means, gmm_responsibilities, vocab_size=VOCAB_SIZE):
    """VARIANT C — soft DIME construction. Replaces the hard indicator z_ic with the
    full responsibility gamma_ic everywhere a count would have been used:

        mu_c^soft   = sum_i gamma_ic x_i / sum_i gamma_ic   (already = gmm_means — the
                                                              GMM's own M-step mean IS
                                                              this quantity, no extra
                                                              work needed)
        P_soft(y=v|c) = sum_i gamma_ic * 1[y_i=v] / sum_i gamma_ic

    The token distribution is the only genuinely new computation: every datastore
    point contributes FRACTIONALLY to every cluster's distribution, not just its
    single hard-assigned cluster. Computed as one sparse matmul
    (gamma.T @ one_hot(y)) rather than a Python loop over N*B pairs — in practice
    gamma_ic underflows to exact float 0.0 for points far from component c (the
    Gaussian density itself underflows), so the resulting per-cluster distributions
    stay naturally sparse despite nothing being explicitly thresholded.
    Returns compressed_dists (Counter per cluster, float-weighted counts) and the
    exact mixture weights alpha_c = N_c^eff / N for reporting."""
    N, B = gmm_responsibilities.shape
    one_hot = csr_matrix(
        (np.ones(N, dtype=np.float32), (np.arange(N), ds_values.astype(np.int64))),
        shape=(N, vocab_size),
    )
    # sparse @ dense (not dense @ sparse — that order is ambiguous across numpy/scipy
    # versions) gives a dense [vocab_size, B] result directly; transpose to [B, vocab_size].
    soft_counts = np.asarray(one_hot.T @ gmm_responsibilities).T

    compressed_dists = []
    for c in range(B):
        row = soft_counts[c]
        nonzero = np.nonzero(row)[0]
        compressed_dists.append(Counter({int(tok): float(row[tok]) for tok in nonzero}))

    N_c_eff = gmm_responsibilities.sum(axis=0)
    alpha_c = N_c_eff / N

    return np.array(compressed_dists, dtype=object), N_c_eff, alpha_c


# ============================================================
# Boundary analysis
# ============================================================

def kmeans_boundary_margin(query_keys, kmeans_centroids):
    """For every query point, distance to nearest (d1) and second-nearest (d2)
    k-means centroid, and margin = d2 - d1. Small margin = near a k-means
    Voronoi boundary. Reuses NearestNeighbors exactly like the rest of the project
    (centroids as the 'datastore', query points queried against it)."""
    index = NearestNeighbors(metric="euclidean").fit(kmeans_centroids)
    distances, _ = index.kneighbors(query_keys, n_neighbors=2)
    d1, d2 = distances[:, 0], distances[:, 1]
    return d1, d2, d2 - d1


def gmm_responsibility_gap(responsibilities):
    """Per point: largest responsibility gamma_(1), second-largest gamma_(2),
    gap = gamma_(1) - gamma_(2), and responsibility entropy H_i = -sum_c gamma_ic log gamma_ic.
    responsibilities: [N, B]."""
    sorted_resp = np.sort(responsibilities, axis=1)[:, ::-1]  # descending
    top1 = sorted_resp[:, 0]
    top2 = sorted_resp[:, 1]
    gap = top1 - top2
    safe = np.clip(responsibilities, 1e-12, None)
    entropy = -(responsibilities * np.log(safe)).sum(axis=1)
    return top1, top2, gap, entropy


def boundary_point_report(margin, gap, entropy, kmeans_assignment, gmm_hard_assignment, responsibilities, percentiles=(1, 5, 10)):
    """For the lowest-margin X% of points (by k-means margin), report how GMM
    behaves there specifically — the direct test of 'do boundary points behave
    differently under GMM'."""
    report = {}
    N = len(margin)
    order = np.argsort(margin)  # ascending: smallest margin first
    for p in percentiles:
        n_take = max(1, int(N * p / 100))
        idx = order[:n_take]
        disagreement = (kmeans_assignment[idx] != gmm_hard_assignment[idx]).mean()
        # how spread-out the soft membership is for these points: effective number
        # of components holding non-negligible mass, via exp(entropy) (perplexity-style)
        soft_spread = np.exp(entropy[idx]).mean()
        report[f"bottom_{p}pct"] = {
            "n_points": int(n_take),
            "mean_top1_responsibility": float(responsibilities[idx].max(axis=1).mean()),
            "mean_top2_responsibility": float(np.sort(responsibilities[idx], axis=1)[:, -2].mean()),
            "mean_responsibility_gap": float(gap[idx].mean()),
            "mean_entropy": float(entropy[idx].mean()),
            "mean_effective_n_components": float(soft_spread),
            "frac_gmm_hard_disagrees_with_kmeans": float(disagreement),
        }
    return report


# ============================================================
# Cluster-level analysis
# ============================================================

def match_components(centroids_a, centroids_b):
    """Optimal one-to-one matching between two sets of B prototypes (Hungarian
    algorithm on pairwise distance) — a safety check per the user's request ('match
    components sensibly if labels differ'), even though GMM is k-means-seeded so
    component order should already line up in practice."""
    B = centroids_a.shape[0]
    cost = cdist(centroids_a, centroids_b)  # avoids materializing a [B,B,D] temporary
    row_idx, col_idx = linear_sum_assignment(cost)
    n_identity = int((row_idx == col_idx).sum())
    return col_idx, n_identity, B  # col_idx[i] = which b-component matches a-component i


def prototype_movement(centroids_hard, centroids_soft, match_idx=None):
    """||mu_c^soft - mu_c^hard|| per cluster, after matching."""
    if match_idx is not None:
        centroids_soft = centroids_soft[match_idx]
    return np.linalg.norm(centroids_hard - centroids_soft, axis=1)


def js_divergence_counter(counter_a, counter_b):
    """Jensen-Shannon divergence between two Counters, computed only over the union
    of their (sparse) support — avoids materializing a 50257-length dense vector
    per cluster for a comparison that only ever touches a few hundred tokens."""
    support = sorted(set(counter_a.keys()) | set(counter_b.keys()))
    if len(support) == 0:
        return 0.0
    a = np.array([counter_a.get(t, 0.0) for t in support], dtype=np.float64)
    b = np.array([counter_b.get(t, 0.0) for t in support], dtype=np.float64)
    a_sum, b_sum = a.sum(), b.sum()
    if a_sum == 0 or b_sum == 0:
        return float("nan")
    p, q = a / a_sum, b / b_sum
    m = 0.5 * (p + q)

    def kl(x, y):
        mask = x > 0
        return float((x[mask] * np.log(x[mask] / y[mask])).sum())

    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def cluster_level_report(centroids_hard, compressed_dists_hard, N_c_hard,
                          centroids_soft, compressed_dists_soft, N_c_eff_soft, alpha_c_soft):
    match_idx, n_identity, B = match_components(centroids_hard, centroids_soft)
    movement = prototype_movement(centroids_hard, centroids_soft, match_idx)

    js = np.zeros(B, dtype=np.float64)
    for c in range(B):
        js[c] = js_divergence_counter(compressed_dists_hard[c], compressed_dists_soft[match_idx[c]])

    return {
        "n_components_matched_to_same_index": n_identity,
        "n_components_total": int(B),
        "prototype_movement": {
            "mean": float(movement.mean()), "std": float(movement.std()),
            "min": float(movement.min()), "max": float(movement.max()),
        },
        "token_distribution_js_divergence": {
            "mean": float(np.nanmean(js)), "std": float(np.nanstd(js)),
            "min": float(np.nanmin(js)), "max": float(np.nanmax(js)),
        },
        "effective_mass_vs_hard_size": {
            "hard_sizes": {"mean": float(N_c_hard.mean()), "std": float(N_c_hard.std())},
            "soft_effective_mass": {"mean": float(N_c_eff_soft.mean()), "std": float(N_c_eff_soft.std())},
        },
        "mixture_weights_alpha": {
            "mean": float(alpha_c_soft.mean()), "std": float(alpha_c_soft.std()),
            "min": float(alpha_c_soft.min()), "max": float(alpha_c_soft.max()),
        },
    }
