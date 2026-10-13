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
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from gmm import init_gmm_from_kmeans, fit_gmm, _log_gaussian

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


def build_hard_dime_from_assignment(ds_values, hard_assignment, n_clusters):
    """Generic hard-DIME-store builder from any precomputed hard assignment array —
    used to turn a GMM checkpoint's (or a counterfactual model's) hard assignment
    into a DIME store without re-deriving the k-means/GMM-specific construction
    logic each time."""
    compressed_dists = []
    for c in range(n_clusters):
        member_values = ds_values[hard_assignment == c]
        compressed_dists.append(Counter(member_values.tolist()) if len(member_values) > 0 else Counter())
    return np.array(compressed_dists, dtype=object)


# ============================================================
# Assignment-similarity diagnostics (k-means vs GMM-hard, global)
# ============================================================

def assignment_similarity_report(assignment_a, assignment_b, centroids_a, centroids_b):
    """Global similarity between two hard clusterings of the SAME N points. Cluster
    labels are arbitrary, so 'fraction agreeing' is computed only AFTER Hungarian
    matching on prototype distance; ARI/NMI are label-invariant by construction and
    need no matching step."""
    match_idx, n_identity, B = match_components(centroids_a, centroids_b)
    inverse_match = np.empty(B, dtype=np.int64)
    inverse_match[match_idx] = np.arange(B)
    remapped_b = inverse_match[assignment_b]  # relabel b's clusters into a's index space

    frac_agree = float((assignment_a == remapped_b).mean())
    ari = float(adjusted_rand_score(assignment_a, assignment_b))
    nmi = float(normalized_mutual_info_score(assignment_a, assignment_b))

    sizes_a = np.bincount(assignment_a, minlength=B)
    sizes_b_remapped = np.bincount(remapped_b, minlength=B)
    return {
        "n_components_matched_to_same_index": n_identity,
        "fraction_points_agreeing_after_matching": frac_agree,
        "adjusted_rand_index": ari,
        "normalized_mutual_info": nmi,
        "cluster_size_correlation": float(np.corrcoef(sizes_a, sizes_b_remapped)[0, 1]),
    }


def margin_percentile_curve(margin, assignment_a, assignment_b, bin_edges=(0, 1, 5, 10, 25, 50, 100)):
    """Fine-grained version of boundary_point_report's bottom-X% check — the FULL
    curve across bin_edges percentile bands (not just three single-point summaries),
    answering 'does disagreement actually concentrate near the k-means boundary, or
    is it spread out?'."""
    order = np.argsort(margin)
    N = len(margin)
    curve = []
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        i_lo = int(N * lo / 100)
        i_hi = max(i_lo + 1, int(N * hi / 100))
        idx = order[i_lo:i_hi]
        disagreement = float((assignment_a[idx] != assignment_b[idx]).mean())
        curve.append({
            "percentile_band": f"{lo}-{hi}%", "n_points": int(len(idx)),
            "mean_margin": float(margin[idx].mean()),
            "fraction_disagreeing": disagreement,
        })
    return curve


# ============================================================
# Counterfactual decomposition — which GMM degree of freedom drives disagreement?
# ============================================================

def counterfactual_assignments(X, means, alpha, variances_diag, global_spherical_var=None):
    """Given ONE fitted diagonal GMM (means, alpha, variances_diag [B,D]), derive
    four counterfactual hard-assignment rules that progressively add back one
    geometric degree of freedom at a time — isolating which term actually moves
    assignments, using a single underlying fit rather than four separate EM runs:

    MODEL 0 — means only:            argmin_c ||x - mu_c||^2  (= k-means' own rule)
    MODEL 1 — + learned alpha,
              forced EQUAL spherical covariance (one global scalar variance,
              the mean of the fitted per-component spherical variances):
                  score_c = log(alpha_c) - ||x-mu_c||^2 / (2 * var_global)
    MODEL 2 — + component-specific SPHERICAL variance (derived from the fitted
              diagonal covariance by averaging each component's own per-dimension
              variances into one scalar per component):
                  score_c = log(alpha_c) - D/2 log(2*pi*var_c) - ||x-mu_c||^2/(2*var_c)
    MODEL 3 — full fitted diagonal covariance (the actual GMM score):
                  score_c = log(alpha_c) - 1/2 log|Sigma_c| - 1/2 (x-mu_c)^T Sigma_c^-1 (x-mu_c)

    Returns a dict of 4 assignment arrays [N], keyed "model0".."model3"."""
    B, D = means.shape
    log_alpha = np.log(np.clip(alpha, 1e-300, None))

    # Model 0: pure nearest-centroid, no alpha, no covariance at all
    sqnorm = (X ** 2).sum(axis=1, keepdims=True) - 2.0 * X @ means.T + (means ** 2).sum(axis=1)[None, :]
    model0 = sqnorm.argmin(axis=1)

    # Model 1: + alpha, forced equal spherical variance
    per_component_spherical = variances_diag.mean(axis=1)  # [B] — average each component's own diag into a scalar
    if global_spherical_var is None:
        global_spherical_var = float(per_component_spherical.mean())
    score1 = log_alpha[None, :] - sqnorm / (2.0 * global_spherical_var)
    model1 = score1.argmax(axis=1)

    # Model 2: + alpha, + component-specific spherical variance (still isotropic, just not shared)
    var_c = per_component_spherical[None, :]
    score2 = (log_alpha[None, :] - 0.5 * D * np.log(2 * np.pi * var_c) - sqnorm / (2.0 * var_c))
    model2 = score2.argmax(axis=1)

    # Model 3: full diagonal — the actual fitted GMM
    log_gauss_full = _log_gaussian(X, means, variances_diag, "diag")
    score3 = log_alpha[None, :] + log_gauss_full
    model3 = score3.argmax(axis=1)

    return {"model0_means_only": model0, "model1_alpha_equal_spherical": model1,
            "model2_alpha_component_spherical": model2, "model3_full_diagonal": model3}


# ============================================================
# DIME-specific token-information diagnostics
# ============================================================

def conditional_entropy_report(compressed_dists, global_token_counts=None):
    """H(Y|C=c) per cluster, size-weighted H(Y|C), and I(C;Y) = H(Y) - H(Y|C).
    global_token_counts: Counter or dict of the FULL datastore's token frequency —
    if not given, it's reconstructed by summing every cluster's own Counter (exact
    either way, just saves a pass over the raw data when the caller already has it)."""
    if global_token_counts is None:
        global_token_counts = Counter()
        for c in compressed_dists:
            global_token_counts.update(c)

    total = sum(global_token_counts.values())
    p_y = np.array(list(global_token_counts.values()), dtype=np.float64) / total
    H_Y = float(-(p_y * np.log(p_y)).sum())

    cluster_entropy = np.zeros(len(compressed_dists), dtype=np.float64)
    cluster_purity = np.zeros(len(compressed_dists), dtype=np.float64)
    cluster_n_distinct = np.zeros(len(compressed_dists), dtype=np.int64)
    cluster_size = np.zeros(len(compressed_dists), dtype=np.float64)

    for i, c in enumerate(compressed_dists):
        n = sum(c.values())
        cluster_size[i] = n
        cluster_n_distinct[i] = len(c)
        if n > 0:
            probs = np.array(list(c.values()), dtype=np.float64) / n
            cluster_entropy[i] = -(probs * np.log(probs)).sum()
            cluster_purity[i] = probs.max()

    N_total = cluster_size.sum()
    p_c = cluster_size / N_total
    H_Y_given_C = float((p_c * cluster_entropy).sum())
    I_C_Y = H_Y - H_Y_given_C

    return {
        "H_Y": H_Y, "H_Y_given_C": H_Y_given_C, "I_C_Y": I_C_Y,
        "cluster_entropy": {"mean": float(cluster_entropy.mean()), "std": float(cluster_entropy.std()),
                             "min": float(cluster_entropy.min()), "max": float(cluster_entropy.max())},
        "cluster_purity": {"mean": float(cluster_purity.mean()), "std": float(cluster_purity.std()),
                            "min": float(cluster_purity.min()), "max": float(cluster_purity.max())},
        "cluster_n_distinct_tokens": {"mean": float(cluster_n_distinct.mean()), "std": float(cluster_n_distinct.std())},
        "cluster_size": {"mean": float(cluster_size.mean()), "std": float(cluster_size.std())},
    }
    }
