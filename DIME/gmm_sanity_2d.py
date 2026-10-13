"""Synthetic 2D sanity check for the GMM/k-means boundary experiment.

Mathematical claim being verified: if a GMM has EQUAL mixture weights and EQUAL
spherical covariance across all components, its posterior-argmax decision boundary
is not just SIMILAR to k-means' nearest-centroid boundary — it is EXACTLY identical,
as an algebraic fact, not an approximation. Proof: under equal alpha_c = alpha and
equal spherical variance_c = var for all c,

    log(alpha_c * N(x | mu_c, var*I)) = log(alpha) - (D/2) log(2*pi*var) - ||x-mu_c||^2 / (2*var)

The first two terms are IDENTICAL across all c, so

    argmax_c [log(alpha_c) + log N(x|mu_c,var*I)] = argmax_c [-||x-mu_c||^2] = argmin_c ||x-mu_c||^2

which is exactly k-means' nearest-centroid rule. This script confirms that
identically (should show 100.000% agreement, not "close to 100%" — any deviation
below exact agreement indicates an implementation bug, not approximation error),
then shows how each additional GMM degree of freedom (unequal alpha, unequal
spherical variance, diagonal covariance) bends the boundary away from k-means'
straight Voronoi lines. All boundaries are visualized by evaluating both decision
rules over a dense grid, not just the training points.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

from gmm import _log_gaussian, init_gmm_from_kmeans, fit_gmm

FIG_DIR = os.path.join(os.path.dirname(__file__), "figures")
os.makedirs(FIG_DIR, exist_ok=True)

rng = np.random.default_rng(42)


def make_blobs(centers, spreads, n_per_cluster):
    """centers: list of (x,y). spreads: list of (sx, sy) per-axis std devs —
    isotropic (sx==sy) by default, anisotropic when explicitly varied."""
    X, y = [], []
    for c, (cx, cy) in enumerate(centers):
        sx, sy = spreads[c]
        pts = rng.normal(loc=[cx, cy], scale=[sx, sy], size=(n_per_cluster[c], 2))
        X.append(pts)
        y.extend([c] * n_per_cluster[c])
    return np.concatenate(X, axis=0), np.array(y)


def kmeans_labels_on_grid(grid, centroids):
    d2 = ((grid[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2)
    return d2.argmin(axis=1)


def gmm_labels_on_grid(grid, means, variances, weights, covariance_type):
    log_w = np.log(weights)
    log_gauss = _log_gaussian(grid, means, variances, covariance_type)
    return (log_w[None, :] + log_gauss).argmax(axis=1)


def plot_boundary(X, y, grid, grid_labels_kmeans, grid_labels_gmm, xx, yy, title, path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    for ax, labels, name in zip(axes, [grid_labels_kmeans, grid_labels_gmm], ["k-means", "GMM"]):
        ax.contourf(xx, yy, labels.reshape(xx.shape), alpha=0.25, cmap="tab10")
        ax.scatter(X[:, 0], X[:, 1], c=y, cmap="tab10", s=8, edgecolors="none")
        ax.set_title(f"{name} boundary")
        ax.set_xlabel("x1")
        ax.set_ylabel("x2")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print(f"  saved {path}")


def make_grid(X, pad=2.0, n=300):
    x_min, x_max = X[:, 0].min() - pad, X[:, 0].max() + pad
    y_min, y_max = X[:, 1].min() - pad, X[:, 1].max() + pad
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, n), np.linspace(y_min, y_max, n))
    grid = np.stack([xx.ravel(), yy.ravel()], axis=1)
    return grid, xx, yy


def run_scenario(name, X, y, means, variances, weights, covariance_type, kmeans_centroids):
    grid, xx, yy = make_grid(X)
    labels_km = kmeans_labels_on_grid(grid, kmeans_centroids)
    labels_gmm = gmm_labels_on_grid(grid, means, variances, weights, covariance_type)
    agreement = (labels_km == labels_gmm).mean()
    print(f"[{name}] grid-point agreement between k-means and GMM boundary: {agreement*100:.3f}%")
    plot_boundary(X, y, grid, labels_km, labels_gmm, xx, yy, name, os.path.join(FIG_DIR, f"sanity_{name}.png"))
    return float(agreement)


def main():
    results = {}

    # --- Scenario 1: equal weights, equal spherical variance -> must EXACTLY match k-means ---
    centers = [(-4, 0), (4, 0), (0, 4)]
    spreads = [(1.0, 1.0)] * 3
    n_per = [150, 150, 150]
    X, y = make_blobs(centers, spreads, n_per)
    km = KMeans(n_clusters=3, n_init=10, random_state=42).fit(X)

    equal_means = km.cluster_centers_
    equal_variances = np.full(3, 1.0)   # deliberately set EQUAL and matching the true spread
    equal_weights = np.full(3, 1.0 / 3)
    results["equal_weights_equal_spherical"] = run_scenario(
        "equal_weights_equal_spherical", X, y, equal_means, equal_variances, equal_weights,
        "spherical", km.cluster_centers_,
    )

    # --- Scenario 2: same geometry, but UNEQUAL mixture weights (manually perturbed) ---
    unequal_weights = np.array([0.7, 0.15, 0.15])
    results["unequal_weights"] = run_scenario(
        "unequal_weights", X, y, equal_means, equal_variances, unequal_weights,
        "spherical", km.cluster_centers_,
    )

    # --- Scenario 3: equal weights, but UNEQUAL spherical variances ---
    unequal_variances = np.array([0.4, 1.0, 3.0])
    results["unequal_spherical_variance"] = run_scenario(
        "unequal_spherical_variance", X, y, equal_means, unequal_variances, equal_weights,
        "spherical", km.cluster_centers_,
    )

    # --- Scenario 4: genuinely anisotropic blobs, real EM fit with diagonal covariance ---
    aniso_centers = [(-4, 0), (4, 0), (0, 4)]
    aniso_spreads = [(2.5, 0.5), (0.5, 2.5), (1.5, 1.5)]  # first two stretched along different axes
    X2, y2 = make_blobs(aniso_centers, aniso_spreads, n_per)
    km2 = KMeans(n_clusters=3, n_init=10, random_state=42).fit(X2)
    init_means, init_var, init_w = init_gmm_from_kmeans(X2, km2.cluster_centers_, km2.labels_, "diag")
    means_d, var_d, w_d, _, loglik = fit_gmm(X2, init_means, init_var, init_w, covariance_type="diag",
                                              tol=1e-6, max_iter=50, batch_size=1000, verbose=False)
    print(f"[diagonal_anisotropic] EM log-likelihood history: {[round(v,4) for v in loglik]}")
    results["diagonal_anisotropic"] = run_scenario(
        "diagonal_anisotropic", X2, y2, means_d, var_d, w_d, "diag", km2.cluster_centers_,
    )

    print("\n=== SANITY CHECK SUMMARY ===")
    for name, agreement in results.items():
        print(f"{name}: {agreement*100:.3f}% grid agreement with k-means")
    print("\nExpected: 'equal_weights_equal_spherical' should be ~100% (exact algebraic "
          "identity — any shortfall indicates an implementation bug). The other three "
          "should show visibly lower agreement, each for a different reason (weights, "
          "variance, or full anisotropic covariance), demonstrating which GMM degree of "
          "freedom is responsible for the boundary change.")

    return results


if __name__ == "__main__":
    main()
