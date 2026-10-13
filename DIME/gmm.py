"""Hand-rolled batched EM for a Gaussian Mixture Model — used only to construct
DIME clusters differently (Variants B/C of the GMM experiment), never at retrieval
time. Deliberately NOT sklearn.mixture.GaussianMixture: we want k-means-seeded init,
a configurable covariance type, logged per-iteration log-likelihood, and a batched
E-step that never materializes the full N x B responsibility matrix during fitting
(only once, at the very end, when it's actually needed downstream).

Supports covariance_type in {"spherical", "diag"}. Full covariance is deliberately
not implemented: at D=768/1024 hidden-state dims, a full covariance matrix per
component is a D x D matrix (768^2 ~= 590K floats) and B of those is not a "small
first experiment" — diagonal already lets each dimension have its own scale, which
is the meaningful step up from spherical for this study.
"""
import numpy as np
from scipy.special import logsumexp


def _log_gaussian_spherical(X, means, variances):
    """X: [N, D]. means: [B, D]. variances: [B] (one scalar per component).
    Returns log N(x_i | mu_c, var_c * I): [N, B]."""
    N, D = X.shape
    sqnorm_diff = (
        (X ** 2).sum(axis=1, keepdims=True)          # [N, 1]
        - 2.0 * X @ means.T                           # [N, B]
        + (means ** 2).sum(axis=1)[None, :]           # [1, B]
    )  # [N, B] == ||x_i - mu_c||^2, computed without forming the [N,B,D] difference tensor
    var = variances[None, :]  # [1, B]
    return -0.5 * D * np.log(2 * np.pi * var) - sqnorm_diff / (2.0 * var)


def _log_gaussian_diag(X, means, variances):
    """X: [N, D]. means: [B, D]. variances: [B, D] (per-component, per-dimension).
    Returns log N(x_i | mu_c, diag(var_c)): [N, B]."""
    # log N = -1/2 * sum_d [ log(2*pi*var_cd) + (x_id - mu_cd)^2 / var_cd ]
    # Expand (x_id - mu_cd)^2 / var_cd = x_id^2/var_cd - 2*x_id*mu_cd/var_cd + mu_cd^2/var_cd
    # so every term is an [N,D] @ [D,B]-style matmul against 1/var-weighted means — no [N,B,D] tensor.
    inv_var = 1.0 / variances              # [B, D]
    term_xx = (X ** 2) @ inv_var.T         # [N, B]
    term_x_mu = X @ (means * inv_var).T    # [N, B]
    term_mm = (means ** 2 * inv_var).sum(axis=1)[None, :]  # [1, B]
    log_det = np.log(2 * np.pi * variances).sum(axis=1)[None, :]  # [1, B]
    return -0.5 * (log_det + term_xx - 2.0 * term_x_mu + term_mm)


def _log_gaussian(X, means, variances, covariance_type):
    if covariance_type == "spherical":
        return _log_gaussian_spherical(X, means, variances)
    elif covariance_type == "diag":
        return _log_gaussian_diag(X, means, variances)
    else:
        raise ValueError(f"unsupported covariance_type: {covariance_type!r}")


def init_gmm_from_kmeans(X, centroids, assignment, covariance_type, reg_covar=1e-6):
    """Seeds the GMM from an already-fitted k-means solution, per the user's explicit
    request for a stable, meaningful comparison (random init would make Variant B/C
    differences partly about init luck, not about the GMM boundary itself).
    X: [N, D]. centroids: [B, D]. assignment: [N] int, k-means hard labels."""
    N, D = X.shape
    B = centroids.shape[0]
    means = centroids.copy()
    counts = np.bincount(assignment, minlength=B).astype(np.float64)
    weights = counts / N

    if covariance_type == "spherical":
        variances = np.full(B, reg_covar, dtype=np.float64)
        for c in range(B):
            if counts[c] > 0:
                diff = X[assignment == c] - means[c]
                variances[c] = (diff ** 2).sum() / (counts[c] * D) + reg_covar
    else:  # diag
        variances = np.full((B, D), reg_covar, dtype=np.float64)
        for c in range(B):
            if counts[c] > 0:
                diff = X[assignment == c] - means[c]
                variances[c] = (diff ** 2).mean(axis=0) + reg_covar

    return means, variances, weights


def fit_gmm(X, init_means, init_variances, init_weights, covariance_type="diag",
            tol=1e-4, max_iter=50, batch_size=5000, reg_covar=1e-6, verbose=True):
    """Batched EM. Never holds the full [N, B] responsibility matrix during fitting —
    each batch's responsibilities are used immediately to accumulate sufficient
    statistics (N_c, sum_x, sum_x^2) and then discarded. Returns the FINAL
    responsibilities as a deliberate one-time full pass after convergence, since
    Variant C's soft construction and the boundary analysis both need them.

    Returns: means [B,D], variances ([B] or [B,D]), weights [B],
             responsibilities [N,B], loglik_history (list of float)."""
    N, D = X.shape
    means = init_means.copy()
    variances = init_variances.copy()
    weights = init_weights.copy()
    B = means.shape[0]

    loglik_history = []
    prev_ll = -np.inf

    for it in range(max_iter):
        log_weights = np.log(np.clip(weights, 1e-300, None))
        total_ll = 0.0
        N_c = np.zeros(B, dtype=np.float64)
        sum_x = np.zeros((B, D), dtype=np.float64)
        if covariance_type == "spherical":
            sum_sqnorm = np.zeros(B, dtype=np.float64)
        else:
            sum_x2 = np.zeros((B, D), dtype=np.float64)

        for start in range(0, N, batch_size):
            Xb = X[start:start + batch_size]
            log_gauss = _log_gaussian(Xb, means, variances, covariance_type)  # [b, B]
            log_joint = log_weights[None, :] + log_gauss                      # [b, B]
            log_norm = logsumexp(log_joint, axis=1, keepdims=True)            # [b, 1]
            total_ll += log_norm.sum()
            gamma_b = np.exp(log_joint - log_norm)                            # [b, B], then discarded

            N_c += gamma_b.sum(axis=0)
            sum_x += gamma_b.T @ Xb
            if covariance_type == "spherical":
                sum_sqnorm += gamma_b.T @ (Xb ** 2).sum(axis=1)
            else:
                sum_x2 += gamma_b.T @ (Xb ** 2)

        # M-step, from accumulated sufficient statistics only
        N_c_safe = np.clip(N_c, 1e-12, None)
        weights = N_c / N
        means = sum_x / N_c_safe[:, None]

        if covariance_type == "spherical":
            # Var = (E[||x||^2] - ||mu||^2) / D, the isotropic sum-of-squares identity
            variances = (sum_sqnorm / N_c_safe - (means ** 2).sum(axis=1)) / D + reg_covar
            variances = np.clip(variances, reg_covar, None)
        else:
            variances = sum_x2 / N_c_safe[:, None] - means ** 2 + reg_covar
            variances = np.clip(variances, reg_covar, None)

        loglik_history.append(total_ll / N)
        if verbose:
            print(f"  EM iter {it+1}: mean log-likelihood = {total_ll/N:.6f}")

        if it > 0 and abs(total_ll / N - prev_ll) < tol:
            if verbose:
                print(f"  EM converged at iter {it+1} (tol={tol})")
            break
        prev_ll = total_ll / N

    # Final full E-step pass — the one deliberate place we materialize all of gamma.
    log_weights = np.log(np.clip(weights, 1e-300, None))
    responsibilities = np.zeros((N, B), dtype=np.float32)
    for start in range(0, N, batch_size):
        Xb = X[start:start + batch_size]
        log_gauss = _log_gaussian(Xb, means, variances, covariance_type)
        log_joint = log_weights[None, :] + log_gauss
        log_norm = logsumexp(log_joint, axis=1, keepdims=True)
        responsibilities[start:start + batch_size] = np.exp(log_joint - log_norm).astype(np.float32)

    return means, variances, weights, responsibilities, loglik_history
