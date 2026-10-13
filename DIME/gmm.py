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


def _log_gaussian_tied_diag(X, means, variance_shared):
    """X: [N, D]. means: [B, D]. variance_shared: [D] — ONE diagonal covariance
    shared by every component (means still differ per-component, only the spread
    around each mean is forced identical). Returns log N: [N, B]."""
    inv_var = 1.0 / variance_shared  # [D]
    term_xx = ((X ** 2) @ inv_var)[:, None]              # [N, 1] — same for every c, broadcasts
    term_x_mu = X @ (means * inv_var[None, :]).T          # [N, B]
    term_mm = (means ** 2 * inv_var[None, :]).sum(axis=1)[None, :]  # [1, B]
    log_det = np.log(2 * np.pi * variance_shared).sum()   # scalar — identical for every c
    return -0.5 * (log_det + term_xx - 2.0 * term_x_mu + term_mm)


def _log_gaussian(X, means, variances, covariance_type):
    if covariance_type == "spherical":
        return _log_gaussian_spherical(X, means, variances)
    elif covariance_type == "diag":
        return _log_gaussian_diag(X, means, variances)
    elif covariance_type == "tied_diag":
        return _log_gaussian_tied_diag(X, means, variances)
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
    elif covariance_type == "diag":
        variances = np.full((B, D), reg_covar, dtype=np.float64)
        for c in range(B):
            if counts[c] > 0:
                diff = X[assignment == c] - means[c]
                variances[c] = (diff ** 2).mean(axis=0) + reg_covar
    elif covariance_type == "tied_diag":
        pooled_sumsq = np.zeros(D, dtype=np.float64)
        for c in range(B):
            if counts[c] > 0:
                diff = X[assignment == c] - means[c]
                pooled_sumsq += (diff ** 2).sum(axis=0)
        variances = pooled_sumsq / N + reg_covar  # [D], shared by every component
    else:
        raise ValueError(f"unsupported covariance_type: {covariance_type!r}")

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


def fit_gmm_rich(X, init_means, init_variances, init_weights, covariance_type="diag",
                  rel_tol=1e-5, patience=3, max_iter=300, batch_size=5000, reg_covar=1e-6,
                  dead_component_threshold=1.0, checkpoint_every=10, verbose=True):
    """Sibling of fit_gmm (kept separate rather than changing fit_gmm's return shape,
    which would silently break its existing callers) — adds everything the deep-dive
    GMM-vs-k-means study needs: real convergence (relative log-likelihood improvement
    below rel_tol for `patience` consecutive iterations, not just hitting max_iter),
    per-iteration diagnostics, and periodic checkpoints for the
    'does longer EM training actually help DIME, or only log-likelihood' question.

    Adds covariance_type="tied_diag" (one diagonal covariance shared across all B
    components — M-step pools each component's sum-of-squared-deviations around its
    OWN mean, then divides by the total N, not each component's own N_c).

    Hard-assignment-changed-fraction and mean-top-responsibility are tracked WITHOUT
    ever storing the full [N,B] responsibility matrix during the loop — only the
    per-point argmax label and per-point max responsibility (two length-N arrays) are
    kept across iterations, exactly like the existing memory discipline in fit_gmm.

    Returns a dict: means, variances, weights, responsibilities (final, full [N,B]),
    history (list of per-iteration diagnostic dicts), checkpoints (list of dicts
    at every `checkpoint_every`-th iteration, each with means/variances/weights/
    hard_assignment/iteration), converged (bool), n_iterations (int)."""
    N, D = X.shape
    means = init_means.copy()
    variances = init_variances.copy()
    weights = init_weights.copy()
    B = means.shape[0]

    history = []
    checkpoints = []
    prev_ll = None
    prev_hard_assignment = None
    below_tol_streak = 0
    converged = False
    final_iter = max_iter

    for it in range(max_iter):
        log_weights = np.log(np.clip(weights, 1e-300, None))
        total_ll = 0.0
        N_c = np.zeros(B, dtype=np.float64)
        sum_x = np.zeros((B, D), dtype=np.float64)
        if covariance_type == "spherical":
            sum_sqnorm = np.zeros(B, dtype=np.float64)
        else:
            sum_x2 = np.zeros((B, D), dtype=np.float64)

        hard_assignment = np.zeros(N, dtype=np.int32)
        top_resp = np.zeros(N, dtype=np.float64)

        for start in range(0, N, batch_size):
            Xb = X[start:start + batch_size]
            log_gauss = _log_gaussian(Xb, means, variances, covariance_type)
            log_joint = log_weights[None, :] + log_gauss
            log_norm = logsumexp(log_joint, axis=1, keepdims=True)
            total_ll += log_norm.sum()
            gamma_b = np.exp(log_joint - log_norm)  # [b, B] — used then discarded

            hard_assignment[start:start + batch_size] = gamma_b.argmax(axis=1)
            top_resp[start:start + batch_size] = gamma_b.max(axis=1)

            N_c += gamma_b.sum(axis=0)
            sum_x += gamma_b.T @ Xb
            if covariance_type == "spherical":
                sum_sqnorm += gamma_b.T @ (Xb ** 2).sum(axis=1)
            else:
                sum_x2 += gamma_b.T @ (Xb ** 2)

        N_c_safe = np.clip(N_c, 1e-12, None)
        weights = N_c / N
        means = sum_x / N_c_safe[:, None]

        if covariance_type == "spherical":
            variances = (sum_sqnorm / N_c_safe - (means ** 2).sum(axis=1)) / D + reg_covar
            variances = np.clip(variances, reg_covar, None)
        elif covariance_type == "diag":
            variances = sum_x2 / N_c_safe[:, None] - means ** 2 + reg_covar
            variances = np.clip(variances, reg_covar, None)
        elif covariance_type == "tied_diag":
            per_component_sumsq = sum_x2 - N_c[:, None] * (means ** 2)  # [B,D], each c's own spread around its own mean
            variances = per_component_sumsq.sum(axis=0) / N + reg_covar  # [D], pooled across components
            variances = np.clip(variances, reg_covar, None)
        else:
            raise ValueError(f"unsupported covariance_type: {covariance_type!r}")

        mean_ll = total_ll / N
        abs_improve = (mean_ll - prev_ll) if prev_ll is not None else float("nan")
        rel_improve = (abs(abs_improve) / (abs(prev_ll) + 1e-12)) if prev_ll is not None else float("nan")
        frac_changed = (float((hard_assignment != prev_hard_assignment).mean())
                         if prev_hard_assignment is not None else float("nan"))
        n_dead = int((N_c < dead_component_threshold).sum())

        diag_entry = {
            "iteration": it + 1,
            "mean_log_likelihood": float(mean_ll),
            "abs_improvement": float(abs_improve) if prev_ll is not None else None,
            "rel_improvement": float(rel_improve) if prev_ll is not None else None,
            "frac_assignment_changed": frac_changed if prev_hard_assignment is not None else None,
            "mean_top_responsibility": float(top_resp.mean()),
            "n_dead_components": n_dead,
            "min_alpha": float(weights.min()),
            "max_alpha": float(weights.max()),
        }
        history.append(diag_entry)
        if verbose:
            print(f"  EM iter {it+1}: ll={mean_ll:.6f}  rel_improve={diag_entry['rel_improvement']}  "
                  f"frac_changed={diag_entry['frac_assignment_changed']}  dead={n_dead}")

        if (it + 1) % checkpoint_every == 0:
            checkpoints.append({
                "iteration": it + 1, "means": means.copy(), "variances": variances.copy(),
                "weights": weights.copy(), "hard_assignment": hard_assignment.copy(),
                "mean_log_likelihood": float(mean_ll),
            })

        if prev_ll is not None and rel_improve < rel_tol:
            below_tol_streak += 1
        else:
            below_tol_streak = 0

        prev_ll = mean_ll
        prev_hard_assignment = hard_assignment

        if below_tol_streak >= patience:
            converged = True
            final_iter = it + 1
            if verbose:
                print(f"  EM converged at iter {final_iter}: relative improvement < {rel_tol} "
                      f"for {patience} consecutive iterations")
            break

    if not converged and verbose:
        print(f"  EM did NOT converge within max_iter={max_iter} iterations "
              f"(last rel_improve={history[-1]['rel_improvement']})")

    # Always checkpoint the final state too, even if it doesn't land on checkpoint_every
    if not checkpoints or checkpoints[-1]["iteration"] != final_iter:
        checkpoints.append({
            "iteration": final_iter, "means": means.copy(), "variances": variances.copy(),
            "weights": weights.copy(), "hard_assignment": prev_hard_assignment.copy(),
            "mean_log_likelihood": float(prev_ll),
        })

    # Final full E-step pass for the responsibility matrix (same discipline as fit_gmm —
    # the only point in the whole run where the full [N,B] matrix is materialized).
    log_weights = np.log(np.clip(weights, 1e-300, None))
    responsibilities = np.zeros((N, B), dtype=np.float32)
    for start in range(0, N, batch_size):
        Xb = X[start:start + batch_size]
        log_gauss = _log_gaussian(Xb, means, variances, covariance_type)
        log_joint = log_weights[None, :] + log_gauss
        log_norm = logsumexp(log_joint, axis=1, keepdims=True)
        responsibilities[start:start + batch_size] = np.exp(log_joint - log_norm).astype(np.float32)

    return {
        "means": means, "variances": variances, "weights": weights,
        "responsibilities": responsibilities, "history": history, "checkpoints": checkpoints,
        "converged": converged, "n_iterations": final_iter, "covariance_type": covariance_type,
    }
