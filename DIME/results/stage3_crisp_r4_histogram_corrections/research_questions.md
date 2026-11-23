# CRISP-R4 Research Questions for Subsequent Mathematical Study

## Functional analysis
- Under what conditions on tau_local and the member-point geometry is the within-cluster evidence function
  s_{j,y}(.) well-approximated by a finite-rank RKHS expansion? (Relevant: this experiment's RBF/linear-anchor
  bases are specific finite dictionaries, not a principled RKHS basis -- E2's rank curves are the empirical trace of
  an unknown underlying operator's spectrum.)
- Does the logit-correction map delta -> softmax(log p + delta) have a tractable Lipschitz constant in terms of
  the basis's operator norm, bounding reconstruction error propagation from coefficient error to distribution error?
  (Relevant: Part D's normalization-bound reasoning in R3 was for the additive case; the multiplicative/softmax
  case needs its own bound.)

## RKHS and kernel methods
- Would a Nystrom or random-features approximation of the exact Gaussian/exponential kernel s_{j,y}(q) give a
  principled alternative to the anchor-based finite bases used throughout this series? Keywords: Nystrom
  approximation, random Fourier features, kernel ridge regression.
- Is there a reproducing kernel under which DIME's histogram p_j is literally the kernel-mean embedding of the
  cluster, and the logit correction a perturbation of that embedding? Keywords: kernel mean embedding, maximum
  mean discrepancy.

## Approximation theory
- What is the minimax rate for approximating a family of conditional distributions F_j(q) by a rank-r
  histogram-correction model, as a function of the intrinsic dimensionality of the query manifold routed to
  cluster j? Keywords: nonparametric regression minimax rates, covering numbers.
- Does the "required rank" observed empirically (Observation 5) relate to any known complexity measure of the
  cluster's label-conditional geometry (e.g., margin between label-conditional sub-point-clouds)?

## Geometry
- The anchor-difference basis is centroid-index-specific (TRI1's own construction); R3 found centroid-based
  anchors can be severely rank-deficient relative to farthest-point-sampled anchors in some clusters. Does this
  interact with the logit-correction's fitting stability the same way it did for B2/B3's raw/direct objectives?
  (Not tested directly in this delivery -- R3's anchor-variant diagnostic was not re-run for B4/B5.)
- What is the right notion of "distance to the cluster boundary" for diagnosing where B4's generalization degrades
  (Part 7.4's boundary-generalization question)? Keywords: Voronoi cell geometry, boundary effects in local models.

## Information theory
- Observation 4 suggests log-odds reconstruction error and distributional TV error are correlated but distinct
  quantities -- is there an exact information-theoretic identity (e.g., via the exponential-family/natural-parameter
  duality) relating error in the natural parameters (log-odds) to error in the mean parameters (probabilities)?
  Keywords: exponential family, natural vs. mean parameterization, Bregman divergence.
- Does JS divergence or KL divergence between F_j(q) and F~_j(q) decompose naturally into a term explained by the
  histogram p_j and a term explained by the fitted correction, given the multiplicative parameterization?

## Statistical estimation
- The per-cluster ridge lambda is chosen by TUNE-split grid search independently per cluster -- would a
  hierarchical/empirical-Bayes shrinkage of lambda ACROSS clusters (borrowing strength from similar clusters)
  improve small-cluster estimates without needing per-cluster tuning data? Keywords: empirical Bayes, hierarchical
  shrinkage.
- What is the effective sample size for the cluster-level aggregate statistics reported here, given that
  query-pairs within a cluster are not independent? (Partially addressed via R2's cluster-level bootstrap, reused
  implicitly in spirit but not re-applied to every R4 statistic -- a gap worth closing.)

## Compression architecture
- Given Observation 5's per-cluster rank variation, what is the actual rate-distortion-optimal allocation
  (E8, deferred here) and how much does it improve over the fixed-rank scheme at EQUAL total storage?
- Is there a principled way to share basis coefficients ACROSS clusters (e.g., via a shared low-rank factor
  across all 3,811 clusters' theta tensors) rather than fitting each cluster independently, reducing total storage
  below the current cluster-independent accounting in storage_accounting.csv?
