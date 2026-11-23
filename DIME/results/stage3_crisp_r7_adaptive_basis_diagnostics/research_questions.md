# CRISP-R7 Research Questions for Subsequent Study

## Approximation theory
- Orthogonal greedy selection is one-step-optimal, not globally optimal for a fixed rank r -- how far from the
  globally best r-subset (over the N_POOL candidate pool) does it land in practice, and is there a tractable
  exact or approximate algorithm (e.g. via submodularity) that would close that gap?
- Does the output-space-vs-query-feature-space gap (Phase F) scale predictably with cluster population or label
  entropy, or does it depend more on the GEOMETRY of how nearby centroids relate to the cluster's own members?

## Geometry
- R4's fixed nearest-centroid anchors are a prefix of a SORTED list -- is there a precise geometric condition
  (e.g. on the local curvature of the centroid manifold) that predicts when this ordering produces redundant
  (ill-conditioned) directions at higher rank, versus genuinely new information?

## Statistical estimation
- B1/B2's gauge-fixed lstsq diagnostic needs no ridge regularization in this delivery (plain least squares) --
  would a properly cross-validated ridge penalty on the SAME gauge change the fixed-vs-adaptive comparison's
  conclusion, and is the gauge choice itself a confound that should be swept rather than fixed by convention?

## Compression architecture
- If adaptive (residual-guided) basis selection shows a genuine, memory-matched improvement over R4's fixed
  basis, what would it cost to make the selection itself part of a DEPLOYABLE representation (i.e., store WHICH
  centroids were selected per cluster, not just assume a fixed prefix) -- is the extra anchor-identity storage
  (already counted in memory_results.csv) the dominant cost, or is it negligible relative to coefficients?
