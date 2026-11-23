# Y1 Research Questions for Subsequent Study

## Approximation theory
- Does the empirical rank needed for accurate held-out reconstruction of F_j^(m) scale predictably with m, or
  does small m (near-one-hot targets) require a QUALITATIVELY different representation (e.g. a classifier-style
  decision boundary) rather than a smooth low-rank basis?
- Family A (no histogram prior) and family B (histogram-prior correction) share the same basis -- is there a
  closed-form relationship between their respective achievable errors at equal rank, given that F_j^(m) can have
  exact zeros that the histogram-prior parameterization structurally cannot represent at delta=0 (since p_j(y)=0
  is impossible for y in the cluster's support, but the TARGET can still be exactly 0 for small m)?

## Geometry
- Is there a precise notion of "effective local radius" (how far the m-th nearest neighbor typically is) that
  predicts whether F_j^(m) will be well-approximated by the existing anchor-difference basis, which is defined
  in terms of distances to OTHER CENTROIDS, not to the cluster's own members?

## Statistical estimation
- Given Y1's m-grid finding (see rank_results.csv), is there a principled way to choose m itself as a learned
  or adaptive parameter per cluster, rather than a fixed diagnostic grid?

## Compression architecture
- If Y1 establishes genuine compressibility (Q10 in report.md), what would change about CRISP's own R4/R5/R6
  pipeline if the LOCAL-m target (rather than the full-cluster target) were adopted as the production objective?
  This is explicitly a Y2-or-later question, not attempted here.
