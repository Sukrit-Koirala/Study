# Y1 (Nonlinear) Research Questions for Subsequent Study

## Approximation theory
- Is there a precise relationship between the neighbor-margin distribution under Q_C and the MINIMAL number of
  RBF bumps needed for a given held-out TV target, beyond the empirical correlation measured here?
- A1's convexity result holds for EVERY m -- does the combinatorial complexity of achievable regions (not
  computed here, only convexity) follow any of the known low-dimensional order-k Voronoi complexity bounds in
  d=768, or does high dimensionality change which bounds are even relevant?

## Geometry
- N1 (shared anchors) and N2 (cluster-specific anchors) represent two extremes -- is there an intermediate
  "partially shared" scheme (e.g. anchors shared within a group of geometrically similar clusters) that captures
  most of N2's adaptivity benefit at closer to N1's amortized storage cost?

## Statistical estimation
- The learning curve reuses a SINGLE bandwidth/lambda tuned at the full training-set size for every smaller
  subsample -- would re-tuning at each subsample size change the curve's shape, and is that re-tuning itself
  stable at very small n_train?

## Compression architecture
- If N2 shows a genuine, memory-accounted improvement over A0 in the small-m regime, what would a PRODUCTION
  version need to store the cluster-specific anchors efficiently (e.g. a lower-precision or compressed
  representation of the 768-dim anchor vectors) without erasing the adaptivity advantage this delivery measured?
