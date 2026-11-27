# Mathematical Analysis (Phase A)

## Centering identity (S0)

The Euclidean distance between a query and an observation equals the distance between their centroid-relative residuals exactly, by translation invariance -- verified numerically to 1e-9 tolerance, not merely asserted.

## Exact-reduction identity

Rule (i), the all-member weighted-sum reconstruction, reduces EXACTLY to R1's exact_evidence when every observation is its own weight-1 prototype (component count equals population per token) -- verified to under 1e-8 max absolute error on a synthetic two-token cluster.

## Two distinct, explicitly documented approximation rules (Section 8, point D3)

Rule (i), ALL-MEMBER weighted sum, is used for the all-member predictive target and the geometric-compression comparison: exact at full component count, approximate otherwise.

Rule (ii), TOP-m TRUNCATED: prototypes are pooled across tokens for the label-conditioned case, or drawn from the single shared codebook for the label-agnostic case; the m closest prototypes by raw distance (not weight-adjusted) form the truncated candidate set, and each selected prototype's softmax contribution is scaled by its own multiplicity, or by its per-token histogram fraction in the agnostic case. This is one reasonable, explicitly stated rule, never silently substituted for rule (i) -- the two are evaluated and reported separately throughout this study.

## Compressibility metric

r_min(eps) is the smallest component count r in the grid [1, 2, 4, 8] whose mean squared within-token distortion is at most eps times the r=1 (single-prototype) distortion, for eps in [0.5, 0.25, 0.1]. Falls back to the largest grid value if no eps threshold is reached.

## Phase B proxies

Support/dominance region shape in 768 dimensions is not claimed to be fully recoverable from finite held-out samples. Two bounded, explicit proxies are reported instead: direct support/dominance counts over sampled held-out queries, and a k=1-vs-k=2 within-token k-means inertia-reduction-ratio multimodality proxy.

