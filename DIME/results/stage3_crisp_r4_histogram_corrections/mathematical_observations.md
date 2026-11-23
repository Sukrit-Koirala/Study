# CRISP-R4 Mathematical Observations

## Observation 1 -- The softmax shift-invariance does not create non-uniqueness in this fitting procedure

**Claim:** adding any query-dependent function c(q) to delta_j,y(q) for every label y leaves the reconstructed
distribution F~_j(q) unchanged, yet the ridge-fitted theta used in this script IS unique (given full column rank
plus ridge regularization).

**Evidence:** the fitting loss sum_y ||Phi theta_y - z_y||^2 decomposes column-wise by label (no cross-label
coupling), so each label's theta_y is the unique normal-equations solution for the SPECIFIC canonical target
z_{q,y} = log(F_j(q)(y)/p_j(y)) -- not an abstract equivalence class. The invariance only matters if one tries to
recover "the" delta from an already-known F~ ambiguously; this script's fitting procedure never does that, since it
regresses directly onto a determinate numerical target.

**Counterexample/caveat:** if a future variant trained directly on F~ via an iterative (non-closed-form) objective
without ever materializing the canonical z target, the shift-invariance WOULD create a flat direction in parameter
space, and gauge-fixing (e.g., centering) would matter for conditioning.

**Status:** proven (direct algebraic argument from the per-column decomposition of the ridge loss).

**Technique:** linear algebra / convex optimization (separable least squares).

## Observation 2 -- Centering the regression target changes conditioning only marginally at this scale

**Claim:** centering z_{q,y} across labels before fitting (subtracting the per-query mean) changes the fitted
coefficient norm and/or held-out TV only slightly relative to the uncentered target.

**Evidence:** centering ablation over a 20-cluster subset: mean theta norm
uncentered=0.3868 vs centered=0.3397;
mean held-out TV uncentered=0.24359 vs centered=0.24359.

**Counterexample/caveat:** this was tested on only 20 clusters at a single rank
(r=8) and single ridge lambda (0.1) -- not swept across the full rank/lambda grid.

**Status:** empirically supported at the scale tested, not proven in general.

**Technique:** numerical linear algebra (condition number / regularization path analysis) would sharpen this.

## Observation 3 -- B4/B5's softmax reconstruction structurally guarantees support preservation; B2/B3 do not

**Claim:** the histogram-correction representation (B4/B5) always produces a full-support, exactly-normalized
distribution with no post-hoc correction, while B2/B3's unconstrained ridge onto raw evidence or raw distribution
can assign negative values that get clipped to exactly zero, silently dropping labels from the reconstructed support.

**Evidence:** see E3's mean_support_preserved_fraction field in the JSON summary -- B4/B5 report exactly 1.0 by
construction; B2/B3 can fall below 1.0 whenever fraction_clipped > 0.

**Status:** proven for B4/B5 (softmax is never exactly zero for finite logits); empirically observed for B2/B3.

**Technique:** this is exactly the practical content of Part 7.5's probability-validity requirement.

## Observation 4 -- Relative log-odds reconstruction error correlates with pointwise distributional error

**Claim:** clusters where B4 reconstructs changes in relative log-odds R_{y,z}(q) poorly also tend to show larger
pointwise TV error.

**Evidence:** correlation between mean |delta R| error and B4's pointwise TV across 66 clusters:
0.76997719146115.

**Counterexample/caveat:** label pairs were bounded to the top-10 highest-probability labels per
cluster, not an unbiased sample of all possible pairs -- rare-label pairwise behavior is not covered by this
observation.

**Status:** empirically supported.

**Technique:** information geometry (log-odds as a natural coordinate on the probability simplex) is the natural
framework for sharpening this.

## Observation 5 -- Required functional rank varies across clusters and correlates with histogram entropy/support size

**Claim:** different clusters need different ranks to reach near-asymptotic (within 10% of rank-32) fidelity, and
this requirement is NOT uniform across the sample.

**Evidence:** correlation between per-cluster required rank (for B4) and cluster entropy: 0.5625568959226998; and
with number of distinct tokens: 0.6907621356768113. See H6_adaptive_rank_signal in the JSON summary for the full
per-cluster breakdown.

**Status:** empirically supported --
this motivates but does not itself implement adaptive rank allocation (E8, deferred).

**Technique:** rate-distortion theory (per-source bit allocation) is the natural framework if E8 is pursued.
