# CRISP-R5 Mathematical Observations

## Observation 1 -- Local fidelity does not mechanically imply global fidelity

**Claim:** R4 showed B4 substantially improves WITHIN-cluster TV over DIME-V1 (reconstructed fresh here across all
3811 clusters, coverage statistics in crisp_fitting_coverage). R5 tests whether this survives centroid
mixing and global aggregation.

**Evidence:** compare experiment_A_global_fidelity's S4 (CRISP) vs. S1 (DIME) mean_tv/mean_js directly against the
within-cluster improvement reconstructed during Phase 3 fitting (tune-split TV during fitting vs. S1's rank-0
TV). A smaller global gap than the within-cluster gap would be DIRECT evidence that aggregation partially masks
local improvement; a comparable or larger global gap would support local-to-global transfer.

**Status:** empirically evaluated in this script's own E2/E3/E4-equivalent (Experiment A/C), not assumed.

**Technique:** this is a question about how distributional error propagates through a convex combination
(DIME's beta_j(q)-weighted mixture) -- relevant: data-processing-style inequalities for divergences under mixing.

## Observation 2 -- Routing error, aggregation error, and compression error are NOT additive

**Claim:** tv_S0_S2 (routing) + tv_S2_S3 (aggregation) + tv_S3_S4 (compression) does not equal tv_S0_S4 in general,
because TV satisfies the triangle inequality (an upper bound), not an equality, and errors can partially cancel.

**Evidence:** see experiment_B_error_attribution's component_correlation_with_final_error -- if all three
correlate positively but the SUM of means exceeds the mean of tv_S0_S4, this is direct confirmation of
sub-additivity (cancellation), reported explicitly rather than assumed.

**Status:** mathematically guaranteed as an upper bound (triangle inequality); whether cancellation is SUBSTANTIAL
in practice is an empirical question answered by the numbers above.

**Technique:** this is exactly the content of the triangle-inequality slack; a tighter mathematical treatment would
bound the gap via the specific structure of the convex combination rather than generic TV subadditivity.

## Observation 3 -- Candidate-set recall structurally upper-bounds attainable raw-kNN fidelity

**Claim:** no local function (exact or compressed) evaluated only on members of the top-20 DIME-selected clusters
can ever recover the raw top-20 neighbors that routing excludes.

**Evidence:** routing_overlap_full_scale's frac_queries_with_full_top20_recall gives the fraction of queries where
NO excluded neighbor exists (an exact ceiling); on the remaining queries, S2/S3/S4/S5/S6 all inherit whatever
error routing_diagnostics.csv's raw_top20_neighbor_recall_in_candidate_set shows directly.

**Status:** proven as a structural ceiling (S2 by definition cannot see outside U(q)); the empirical MAGNITUDE of
the ceiling's cost is given by the routing-overlap numbers above.

**Technique:** this is a candidate-generation recall question, analogous to retrieval-augmented generation's
"retriever recall upper-bounds generator accuracy" framing.

## Observation 4 -- Evidence-mass-aware aggregation's effect size, holding local functions fixed

**Claim:** comparing S3 (centroid-softmax-weighted exact oracle) against S5 (mass-weighted exact oracle) isolates
exactly how much DIME's beta_j(q) choice (vs. the real evidence mass m_j(q)) matters, independent of any
compression question.

**Evidence:** see experiment_B_error_attribution's mean tv_S3_S5. A near-zero value would mean centroid-softmax
weighting is already a good proxy for real evidence mass; a large value motivates investigating S7 (compressed
mass predictor) as a genuinely separate, worthwhile follow-up -- not assumed in advance.

**Status:** empirically evaluated; see Q7 in report.md for the explicit S7-justification verdict.

**Technique:** this is a question of how well a DISTANCE-based proxy (softmax over centroid distance) approximates
an INTEGRAL quantity (the total kernel mass over all members) -- relevant to kernel density estimation literature
on bandwidth/proxy mismatch.
