# CRISP-R6 Mathematical Observations

## Observation 1 -- Lower TV/JS does not mechanically imply lower interpolated NLL

**Claim:** R5 found CRISP closer to raw kNN in global TV/JS than DIME, yet R5's own FLOORED retrieval-only NLL
showed CRISP worse than DIME. R6 tests whether this conflict persists under proper, floor-free interpolation.

**Evidence:** compare experiment_A_matched_nll's M2 vs M3 mean_nll directly -- CRISP beats DIME at alpha=0.25
(4.00564 vs 4.02087). NLL is a LOG-scale, SINGLE-coordinate functional of
the retrieval distribution (only the true token's probability matters), while TV/JS are FULL-support divergences --
these can disagree because a method can improve the AVERAGE shape of a distribution while occasionally assigning
much lower probability to the one coordinate (the true token) that NLL actually scores.

**Status:** empirically resolved by this script's own alpha=0.25 matched comparison; NOT assumed from R5's TV/JS
finding either way.

**Technique:** this is exactly the mismatch between an L1/f-divergence-style distributional metric and a
log-loss (proper scoring rule) evaluated at one realized outcome -- relevant literature: proper scoring rules,
calibration vs. sharpness tradeoffs.

## Observation 2 -- Interpolation with a strictly positive LM prior removes the floor-sensitivity problem entirely

**Claim:** for alpha<1, -log[(1-alpha)*P_LM + alpha*P_R] is insensitive to whether P_R is EXACTLY zero or merely
very small, because the GPT-2 term alone already lower-bounds the mixture away from zero. R5's floored
retrieval-only NLL (alpha=1 implicitly, with an artificial floor) is NOT representative of the actual deployed
interpolated system.

**Evidence:** alpha1_report in the JSON summary shows the exact zero-probability rate for every method; comparing
those n_zero counts against experiment_A_matched_nll's alpha=0.25 mean_nll values shows the mixed NLL is governed
by p_lm_true's distribution in the many-zero-probability regime, not by the retrieval method's floor choice.

**Status:** proven algebraically (p_mix >= (1-alpha)*p_lm_true > 0 strictly); the magnitude of how much this
matters in practice is the empirical content of alpha1_report.

**Technique:** this is a direct consequence of log-sum lower-bounding; no further technique needed to prove it,
though quantifying HOW MUCH residual sensitivity remains near alpha=1 is an open empirical question (see
research_questions.md).

## Observation 3 -- Oracle headroom vs. recovered headroom separates "CRISP's own limitation" from "DIME's inherent limitation"

**Claim:** M2 (DIME) vs. M4 (exact oracle) isolates how much NLL improvement is even POSSIBLE under perfect local
reconstruction (holding centroid routing/mixing fixed); M2 vs. M3 (CRISP) isolates how much of that CRISP actually
captures.

**Evidence:** experiment_D's crisp_fraction_of_oracle_headroom_recovered = 0.16439405748762875. A value near 1 means
CRISP is close to saturating the achievable local-reconstruction improvement; a value near 0 or negative means
CRISP's specific approximation (not the underlying local-reconstruction idea) is the bottleneck.

**Status:** empirically measured directly; interpretation (whether the number is "good") is left to report.md's
verdict section, not asserted here.

**Technique:** this is a standard oracle-vs-model decomposition; a sharper treatment would need a formal
regret bound relating within-cluster approximation error (R4's own TV metric) to this NLL-space headroom-recovery
fraction.
