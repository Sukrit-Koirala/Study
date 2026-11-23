# R-extra Audit (formerly named Y1)

R-extra = run_y1_per_cluster_function.py, results under results/stage3_y1_per_cluster_function/ -- UNCHANGED,
not overwritten, not renamed on disk. Only the LABEL used to refer to it changes, per this spec's own instruction.

## Real results (read directly from the existing report.md, not reproduced)

Of the base-20 cluster sample, only 11 clusters passed even R-extra's LOOSE fit>=5/tune>=3/test>=3 filter (9
excluded for insufficient queries) -- exactly the under-sampling problem this delivery's Phase E fixes via a
broader candidate pool and stricter thresholds (MIN_FIT/TUNE/TEST = 30/15/20 here vs. R-extra's 5/3/3).

At m=1, DIME's own histogram TV against the exact local-1 target was 0.974 (near-maximal); R-extra's best
functional model (family B, histogram prior, affine anchor-difference basis) still averaged 0.884 TV at rank 8 --
barely better than noise. Query-change preservation FAILED for every single (m, family) combination tested
(beats_dime_zero_change: False in all 8 cases). R-extra's own Q10 verdict ("Justified") rested specifically on
family B's performance at m=20 (72.7% win-rate over the histogram) -- not on the small-m regime this new Y1
investigates with a genuinely nonlinear family.

## Basis construction (confirmed by direct inspection, not assumed)

R-extra's family B used exactly R4's own machinery: log_ratio_target + fit_ridge + reconstruct_logit, with
linear_anchor_diff_basis (TRI1's AFFINE anchor-difference features, phi_l(q) = -2q.(a_l-a_0)+const) as the basis
-- reused UNCHANGED here for A0, not reimplemented. R-extra's family A (no histogram prior) used a direct,
centered log-target with the SAME affine basis -- also affine in q, for the same reason.

## Why this matters for the new Y1

Both of R-extra's families are AFFINE in q (R4's own basis, confirmed in R7's audit). This new Y1 asks whether a
GENUINELY nonlinear basis (RBF, verified non-affine in check_non_affine_expressivity()) does better on exactly
the regime (small m, query-change preservation) where R-extra's real results were weakest.
