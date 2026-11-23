# Y2 Experimental Design

## Preregistered hypotheses

H1: a sufficiently expressive model (MLP or RBF-large) can memorize a fixed finite training set to
train_tv < 0.05 at m={1,8,n}.
H2: held-out TV improves by more than 0.05 (absolute) over Y1's real A0 benchmark
(0.8293) for at least one new model family.
H3: query-change L1 error beats the DIME zero-change baseline for at least one model, in at least the natural-
pairs category.
H4: held-out TV decreases monotonically (or near-monotonically) as n_train increases, for at least one model.
H5: boundary-enriched training improves natural held-out TV relative to equal-size natural-only training.

## Selected clusters

[50, 647, 373, 873, 603]

## Operational thresholds (defined BEFORE inspecting new held-out results, Section 11)

near_memorization_tv=0.05, held_out_tv_improvement_required=0.05
(vs. Y1's real A0=0.8293, D0=0.8862, N2=0.8301), query_change_must_beat_zero_change=True,
generalization_gap_threshold=0.15.

## Training objective

MLP: soft-target cross-entropy -(T*log_softmax).sum(dim=1).mean(), correctly zero for every exact-zero teacher
label (0*finite=0, never NaN). RBF/A0: R4's own log-ratio-target ridge regression, reused unchanged.

## Stage gates

Y2-A (capacity/memorization) -> Y2-B (learnability/data) -> Y2-C (geometry/failure attribution), run
sequentially; later stages' scope was NOT expanded beyond this delivery's stated bounds, since the Y2-A/B
results (see report.md) did not show a clear justification for a larger grid.
