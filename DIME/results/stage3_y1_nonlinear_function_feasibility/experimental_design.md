# Y1 (Nonlinear) Experimental Design

## Cluster/query sampling (Phase E)

Scanned 150 eligible candidates (population >= 50, up to 150);
86 met the STRICTER sufficiency bar (fit>=30/tune>=15/test>=20, vs.
R-extra's 5/3/3). Selected the top 15 by total routed-query count: [50, 647, 373, 873, 603, 923, 965, 1598, 795, 2059, 2622, 374, 1733, 3620, 497].

## Splits

Per-cluster fit/tune/test splits use the SAME position-level controller_train(fit/tune)/val(test) convention as
every prior script in this project (no document/sequence-grouped split exists in the cached arrays -- stated as
a limitation, not assumed away). No held-out (test) response is ever used to select anchors or tune
bandwidth/rank/lambda -- N2's anchors are farthest-point-sampled from the FIT split only; bandwidth and ridge
lambda are both selected on the TUNE split.

## Models

D0 (fixed histogram), A0 (R-extra's affine family, reused unchanged), N1 (RBF, globally shared fixed anchors),
N2 (RBF, cluster-specific training-derived anchors) -- see module docstring for exact definitions.

## Primary configuration

m=8, tau=2.0, primary rank=8, rank grid=[4, 8, 16, 32], bandwidth scale grid=
[0.5, 1.0, 2.0] (relative to each anchor set's own median anchor-to-fit-query distance).

## Secondary analyses

m-sweep ([1, 4, 8, 20]) at the primary rank; a learning curve (N2, primary rank, varying n_train in
[10, 20, 40, 80, 150]); a neighbor-margin quantile breakdown of A0 vs. N2 error.
