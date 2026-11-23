# Y1 Functional Complexity Observations

## Rank-8 held-out projection error vs. cluster properties

Correlation between held-out relative reconstruction error (rank=8) and number of distinct tokens:
0.7069367422448913

Correlation between held-out relative reconstruction error (rank=8) and local neighbor count m:
-0.18825337177517984

## Interpretation notes

A positive correlation with n_tokens would suggest label-diverse clusters need higher rank to represent their
response geometry. A correlation with m indicates the choice of local neighbor count itself changes how
low-dimensional the target response is -- e.g. m=1's near-one-hot targets may show a DIFFERENT (not necessarily
lower) empirical rank than m=20's smoother targets, since one-hot targets are trivially rank-1 in a sense but
highly discontinuous in q.

Per Section 7's explicit instruction, a low measured rank in the response-matrix SVD is NOT automatically
evidence that the function predicts unseen responses well -- see rank_results.csv's mean_test_tv column (a
genuine held-out generalization measure) alongside this file's correlations (a measure of the TARGET's intrinsic
geometry, computed from the FIT split only) for the distinction the spec requires.

See the JSON summary's functional_complexity (not persisted per-cluster here to avoid an extremely large file;
see cluster_results.csv for the per-cluster/m/rank held-out test_tv instead) for the full per-cluster singular-value
spectra if deeper inspection is needed.
