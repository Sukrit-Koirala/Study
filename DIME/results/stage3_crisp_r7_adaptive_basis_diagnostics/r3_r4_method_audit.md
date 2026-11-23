# R3/R4 Method Audit (Phase A)

## R3 audit

- **Quantity approximated (Part A/B/C):** RAW EVIDENCE s_y(q) (R1's exact_evidence output), NOT the normalized
  distribution and NOT a logit correction -- fit_ridge is called directly on fit_ev in Parts A and B. Part C
  explicitly compares this (objective A) against objective B (normalized distribution) and objective C
  (distribution + change penalty), run_crisp_r3_failure_diagnosis.py lines ~554-589.
- **Matrix decomposed by SVD (Part F):** `M_fit - p_j[None,:]`, i.e. the CENTERED, NORMALIZED distribution matrix
  from FIT queries (run_crisp_r3_failure_diagnosis.py, functional_rank_analysis, lines 249-265).
- **Singular vector space:** TOKEN-OUTPUT space (columns = tokens), NOT query-feature space. There is no phi(q)
  basis anywhere in Part F -- it decomposes the responses themselves.
- **Projection error calculation:** `coords_test = M_centered_test @ basis_r.T` -- this uses the TRUE held-out
  response to find its coordinates in the FIT-derived subspace. This is an ORACLE projection, not a prediction
  from q. Confirmed directly from source, not assumed. R7's query_feature_prediction_quality() adds the missing
  query-feature-based prediction for direct comparison.
- **High-projection-error clusters:** Part F ran on only 4 diagnostic clusters (2 "failing": 923/1465 by spec
  citation; 2 "successful", chosen by lowest rank-8 test TV among the base-20 sample) -- not a population-level
  statement.
- **Rank grid used by functional_rank_analysis:** R3's own module-level RANK_GRID=[2,4,8,16,32] (no rank=1) --
  this is a FREE VARIABLE resolved from R3's own module globals at call time, a fact that mattered when this
  function was reused in the Y1 delivery (Y1 needed rank=1 and had to iterate the function's own returned keys
  rather than assume its own grid, caught via self-review before running).
- **Sample counts/regularization:** MAX_FIT/TUNE/TEST=150/50/100; RIDGE_LAMBDA_GRID=[0.01,0.1,1.0,10.0]; Part B's
  own `fit_lstsq`/`fit_standardized_ridge` are DIAGNOSTIC alternatives, never adopted into R4's production path.

## R4 audit

- **phi_l(q) definition:** phi_l(q) = -2*q.(a_l - a_0) + ||a_l||^2 - ||a_0||^2 (TRI1's compute_diff_features,
  run_e2tri1_anchor_conditioned_compression.py lines 161-176) -- AFFINE (linear) in q. a_0 is the cluster's OWN
  centroid; a_l are OTHER centroids.
- **Anchor/basis selection:** select_anchors_nearest(dist_matrix, c, k) = the k NEAREST other centroids by
  centroid-centroid Euclidean distance, in ascending order (run_e2tri1_anchor_conditioned_compression.py lines
  137-138). Anchors are DIME CENTROIDS, never datastore vectors, query vectors, or learned parameters.
- **Nested or independent basis sets across rank:** NESTED -- `a_idx = get_anchors(c)[:r]` (R4 line ~560) takes
  a PREFIX of the same sorted anchor list for every rank. Rank 16 is rank 8 plus the next 8 nearest centroids,
  confirmed directly, not assumed.
- **Kernel type/bandwidth:** none -- this is a pure linear (affine) feature map, not an RBF kernel (R3's Part A
  RBF variant, with a gamma bandwidth, was a SEPARATE diagnostic never adopted into R4's production basis).
- **Shared vs. cluster-specific:** CLUSTER-SPECIFIC -- a_0 is cluster c's own centroid, so the feature map
  differs per cluster even for the same anchor indices.
- **Coefficient fitting:** closed-form ridge (fit_ridge, PLAIN, non-standardized), 4-point lambda grid, selected
  on the TUNE split by held-out TV (R4 lines 348-357, B4 block of fit_rank_all_methods).
- **Loss function / target:** z_{q,y} = log(F(q)(y)/p_j(y)) (log_ratio_target, R4 lines 205-215) -- a specific
  gauge of the exact correction, NOT the spec's own suggested gauge (see mathematical_analysis.md).
- **Normalization/clipping:** B4/B5's softmax reconstruction needs NO clipping (always valid by construction);
  B2/B3 (raw-evidence/direct-distribution objectives) DO need clip-and-renormalize.
- **Storage per basis function:** one anchor index (4 bytes) shared across all tokens, plus one coefficient per
  (token, rank) pair (4 bytes each) -- b4_bytes_for(n_tokens, rank), R4 lines 234-237.

## Required outcome -- which diagnostic measures what

1. **Output-subspace approximation error:** R3 Part F (oracle, as audited above).
2. **Query-feature approximation error:** NOT measured anywhere in R3 or R4 directly -- R4's own held-out TV/JS
   conflates query-feature prediction quality with the choice of objective (log-ratio vs. raw evidence) and with
   regularization; R7's query_feature_prediction_quality() isolates this directly for the first time.
3. **Optimization or estimation error:** R3 Part B (fitting-method comparison, holding basis+target fixed).

These three remain reported as DISTINCT quantities throughout R7, per the spec's explicit instruction, never
conflated into a single "error vs. rank" number.
