# Y1 (Nonlinear) Synthetic Tests

All seven required sanity checks (Section 11) ran BEFORE any real cache was touched; the script asserts on
failure rather than proceeding. Results from this run:

{
  "check1_exact_target_consistency": true,
  "check2_nearest_one_onehot": true,
  "check3_probability_validity": true,
  "check4_non_affine_expressivity": true,
  "check4_affine_accuracy": 0.525,
  "check4_rbf_accuracy": 0.9775,
  "check5_discontinuity_behavior": true,
  "check6_gauge_invariance": true,
  "check7_storage_accounting_by_construction": true,
  "all_passed": true
}

## Check 4 detail (non-affine expressivity)

XOR-style synthetic target (label = XOR(sign(x1), sign(x2)), no linearly-separating hyperplane exists by
construction): affine-logit accuracy = 0.525, RBF-logit accuracy (4 Gaussian
bumps at the quadrant centers) = 0.978. This directly demonstrates the RBF family's
genuine non-affine expressivity on a target an affine family cannot represent, rather than merely asserting the
RBF feature is "different" from R4's affine family by construction.

## Check 5 detail (discontinuity)

A 1-D, 2-member synthetic cluster at q=-1 (label 0) and q=+1 (label 1), m=1: the exact local-1 target is verified
to be EXACTLY one-hot on label 0 for all probed q<0 and EXACTLY one-hot on label 1 for all probed q>0, with the
flip occurring precisely at q=0 and no smooth transition region -- confirming the piecewise-constant,
discontinuous-at-the-boundary characterization in mathematical_analysis.md's A2, not merely asserting it.
