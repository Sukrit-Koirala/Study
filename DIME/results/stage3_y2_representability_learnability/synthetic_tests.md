# Y2 Synthetic Tests

{
  "mlp_solves_xor": true,
  "mlp_xor_accuracy": 0.9925,
  "boundary_jump_formula_matches_observed": true,
  "observed_jump_tv": 1.0,
  "theoretical_jump_tv": 1.0,
  "all_passed": true
}

Y1's own 7 synthetic checks (exact-target consistency, m=1 one-hot behavior, probability validity, non-affine
XOR expressivity for the RBF family, discontinuity behavior, gauge invariance, storage accounting) are reused
unchanged and were not re-run here to avoid duplicating already-verified work -- see
results/stage3_y1_nonlinear_function_feasibility/synthetic_tests.md for that record. This file adds only the
two checks genuinely new to Y2: that the MLP family can solve a target (XOR) no affine model can, and that the
EXACT boundary-jump formula derived in mathematical_analysis.md matches the real teacher's observed jump, not
merely that a jump exists (which Y1 already checked).
