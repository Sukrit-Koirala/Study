# CRISP-R7 Mathematical Analysis

## Approximation spaces

V_r = span{phi_1,...,phi_r}, phi_l(q) = -2 q.(a_l-a_0) + ||a_l||^2-||a_0||^2 (TRI1's anchor-difference feature,
reused unchanged). E_r(f_y)^2 = min_{g in V_r} E_q[(f_y(q)-g(q))^2] is the population-optimal approximation
error within V_r -- UNKNOWN in general; this script reports the EMPIRICAL training optimum (Theta* =
argmin_Theta ||Phi Theta - T||_F^2, solved by np.linalg.lstsq) and its HELD-OUT error as two DISTINCT quantities,
never conflated with the true (unknown) population optimum or with R4's own probability-loss-trained model
(which optimizes a different, log-ratio-gauge, ridge-regularized objective).

## Gauge-fixing

delta*_y(q) = log(s_y(q)) - log(n_y) (computed in compute_delta_star, from R1's own raw evidence s_y(q)).
t_y(q) = delta*_y(q) - sum_y p_j(y) delta*_y(q) (gauge_fix). R4's own target z_{q,y}=log(F(y)/p_j(y)) is a
DIFFERENT, equally valid gauge of the SAME equivalence class -- z = delta* - [log(sum_z s_z(q)) - log(N)], and
the bracketed term is common to every label y at a fixed q. softmax(z) == softmax(delta*) == softmax(t) exactly
(verified numerically in synthetic_correctness_checks -- gauge_shift_invariance_exact). Single-label clusters
are excluded from the sample (t is identically zero there, trivially "perfectly fit" by any basis, uninformative).

## Residual-guided (orthogonal greedy) selection

S(phi) = ||E_q[u(q)E(q)]||_2^2 / E_q[u(q)^2], u = phi - P_{V_r}phi, implemented batched over the full N_POOL=64
candidate pool per step (orthogonal_greedy_select). This is the one-step-optimal score for the CURRENT space and
squared-error objective -- it does NOT guarantee the globally best r-subset (a known limitation of greedy/OMP
methods), and it optimizes squared error in the gauge-fixed correction space, NOT the final softmax
distribution's TV/JS/cross-entropy directly (those are evaluated separately, never assumed to track the greedy
objective).

## Output-space vs. query-feature-space separation

query_feature_prediction_quality() fits coords = Phi(q).theta_coord to PREDICT the output-subspace coordinates
(found via R3's own fit-side SVD), then compares: (a) oracle_output_space_rel_error (R3's original diagnostic,
using the TRUE held-out response's coordinates), (b) achievable_rel_error_using_predicted_coords (using
Phi(q)-PREDICTED coordinates instead). The GAP between (a) and (b) is the query-feature approximation error the
spec's Phase F explicitly asks to separate from output-space sufficiency. A Pythagorean (orthogonal) decomposition
is valid for THIS linear reconstruction step (both are least-squares projections in the SAME output subspace),
but is NOT applied to TV/JS after softmax, per the spec's explicit caution -- TV/JS errors are reported as their
own separately-measured quantities throughout.

## Assumptions and limitations

- The candidate pool (N_POOL=64 nearest centroids) is itself a FIXED choice -- greedy selection chooses
  WHICH of these to use, not arbitrary directions in R^768. A genuinely larger or different pool could change
  which clusters show adaptive improvement; not swept here, per the spec's own "do not expand unless justified"
  instruction.
- R-series validation data has been used repeatedly across R1-R6; this delivery remains EXPLORATORY per the
  spec's own instruction, not an independent confirmatory test.
- No document/sequence-grouped query split exists in the cached arrays; the existing position-level split is
  used, stated as a limitation.
