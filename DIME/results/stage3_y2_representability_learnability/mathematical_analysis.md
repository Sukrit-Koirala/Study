# Y2 Mathematical Analysis

## Target regularity (extends Y1's A1/A2 analysis, not re-derived from scratch)

Y1 already proved R_S = {q: N_m^C(q)=S} is convex (an intersection of perpendicular-bisector half-spaces) and
characterized which neighbor exchanges cause discontinuities. Y2 adds the EXACT quantitative boundary-jump
derivation (Section 7/C2 of the spec):

At an ordinary single exchange where member i leaves the top-m set and member j enters, the tie condition
d(q,h_i)=d(q,h_j) gives EQUAL kernel weights w_i(q)=w_j(q)=w at the boundary. Assuming every other selected
neighbor is unchanged, the normalizer Z is also continuous through the crossing (it is a sum over the SAME set
of m-1 unchanged members plus the one tied weight w, counted once on either side), so:

\[
F_+(q) - F_-(q) = \frac{w}{Z}(e_{y_j} - e_{y_i})
\]

For y_i != y_j: TV(F_+,F_-) = w/Z exactly (a one-directional mass transfer of exactly w/Z from label y_i to y_j).
For y_i == y_j: the right-hand side is identically zero -- NO jump, confirming Y1's qualitative finding with an
exact quantity, not just a sign. This was VERIFIED NUMERICALLY (not merely derived) in
check_boundary_jump_formula() on a synthetic m=1 full-flip case (theoretical=observed=1.0 exactly) and again
directly on REAL cluster data in boundary_results.csv (same_label_mean_exact_jump_tv vs.
different_label_mean_exact_jump_tv, see report).

**Assumptions stated explicitly:** an ORDINARY single exchange (no higher-order ties among 3+ members at
once), and that the OTHER m-1 selected neighbors' weights do not themselves change discontinuously at this same
q (true generically, since their own distances vary smoothly away from their own tie points, which are
independent lower-measure events). Not every geometric neighbor exchange is a predictive discontinuity -- only
different-label exchanges are, exactly as Y1 already established qualitatively; this derivation makes it exact.

## Approximation guarantees for the MLP family

A sufficiently wide one-hidden-layer network with a non-polynomial activation (ReLU qualifies) is a universal
approximator of continuous functions on compact sets (classical universal approximation theorem) -- this is an
EXISTENCE result, not evidence of practical finite-sample learnability, favorable optimization, or efficient
storage, per the spec's own explicit caution. F_C^(m) is bounded (hence in L^2(Q_C) regardless of continuity, as
Y1 already established), so sufficiently expressive families CAN approximate it in expected TV/squared error
even where it is discontinuous, PROVIDED the query distribution assigns zero mass to the exact discontinuity set
-- the same L^2-density argument Y1 made for the RBF family applies identically to the MLP family, since neither
argument depends on the specific function class, only on boundedness and the discontinuity set's measure. This
is explicitly NOT the same as uniform approximation across all of R^768 (impossible near a genuine jump, per
Y1's two-sided-limit argument), and it is explicitly NOT evidence that a SPECIFIC finite MLP, trained on a SPECIFIC
finite query sample with a SPECIFIC optimizer, will actually achieve this -- that is exactly what
memorization_results.csv and learning_curve_results.csv measure empirically, not assumed from the theorem.

## Critical distinctions maintained throughout (Section 3's explicit requirement)

Five separate questions, measured by five separate experiments: (1) existence of a good approximant --
NOT measured directly (R_H^* is unknown); (2) architecture's capacity to represent it -- Y2-A3's memorization
test; (3) optimizer's ability to find good parameters -- also Y2-A3 (persistent high training error after
memorization-test optimization effort implicates optimization or capacity, not data); (4) generalization from a
finite sample -- Y2-B2's learning curves; (5) compressibility of the result -- explicitly OUT OF SCOPE for Y2,
per the spec's own "do not introduce memory budgets" instruction.
