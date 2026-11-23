# Y1 (Nonlinear) Mathematical Analysis

## A1 -- Partition of query space

For fixed cluster members {(h_i,y_i)}, R_S = {q : N_m^C(q) = S} (S a fixed size-m subset) is the intersection
of m*(n-m) half-spaces {q : ||q-h_i|| < ||q-h_j||} (i in S, j not in S) -- each an exact perpendicular-bisector
half-space under Euclidean distance. R_S is therefore CONVEX (an intersection of half-spaces), hence CONNECTED,
PROVEN directly from the definition, not assumed. As m increases, the number of achievable nonempty regions
changes, but the convexity argument holds for every m identically -- it does not depend on m's value, only on
the half-space characterization of "closer than." The boundary between adjacent regions (where the m-th and
(m+1)-th nearest member tie in distance) is itself a single hyperplane -- a Lebesgue-measure-zero subset of
R^{768}. Label duplication (two members at the boundary sharing a label) changes WHICH underlying observation is
counted but not necessarily the resulting distribution -- see A2.

## A2 -- Continuity and smoothness

Within a fixed region R_S, F_C^(m)(q) is a ratio of sums of exp(-||q-h_i||/tau) terms (i in S), each real-analytic
in q wherever q != h_i -- so F_C^(m) is real-analytic (hence smooth) in the INTERIOR of every region. AT a
boundary where the m-th and (m+1)-th nearest exchange:
- SAME-label exchange: the two members' kernel weights are EQUAL exactly at the tie (d_i=d_j there), so the
  labeled sum is the SAME value immediately before, at, and after the crossing -- NO discontinuity, verified
  directly on a synthetic two-point cluster in check_discontinuity_behavior() (same-label case trivially continuous
  by this argument, not separately coded since the one-point-swap construction already isolates the mechanism).
- DIFFERENT-label exchange: one label's contribution LOSES exactly the shared tie-weight and the other GAINS it
  -- a GENUINE finite jump of that weight's share of the total mass, verified directly (not merely asserted) in
  check_discontinuity_behavior(): a 1-D, 2-member cluster at -1 (label 0) and +1 (label 1), m=1, shows an EXACT
  flip from (1,0) to (0,1) at q=0, with zero smooth transition region.
- Weight changes WITHOUT membership changes are the smooth, real-analytic interior behavior above -- no special
  treatment needed.
For m=1, F_C^(1) is PIECEWISE CONSTANT, with jump discontinuities exactly on the (n-1)-dimensional Voronoi
boundary of the cluster's own members (the order-1 Voronoi diagram). For m>1, F_C^(m) is piecewise SMOOTH
(real-analytic on each order-m Voronoi cell), with jumps only at order-(m,m+1) boundaries.

## A3 -- Approximation in different norms

F_C^(m)(q) is a probability vector (each component in [0,1]), hence BOUNDED, hence trivially in
H_C = L^2(Q_C; R^{L_C}) for ANY probability measure Q_C, regardless of continuity -- boundedness alone suffices
for L^2 membership under a finite measure. However, F_C^(1) (and F_C^(m) generally) CANNOT be uniformly
approximated by continuous functions to within less than half the jump size at a genuine discontinuity: if g is
continuous and |F-g|<eps everywhere, then as q approaches the boundary from both sides, continuity of g forces
g's value to be within eps of BOTH one-sided limits of F, whose difference is the jump size Delta -- so
Delta <= 2*eps, i.e. eps >= Delta/2, a classical two-sided-limit argument. In EXPECTED squared error, continuous
(even smooth) functions ARE dense in L^2(Q_C) whenever Q_C assigns exactly zero probability to the discontinuity
hyperplane itself -- true whenever Q_C is absolutely continuous w.r.t. Lebesgue measure (or more generally,
whenever Q_C has no atom on that specific hyperplane), which is the generic, expected case for continuous
GPT-2 hidden-state query distributions. This is the precise reason a discontinuous target is not automatically
"unlearnable" in expectation even though it is provably unlearnable uniformly.

**The spec's own caution, respected explicitly, not assumed away:** the THEOREM above only requires Q_C(exact
boundary)=0, not that Q_C assigns negligible mass to a NEIGHBORHOOD of the boundary. If real query data
concentrates substantially NEAR (not exactly on) label-switching boundaries, finite-sample approximation there
remains genuinely harder in practice (higher variance, more basis functions needed locally) even though the
infinite-capacity L^2-density theorem still holds. This gap between infinite-capacity theory and finite-sample
practice is measured directly via the margin diagnostic (Phase F / margin_results, not merely asserted away).

## A4 -- Geometric complexity

Measurable diagnostics computed in this delivery: number of distinct labels (n_tokens), the neighbor margin
gamma_m(q) = d_(m+1)(q)-d_(m)(q) and its distribution across held-out queries (small margins flag queries near a
switching boundary), and the empirical conditioning (singular-value spectrum) of the realized RBF feature matrix
per cluster. The ambient dimension d=768 is NOT assumed equal to the query representations' effective
dimensionality -- the feature-conditioning plot (7_feature_conditioning.png) reports the REALIZED singular-value
decay directly rather than assuming full rank.

## Required mathematical conclusion

Given A1-A4, an approximation family for F_C^(m) should ideally be: (i) capable of NONLINEAR dependence on q
(to represent curvature near non-affine label boundaries -- verified achievable, not merely assumed, in
check_non_affine_expressivity()); (ii) SPATIALLY LOCALIZED (an RBF bump naturally concentrates its influence near
its own anchor, unlike an affine feature's global linear gradient); (iii) able to place multiple bumps for the
SAME label in geometrically separated regions (RBF features trivially support this via multiple anchors per
label-relevant region; an affine family cannot represent two disjoint regions for one label without a non-affine
combination); (iv) flexible enough near switching boundaries (bandwidth controls this trade-off directly); and
(v) compact and finite-sample-stable (the learning-curve experiment measures this last property directly, not
assumed). These are candidate properties motivating the RBF family tested here, not claims that every successful
family must have them.
