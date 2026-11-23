# Y1 Audit (Phase 1, per Y2's own instruction)

## Real results (read directly from results/stage3_y1_nonlinear_function_feasibility/y1_nonlinear_summary.json)

Y1's real sample: 15 clusters selected from 86/150 scanned candidates meeting fit>=30/tune>=15/test>=20 (its own
per_cluster_counts show 45-63 fit queries/cluster -- scarce). At m=8, rank=8 (primary config): D0=0.8862,
A0=0.8293, N1=0.8510, N2=0.8301 mean held-out TV. N2 did NOT significantly beat A0 on real data
(frac_clusters_N2_beats_A0=0.533, 95% CI [-0.0092,+0.0118]) despite clearly beating A0 on Y1's own pilot's trivial
synthetic structure -- confirming the pilot/real gap this project has repeatedly found, not papered over here either.
Query-change preservation failed for every model (A0=2.80, N1=2.07, N2=2.41, all worse than D0's 1.77). The
m-sweep showed nonlinear models helping at low m but losing to the histogram at m=20. Y1's own verdict:
"Promising but insufficient evidence."

## Y1's exact prior implementation (confirmed by direct inspection of
run_y1_nonlinear_function_feasibility.py, reused UNCHANGED here for D0/A0/N1/N2)

- Teacher: run_y1_per_cluster_function's local_m_distribution -- hard top-m selection by real Euclidean distance,
  then a softmax restricted to those m members. Reduces exactly to the all-member exact_evidence target when
  m >= population (verified in Y1's own check 1, not re-verified here to avoid duplicating that work).
- A0: R4's own affine anchor-difference basis (confirmed affine in q by R7's audit), fit via log_ratio_target +
  ridge + reconstruct_logit.
- N1/N2: RAW (never logged) Gaussian RBF features -- genuinely non-affine, verified on a synthetic XOR target in
  Y1's check_non_affine_expressivity (affine accuracy 0.525, RBF accuracy 0.978).
- Routing: nearest-centroid (k=1) assignment of controller_train/val queries to clusters, identical convention
  to every prior script in this project.
- Distance/temperature convention: Euclidean distance, tau=2.0 -- unchanged here.

## Causes of exclusion / query-sample limitation (direct, not assumed)

Y1's own audit scanned 150 candidates (population>=50) and found only 86 meeting its fit>=30/tune>=15/test>=20
bar -- the remaining 64 candidates had fewer routed controller_train/val queries than that bar, a direct
consequence of this project's fixed cache sizes (90,170 controller_train + 77,089 val queries routed via
nearest-centroid assignment across 3,811 clusters, averaging ~44 queries/cluster) spread unevenly by the
underlying MiniBatchKMeans partition. Y2's own audit (broader pool, higher bar, Section 5 above) reaches the
same structural ceiling -- there is no additional natural query data beyond these three cache files.

## Were Y1's nonlinear models limited primarily by architecture, anchors, bandwidth, data, or optimization?

Y1 itself did not separate these explanations (it compared only N1/N2 against A0, both still rank<=32, both
still fit by closed-form ridge, with a 3-point bandwidth sweep). Y2's explicit purpose is to separate exactly
these candidate explanations -- the memorization test (Y2-A3) isolates architecture/optimization from data/
generalization; the learning curve (Y2-B2) isolates data availability; the capacity sweep (MLP, RBF-large)
isolates whether rank<=32 itself was the bottleneck. See capacity_results.csv and memorization_results.csv for
the resulting evidence, not asserted here in advance.
