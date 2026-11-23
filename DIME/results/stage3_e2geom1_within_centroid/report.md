# DIME Stage 3 -- E2-GEOM1: Diagnosing Within-Centroid Predictive Geometry

Matched config (k=20, tau=2.0, alpha=0.25), 77089 val queries, 3811 clusters.

## Main comparison (mean val NLL, full scale)
- raw_kNN: 3.9387
- A_dime_v1: 4.0209
- B_full_space_local: 3.9159
- C_anchor_space_local: 4.0128
- D_tri1_compressed: 4.0127

## Experiment A -- spatial structure (sampled clusters)
Per-cluster separability: raw-space vs. shuffled-label control vs. anchor-space, and multimodality fraction -- see experiment_A_per_cluster in the JSON summary for the full per-cluster table.

## Experiment B -- query-level enrichment (subsample)
B1 (correct-token enrichment): {'mean_enrichment_all': 0.04872560380329565, 'mean_enrichment_given_nonzero_prior': 0.06397794617029366, 'frac_enriched_positive': 0.4778, 'n_subsample': 5000}
B2 (query-dependent variation vs. DIME-V1 prior): {'mean_js_local_vs_dime_prior': 0.23105852607445332, 'n_subsample': 5000}
B4 (shuffled-label control): real=3.8854 vs. shuffled_avg=4.0594

## Experiment D -- difficult-query failure attribution
- worst-1%: raw=2.3590, A=7.0754, B=3.8554, C=6.7266, D=6.9083; D1(no support)=0.034, D2(B recovers)=0.956, D3(C fails where B succeeds)=0.183, D4(D fails where C succeeds)=0.209
- worst-5%: raw=2.6604, A=5.2744, B=3.5970, C=5.1124, D=5.1628; D1(no support)=0.020, D2(B recovers)=0.957, D3(C fails where B succeeds)=0.422, D4(D fails where C succeeds)=0.164
- worst-10%: raw=2.5713, A=4.3060, B=3.2204, C=4.2026, D=4.2255; D1(no support)=0.012, D2(B recovers)=0.951, D3(C fails where B succeeds)=0.535, D4(D fails where C succeeds)=0.120

## Experiment E -- anchor-dimension sweep (point-level System C, subsample)
{'nearest': {1: 4.01384307407857, 2: 4.009790729773778, 4: 4.007217064642637, 8: 4.005811889392748, 16: 4.004008727934892, 32: 4.0047836089030175}, 'diverse': {1: 4.015831963940846, 2: 4.01593468468487, 4: 4.015880134841831, 8: 4.01578953912025, 16: 4.0156270702305985, 32: 4.015373636111473}, 'random_projection': {1: 4.014051564464249, 2: 4.014741038048418, 4: 4.0130558970854, 8: 4.013859155821921, 16: 4.01344157901711, 32: 4.010984697740119}}

## Answers to the research questions (from the measurements above only)
**Q1/Q2** (real spatial organization vs. the synthetic Paris/Berlin assumption): see Experiment A's per-cluster purity/cohesion/separability table -- judge directly from those numbers, not asserted here.
**Q3** (does full-space local retrieval improve over DIME-V1 under fixed routing?): System B=3.9159 vs. System A=4.0209 (raw kNN=3.9387 for reference).
**Q4** (information lost to anchor projection, B vs. C): System C=4.0128 vs. System B=3.9159.
**Q5** (information lost to per-token aggregation, C vs. D): System D=4.0127 vs. System C=4.0128.
**Q6** (which failure mechanism dominates)/**Q7** (fraction of the DIME-vs-raw gap System B recovers)/**Q8** (does the evidence justify further anchor-conditioned development?): judge directly from the main comparison, Experiment D's failure attribution, and Experiment E's K-sweep above -- not pre-decided here.

## Limitations
- This is a diagnostic: Systems B and C retain original or anchor-projected individual datastore points per query and are explicitly NOT a proposed deployment architecture.
- Experiment A bounded to 15 largest-population clusters.
- Experiment B's query-level analyses (B1/B2/B4) run on a 5000-query val subsample, not full val -- B3 reuses System B's full-val NLL directly.
- Experiment E's K-sweep run on a 3000-query val subsample, not full val; the full-val K=8/nearest point is covered by System C's main-experiment number above.
- System B's secondary bounded-neighbor-count variant was scoped out of this delivery entirely in favor of the full-cluster primary variant, given the latter already has a real cross-checkable precedent (E2's S3).
- Each system uses a different estimator; differences are empirical performance comparisons, not an exact additive decomposition of information loss.