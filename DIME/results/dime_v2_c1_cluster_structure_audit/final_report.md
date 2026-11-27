# Final Report -- DIME-V2 Experiment C1

## Q1. Construction

See construction_audit.md for the exact source references and this run's fingerprint.

## Q2. Why 3811 clusters

`max(100, round(N/100))` with N=381127 -> 3811. Persisted tier1 config's n_clusters=3811. These MATCH.

## Q3. Population imbalance

0 empty, 3 singleton, of 3811 configured. Lorenz gap=0.16836678944911296. See structural_summary.md for the full percentile/threshold breakdown.

## Q4-Q5. Geometry and dimensionality

See cluster_statistics.csv (all clusters) and dimensionality_results.csv (40 rows, the bounded stratified sample -- NOT all clusters).

## Q6-Q8. Token composition

See cluster_statistics.csv's V_j/rho_j/entropy columns and structural_summary.md's stratified breakdown.

## Q9. Structural relationships

- n_j <-> R_j (scope=all_nonempty_clusters): pearson_r=-0.06608748612642691, spearman_r=-0.06518019175647795, n=3811
- n_j <-> rms_dispersion (scope=all_nonempty_clusters): pearson_r=-0.05222833039207648, spearman_r=-0.10307078419993454, n=3811
- n_j <-> V_j (scope=all_nonempty_clusters): pearson_r=0.8124729946186275, spearman_r=0.8060557302676246, n=3811
- rms_dispersion <-> entropy (scope=all_nonempty_clusters): pearson_r=0.22370564200109772, spearman_r=0.1972543779638052, n=3811
- rho_j <-> rms_dispersion (scope=all_nonempty_clusters): pearson_r=-0.19097424058596627, spearman_r=-0.28746518014768, n=3811
- traffic_top20_val <-> n_j (scope=all_nonempty_clusters): pearson_r=0.8537200287664936, spearman_r=0.8376803231067741, n=3811
- traffic_top20_val <-> rms_dispersion (scope=all_nonempty_clusters): pearson_r=-0.08807435007722761, spearman_r=-0.15820381056331645, n=3811
- traffic_top20_val <-> V_j (scope=all_nonempty_clusters): pearson_r=0.7540740229520853, spearman_r=0.7320551949871911, n=3811
- traffic_top1_val <-> n_j (scope=all_nonempty_clusters): pearson_r=0.9121542243080725, spearman_r=0.9145411404479405, n=3811
- n_j <-> effective_rank (scope=dimensionality_sample_n=40): pearson_r=0.9296893850648972, spearman_r=0.9346092545118042, n=40
- rms_dispersion <-> effective_rank (scope=dimensionality_sample_n=40): pearson_r=0.32642344118687816, spearman_r=0.16435272045028146, n=40

## Q10. Routing traffic

Reported as TWO separate quantities -- top-1 nearest-centroid traffic (diagnostic-grouping convention) and top-20 matched-config traffic (production mixing convention, mix_dime_and_lm) -- never conflated. See cluster_statistics.csv's traffic_top1_* and traffic_topk_* columns.

## Q11-Q12. Relevance and limitations

See DIME_V2_PREFLIGHT_REVIEW.md Section F for unresolved issues (sklearn version provenance, no prior saved ground-truth artifact, exact extraction CLI invocation). This run's own fingerprint is now saved for future comparison: {"centroids_sha256": "e7adc93629e102ad7dc16fee9b1c0b1a29be599c44e1f5e42fedbf514d96957e", "assignment_sha256": "9dae0c746a8f89dc6f6a3e96b4e0cead73d00965f474d7c3795bdffe6767f13e"}.

No claim is made that geometric compactness or token repetition alone proves predictive compressibility -- that question is explicitly deferred to a later experiment.
