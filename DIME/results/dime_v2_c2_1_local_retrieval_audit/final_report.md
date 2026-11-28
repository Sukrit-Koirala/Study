# C2.1 Final Report -- Exact Local Retrieval Behavior Audit

Sampled clusters: 67. Query-cluster pairs analyzed: 6545. Nearby-query pairs analyzed: 6545.

## Q1. Candidate-set stability across nearby queries

Mean Jaccard similarity of selected sets between nearby query pairs: mean=0.5670, median=0.5506, n=67 (centroid-mass-weighted: mean=0.5282, median=0.4915, n=67). Selection margin (only where n_C > m): mean=0.1752, median=0.1093, n=64. Review point 3 -- stability by query-query distance (all sampled pairs, pooled across clusters, quartiles of query_query_distance):

- Q1 (n=1637, distance in [0.000, 15.123]): mean_jaccard=0.6882, mean_tv_total=0.2274, mean_js_total=0.0797, frac_stability_guaranteed=0.0024
- Q2 (n=1637, distance in [15.123, 17.907]): mean_jaccard=0.5408, mean_tv_total=0.3624, mean_js_total=0.153, frac_stability_guaranteed=0.0
- Q3 (n=1637, distance in [17.907, 20.295]): mean_jaccard=0.5107, mean_tv_total=0.405, mean_js_total=0.1822, frac_stability_guaranteed=0.0
- Q4 (n=1637, distance in [20.295, 56.030]): mean_jaccard=0.4913, mean_tv_total=0.4703, mean_js_total=0.2221, frac_stability_guaranteed=0.0

Note: these pairs are a deterministic 1-nearest-neighbor pairing, not an independent sample (see `validation_report.md`'s pair-reuse caveat); the stratification is descriptive.

## Q2. Candidate changes vs. label changes

Fraction of nearby pairs whose candidate-set change altered the represented token labels: mean=0.8891, median=1.0000, n=67.

## Q3. Competition weight concentration

Mean effective number of contributing candidates (N_eff): mean=11.8005, median=12.4216, n=67. Mean top-3 weight mass: mean=0.4247, median=0.3888, n=67. Mean centroid mixture weight for the routed query-cluster pair itself: mean=0.0530, median=0.0486, n=67.

## Q4/Q5. Competition-change vs. selection-change contributions to predictive variation

Mean TV from competition-weight change alone (candidates fixed): mean=0.2680, median=0.2867, n=67. Mean TV from candidate-selection change alone (query fixed): mean=0.1749, median=0.1556, n=67. Mean total predictive TV between nearby queries: mean=0.3643, median=0.3652, n=67 (centroid-mass-weighted: mean=0.4084, median=0.4188, n=67). These two components are NOT asserted to sum to the total (the decomposition is path-dependent); see Q11 for forward/reverse-path agreement.

## Q6. Exact local token-distribution variation across the routed-query population

Pairwise local variation: mean TV between F_C(q) at nearby queries = mean=0.3643, median=0.3652, n=67; mean JS divergence = mean=0.1580, median=0.1404, n=67. Query-dependence statistic: I_C (val, in-sample) = mean=1.0485, median=1.0038, n=67 (mutual information in nats between the query and the retrieved token, under the sampled routed-query population; split-half sensitivity in Q11).

## Q7/Q8. Static histogram vs. query-distribution-adapted constant baseline

Mean TV(F_C(q_val), static histogram p_C): mean=0.5858, median=0.6065, n=67 (centroid-mass-weighted: mean=0.5881, median=0.6109, n=67; missing-support fraction: mean=0.0000, median=0.0000, n=67, expected ~0 since every candidate label is necessarily a member of the cluster's own static histogram). Mean TV(F_C(q_val), train-derived constant bar_F_train): mean=0.5245, median=0.5433, n=67 (centroid-mass-weighted: mean=0.5338, median=0.5414, n=67; missing-support fraction: mean=0.0263, median=0.0000, n=67 -- nonzero here is expected whenever a held-out val query's candidates include a label the controller_train sample for that cluster never selected).

## Q9. C1 structural correlates of C2 behavior

- n_j vs mean_jaccard (all_sampled_clusters): pearson_r=-0.8186844546646315, spearman_r=-0.8976308819955032, n=67
- n_j vs mean_jaccard (stratified_only_all_sampled_clusters): pearson_r=-0.8316589721994136, spearman_r=-0.8580957357326815, n=60
- rms_dispersion vs mean_margin (sampled_clusters_with_n_j_gt_m): pearson_r=0.7346628780011193, spearman_r=0.43727106227106216, n=64
- rms_dispersion vs mean_margin (stratified_only_sampled_clusters_with_n_j_gt_m): pearson_r=0.2091082281448111, spearman_r=0.32805376972530687, n=59
- entropy vs mean_tv_total (all_sampled_clusters): pearson_r=0.9485719676000207, spearman_r=0.9185058074920515, n=67
- entropy vs mean_tv_total (stratified_only_all_sampled_clusters): pearson_r=0.9436136817345125, spearman_r=0.9265758069219695, n=60
- distinct_token_fraction vs frac_label_changing_pairs (all_sampled_clusters): pearson_r=0.22356317517572802, spearman_r=0.2795348252524727, n=67
- distinct_token_fraction vs frac_label_changing_pairs (stratified_only_all_sampled_clusters): pearson_r=0.4435900951943941, spearman_r=0.4159033202707748, n=60
- singleton_token_fraction vs mean_tv_total (all_sampled_clusters): pearson_r=0.6414851102931644, spearman_r=0.5246752236211262, n=67
- singleton_token_fraction vs mean_tv_total (stratified_only_all_sampled_clusters): pearson_r=0.6988756258799909, spearman_r=0.5834884839792817, n=60
- obs_frac_ge_2 vs mean_tv_total (all_sampled_clusters): pearson_r=-0.6186225274999667, spearman_r=-0.6436710534725173, n=67
- obs_frac_ge_2 vs mean_tv_total (stratified_only_all_sampled_clusters): pearson_r=-0.7525932900963408, spearman_r=-0.7359602104100998, n=60
- traffic_top20_val vs I_C_val (all_sampled_clusters): pearson_r=0.507594878473871, spearman_r=0.2986760320207696, n=67
- traffic_top20_val vs I_C_val (stratified_only_all_sampled_clusters): pearson_r=0.34464685820626606, spearman_r=0.16969073890237396, n=60
- top_token_fraction vs mean_tv_vs_static (all_sampled_clusters): pearson_r=-0.88507161243933, spearman_r=-0.8428436039514405, n=67
- top_token_fraction vs mean_tv_vs_static (stratified_only_all_sampled_clusters): pearson_r=-0.8796303386806583, spearman_r=-0.8484654827659676, n=60

## Q10. Clusters with high geometric/candidate variability but low predictive variability

See `failure_cases.md` for predeclared-criterion examples drawn from the sampled data; this is NOT asserted to characterize all 3,811 clusters.

## Q11. Observations robust across query splits, analysis choices, and numerical settings

Forward/reverse counterfactual-path agreement (review points 2/6): mean absolute difference between tv_weight_change and its reverse-path counterpart = 0.040352; between tv_selection_change and its reverse = 0.049913 (n=6545 pairs). Small values indicate the decomposition is not an artifact of which query was treated as the anchor.

Split-half I_C sensitivity: mean absolute difference between I_C_val_half1 and I_C_val_half2 across sampled clusters = 0.049122 (n=67 clusters with >=2 queries in each half); Pearson correlation between the two halves = 0.9917. A high correlation / low mean difference indicates I_C is not dominated by small-sample noise at this sample size; a low correlation means I_C estimates at this sample size should be read as noisy, not as precise per-cluster values.

Boundary-tie prevalence (review point 7): mean fraction of sampled queries landing on an exact local top-m boundary tie across clusters = mean=0.0152, median=0.0000, n=67; of those, mean fraction where the tied pair carries different labels = mean=0.3333, median=0.0000, n=3 -- nonzero here means the stable-sort tie-breaking convention is doing real (not merely cosmetic) work in determining F_C(q) for some queries.

## Q12. What C2.1 alone does NOT justify

C2.1 measures exact local retrieval behavior under one diagnostic (m_local, tau_local) choice, over a STRATIFIED SAMPLE of clusters, not the full population -- and the deliberately-included unusual clusters (see `sampling_report.md`) have no inclusion probability and must not be pooled into population-level estimates (Q9's `stratified_only` scope exists for this reason). It does NOT measure distortion-versus-storage (that is C3), does NOT establish which structural property CAUSES compression difficulty (that is C4), and query-dependent variation measured here is NOT, by itself, proof that a cluster's exact retrieval behavior is hard to compress.
