# C2.1 Validation Report

## Synthetic checks: 20/20 passed

- [x] local_candidate_selection_known_values
- [x] margin_undefined_when_nc_le_m
- [x] local_weights_known_values
- [x] exact_local_distribution_shared_token
- [x] candidate_change_without_token_change
- [x] label_changing_replacement
- [x] fixed_candidate_competition_change
- [x] tv_js_kl_known_values
- [x] static_vs_query_dependent
- [x] mean_distribution_forward_kl_identity
- [x] train_val_baseline_construction_and_evaluation
- [x] centroid_mixture_weights_known_values
- [x] counterfactual_decomposition_known_values
- [x] selection_margin_stability_guarantee
- [x] boundary_tie_handling
- [x] c1_artifact_hash_verification
- [x] missing_c1_artifact_raises
- [x] missing_fingerprint_raises
- [x] routing_duplicate_and_normalization
- [x] stratified_cluster_sample_covers_axes

## Real-data invariants checked at runtime

- C1 artifact centroids/assignment hash verified: e7adc93629e102ad.../9dae0c746a8f89dc...
- Datastore row count matches C1 assignment length: 381127
- Datastore/controller_train/val keys all dimension 768, length-matched to their values, and finite.
- Recomputed per-cluster population from the loaded assignment matches C1's cluster_statistics.csv n_j column exactly.
- Routing index matrices (90170, 20)/(77089, 20) contain no duplicate centroid ids per query.
- Centroid mixture weights verified to sum to 1 for every controller_train/val query.
- Recomputed routing traffic matches C1's recorded traffic within 1e-06.

## Known limitations

- Query identifiers are positional row indices into the controller_train/val cache arrays (the cache format has no row-id field) -- structural, not independently verifiable semantic provenance, same documented limitation as C1.
- Historical alignment of these cluster IDs with older CRISP-R5 cluster IDs is unproven and not assumed anywhere in this script.
- The counterfactual TV decomposition (tv_weight_change, tv_selection_change) is explicitly path-dependent and NOT asserted to sum exactly to tv_total.
- Sampling is bounded to a stratified subset of clusters; aggregate statistics are NOT extrapolated to all 3,811 clusters without accounting for inclusion probabilities -- see `sampling_report.md` for per-cluster inclusion probabilities where calculable.
- Review point 3: nearby query PAIRS are a deterministic 1-nearest-neighbor pairing, not an independent sample -- a single query can be the nearest neighbor of several other sampled queries simultaneously (directed-pair reuse). `cluster_behavior_statistics.csv`'s `max_pair_reuse_count` reports the largest such reuse count per cluster; pair-level statistics should be read as a set of possibly-correlated observations, not i.i.d. draws.
- Review point 3: the margin-vs-displacement stability guarantee (`gamma_m(q) > 2*||q-q'|| => selected sets are equal`) is actively ASSERTED, not just reported, on every pair in `analyze_cluster` -- a violation would abort the run with an AssertionError naming the offending cluster.
