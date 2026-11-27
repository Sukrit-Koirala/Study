# Validation Report

## Synthetic checks (run before any real data was touched)

- population_sum_identity: passed=True
- population_stats_singleton_vs_empty: passed=True
- geometry_known_values: passed=True
- singleton_cluster_geometry: passed=True
- effective_rank_bounds: passed=True
- token_composition_known_values: passed=True
- routing_traffic_known_values: passed=True
- routing_normalization: passed=True
- duplicate_index_rejection: passed=True
- spearman_matches_scipy: passed=True
- cluster_to_members_matches_manual: passed=True
- fingerprint_matches_persisted_artifact: passed=True
- 2d_stratification_covers_both_dimensions: passed=True

## Real-data invariants (assert_real_data_invariants)

Key/label row-count match, key dimensionality, finite coordinates (keys, labels, centroids), population-sum identity, assignment length/range, compressed_dists length, centroid shape, per-cluster token-sum-equals-population -- all asserted (not merely logged) before any statistic was computed; a failure here stops the run.

**Limitation, stated explicitly**: the key/label check verifies only STRUCTURAL alignment (equal row counts). True semantic row-for-row provenance cannot be independently verified from this cache format, which stores no row-id field -- this is a reported limitation, not a verified guarantee of correct alignment.

## Routing invariants (assert_routing_invariants)

Shape, in-range indices, PER-QUERY INDEX UNIQUENESS (no duplicate retrieved-centroid index within a single query's top-k row -- checked explicitly before the traffic normalization check below is treated as meaningful), and traffic-sum normalization.

- sum_traffic_top1_controller_train: sum=1.0
- sum_traffic_top1_val: sum=1.0
- sum_traffic_top20_controller_train: sum=20.0
- sum_traffic_top20_val: sum=20.000000000000004

## Reproducibility status

This run's construction fingerprint: centroids_sha256=e7adc93629e102ad..., assignment_sha256=9dae0c746a8f89dc..., environment={'python_version': '3.13.12', 'numpy_version': '2.4.6', 'sklearn_version': '1.9.0', 'scipy_version': '1.18.0', 'pandas_version': '3.0.2', 'platform': 'Linux-6.1.0-52-amd64-x86_64-with-glibc2.36'}.
