# CRISP-R7: Functional Approximation Diagnostics and Adaptive Basis Selection

## 1. Research question
See module docstring and r3_r4_method_audit.md.

## 2. Audit summary
See r3_r4_method_audit.md for the full audit.

## 3. Mathematical framing
See mathematical_analysis.md.

## 4. Main comparison (B0 vs. B3, memory-matched, rank=8)
{'B0_mean_tv': 0.27997518453630355, 'B3_mean_tv': 0.2746404377274678, 'B4_mean_tv': 0.28447607584455853, 'B1_mean_tv': 0.4126084612780412, 'B2_mean_tv': 0.4062603188615689, 'frac_clusters_B3_beats_B0': 0.696969696969697, 'mean_paired_diff_B3_minus_B0': -0.005334787878787878, 'ci_95_low': -0.013651617803030303, 'ci_95_high': 0.0034327818181818025, 'significant_different_from_zero': False, 'significantly_better': False}

## 5. Output-space vs. query-feature-space gap
{'oracle_output_space_rel_error': 0.6722628204920447, 'achievable_rel_error_using_predicted_coords': 1.230025782220389, 'coord_prediction_rel_error': 1.4545838710803138}

## 6. Query-change preservation
{'B0': {'n_pairs': 2532, 'mean_e_delta': 0.8704500513428121, 'mean_dime_zero_change_error': 0.9215704482622432}, 'B1': {'n_pairs': 2532, 'mean_e_delta': 0.9851188803317534, 'mean_dime_zero_change_error': 0.9215704482622432}, 'B2': {'n_pairs': 2532, 'mean_e_delta': 0.9578973597946288, 'mean_dime_zero_change_error': 0.9215704482622432}, 'B3': {'n_pairs': 2532, 'mean_e_delta': 0.8460258578199051, 'mean_dime_zero_change_error': 0.9215704482622432}, 'B4': {'n_pairs': 2532, 'mean_e_delta': 0.8752018858609795, 'mean_dime_zero_change_error': 0.9215704482622432}}

## 7. Memory-matched comparison across ranks
[{'rank': 2, 'B0_bytes': 740, 'B3_bytes': 740, 'B0_mean_tv': 0.348057, 'B3_mean_tv': 0.333497, 'memory_matched': True}, {'rank': 4, 'B0_bytes': 1236, 'B3_bytes': 1236, 'B0_mean_tv': 0.325861, 'B3_mean_tv': 0.304134, 'memory_matched': True}, {'rank': 8, 'B0_bytes': 2228, 'B3_bytes': 2228, 'B0_mean_tv': 0.279975, 'B3_mean_tv': 0.27464, 'memory_matched': True}, {'rank': 16, 'B0_bytes': 4212, 'B3_bytes': 4212, 'B0_mean_tv': 0.276081, 'B3_mean_tv': 0.275439, 'memory_matched': True}, {'rank': 32, 'B0_bytes': 8180, 'B3_bytes': 8180, 'B0_mean_tv': 0.268219, 'B3_mean_tv': 0.269879, 'memory_matched': True}]

## 8. Final verdict
**Inconclusive**
{'Q1': "R3's Part F measures OUTPUT-SUBSPACE approximation (oracle projection using the true held-out response), not query-feature prediction. See r3_r4_method_audit.md.", 'Q2': "R4's basis is V_r = span of r anchor-difference features to the r NEAREST other centroids, affine in q, NESTED across rank, cluster-specific. See r3_r4_method_audit.md.", 'Q3_basis_expanded_effective_space': False, 'Q4_high_rank_redundant': True, 'Q5_empirical_improves': True, 'Q6_generalizes_held_out': True, 'Q7_improves_fidelity_under_B4_objective': False, 'Q8_query_change_preservation': {'B0': {'n_pairs': 2532, 'mean_e_delta': 0.8704500513428121, 'mean_dime_zero_change_error': 0.9215704482622432}, 'B1': {'n_pairs': 2532, 'mean_e_delta': 0.9851188803317534, 'mean_dime_zero_change_error': 0.9215704482622432}, 'B2': {'n_pairs': 2532, 'mean_e_delta': 0.9578973597946288, 'mean_dime_zero_change_error': 0.9215704482622432}, 'B3': {'n_pairs': 2532, 'mean_e_delta': 0.8460258578199051, 'mean_dime_zero_change_error': 0.9215704482622432}, 'B4': {'n_pairs': 2532, 'mean_e_delta': 0.8752018858609795, 'mean_dime_zero_change_error': 0.9215704482622432}}, 'Q9_holds_memory_matched': True, 'Q10_rank_plateau_implication': 0.5577629617283444, 'findings': {'basis_selection_mismatch': False, 'fitting_or_generalization_limitation': False, 'insufficient_representational_capacity': False}, 'final_classification': 'Inconclusive'}

## Limitations
- Cluster sample reproduces R4's exact procedure (66 clusters after excluding 0 single-label clusters) -- not expanded to all 3,811, per the spec's own instruction.
- No document/sequence-grouped split exists in the cached data.
- This delivery remains EXPLORATORY, not an independent confirmatory test (R-series validation data has been used repeatedly across R1-R6).
- The candidate pool for greedy selection is fixed at N_POOL=64 nearest centroids, not swept.