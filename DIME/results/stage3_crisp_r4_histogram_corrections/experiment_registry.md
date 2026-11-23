# CRISP-R4 Experiment Registry

| ID | Dataset/split | Cluster sample | Temperature | Model type | Basis | Rank | Loss | Regularization | Seed | Code revision | Output location | Status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| E1 | controller_train/val, seed=42 clustering | 66 clusters (base20+stratified+sensitivity-enriched) | N/A | sampling | N/A | N/A | N/A | 42 | this script | crisp_r4_summary.json:cluster_sampling | completed |
| E2 | controller_train(fit/tune)/val(test) | 66 clusters | 2.0 | B1-B5 | linear_anchor_diff | [0, 1, 2, 4, 8, 16, 32] | raw_ev/dist/logit/logit+change | ridge(0.01-10) | 42 | this script | crisp_r4_summary.json:E2_rank_vs_fidelity | completed |
| E3 | val(test) | 66 clusters, stratified by sensitivity | 2.0 | B1-B6 | linear_anchor_diff / full-support repr. | 8 | see E2 | see E2 | 42 | this script | crisp_r4_summary.json:E3_pointwise_fidelity_by_sensitivity | completed |
| E4 | val(test) query pairs | 66 clusters, 40 pairs/cluster | 2.0 | B1-B5 | linear_anchor_diff | 8 | see E2 | see E2 | 42 | this script | crisp_r4_summary.json:E4_query_change_preservation | completed |
| E5 | val(test) | 66 clusters | 2.0 | B4 vs B5 | linear_anchor_diff | 8 | logit vs logit+change | ridge + lam_delta[0.1, 1.0, 10.0] | 42 | this script | crisp_r4_summary.json:E5_objective_comparison | completed |
| E6 | val(test) query pairs | 66 clusters, top-10 labels | 2.0 | B4 | linear_anchor_diff | 8 | logit | ridge | 42 | this script | crisp_r4_summary.json:E6_relative_log_odds | completed |
| E7 | N/A (storage accounting) | 66 clusters | N/A | B1-B6 | N/A | [0, 1, 2, 4, 8, 16, 32] | N/A | N/A | 42 | this script | storage_accounting.csv | completed |
| E8 | N/A | N/A | N/A | adaptive rank allocation | N/A | N/A | N/A | N/A | N/A | this script | N/A | NOT RUN -- deferred, see docstring/H6 signal |
| Part7 | controller_train(fit)/val(test), 20-cluster subset | 20 clusters | 2.0 | B4 centering ablation + validity stress test | linear_anchor_diff | 8 | logit, centered vs uncentered | ridge(0.1) | 42 | this script | crisp_r4_summary.json:part7_mathematical_investigations | completed |
