# DIME-V2 Token-Conditioned Geometric Compressibility Study: Final Report

SMOKE=False, n_clusters_total=3811, n_eligible=2678, n_sampled=10.

## Primary comparisons (cluster-level bootstrap, 95% CI)

- tv_all_member_cond_minus_agn: mean=0.016158082119944194, CI=[0.0009923192237592038, 0.034624546207846274], n_clusters=10
- tv_topm_cond_minus_agn: mean=0.010356282469941112, CI=[-0.0011015291317840292, 0.024027885048407416], n_clusters=10
- candidate_recall_cond_minus_agn: mean=0.016007538109333187, CI=[0.002745888789440809, 0.03887167071152734], n_clusters=10
- distortion_cond_minus_agn: mean=7.17051187422336, CI=[2.621006213193699, 11.8592883570959], n_clusters=7

## Outcome classification: C

Token-conditioned residuals are NOT measurably more compact than the label-agnostic baseline at matched storage -- no clear geometric structure to exploit.

## Query-change preservation

mean E_delta (conditioned) = 0.19722598759590235; mean E_delta (agnostic) = 0.1681131445395325; mean E_delta (static zero-change baseline) = 1.0003237680479142; conditioned beats the static baseline on 0.9824999999999999 of pairs; agnostic beats it on 0.9924999999999999 of pairs.

## Limitations

This is a diagnostic study, not a new compressed retrieval algorithm. The controller_train/val query splits reused here have been used repeatedly across every prior CRISP and Y-series experiment in this project, so comparisons remain exploratory, not independently confirmatory. The cluster sample is rho_j-stratified and bounded, not the full cluster population. Phase B's region analysis uses explicit measurable proxies, not full topological recovery. Deliverables beyond the primary named set (cluster_composition_results.csv, token_region_results.csv, distortion_curve_results.csv) are included for full traceability, not to hit a specific file count.

