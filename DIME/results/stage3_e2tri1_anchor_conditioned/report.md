# DIME Stage 3 -- E2-TRI1: Anchor-Conditioned Predictive Compression

Matched config (k=20, tau=2.0, alpha=0.25), 77089 val queries, 3811 clusters. Chosen anchor-selection method: **nearest** (Experiment A comparison: {'nearest': {'mean_accuracy': 0.009527377347803343, 'n_clusters_evaluated': 15}, 'random': {'mean_accuracy': 0.007105157713060217, 'n_clusters_evaluated': 15}, 'diverse': {'mean_accuracy': 0.008593533612951213, 'n_clusters_evaluated': 15}}).

## Experiment D -- ablation grid (mean NLL, matched config, val)
- A0_raw: 3.9387
- A1_dime: 4.0209
- A2_radial: 4.0210
- A3_diff_K1: 4.0174
- A4_diff_K2: 4.0157
- A5_diff_K4: 4.0141
- A6_diff_K8: 4.0127
- A7_diff_K16: 4.0127
- A_gaussian_K8: 4.0146
- A_random_proj_K8: 4.0203

Best K by controller_train-subsample NLL: **K=8** (mean-only diff NLL=4.0127 on val).

## Distributional metrics (val subsample)
Mean JS(DIME-V1, raw) = 0.4443; Mean JS(anchor-conditioned, raw) = 0.4301.

## Difficult-query tail
- worst-1%: raw=2.3590, DIME-V1=7.0754, anchor-conditioned=6.9083
- worst-5%: raw=2.6604, DIME-V1=5.2744, anchor-conditioned=5.1628
- worst-10%: raw=2.5713, DIME-V1=4.3060, anchor-conditioned=4.2255

## Experiment E -- cohesion diagnostic
Spearman correlation (cohesion vs. anchor-minus-DIME true-token-probability delta): {'spearman_rho': 0.14889091767118978, 'p_value': 0.47750833114233526, 'n': 25}

## Compression efficiency
DIME-V1: 12893490 bytes. Anchor mean-only (K=8): 20518162 bytes (59.14% overhead vs. DIME-V1, 57.1x compression vs. raw kNN).

## Answers to the research questions (from the measurements above only)
**Q1** (can anchor-relative signatures distinguish token groups raw frequency can't?): synthetic test passed=True; real separability accuracy vs. K: {1: 0.0038488143841297353, 2: 0.0023352824492465663, 4: 0.009527377347803343, 8: 0.011642465342159673, 16: 0.008441255344637693}.
**Q2** (how much query-conditioned information is recoverable?): see the ablation grid above -- compare A1 (DIME-V1) against A3-A7 (anchor-diff) directly.
**Q3** (is mean sufficient, or is variance/multiple prototypes needed?): compare A6_diff_K8 (4.0127) against A_gaussian_K8 (4.0146).
**Q4** (are existing DIME centroids good enough as anchors?): method comparison above ({'nearest': {'mean_accuracy': 0.009527377347803343, 'n_clusters_evaluated': 15}, 'random': {'mean_accuracy': 0.007105157713060217, 'n_clusters_evaluated': 15}, 'diverse': {'mean_accuracy': 0.008593533612951213, 'n_clusters_evaluated': 15}}); random-projection baseline NLL = 4.0203 vs. anchor-structured K=8 NLL = 4.0127.
**Q5** (does this improve NLL/distribution fidelity under practical budgets?): see the ablation grid, distributional metrics, and compression-efficiency sections above.
**Q6** (does it help the difficult-query tail?): see the difficult-query tail section above.
**Q7** (does the evidence justify further development?): judge directly from Q1-Q6's actual measured numbers above -- not pre-decided here.

## Limitations
- Anchor-selection-method comparison done at a single K=4 on 15 largest-population clusters, not the full grid.
- Shrinkage N0 and base tau_z tuned once on a 8000-query controller_train subsample at K=4, held fixed elsewhere (tau_z rescaled by sqrt(K/4)).
- Gaussian variant and random-projection baseline evaluated only at the single best K, not the full grid.
- Distributional metrics (JS, entropy) computed on a 4000-query val subsample, not full val.
- Tuned-config secondary pass not implemented, consistent with E2's own stated precedent.
- Correlation (e.g. cohesion vs. prediction delta) does not imply causation.