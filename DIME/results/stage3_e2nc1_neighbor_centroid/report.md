# DIME Stage 3 -- E2-NC1: Neighbor-Centroid Conditional Compression

Matched config (k=20, tau=2.0, alpha=0.25), 77089 val queries, 3811 clusters. Chosen lambda=50.0, chosen gamma=1.0.

## Experiment D -- ablation grid (mean NLL, matched config, val)
- N0_raw: 3.9387
- N1_dime: 4.0209
- N2_primary_c2: 4.0200
- N3_primary_c3: 4.0204
- N4_primary_c2_c3_joint: 4.0202
- N5_primary_c2_stronger_smoothing: 4.0207
- N6_primary_c2_hybrid: 4.0200
- N_shuffle_control: 4.0217
- N7_tri1_reference: 4.0127

## Experiment C -- does c2 carry information about y beyond c1?
Naive pooled plug-in I(Y;C2|C1) = 1.88567 nats (upward-biased, not held-out).
Held-out check: mean log-lik with c2 = -16.9494 vs. c1-only = -16.9498 -- improvement = 0.00045 nats/token (n_eval=190247).

## Experiment F -- discrete (N2) vs. continuous (N7/TRI1 K=8)
Per-query improvement overlap vs. DIME-V1: {'n_improved_by_both': 8866, 'n_improved_by_n2_only': 3778, 'n_improved_by_n7_only': 24564, 'n_improved_by_neither': 39881, 'n_total': 77089, 'jaccard_overlap': 0.23828208987314556}
Storage: N2=16005082 bytes, TRI1 K=8=20518162 bytes, DIME-V1=12893490 bytes, raw=1172346652 bytes.

## Difficult-query tail
- worst-1%: raw=2.3590, DIME-V1=7.0754, N2=7.0337, N7(TRI1)=6.9083
- worst-5%: raw=2.6604, DIME-V1=5.2744, N2=5.2534, N7(TRI1)=5.1628
- worst-10%: raw=2.5713, DIME-V1=4.3060, N2=4.2923, N7(TRI1)=4.2255

## Failure mechanism evidence
{'fraction_cj_keys_with_fewer_than_5_observations': 0.64487028709789, 'n2_vs_shuffle_mean_nll_diff': -0.0017149057288348146, 'n2_vs_dime_mean_nll_diff': -0.0008316303383990586, 'note': "a near-zero n2_vs_shuffle_mean_nll_diff would indicate N2's apparent gain over DIME-V1 (if any) is an artifact of the smoothing formula itself, not genuine (c2,y) association information."}

## Answers to the research questions (from the measurements above only)
**Q1** (does c2 identity carry predictive information beyond c1?): held-out improvement = 0.00045 nats/token; naive plug-in MI = 1.88567 nats.
**Q2** (does this recover lost context-token associations?): compare N1 (DIME-V1) against N2-N6 above.
**Q3** (how much NLL improvement from discrete identity alone?): N2 mean NLL = 4.0200 vs. N1 = 4.0209.
**Q4** (better memory/fidelity trade-off than TRI1?): see Experiment F's storage and NLL comparison above.
**Q5** (does c2 add info beyond DIME's existing multi-centroid weights?): compare N2 against the shuffled control (N_shuffle_control, mean NLL=4.0217) -- shuffle breaks the (c2,y) association while preserving marginals, isolating what the existing weighting alone could achieve.
**Q6** (are improvements concentrated in specific regions?): see the difficult-query tail and per-cluster held-out table in the JSON summary.
**Q7** (would finer neighbor rankings/distances be justified?): compare N2/N3/N4 against N7(TRI1) above -- judge directly from the measured numbers, not pre-decided here.

## Recommendation
Judge directly from the Experiment D grid, Experiment F's storage/overlap comparison, and the held-out Experiment C check above whether E2-NC1 should replace, complement, or be discarded relative to E2-TRI1 -- not asserted here in advance of the real numbers.

## Limitations
- Smoothing lambda and hybrid gamma tuned via a small grid, not a continuous search.
- Experiment C's per-cluster held-out detail limited to the 15 largest-population clusters; the pooled estimate covers all clusters.
- Preference-change examples bounded to 20 cases across the same sampled clusters.
- N7's TRI1 reference is a RECONSTRUCTION from TRI1's persisted hyperparameters, verified to reproduce its real mean NLL within tolerance before use, not a re-run of the original script.
- Correlation/overlap analyses in Experiment F do not establish which mechanism causes any given query's change.