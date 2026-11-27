# Synthetic Correctness Checks (S0-S5 + exact-reduction)

All checks run and assert-pass BEFORE any real cache is touched.

- **S0_centering_identity**: passed=True, 
- **exact_reduction**: passed=True, max_err=1.1102230246251565e-16
- **S1_one_compact_region**: passed=True, tv=0.0011541193051198132
- **S2_multiple_regions**: passed=True, tv_r1=0.4848707095269874, tv_r2=0.0007137354208894396, multimodality_reduction_ratio=0.9957558186414982
- **S3_interleaved_labels**: passed=True, tv=0.0011916262608165048
- **S4_rare_but_important**: passed=True, tv_starved=0.36459010117336066, tv_preserved=0.06843250034471443
- **S5_candidate_vs_weight**: passed=True, candidate_recall=1.0, weight_tv=0.02797259204497804
