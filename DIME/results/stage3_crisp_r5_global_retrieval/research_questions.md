# CRISP-R5 Research Questions for Subsequent Study

## Functional analysis / approximation theory
- Does local conditional-distribution fidelity (as measured in R1-R4) imply any FORMAL bound on globally
  aggregated distributional error under DIME's convex-combination mixing, or only an empirical correlation?
  Keywords: data-processing inequality for f-divergences, convexity of TV/KL under mixtures.
- What determines the rank needed for accurate GLOBAL (not just within-cluster) reconstruction, given that
  different clusters are selected with different frequencies and different beta_j(q) weights across queries?

## Geometry / information theory
- How does centroid candidate-set recall (Observation 3) constrain attainable raw-kNN fidelity as a function of
  the number of selected centroids k and the within-cluster radius distribution? Is there a closed-form
  relationship, or only an empirical one?
- How does evidence-mass approximation (S3 vs. S5's tv_S3_S5) affect cross-cluster probability weighting --
  specifically, is the centroid-softmax-vs-real-mass gap systematically related to cluster population or
  within-cluster variance?

## RKHS and kernel methods
- Can RKHS approximation methods (e.g. random features, Nystrom) efficiently represent the histogram
  log-correction functions delta_{j,y}(q) with provable approximation guarantees, rather than the anchor-difference
  basis used here (which has no such guarantee)?

## Statistical estimation / compression architecture
- How should approximation capacity (CRISP's rank r) be allocated across clusters under a FIXED total memory
  budget, given R4's finding that required rank varies by cluster -- and does the ANSWER change once global
  retrieval frequency (not just within-cluster difficulty) is accounted for, since a rarely-selected cluster's
  compression error matters less to the global result?
- Is a compressed mass predictor (S7, not implemented here) worth its own storage cost, given the measured
  tv_S3_S5 / tv_S5_S6 gaps? This script's own numbers are the evidence to judge this with.

## Experimental design
- R5's main comparison is confirmatory-but-not-fully-independent (rank=8 and the logit-correction representation
  were chosen partly by looking at R1-R4's own val-set results). What would a genuinely held-out confirmatory
  test require, given this project's only two splits (controller_train, val)?
