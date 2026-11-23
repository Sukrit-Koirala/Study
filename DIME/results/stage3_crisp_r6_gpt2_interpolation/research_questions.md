# CRISP-R6 Research Questions for Subsequent Mathematical Study

## Proper scoring rules / information theory
- Observation 1 shows TV/JS improvement does not guarantee NLL improvement. Is there a PRECISE condition
  (e.g. on the correlation between a method's per-token support coverage and its probability mass on rare
  correct tokens) under which a TV/JS improvement DOES guarantee an NLL improvement? Keywords: proper scoring
  rules, Bregman divergence, calibration-sharpness decomposition.
- How much residual sensitivity to near-zero retrieval probability remains as alpha approaches 1 (Observation 2)?
  Is there a closed-form expression for d(NLL)/d(alpha) at alpha=1 in terms of the retrieval distribution's mass
  on the true token?

## Approximation theory / statistical estimation
- Is there a formal regret bound relating CRISP's within-cluster TV error (measured in R4) to its NLL-space
  "fraction of oracle headroom recovered" (Observation 3)? Currently only an empirical correlation can be
  measured, not a provable bound.
- Given R5's finding that required CRISP rank varies by cluster, and R6's finding of which clusters/confidence
  groups dominate the NLL gap, would a DIFFERENT (NLL-aware, not TV-aware) capacity allocation rule change which
  clusters are prioritized for higher rank?

## Experimental design
- R6's "individually tuned alpha" comparison is explicitly exploratory (no untouched third split exists). What
  is the smallest additional held-out set this project could carve out (e.g. a contiguous document-level
  held-back block) that would make a genuinely independent tuned-alpha confirmation possible?
- The confidence-group and support-group analyses (Experiment C) use predetermined quartile bins -- would a
  continuous (non-binned) regression of delta-NLL on p_lm_true reveal a smoother, more precise relationship?

## Mechanism
- Experiment D's "aggregation imposes a major limitation" check compares M4-vs-M1 against M2-vs-M1 at a fixed,
  predetermined threshold (30%). Is there a principled (not arbitrary) threshold derivable from the geometry of
  DIME's centroid partition (e.g. related to average within-cluster radius vs. between-cluster separation)?
