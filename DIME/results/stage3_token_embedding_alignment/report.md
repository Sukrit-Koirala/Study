# DIME Stage 3 -- Token Embedding-Predictive Relationship Alignment

Matched config (k=20, tau=2.0, alpha=0.25), 77089 val queries, 20 centroids selected across 4 geometric strata + 4 degradation-based picks.

## Experiment A -- embedding vs. predictive similarity
Mean embedding-vs-predictive ranking Spearman correlation across 12 valid centroids: **0.15868343493639384** (std 0.10950021288611757).

## Experiment B -- neighborhood overlap
- k=3: mean embedding/predictive neighbor overlap = 0.278 (random baseline = 0.213)
- k=5: mean embedding/predictive neighbor overlap = 0.420 (random baseline = 0.357)

## Experiment C -- predictive discrimination failures
30 token pairs found with high embedding similarity (>0.5) but low predictive-profile similarity (<0.1), across 9 centroids.

## Experiment D -- global vs. centroid-local consistency
20 tokens appeared in >=2 selected centroids; 3 had sufficient shared comparison tokens for a cross-region check (mean cross-region Spearman = 0.38809523809523805).

## Answers to the four research questions (from the measurements above only)
**Q1** (do embeddings encode predictive relationships?): mean ranking correlation 0.15868343493639384 -- interpret directly from this number and the full per-centroid table in summary.json, not asserted here.
**Q2** (consistent across centroids, or context-dependent?): see Experiment D's cross-region Spearman distribution in summary.json.
**Q3** (do embeddings predict DIME's specific lost distinctions?): see Experiment C's case count and region_context entries.
**Q4** (next step -- direct embedding graph, NCA, or other?): not pre-decided here; judge from Q1-Q3's actual measured evidence once reviewed.

## Limitations
- 20 of 3811 centroids sampled (bounded for cost); per-centroid token sets bounded to top-15 most frequent tokens.
- Experiment C's `region_context` reports region-level mean JS divergence and mean delta_L (raw-vs-DIME full distributions), not a per-pair causal attribution of either to specific token pairs.
- Correlation does not imply causation anywhere in this report.