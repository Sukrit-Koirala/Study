# Failure Cases

## Clusters where label-conditioned did worse than label-agnostic (all-member TV)

4/10 clusters.

```
 cluster  tv_all_member_conditioned  tv_all_member_agnostic
     883                   0.159685                0.128057
    1118                   0.214654                0.157440
    2533                   0.212024                0.210029
    2534                   0.275502                0.204759
```

## Clusters where token-conditioned geometry was less compact than label-agnostic

4/10 clusters.

## Tokens whose own residuals were not more compact than a random same-size baseline

5/24 analyzed tokens (compactness_ratio >= 1.0 means the token-conditioned group is no tighter than an equal-size random group of cluster members -- a direct counterexample to the compressibility hypothesis where it occurs).

