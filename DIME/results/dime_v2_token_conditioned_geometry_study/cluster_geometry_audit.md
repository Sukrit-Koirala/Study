# Cluster Geometry Audit (Phase A1/A2)

Sampled 10 clusters, rho_j-stratified (low/mid/high average observations per token).

## Composition (Phase A1)

```
           cluster         n_j  n_distinct_tokens      rho_j    entropy  singleton_frac  top_label_mass_frac
count    10.000000   10.000000           10.00000  10.000000  10.000000       10.000000            10.000000
mean   1938.000000  110.400000           63.90000   8.019513   3.495439        0.751535             0.193541
std     964.617368   27.305677           36.95177  19.858673   1.361552        0.162880             0.289757
min     883.000000   62.000000            2.00000   1.024390   0.045395        0.500000             0.029412
25%    1312.250000   89.250000           43.50000   1.241177   3.447658        0.604911             0.053704
50%    1404.000000  114.000000           58.00000   1.581845   3.645412        0.772619             0.083352
75%    2533.750000  134.250000           81.25000   2.313095   4.339133        0.863495             0.195663
max    3531.000000  143.000000          120.00000  64.500000   4.731673        0.987805             0.992248
```

## Within-token geometry vs. size-matched random baseline (Phase A2)

Fraction of analyzed tokens whose own residuals are more compact than an equal-size random same-cluster baseline: 0.7916666666666666

```
       mean_dist_to_centroid  effective_dim_90pct  compactness_ratio  multimodality_reduction_ratio
count              24.000000            24.000000          24.000000                       9.000000
mean               13.888808             7.250000           0.925475                       0.176376
std                 1.670766             4.618018           0.097994                       0.071613
min                10.633242             3.000000           0.721126                       0.104440
25%                12.863202             4.000000           0.883325                       0.121170
50%                14.202801             5.000000           0.960193                       0.147518
75%                15.159805             9.000000           0.991207                       0.193259
max                16.591264            17.000000           1.032121                       0.294217
```

