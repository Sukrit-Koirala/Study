# Structural Summary (Part 3/4/6)

Configured clusters: 3811; nonempty: 3811; empty: 0; singleton (n_j=1): 3.

Population mean=100.00708475465757, median=91.0, p5=19.0, p95=215.0, Lorenz imbalance gap=0.16836678944911296.

- fraction of clusters with population lt_20: cluster-weighted=0.05221726580949882, observation-weighted=0.005911415354986658
- fraction of clusters with population lt_60: cluster-weighted=0.2896877459984256, observation-weighted=0.10283448824145232

## Population-stratified summary (ALL nonempty clusters, 5 quintiles)

- quintile 0 (n_j in [1, 46], 783 clusters): mean R_j=24.735753348174267, mean sigma_j=16.438833845280573, mean V_j=17.320561941251597, mean rho_j=3.6916422419928185, mean entropy=2.2796090794075234
- quintile 1 (n_j in [47, 75], 745 clusters): mean R_j=24.66516092375113, mean sigma_j=15.763166119629032, mean V_j=36.11543624161074, mean rho_j=4.155551772606443, mean entropy=3.026246911078059
- quintile 2 (n_j in [76, 108], 768 clusters): mean R_j=24.23134791389758, mean sigma_j=15.6241126071901, mean V_j=53.99609375, mean rho_j=4.316890454731932, mean entropy=3.400665472909969
- quintile 3 (n_j in [109, 148], 760 clusters): mean R_j=24.18755601893745, mean sigma_j=15.77010467188436, mean V_j=77.33157894736843, mean rho_j=3.0591642967174453, mean entropy=3.7941941284083147
- quintile 4 (n_j in [149, 368], 755 clusters): mean R_j=23.222682861845033, mean sigma_j=15.559835951800894, mean V_j=119.12582781456953, mean rho_j=2.023067193014639, mean entropy=4.188651629849827

Geometric compactness and token repetition are reported here as DESCRIPTIVE covariates only -- this script draws no conclusion that either implies predictive compressibility (Parts 4.6/6.2/6.4/8).

## Part 5 dimensionality sample coverage (2-D population x dispersion grid)

40/40 grid cells occupied; 40 clusters sampled (one per occupied cell, lowest cluster id, deterministic). Full per-cell population and the exact sampled cluster id are in dimensionality_sample_strata.csv -- NOT summarized here as an all-cluster statistic, since it describes only the bounded sample.

Sampled cluster IDs: [0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 13, 15, 16, 18, 22, 24, 28, 29, 30, 31, 32, 36, 38, 41, 46, 47, 49, 55, 58, 62, 64, 68, 75, 78, 90, 95, 127, 157, 168, 671]
