# C2.1 Sampling Report

## Cluster sampling

- Stratification axes (from C1's `cluster_statistics.csv`, never recomputed): ['n_j', 'rms_dispersion', 'distinct_token_fraction', 'traffic_top20_val']
- Bins per axis: 3 (total possible cells: 81)
- Eligible (nonempty) clusters: 3811
- Occupied cells: 77; cells kept after budget subsampling: 60 (cell_keep_prob=0.7792207792207793)
- Clusters sampled via stratification: 60
- Random seed: 42 (shuffle before per-cell representative selection -- NOT always the lowest id)

## Unusual/extreme clusters (labeled separately, not part of the stratified count)

- cluster 647: max_population
- cluster 3323: max_dispersion
- cluster 15: max_singleton_token_fraction
- cluster 1598: max_traffic_top20_val
- cluster 92: min_population_above_singleton
- cluster 3072: max_V_j
- cluster 13: max_top_token_fraction

## Selected cluster IDs (sorted, union of stratified + unusual)

13, 15, 87, 92, 93, 149, 214, 220, 286, 354, 393, 599, 647, 776, 869, 1006, 1027, 1032, 1135, 1159, 1352, 1378, 1402, 1421, 1499, 1512, 1553, 1586, 1588, 1598, 1820, 1897, 2016, 2018, 2066, 2140, 2250, 2252, 2266, 2272, 2329, 2400, 2407, 2530, 2531, 2689, 2715, 2736, 2912, 2913, 2981, 3004, 3013, 3053, 3055, 3072, 3122, 3159, 3166, 3250, 3280, 3323, 3333, 3410, 3449, 3760, 3798

## Per-cluster stratum, cell population, and approximate inclusion probability

Review point 5: `cell` is this cluster's bin-tuple over the stratification axes; `cell_population` is how many ELIGIBLE clusters shared that cell (not just the one chosen); `inclusion_probability` is `(1/cell_population) * cell_keep_prob` for clusters reached via the random stratified mechanism (`sampling_mode=stratified` or `both`) -- an approximation (it ignores that a cluster could in principle occupy a cell with unusual marginal properties), not an exact design-based weight. Deliberately-included unusual clusters (`sampling_mode=unusual`) have NO inclusion probability -- they were not randomly selected, and must not be pooled into population-level/representativeness estimates. `structural_relationships.csv` reports a `stratified_only` scope for exactly this reason.

 cluster sampling_mode         cell  cell_population  inclusion_probability
      13       unusual (0, 0, 0, 0)              157                    NaN
      15       unusual (0, 2, 2, 0)              221                    NaN
      87    stratified (2, 2, 1, 0)                2               0.389610
      92       unusual (0, 0, 2, 0)               80                    NaN
      93    stratified (2, 0, 0, 1)               47               0.016579
     149    stratified (0, 2, 1, 0)              135               0.005772
     214    stratified (2, 1, 2, 1)               25               0.031169
     220    stratified (0, 2, 2, 0)              221               0.003526
     286    stratified (0, 2, 1, 2)                3               0.259740
     354    stratified (2, 1, 1, 1)               38               0.020506
     393    stratified (2, 0, 2, 1)               12               0.064935
     599    stratified (1, 0, 0, 2)               32               0.024351
     647       unusual (2, 1, 1, 2)              197                    NaN
     776    stratified (0, 1, 1, 2)                2               0.389610
     869    stratified (1, 1, 0, 0)               22               0.035419
    1006    stratified (2, 1, 0, 1)               41               0.019005
    1027    stratified (2, 0, 2, 2)               85               0.009167
    1032    stratified (1, 2, 0, 1)               81               0.009620
    1135    stratified (2, 0, 1, 1)               25               0.031169
    1159    stratified (1, 2, 1, 0)               28               0.027829
    1352    stratified (2, 1, 1, 0)                1               0.779221
    1378    stratified (0, 1, 2, 0)               48               0.016234
    1402    stratified (2, 0, 0, 0)                7               0.111317
    1421    stratified (1, 0, 1, 2)               38               0.020506
    1499    stratified (2, 1, 0, 2)              147               0.005301
    1512    stratified (0, 1, 1, 1)               31               0.025136
    1553    stratified (0, 2, 0, 1)               29               0.026870
    1586    stratified (1, 1, 0, 2)               37               0.021060
    1588    stratified (1, 0, 1, 1)               64               0.012175
    1598       unusual (2, 0, 2, 2)               85                    NaN
    1820    stratified (2, 1, 2, 2)              142               0.005487
    1897    stratified (1, 2, 2, 0)               56               0.013915
    2016    stratified (1, 0, 2, 0)               15               0.051948
    2018    stratified (0, 0, 1, 0)               89               0.008755
    2066    stratified (0, 1, 2, 2)                2               0.389610
    2140    stratified (2, 1, 0, 0)                2               0.389610
    2250    stratified (0, 0, 1, 1)               38               0.020506
    2252    stratified (0, 1, 2, 1)               30               0.025974
    2266    stratified (1, 1, 0, 1)               75               0.010390
    2272    stratified (1, 2, 2, 1)              137               0.005688
    2329    stratified (2, 2, 2, 2)               94               0.008290
    2400    stratified (1, 2, 2, 2)               17               0.045837
    2407    stratified (0, 2, 1, 1)               24               0.032468
    2530    stratified (0, 0, 0, 2)                2               0.389610
    2531    stratified (1, 1, 1, 0)               23               0.033879
    2689    stratified (1, 2, 1, 2)               16               0.048701
    2715    stratified (1, 2, 0, 2)               10               0.077922
    2736    stratified (0, 0, 2, 1)               33               0.023613
    2912    stratified (1, 1, 2, 1)               61               0.012774
    2913    stratified (1, 0, 0, 0)               73               0.010674
    2981    stratified (1, 0, 2, 1)               21               0.037106
    3004    stratified (1, 1, 1, 2)               52               0.014985
    3013    stratified (1, 1, 2, 2)               41               0.019005
    3053    stratified (1, 0, 0, 1)               98               0.007951
    3055    stratified (0, 1, 0, 0)               37               0.021060
    3072       unusual (2, 0, 2, 2)               85                    NaN
    3122    stratified (2, 2, 1, 2)               63               0.012369
    3159    stratified (1, 2, 0, 0)               35               0.022263
    3166    stratified (0, 0, 0, 1)               53               0.014702
    3250    stratified (2, 2, 0, 1)               20               0.038961
    3280    stratified (0, 0, 1, 2)                8               0.097403
    3323       unusual (0, 2, 2, 0)              221                    NaN
    3333    stratified (2, 2, 1, 1)               10               0.077922
    3410    stratified (2, 0, 1, 2)              117               0.006660
    3449    stratified (2, 0, 2, 0)                3               0.259740
    3760    stratified (0, 2, 2, 2)                1               0.779221
    3798    stratified (0, 1, 0, 2)                2               0.389610

## Per-cluster query availability and weighting convention

Primary within-cluster descriptive analysis treats sampled routed val query-cluster pairs UNIFORMLY (unbiased within-cluster sampling of Q_C^(20)); centroid mixture-weighted aggregates (`massweighted_*` columns) are reported separately in `cluster_behavior_statistics.csv` and never averaged together with the uniform-weight (`mean_*`) aggregates under the same column name.

 cluster  n_j  n_val_available  n_val_sampled  n_ct_available  n_ct_sampled  is_trivial_selection
      13   27               65             65              60            60                 False
      15   15              115            100             179           100                  True
      87  149              264            100             303           100                 False
      92    2              166            100             248           100                  True
      93  125              358            100             379           100                 False
     149   13               61             61              69            69                  True
     214  121              312            100             400           100                 False
     220   62              166            100             182           100                 False
     286   61              476            100             505           100                 False
     354  128              295            100             390           100                 False
     393  121              443            100             500           100                 False
     599   83              573            100             677           100                 False
     647  368             1220            100            1461           100                 False
     776   51              527            100             706           100                 False
     869   68              109            100             118           100                 False
    1006  124              432            100             448           100                 False
    1027  252             1032            100            1171           100                 False
    1032   75              342            100             457           100                 False
    1135  125              441            100             557           100                 False
    1159   66              220            100             255           100                 False
    1352  126              195            100             310           100                 False
    1378   42               99             99              89            89                 False
    1402  124              241            100             322           100                 False
    1421   99              514            100             575           100                 False
    1499  198              971            100            1164           100                 False
    1512   59              356            100             442           100                 False
    1553   65              284            100             321           100                 False
    1586   74              487            100             595           100                 False
    1588  116              362            100             442           100                 False
    1598  316             1396            100            1612           100                 False
    1820  271             1112            100            1346           100                 False
    1897   82              264            100             318           100                 False
    2016   86              231            100             264           100                 False
    2018   41              173            100             192           100                 False
    2066   52              685            100             751           100                 False
    2140  128              255            100             319           100                 False
    2250   49              389            100             353           100                 False
    2252   61              460            100             516           100                 False
    2266   81              407            100             430           100                 False
    2272   75              310            100             319           100                 False
    2329  170              624            100             745           100                 False
    2400   96              563            100             652           100                 False
    2407   64              385            100             506           100                 False
    2530   25              501            100             572           100                 False
    2531   70              230            100             341           100                 False
    2689   74              505            100             599           100                 False
    2715   69              525            100             658           100                 False
    2736   52              284            100             392           100                 False
    2912   87              394            100             420           100                 False
    2913  101              248            100             293           100                 False
    2981   93              425            100             576           100                 False
    3004   92              645            100             743           100                 False
    3013  111              621            100             760           100                 False
    3053   80              294            100             338           100                 False
    3055   29               81             81             110           100                 False
    3072  312             1016            100            1265           100                 False
    3122  214             1031            100            1236           100                 False
    3159   69              157            100             176           100                 False
    3166   63              427            100             524           100                 False
    3250  125              311            100             356           100                 False
    3280   63              493            100             567           100                 False
    3323   22               39             39              33            33                 False
    3333  137              451            100             535           100                 False
    3410  226             1063            100            1192           100                 False
    3449  122              216            100             284           100                 False
    3760   63              596            100             674           100                 False
    3798   30              577            100             682           100                 False

## Exclusions

3751 eligible clusters not selected by stratification.
